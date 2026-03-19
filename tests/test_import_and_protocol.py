import asyncio
from importlib import import_module
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cvmmap = import_module("cvmmap")
cvmmap_msg = import_module("cvmmap.msg")
control_pb2 = import_module("cvmmap.control_pb2")


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "protocol"
CORE_PROTOCOL_FIXTURE_DIR = Path(
    os.environ.get(
        "CVMMAP_CORE_PROTOCOL_FIXTURE_DIR",
        Path(__file__).resolve().parents[1].parent
        / "cv-mmap"
        / "core"
        / "fixtures"
        / "protocol",
    )
)
CORE_FIXTURE_PATH = Path(
    os.environ.get(
        "CVMMAP_CORE_URI_FIXTURE",
        Path(__file__).resolve().parent
        / "fixtures"
        / "cvmmap_core"
        / "uri_targets.json",
    )
)


def _load_fixture_bytes(filename: str) -> bytes:
    return bytes.fromhex((FIXTURE_DIR / filename).read_text(encoding="utf-8"))


def _patch_v2_depth_unit(metadata_region: bytes, depth_unit: int) -> bytes:
    patched = bytearray(metadata_region)
    patched[0x2C] = depth_unit
    return bytes(patched)


def _load_core_uri_fixture() -> dict:
    return json.loads(CORE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _require_core_protocol_fixture_dir() -> Path:
    if CORE_PROTOCOL_FIXTURE_DIR.is_dir():
        return CORE_PROTOCOL_FIXTURE_DIR
    pytest.skip(
        f"cv-mmap core protocol fixtures not available: {CORE_PROTOCOL_FIXTURE_DIR}"
    )


def _load_core_protocol_fixture_bytes(filename: str) -> bytes:
    fixture_dir = _require_core_protocol_fixture_dir()
    return (fixture_dir / filename).read_bytes()


def _load_core_protocol_manifest() -> dict:
    fixture_dir = _require_core_protocol_fixture_dir()
    return json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))


class _FakeSharedMemory:
    def __init__(self, data: bytes) -> None:
        self.buf = memoryview(data)


class _FutureReturningSocket:
    def __init__(self, message: bytes) -> None:
        self._message = message
        self.closed = False

    def recv(self) -> asyncio.Future[bytes]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()
        future.set_result(self._message)
        return future

    def close(self) -> None:
        self.closed = True


def test_import_core_symbols() -> None:
    assert getattr(cvmmap, "CvMmapClient") is not None
    assert getattr(cvmmap, "FrameInfo") is not None


def test_sync_message_roundtrip() -> None:
    sync_message_type = getattr(cvmmap, "SyncMessage")
    msg = sync_message_type(frame_count=42, timestamp_ns=123456789, label="camera_0")
    encoded = msg.marshal()
    decoded = sync_message_type.unmarshal(encoded)

    assert decoded.frame_count == 42
    assert decoded.timestamp_ns == 123456789
    assert decoded.label == "camera_0"


def test_protocol_struct_sizes() -> None:
    sync_message_type = getattr(cvmmap, "SyncMessage")
    frame_info_type = getattr(cvmmap, "FrameInfo")

    assert sync_message_type.size() == 48
    assert frame_info_type.size() == 12
    assert cvmmap.BodyTrackingMessageHeader.size() == 64
    assert cvmmap.BodyTrack.size() == 3248
    assert cvmmap.FrameMetadataV2Header.size() == 64
    assert cvmmap.FramePlaneDescriptorV2.size() == 24
    assert cvmmap.FrameMetadataV2.size() == 256


def test_control_message_header_sizes() -> None:
    """Assert control message headers match the actual wire layout.

    The producer C++ structs are larger because of unsent tail padding after the
    flexible-array length fields:

    - request wire header: 34 bytes, C++ sizeof(...) == 36
    - response wire header: 38 bytes, C++ sizeof(...) == 40
    """
    assert cvmmap.ControlMessageRequest.header_size() == 34
    assert cvmmap.ControlMessageResponse.header_size() == 38


def test_control_request_and_response_payload_sizes() -> None:
    assert cvmmap.SourceInfo.size() == 48
    assert cvmmap.SeekTimestampRequest.size() == 12
    assert cvmmap.SeekResult.size() == 28
    assert cvmmap.RecordingStartRequest.size() == 8
    assert cvmmap.RecordingStatus.size() == 20


def test_source_info_roundtrip() -> None:
    payload = struct.pack(
        cvmmap.SourceInfo.PACK_FMT,
        cvmmap.SourceInfo.size(),
        cvmmap.SOURCE_KIND_FINITE,
        cvmmap.TIMESTAMP_DOMAIN_UNIX_EPOCH_NS,
        cvmmap.SOURCE_INFO_FLAG_CAN_SEEK
        | cvmmap.SOURCE_INFO_FLAG_AUTO_LOOP
        | cvmmap.SOURCE_INFO_FLAG_HAS_DEPTH
        | cvmmap.SOURCE_INFO_FLAG_HAS_BODY
        | cvmmap.SOURCE_INFO_FLAG_CAN_RECORD,
        100,
        250,
        150,
        175,
        9,
        0,
    )

    parsed = cvmmap.SourceInfo.unmarshal(payload)
    assert parsed.source_kind == cvmmap.SOURCE_KIND_FINITE
    assert parsed.timestamp_domain == cvmmap.TIMESTAMP_DOMAIN_UNIX_EPOCH_NS
    assert parsed.timeline_start_ns == 100
    assert parsed.timeline_end_ns == 250
    assert parsed.duration_ns == 150
    assert parsed.current_timestamp_ns == 175
    assert parsed.current_frame_count == 9
    assert parsed.can_seek is True
    assert parsed.auto_loop is True
    assert parsed.has_depth is True
    assert parsed.has_body is True
    assert parsed.can_record is True


def test_seek_payload_roundtrip() -> None:
    request = cvmmap.SeekTimestampRequest(target_timestamp_ns=123456789)
    assert request.marshal() == struct.pack(
        cvmmap.SeekTimestampRequest.PACK_FMT,
        cvmmap.SeekTimestampRequest.size(),
        0,
        123456789,
    )

    payload = struct.pack(
        cvmmap.SeekResult.PACK_FMT,
        cvmmap.SeekResult.size(),
        1,
        0,
        123456789,
        123456999,
        0,
        0,
    )
    parsed = cvmmap.SeekResult.unmarshal(payload)
    assert parsed.requested_timestamp_ns == 123456789
    assert parsed.landed_timestamp_ns == 123456999
    assert parsed.landed_frame_count == 0
    assert parsed.exact_match is True


def test_recording_payload_roundtrip() -> None:
    request = cvmmap.RecordingStartRequest(output_path="/tmp/example.svo2")
    assert request.marshal() == struct.pack(
        cvmmap.RecordingStartRequest.PACK_FMT,
        cvmmap.RecordingStartRequest.size(),
        0,
        len(b"/tmp/example.svo2"),
        0,
    ) + b"/tmp/example.svo2"

    payload = struct.pack(
        cvmmap.RecordingStatus.PACK_FMT,
        cvmmap.RecordingStatus.size(),
        cvmmap.RECORDING_FORMAT_SVO,
        0,
        cvmmap.RECORDING_STATUS_FLAG_CAN_RECORD
        | cvmmap.RECORDING_STATUS_FLAG_IS_RECORDING
        | cvmmap.RECORDING_STATUS_FLAG_LAST_FRAME_OK,
        len(b"/tmp/example.svo2"),
        42,
        40,
        0,
    ) + b"/tmp/example.svo2"
    parsed = cvmmap.RecordingStatus.unmarshal(payload)
    assert parsed.recording_format == cvmmap.RECORDING_FORMAT_SVO
    assert parsed.can_record is True
    assert parsed.is_recording is True
    assert parsed.is_paused is False
    assert parsed.last_frame_ok is True
    assert parsed.frames_ingested == 42
    assert parsed.frames_encoded == 40
    assert parsed.active_path == "/tmp/example.svo2"


def test_recording_start_request_rejects_empty_path() -> None:
    try:
        cvmmap.RecordingStartRequest(output_path="").marshal()
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected empty recording output path to be rejected")


def test_control_response_truncated_payload_rejected() -> None:
    header = struct.pack(
        cvmmap.ControlMessageResponse.marshal_format(),
        cvmmap_msg.CONTROL_MESSAGE_RESPONSE_MAGIC,
        cvmmap_msg.VERSION_MAJOR,
        cvmmap_msg.VERSION_MINOR,
        cvmmap_msg.CONTROL_MSG_CMD_GET_SOURCE_INFO,
        cvmmap_msg.CONTROL_RESPONSE_OK,
        b"example".ljust(cvmmap_msg.LABEL_LEN_MAX, b"\0"),
        48,
    )
    try:
        cvmmap.ControlMessageResponse.unmarshal(header + (b"\0" * 47))
    except ValueError as exc:
        assert "response payload" in str(exc)
    else:
        raise AssertionError("Expected truncated response payload to be rejected")


def test_cpp_sync_fixture_roundtrip() -> None:
    fixture = _load_core_protocol_manifest()["sync_valid"]
    payload = _load_core_protocol_fixture_bytes(fixture["file"])

    decoded = cvmmap.SyncMessage.unmarshal(payload)

    assert len(payload) == fixture["size"]
    assert decoded.frame_count == fixture["frame_count"]
    assert decoded.timestamp_ns == fixture["timestamp_ns"]
    assert decoded.label == fixture["label"]
    assert (
        payload
        == cvmmap.SyncMessage(
            frame_count=fixture["frame_count"],
            timestamp_ns=fixture["timestamp_ns"],
            label=fixture["label"],
        ).marshal()
    )


def test_cpp_control_get_source_info_fixtures_parse() -> None:
    manifest = _load_core_protocol_manifest()
    request_fixture = manifest["control_request_get_source_info"]
    response_fixture = manifest["control_response_get_source_info"]

    request_payload = _load_core_protocol_fixture_bytes(request_fixture["file"])
    response_payload = _load_core_protocol_fixture_bytes(response_fixture["file"])

    assert len(request_payload) == request_fixture["size"]
    assert request_payload == cvmmap.ControlMessageRequest(
        label=request_fixture["label"],
        command_id=request_fixture["command_id"],
        request_message=b"",
    ).marshal()

    response = cvmmap.ControlMessageResponse.unmarshal(response_payload)
    source_info = cvmmap.SourceInfo.unmarshal(response.response_message)
    expected = response_fixture["source_info"]

    assert len(response_payload) == response_fixture["size"]
    assert response.command_id == response_fixture["command_id"]
    assert response.response_code == response_fixture["response_code"]
    assert response.label == response_fixture["label"]
    assert source_info.source_kind == expected["source_kind"]
    assert source_info.timestamp_domain == expected["timestamp_domain"]
    assert source_info.flags == expected["flags"]
    assert source_info.timeline_start_ns == expected["timeline_start_ns"]
    assert source_info.timeline_end_ns == expected["timeline_end_ns"]
    assert source_info.duration_ns == expected["duration_ns"]
    assert source_info.current_timestamp_ns == expected["current_timestamp_ns"]
    assert source_info.current_frame_count == expected["current_frame_count"]


def test_cpp_control_seek_fixtures_parse() -> None:
    manifest = _load_core_protocol_manifest()
    request_fixture = manifest["control_request_seek_timestamp_ns"]
    response_fixture = manifest["control_response_seek_timestamp_ns"]

    request_payload = _load_core_protocol_fixture_bytes(request_fixture["file"])
    response_payload = _load_core_protocol_fixture_bytes(response_fixture["file"])

    expected_request = cvmmap.SeekTimestampRequest(
        target_timestamp_ns=request_fixture["target_timestamp_ns"]
    ).marshal()
    assert len(request_payload) == request_fixture["size"]
    assert request_payload == cvmmap.ControlMessageRequest(
        label=request_fixture["label"],
        command_id=request_fixture["command_id"],
        request_message=expected_request,
    ).marshal()

    response = cvmmap.ControlMessageResponse.unmarshal(response_payload)
    seek_result = cvmmap.SeekResult.unmarshal(response.response_message)
    expected = response_fixture["seek_result"]

    assert len(response_payload) == response_fixture["size"]
    assert response.command_id == response_fixture["command_id"]
    assert response.response_code == response_fixture["response_code"]
    assert response.label == response_fixture["label"]
    assert seek_result.requested_timestamp_ns == expected["requested_timestamp_ns"]
    assert seek_result.landed_timestamp_ns == expected["landed_timestamp_ns"]
    assert seek_result.landed_frame_count == expected["landed_frame_count"]
    assert seek_result.exact_match is expected["exact_match"]


def test_cpp_control_recording_fixtures_parse() -> None:
    manifest = _load_core_protocol_manifest()
    start_fixture = manifest["control_request_start_recording"]
    stop_fixture = manifest["control_request_stop_recording"]
    get_status_fixture = manifest["control_request_get_recording_status"]
    status_fixture = manifest["control_response_recording_status"]

    start_request_payload = _load_core_protocol_fixture_bytes(start_fixture["file"])
    stop_request_payload = _load_core_protocol_fixture_bytes(stop_fixture["file"])
    get_status_request_payload = _load_core_protocol_fixture_bytes(
        get_status_fixture["file"]
    )
    response_payload = _load_core_protocol_fixture_bytes(status_fixture["file"])

    expected_request = cvmmap.RecordingStartRequest(
        output_path=start_fixture["output_path"]
    ).marshal()
    assert len(start_request_payload) == start_fixture["size"]
    assert start_request_payload == cvmmap.ControlMessageRequest(
        label=start_fixture["label"],
        command_id=start_fixture["command_id"],
        request_message=expected_request,
    ).marshal()
    assert len(stop_request_payload) == stop_fixture["size"]
    assert stop_request_payload == cvmmap.ControlMessageRequest(
        label=stop_fixture["label"],
        command_id=stop_fixture["command_id"],
        request_message=b"",
    ).marshal()
    assert len(get_status_request_payload) == get_status_fixture["size"]
    assert get_status_request_payload == cvmmap.ControlMessageRequest(
        label=get_status_fixture["label"],
        command_id=get_status_fixture["command_id"],
        request_message=b"",
    ).marshal()

    response = cvmmap.ControlMessageResponse.unmarshal(response_payload)
    recording_status = cvmmap.RecordingStatus.unmarshal(response.response_message)
    expected = status_fixture["recording_status"]

    assert len(response_payload) == status_fixture["size"]
    assert response.command_id == status_fixture["command_id"]
    assert response.response_code == status_fixture["response_code"]
    assert response.label == status_fixture["label"]
    assert recording_status.recording_format == expected["recording_format"]
    assert recording_status.flags == expected["flags"]
    assert recording_status.active_path == expected["path"]
    assert recording_status.frames_ingested == expected["frames_ingested"]
    assert recording_status.frames_encoded == expected["frames_encoded"]


def test_request_client_recording_methods() -> None:
    async def _run() -> None:
        client = object.__new__(cvmmap.CvMmapRequestClient)
        client._target_key = "cvmmap_example"  # type: ignore[attr-defined]
        client._nats = None  # type: ignore[attr-defined]

        async def _ok_request_pb(
            subject: str,
            request_message,
            response_type,
            timeout_ms: int,
        ):
            assert timeout_ms == 5000
            if subject.endswith(".start"):
                assert subject == "cvmmap.cvmmap_example.control.recorder.svo.start"
                assert request_message.output_path == "/tmp/test.svo2"
            response = response_type()
            response.error = control_pb2.ERROR_CODE_OK
            response.format = control_pb2.RECORDING_FORMAT_SVO
            response.can_record = True
            response.is_recording = True
            response.frames_ingested = 10
            response.frames_encoded = 9
            response.active_path = "/tmp/test.svo2"
            return response

        client._request_pb = _ok_request_pb  # type: ignore[method-assign]

        started = await client.start_recording("/tmp/test.svo2")
        assert started.is_recording is True
        assert started.active_path == "/tmp/test.svo2"

        stopped = await client.stop_recording()
        assert stopped.frames_ingested == 10

        status = await client.get_recording_status()
        assert status.frames_encoded == 9

        async def _error_request_pb(
            subject: str,
            request_message,
            response_type,
            timeout_ms: int,
        ):
            response = response_type()
            response.error = control_pb2.ERROR_CODE_UNSUPPORTED
            return response

        client._request_pb = _error_request_pb  # type: ignore[method-assign]
        try:
            await client.get_recording_status()
        except RuntimeError as exc:
            assert "GET_RECORDING_STATUS failed with UNSUPPORTED (-7)" == str(exc)
        else:
            raise AssertionError("Expected get_recording_status() to raise")

    asyncio.run(_run())


def test_request_client_capabilities_merge() -> None:
    async def _run() -> None:
        client = object.__new__(cvmmap.CvMmapRequestClient)
        client._target_key = "cvmmap_example"  # type: ignore[attr-defined]
        client._nats = None  # type: ignore[attr-defined]

        async def _request_pb(
            subject: str,
            request_message,
            response_type,
            timeout_ms: int,
        ):
            response = response_type()
            response.error = control_pb2.ERROR_CODE_OK
            if subject.endswith(".source.capabilities"):
                response.can_seek = False
            elif subject.endswith(".recorder.svo.capabilities"):
                response.available_recording_formats.append(
                    control_pb2.RECORDING_FORMAT_SVO
                )
            elif subject.endswith(".recorder.mcap.capabilities"):
                response.available_recording_formats.append(
                    control_pb2.RECORDING_FORMAT_MCAP
                )
            return response

        client._request_pb = _request_pb  # type: ignore[method-assign]

        capabilities = await client.get_capabilities()
        assert capabilities.can_seek is False
        assert capabilities.available_recording_formats == [
            cvmmap.RECORDING_FORMAT_SVO,
            cvmmap.RECORDING_FORMAT_MCAP,
        ]
        assert capabilities.supports_recording_format(cvmmap.RECORDING_FORMAT_SVO)
        assert capabilities.supports_recording_format(cvmmap.RECORDING_FORMAT_MCAP)

    asyncio.run(_run())


def test_cpp_body_tracking_fixture_parse() -> None:
    manifest = _load_core_protocol_manifest()["body_tracking_valid"]
    payload = _load_core_protocol_fixture_bytes(manifest["file"])

    frame = cvmmap_msg.unmarshal_body_tracking_message(payload)
    expected_body = manifest["body"]

    assert len(payload) == manifest["size"]
    assert frame.frame_count == manifest["frame_count"]
    assert frame.timestamp_ns == manifest["timestamp_ns"]
    assert frame.sdk_timestamp_ns == manifest["sdk_timestamp_ns"]
    assert frame.body_count == manifest["body_count"]
    assert frame.body_format == manifest["body_format"]
    assert frame.body_selection == manifest["body_selection"]
    assert frame.detection_model == manifest["detection_model"]
    assert frame.inference_precision == manifest["inference_precision"]
    assert frame.flags == manifest["flags"]
    assert frame.coordinate_system == manifest["coordinate_system"]
    assert frame.reference_frame == manifest["reference_frame"]
    assert frame.floor_as_origin is manifest["floor_as_origin"]
    assert frame.label == manifest["label"]
    assert len(frame.bodies) == 1
    assert frame.bodies[0].id == expected_body["id"]
    assert frame.bodies[0].tracking_state == expected_body["tracking_state"]
    assert frame.bodies[0].action_state == expected_body["action_state"]
    assert frame.bodies[0].confidence == expected_body["confidence"]
    assert frame.bodies[0].keypoint_count == expected_body["keypoint_count"]
    assert frame.bodies[0].flags == expected_body["flags"]
    assert np.allclose(frame.bodies[0].position, np.array(expected_body["position"]))


def test_v1_parse_pass() -> None:
    metadata_region = _load_fixture_bytes("v1_valid_metadata.hex")
    payload = _load_fixture_bytes("v1_valid_payload.hex")
    metadata = cvmmap_msg.unmarshal_frame_metadata(metadata_region)

    assert isinstance(metadata, cvmmap.FrameMetadata)
    assert metadata.frame_count == 77
    assert metadata.timestamp_ns == 1234567890
    assert metadata.info.width == 4
    assert metadata.info.height == 2
    assert metadata.info.channels == 3
    assert metadata.info.buffer_size == 24
    assert metadata.info.buffer_size == len(payload)


def test_v2_left_only_parse_pass() -> None:
    left_payload = _load_fixture_bytes("v2_left_only_valid_payload.hex")
    metadata = cvmmap_msg.unmarshal_frame_metadata(
        _load_fixture_bytes("v2_left_only_valid_metadata.hex")
    )

    assert isinstance(metadata, cvmmap.FrameMetadataV2)
    left = metadata.left_plane(left_payload)
    assert left.shape == (2, 2, 3)
    assert left.strides == (6, 3, 1)
    assert left[0, 0].tolist() == [1, 2, 3]
    assert left[1, 1].tolist() == [10, 11, 12]
    assert metadata.depth_unit == cvmmap.DEPTH_UNIT_UNKNOWN
    assert metadata.depth_plane(left_payload) is None
    assert metadata.confidence_plane(left_payload) is None


def test_v2_left_depth_parse_pass() -> None:
    payload = _load_fixture_bytes("v2_left_depth_valid_payload.hex")
    metadata = cvmmap_msg.unmarshal_frame_metadata(
        _load_fixture_bytes("v2_left_depth_valid_metadata.hex")
    )

    assert isinstance(metadata, cvmmap.FrameMetadataV2)

    left = metadata.left_plane(payload)
    assert left.shape == (2, 2, 3)
    assert left.strides == (8, 3, 1)
    assert left[0, 1].tolist() == [4, 5, 6]
    assert left[1, 1].tolist() == [10, 11, 12]

    depth = metadata.depth_plane(payload)
    assert depth is not None
    assert metadata.depth_unit == cvmmap.DEPTH_UNIT_UNKNOWN
    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert depth.strides == (12, 4)
    assert np.isclose(depth[0, 0], 1.5)
    assert np.isclose(depth[0, 1], 2.5)
    assert np.isclose(depth[1, 0], 3.5)
    assert np.isclose(depth[1, 1], 4.5)
    assert metadata.confidence_plane(payload) is None


def test_v2_left_depth_confidence_parse_pass() -> None:
    payload = _load_fixture_bytes("v2_left_depth_confidence_valid_payload.hex")
    metadata = cvmmap_msg.unmarshal_frame_metadata(
        _load_fixture_bytes("v2_left_depth_confidence_valid_metadata.hex")
    )

    assert isinstance(metadata, cvmmap.FrameMetadataV2)
    assert metadata.confidence_descriptor is not None
    assert metadata.depth_unit == cvmmap.DEPTH_UNIT_UNKNOWN

    confidence = metadata.confidence_plane(payload)
    assert confidence is not None
    assert confidence.shape == (2, 2)
    assert confidence.dtype == np.uint8
    assert confidence.strides == (2, 1)
    assert confidence.tolist() == [[10, 20], [30, 40]]


def test_v2_explicit_depth_unit_parse_pass() -> None:
    payload = _load_fixture_bytes("v2_left_depth_valid_payload.hex")
    metadata_region = _patch_v2_depth_unit(
        _load_fixture_bytes("v2_left_depth_valid_metadata.hex"),
        cvmmap.DEPTH_UNIT_METER,
    )
    metadata = cvmmap_msg.unmarshal_frame_metadata(metadata_region)

    assert isinstance(metadata, cvmmap.FrameMetadataV2)
    assert metadata.header.depth_unit == cvmmap.DEPTH_UNIT_METER
    assert metadata.depth_unit == cvmmap.DEPTH_UNIT_METER
    depth = metadata.depth_plane(payload)
    assert depth is not None
    assert np.isclose(depth[0, 0], 1.5)


def test_client_confidence_plane_helper() -> None:
    client = cvmmap.CvMmapClient("example")

    try:
        metadata_region = _load_fixture_bytes("v2_left_depth_valid_metadata.hex")
        payload = _load_fixture_bytes("v2_left_depth_valid_payload.hex")
        client._shm = _FakeSharedMemory(metadata_region + payload)
        metadata = client._read_metadata()
        assert client.confidence_plane(metadata) is None

        metadata_region = _load_fixture_bytes(
            "v2_left_depth_confidence_valid_metadata.hex"
        )
        payload = _load_fixture_bytes("v2_left_depth_confidence_valid_payload.hex")
        client._shm = _FakeSharedMemory(metadata_region + payload)
        metadata = client._read_metadata()
        confidence = client.confidence_plane(metadata)

        assert confidence is not None
        assert confidence.tolist() == [[10, 20], [30, 40]]
    finally:
        client._sock.close()


def test_client_depth_unit_helper() -> None:
    client = cvmmap.CvMmapClient("example")

    try:
        metadata_region = _patch_v2_depth_unit(
            _load_fixture_bytes("v2_left_depth_valid_metadata.hex"),
            cvmmap.DEPTH_UNIT_MILLIMETER,
        )
        payload = _load_fixture_bytes("v2_left_depth_valid_payload.hex")
        client._shm = _FakeSharedMemory(metadata_region + payload)
        metadata = client._read_metadata()

        assert client.depth_unit(metadata) == cvmmap.DEPTH_UNIT_MILLIMETER
    finally:
        client._sock.close()


def test_client_async_iterator_accepts_future_returning_recv() -> None:
    client = cvmmap.CvMmapClient("example")
    original_sock = client._sock
    metadata_region = _load_fixture_bytes("v2_left_depth_valid_metadata.hex")
    payload = _load_fixture_bytes("v2_left_depth_valid_payload.hex")
    sync_message = cvmmap.SyncMessage(
        frame_count=1,
        timestamp_ns=123456789,
        label="example",
    )

    original_sock.close()
    client._sock = _FutureReturningSocket(sync_message.marshal())
    client._shm = _FakeSharedMemory(metadata_region + payload)
    client._status_subscription_ready = True

    async def _run() -> None:
        frame, metadata = await anext(client.__aiter__())
        assert frame.shape == (2, 2, 3)
        assert frame[1, 1].tolist() == [10, 11, 12]
        depth = client.depth_plane(metadata)
        assert depth is not None
        assert np.isclose(depth[0, 0], 1.5)

    try:
        asyncio.run(_run())
    finally:
        client.close()


def test_invalid_v2_depth_unit_rejected() -> None:
    metadata_region = _patch_v2_depth_unit(
        _load_fixture_bytes("v2_left_depth_valid_metadata.hex"),
        3,
    )

    try:
        cvmmap_msg.unmarshal_frame_metadata(metadata_region)
    except ValueError as exc:
        assert "depth_unit" in str(exc)
    else:
        raise AssertionError("Expected invalid v2 depth_unit to be rejected")


def test_v2_malformed_descriptor_rejected() -> None:
    payload = _load_fixture_bytes("v2_malformed_descriptor_payload.hex")
    malformed_region = _load_fixture_bytes("v2_malformed_descriptor_metadata.hex")
    assert len(payload) == 12

    try:
        cvmmap_msg.unmarshal_frame_metadata(malformed_region)
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("Expected malformed v2 descriptor to be rejected")


def test_uri_target_defaults() -> None:
    client = cvmmap.CvMmapClient("cvmmap://example")
    assert client.shm_name == "cvmmap_example"
    assert client.zmq_addr == "ipc:///tmp/cvmmap_example"
    assert client.nats_target_key == "cvmmap_example"
    client.close()


def test_uri_target_custom_prefix_namespace() -> None:
    client = cvmmap.CvMmapClient("cvmmap://camera0@/run/cvmmap?namespace=zed")
    assert client.shm_name == "zed_camera0"
    assert client.zmq_addr == "ipc:///run/cvmmap/zed_camera0"
    assert client.nats_target_key == "zed_camera0"
    client.close()


def test_plain_name_rejects_uri_chars() -> None:
    try:
        _ = cvmmap.CvMmapClient("bad/name")
    except ValueError as exc:
        assert "plain cvmmap instance names" in str(exc)
    else:
        raise AssertionError("Expected plain name with slash to be rejected")


def test_uri_fixture_conformance() -> None:
    fixture = _load_core_uri_fixture()

    for case in fixture["cases"]:
        client = cvmmap.CvMmapClient(case["input"])
        assert client._name == case["instance"]
        assert client._namespace == case["namespace"]
        assert client._prefix == case["prefix"]
        assert client._base_name == case["base_name"]
        assert client.shm_name == case["shm_name"]
        assert client.zmq_addr == case["zmq_addr"]
        assert client.nats_target_key == case["nats_target_key"]
        client.close()

    for case in fixture["invalid_cases"]:
        try:
            _ = cvmmap.CvMmapClient(case["input"])
        except ValueError as exc:
            assert case["error_contains"] in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid URI fixture to fail: {case['input']}"
            )


def test_body_tracking_message_parse_pass() -> None:
    header_fmt = cvmmap.BodyTrackingMessageHeader.PACK_FMT
    record_size = cvmmap.BodyTrack.size()
    label = b"example".ljust(24, b"\0")

    body_record = bytearray(record_size)
    struct.pack_into("<iBB", body_record, 0, 7, 1, 0)
    struct.pack_into("<f", body_record, 8, 88.5)
    struct.pack_into("<3f", body_record, 12, 1.0, 2.0, 3.0)
    struct.pack_into("<H", body_record, record_size - 4, 1)
    struct.pack_into("<H", body_record, record_size - 2, 2)

    payload = bytes(body_record)
    header = struct.pack(
        header_fmt,
        cvmmap_msg.BODY_TRACKING_MAGIC,
        0,
        cvmmap_msg.VERSION_MAJOR,
        cvmmap_msg.VERSION_MINOR,
        42,
        1000,
        2000,
        1,
        record_size,
        0,
        0,
        2,
        0,
        1,
        0,
        len(payload),
        label,
    )

    frame = cvmmap_msg.unmarshal_body_tracking_message(header + payload)
    assert frame.frame_count == 42
    assert frame.header.body_count == 1
    assert frame.coordinate_system == 0
    assert frame.reference_frame == 0
    assert frame.floor_as_origin is False
    assert frame.label == "example"
    assert len(frame.bodies) == 1
    assert frame.bodies[0].id == 7
    assert np.allclose(frame.bodies[0].position, np.array([1.0, 2.0, 3.0]))
    assert frame.bodies[0].keypoint_count == 1
    assert frame.bodies[0].flags == 2


def test_body_tracking_message_parses_extended_frame_metadata() -> None:
    header_fmt = cvmmap.BodyTrackingMessageHeader.PACK_FMT
    payload = b"\0" * cvmmap_msg.BODY_TRACKING_BODY_RECORD_SIZE

    header = struct.pack(
        header_fmt,
        cvmmap_msg.BODY_TRACKING_MAGIC,
        7,
        cvmmap_msg.VERSION_MAJOR,
        cvmmap_msg.VERSION_MINOR,
        9,
        111,
        222,
        1,
        cvmmap_msg.BODY_TRACKING_BODY_RECORD_SIZE,
        cvmmap_msg.BODY_FORMAT_BODY_38,
        cvmmap_msg.BODY_KEYPOINT_SELECTION_UPPER_BODY,
        cvmmap_msg.BODY_TRACKING_MODEL_HUMAN_BODY_MEDIUM,
        cvmmap_msg.INFERENCE_PRECISION_FP16,
        cvmmap_msg.BODY_TRACKING_FLAG_FLOOR_AS_ORIGIN,
        11,
        len(payload),
        b"meta".ljust(24, b"\0"),
    )

    frame = cvmmap_msg.unmarshal_body_tracking_message(header + payload)

    assert frame.coordinate_system == 7
    assert frame.reference_frame == 11
    assert frame.floor_as_origin is True
    assert frame.header.coordinate_system == 7
    assert frame.header.reference_frame == 11
    assert frame.header.floor_as_origin is True
    assert frame.flags & cvmmap_msg.BODY_TRACKING_FLAG_FLOOR_AS_ORIGIN


def test_body_tracking_message_invalid_record_size_rejected() -> None:
    header = struct.pack(
        cvmmap.BodyTrackingMessageHeader.PACK_FMT,
        cvmmap_msg.BODY_TRACKING_MAGIC,
        0,
        cvmmap_msg.VERSION_MAJOR,
        cvmmap_msg.VERSION_MINOR,
        1,
        100,
        200,
        1,
        12,
        cvmmap_msg.BODY_FORMAT_BODY_18,
        cvmmap_msg.BODY_KEYPOINT_SELECTION_FULL,
        cvmmap_msg.BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE,
        cvmmap_msg.INFERENCE_PRECISION_FP32,
        0,
        0,
        12,
        b"example".ljust(24, b"\0"),
    )

    try:
        cvmmap_msg.unmarshal_body_tracking_message(header + (b"\0" * 12))
    except ValueError as exc:
        assert "body_record_size" in str(exc)
    else:
        raise AssertionError("Expected invalid body record size to be rejected")


def test_body_tracking_message_invalid_payload_size_rejected() -> None:
    header = struct.pack(
        cvmmap.BodyTrackingMessageHeader.PACK_FMT,
        cvmmap_msg.BODY_TRACKING_MAGIC,
        0,
        cvmmap_msg.VERSION_MAJOR,
        cvmmap_msg.VERSION_MINOR,
        1,
        100,
        200,
        1,
        cvmmap_msg.BODY_TRACKING_BODY_RECORD_SIZE,
        cvmmap_msg.BODY_FORMAT_BODY_18,
        cvmmap_msg.BODY_KEYPOINT_SELECTION_FULL,
        cvmmap_msg.BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE,
        cvmmap_msg.INFERENCE_PRECISION_FP32,
        0,
        0,
        12,
        b"example".ljust(24, b"\0"),
    )

    try:
        cvmmap_msg.unmarshal_body_tracking_message(
            header + (b"\0" * cvmmap_msg.BODY_TRACKING_BODY_RECORD_SIZE)
        )
    except ValueError as exc:
        assert "payload_size_bytes" in str(exc)
    else:
        raise AssertionError("Expected invalid payload size to be rejected")
