from importlib import import_module
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cvmmap = import_module("cvmmap")
cvmmap_msg = import_module("cvmmap.msg")


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "protocol"
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


def _load_core_uri_fixture() -> dict:
    return json.loads(CORE_FIXTURE_PATH.read_text(encoding="utf-8"))


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
    """Assert control message headers match C++ struct sizes exactly.

    C++ static_asserts:
    - sizeof(control_message_request_t) == 36
    - sizeof(control_message_response_t) == 40
    """
    assert cvmmap.ControlMessageRequest.header_size() == 36
    assert cvmmap.ControlMessageResponse.header_size() == 40


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
    assert metadata.depth_plane(left_payload) is None


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
    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert depth.strides == (12, 4)
    assert np.isclose(depth[0, 0], 1.5)
    assert np.isclose(depth[0, 1], 2.5)
    assert np.isclose(depth[1, 0], 3.5)
    assert np.isclose(depth[1, 1], 4.5)


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
    assert client.zmq_body_addr == "ipc:///tmp/cvmmap_example_body"


def test_uri_target_custom_prefix_namespace() -> None:
    client = cvmmap.CvMmapClient("cvmmap://camera0@/run/cvmmap?namespace=zed")
    assert client.shm_name == "zed_camera0"
    assert client.zmq_addr == "ipc:///run/cvmmap/zed_camera0"
    assert client.zmq_body_addr == "ipc:///run/cvmmap/zed_camera0_body"


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
        assert client.zmq_body_addr == case["zmq_body_addr"]

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
    assert frame.label == "example"
    assert len(frame.bodies) == 1
    assert frame.bodies[0].id == 7
    assert np.allclose(frame.bodies[0].position, np.array([1.0, 2.0, 3.0]))
    assert frame.bodies[0].keypoint_count == 1
    assert frame.bodies[0].flags == 2


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
