from importlib import import_module
import json
import os
from pathlib import Path
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


def test_uri_target_custom_prefix_namespace() -> None:
    client = cvmmap.CvMmapClient("cvmmap://camera0@/run/cvmmap?namespace=zed")
    assert client.shm_name == "zed_camera0"
    assert client.zmq_addr == "ipc:///run/cvmmap/zed_camera0"


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

    for case in fixture["invalid_cases"]:
        try:
            _ = cvmmap.CvMmapClient(case["input"])
        except ValueError as exc:
            assert case["error_contains"] in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid URI fixture to fail: {case['input']}"
            )
