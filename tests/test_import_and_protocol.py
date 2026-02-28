from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cvmmap = import_module("cvmmap")


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
