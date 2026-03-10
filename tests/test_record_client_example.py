import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import anyio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_record_client_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "record_client.py"
    )
    spec = spec_from_file_location("record_client_example", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_depth_npz_recorder_writes_depth_and_confidence(tmp_path: Path) -> None:
    module = _load_record_client_module()
    recorder = module.DepthNpzRecorder(tmp_path / "sample_depth.npz")

    recorder.append(
        frame_count=10,
        timestamp_ns=1000,
        depth=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        confidence=np.array([[10, 20], [30, 40]], dtype=np.uint8),
    )
    recorder.append(
        frame_count=11,
        timestamp_ns=2000,
        depth=None,
        confidence=None,
    )

    output_path = recorder.write(fps=29.97)

    assert output_path == tmp_path / "sample_depth.npz"
    saved = np.load(output_path)

    assert saved["depth_mm"].shape == (2, 2, 2)
    assert saved["depth_present_mask"].tolist() == [True, False]
    assert saved["depth_units"].item() == "millimeters"
    assert np.isclose(saved["fps"].item(), 29.97)
    assert np.isnan(saved["depth_mm"][1]).all()
    assert saved["timestamp_ns"].tolist() == [1000, 2000]
    assert saved["frame_count"].tolist() == [10, 11]
    assert saved["confidence"].shape == (2, 2, 2)
    assert saved["confidence_present_mask"].tolist() == [True, False]
    assert saved["confidence"][0].tolist() == [[10, 20], [30, 40]]
    assert saved["confidence"][1].tolist() == [[0, 0], [0, 0]]


def test_depth_npz_recorder_skips_when_no_depth(tmp_path: Path) -> None:
    module = _load_record_client_module()
    recorder = module.DepthNpzRecorder(tmp_path / "sample_depth.npz")

    recorder.append(
        frame_count=1,
        timestamp_ns=1000,
        depth=None,
        confidence=None,
    )

    assert recorder.write() is None
    assert not (tmp_path / "sample_depth.npz").exists()


def test_stop_recording_saves_depth_in_background(tmp_path: Path) -> None:
    module = _load_record_client_module()
    output_path = tmp_path / "sample_depth.npz"
    recorder = module.DepthNpzRecorder(output_path)
    recorder.append(
        frame_count=1,
        timestamp_ns=1000,
        depth=np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        confidence=None,
    )

    async def run() -> None:
        _, _, pending_save = module.stop_recording(
            writer=None,
            depth_recorder=recorder,
            frame_count=1,
            fps=12.5,
        )

        assert pending_save is not None
        assert pending_save.target_path == output_path
        await module.finish_depth_save(pending_save)

    anyio.run(run)

    saved = np.load(output_path)
    assert saved["depth_mm"].tolist() == [[[100.0, 200.0], [300.0, 400.0]]]
    assert np.isclose(saved["fps"].item(), 12.5)


def test_safe_stream_name_sanitizes_uri() -> None:
    module = _load_record_client_module()

    assert (
        module.safe_stream_name("cvmmap://zed@/tmp/cvmmap")
        == "cvmmap_zed_tmp_cvmmap"
    )


def test_depth_to_colormap_ignores_invalid_values() -> None:
    module = _load_record_client_module()
    depth = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        colormap = module.depth_to_colormap(depth)

    assert len(caught) == 0
    assert colormap.shape == (2, 2, 3)
    assert colormap.dtype == np.uint8
