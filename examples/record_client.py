import asyncio
import anyio
import click
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from loguru import logger
import cv2
import numpy as np

from cvmmap import CvMmapClient
from cvmmap.msg import (
    PIXEL_FORMAT_BGR,
    PIXEL_FORMAT_RGB,
    PIXEL_FORMAT_BGRA,
    PIXEL_FORMAT_RGBA,
)
from typing import Optional, Literal, TypeAlias

ViewMode: TypeAlias = Literal["left", "depth", "both"]
LeftOrder: TypeAlias = Literal["auto", "bgr", "rgb", "bgra", "rgba"]
RecordingState: TypeAlias = Literal["idle", "recording", "saving"]


@dataclass
class PendingDepthSave:
    task: asyncio.Task[Optional[Path]]
    target_path: Path


class DepthNpzRecorder:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._records: list[
            tuple[int, int, Optional[np.ndarray], Optional[np.ndarray]]
        ] = []

    def append(
        self,
        frame_count: int,
        timestamp_ns: int,
        depth: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
    ) -> None:
        depth_copy = (
            None if depth is None else np.asarray(depth, dtype=np.float32).copy()
        )
        confidence_copy = None if confidence is None else np.asarray(confidence).copy()
        self._records.append(
            (int(frame_count), int(timestamp_ns), depth_copy, confidence_copy)
        )

    def write(self, fps: Optional[float] = None) -> Optional[Path]:
        if not self._records:
            return None

        first_depth = next(
            (depth for _, _, depth, _ in self._records if depth is not None),
            None,
        )
        if first_depth is None:
            return None

        depth_shape = first_depth.shape
        first_confidence = next(
            (
                confidence
                for _, _, _, confidence in self._records
                if confidence is not None
            ),
            None,
        )

        depth_frames: list[np.ndarray] = []
        depth_present_mask: list[bool] = []
        confidence_frames: list[np.ndarray] = []
        confidence_present_mask: list[bool] = []
        frame_counts: list[int] = []
        timestamp_ns_values: list[int] = []

        for frame_count, timestamp_ns, depth, confidence in self._records:
            frame_counts.append(frame_count)
            timestamp_ns_values.append(timestamp_ns)

            if depth is None:
                depth_frames.append(np.full(depth_shape, np.nan, dtype=np.float32))
                depth_present_mask.append(False)
            else:
                if depth.shape != depth_shape:
                    raise ValueError(
                        "Depth frame shape changed during recording: "
                        f"expected {depth_shape}, got {depth.shape}"
                    )
                depth_frames.append(depth)
                depth_present_mask.append(True)

            if first_confidence is None:
                continue

            if confidence is None:
                confidence_frames.append(
                    np.zeros(first_confidence.shape, dtype=first_confidence.dtype)
                )
                confidence_present_mask.append(False)
                continue

            if confidence.shape != first_confidence.shape:
                raise ValueError(
                    "Confidence frame shape changed during recording: "
                    f"expected {first_confidence.shape}, got {confidence.shape}"
                )
            confidence_frames.append(confidence)
            confidence_present_mask.append(True)

        payload: dict[str, np.ndarray] = {
            "depth_mm": np.stack(depth_frames, axis=0),
            "timestamp_ns": np.asarray(timestamp_ns_values, dtype=np.uint64),
            "frame_count": np.asarray(frame_counts, dtype=np.uint64),
            "depth_present_mask": np.asarray(depth_present_mask, dtype=bool),
            "frame_total": np.asarray(len(self._records), dtype=np.uint64),
            "depth_units": np.asarray("millimeters"),
            "source_depth_dtype": np.asarray(str(first_depth.dtype)),
        }
        if fps is not None:
            payload["fps"] = np.asarray(float(fps), dtype=np.float32)
        if first_confidence is not None:
            payload["confidence"] = np.stack(confidence_frames, axis=0)
            payload["confidence_present_mask"] = np.asarray(
                confidence_present_mask,
                dtype=bool,
            )
            payload["source_confidence_dtype"] = np.asarray(
                str(first_confidence.dtype)
            )

        np.savez_compressed(self.output_path, **payload)
        return self.output_path


async def write_depth_npz(
    depth_recorder: DepthNpzRecorder,
    fps: Optional[float],
) -> Optional[Path]:
    return await anyio.to_thread.run_sync(depth_recorder.write, fps)


def stop_recording(
    writer: Optional[cv2.VideoWriter],
    depth_recorder: Optional[DepthNpzRecorder],
    frame_count: int,
    fps: Optional[float],
) -> tuple[None, None, Optional[PendingDepthSave]]:
    if writer is not None:
        writer.release()

    logger.info(f"Stopped recording. Saved {frame_count} frames.")
    if depth_recorder is None:
        return None, None, None

    logger.info(
        f"Saving depth NPZ in background to {depth_recorder.output_path}. "
        "Preview stays live while the file is finalized."
    )
    pending_save = PendingDepthSave(
        task=asyncio.create_task(write_depth_npz(depth_recorder, fps)),
        target_path=depth_recorder.output_path,
    )
    return None, None, pending_save


async def finish_depth_save(pending_save: PendingDepthSave) -> None:
    try:
        depth_output_path = await pending_save.task
    except Exception:
        logger.exception(f"Failed to save depth NPZ to {pending_save.target_path}")
        return

    if depth_output_path is None:
        logger.info(
            "No depth planes were recorded during this session; "
            "skipped depth NPZ export."
        )
        return

    logger.success(f"Saved depth NPZ to {depth_output_path}")


def safe_stream_name(stream_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stream_name).strip("._-")
    return slug or "stream"


def to_bgr(frame: np.ndarray, pixel_format: int, left_order: LeftOrder) -> np.ndarray:
    order = left_order
    if order == "auto":
        if pixel_format == PIXEL_FORMAT_BGR:
            order = "bgr"
        elif pixel_format == PIXEL_FORMAT_RGB:
            order = "rgb"
        elif pixel_format == PIXEL_FORMAT_BGRA:
            order = "bgra"
        elif pixel_format == PIXEL_FORMAT_RGBA:
            order = "rgba"
        else:
            order = "bgr"

    if order == "bgr":
        return frame
    if order == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if order == "bgra":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if order == "rgba":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported left order: {order}")


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    values = d[valid]
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 95.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        high = low + 1.0

    clipped = np.clip(d, low, high)
    scaled = np.zeros(d.shape, dtype=np.uint8)
    scaled[valid] = (
        (clipped[valid] - low) * (255.0 / (high - low))
    ).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def compose_display(
    left_bgr: np.ndarray,
    depth_bgr: Optional[np.ndarray],
    view_mode: ViewMode,
) -> np.ndarray:
    if view_mode == "left":
        return left_bgr
    if view_mode == "depth":
        if depth_bgr is None:
            return np.zeros_like(left_bgr)
        return depth_bgr
    if view_mode == "both":
        if depth_bgr is None:
            right = np.zeros_like(left_bgr)
        else:
            right = depth_bgr
        if right.shape[:2] != left_bgr.shape[:2]:
            right = cv2.resize(
                right,
                (left_bgr.shape[1], left_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.hstack([left_bgr, right])
    raise ValueError(f"Unsupported view mode: {view_mode}")


# We use asyncio directly since cvmmap is built on zmq.asyncio.
# The core loop must run asynchronously to process frames as they arrive.
async def record_client(
    stream_name: str,
    out_path: Path,
    force_fps: Optional[float] = None,
    view_mode: ViewMode = "left",
    left_order: LeftOrder = "auto",
    first_frame_timeout: float = 5.0,
):
    out_path.mkdir(parents=True, exist_ok=True)
    view_cycle: list[ViewMode] = ["left", "depth", "both"]
    if view_mode not in view_cycle:
        raise ValueError(f"Unsupported view_mode: {view_mode}")
    view_idx = view_cycle.index(view_mode)

    logger.info("Controls:")
    logger.info("  [SPACE] - Start/Stop recording")
    logger.info("  [m] - Cycle view mode (left/depth/both)")
    logger.info("  [q] or [ESC] - Quit")

    client = CvMmapClient(stream_name)
    frame_iter = client.__aiter__()

    try:
        async with asyncio.timeout(first_frame_timeout):
            first_frame, first_meta = await anext(frame_iter)
    except TimeoutError as exc:
        raise click.ClickException(
            f"No frame received within {first_frame_timeout:.2f}s from stream '{stream_name}'"
        ) from exc

    # State variables for recording
    recording_state: RecordingState = "idle"
    writer: Optional[cv2.VideoWriter] = None
    depth_recorder: Optional[DepthNpzRecorder] = None
    recording_fps: Optional[float] = None
    pending_depth_save: Optional[PendingDepthSave] = None
    frame_count = 0

    # FPS estimation using timestamps from metadata
    last_timestamps: list[int] = []
    MAX_FPS_SAMPLES = 60
    try:
        current_frame = first_frame
        current_meta = first_meta
        while True:
            if pending_depth_save is not None and pending_depth_save.task.done():
                await finish_depth_save(pending_depth_save)
                pending_depth_save = None
                recording_state = "idle"

            frame = current_frame
            meta = current_meta
            # Update FPS estimation queue using metadata timestamps (nanoseconds)
            last_timestamps.append(meta.timestamp_ns)
            if len(last_timestamps) > MAX_FPS_SAMPLES:
                last_timestamps.pop(0)
            left_bgr = to_bgr(frame, meta.info.pixel_format, left_order)
            depth = client.depth_plane(meta)
            confidence = client.confidence_plane(meta)
            depth_bgr = None if depth is None else depth_to_colormap(depth)
            active_view_mode: ViewMode = view_cycle[view_idx]
            display_frame = compose_display(left_bgr, depth_bgr, active_view_mode)
            if recording_state != "idle":
                display_frame = display_frame.copy()

            if recording_state == "recording":
                _ = cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                _ = cv2.putText(
                    display_frame,
                    f"REC {frame_count}",
                    (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            elif recording_state == "saving":
                _ = cv2.circle(display_frame, (30, 30), 10, (0, 215, 255), -1)
                _ = cv2.putText(
                    display_frame,
                    "SAVING",
                    (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 215, 255),
                    2,
                )
            cv2.namedWindow(f"Preview: {stream_name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Preview: {stream_name}", display_frame)

            # --- RECORDING ---
            if recording_state == "recording" and writer is not None:
                # writer.write blocks slightly, but NVENC/hardware encoders are fast.
                # If disk I/O blocks too much, this could cause ZeroMQ conflate to
                # drop frames (which is intended behavior for real-time systems to avoid memory explosion).
                writer.write(display_frame)
                if depth_recorder is not None:
                    depth_recorder.append(
                        frame_count=meta.frame_count,
                        timestamp_ns=meta.timestamp_ns,
                        depth=depth,
                        confidence=confidence,
                    )
                frame_count += 1

            # --- INPUT HANDLING ---
            # cv2.waitKey(1) allows OpenCV GUI events to process
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break

            elif key == ord(" "):  # Spacebar toggles recording
                if recording_state == "idle":
                    # START RECORDING
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    stream_slug = safe_stream_name(stream_name)
                    filename = out_path / f"record_{stream_slug}_{timestamp}.mp4"

                    height: int = int(display_frame.shape[0])
                    width: int = int(display_frame.shape[1])

                    # Calculate FPS from rolling average if enough samples, else default to 30.0
                    fps = 30.0
                    if force_fps is not None:
                        fps = force_fps
                        logger.info(f"Using forced FPS from CLI: {fps}")
                    elif len(last_timestamps) >= 2:
                        duration_ns = last_timestamps[-1] - last_timestamps[0]
                        if duration_ns > 0:
                            fps_estimated = (len(last_timestamps) - 1) / (
                                duration_ns / 1e9
                            )
                            fps = round(fps_estimated, 2)
                            logger.info(
                                f"Estimated FPS based on {len(last_timestamps)} samples: {fps}"
                            )
                        else:
                            logger.warning("Duration is 0, using default 30.0 FPS")
                    else:
                        logger.warning(
                            "Not enough samples to estimate FPS, using default 30.0"
                        )
                    # --- CHOOSING THE CODEC ---
                    # FOURCC (Four-Character Code) specifies the codec.
                    #
                    # For hardware acceleration (NVIDIA GPU):
                    # Use 'h264_nvenc' or 'hevc_nvenc'.
                    # Note: OpenCV must be compiled with FFmpeg + NVENC support to use hardware encoders directly via VideoWriter.
                    # Usually, the string 'avc1' or 'mp4v' combined with the right backend (CAP_FFMPEG) is used.
                    #
                    # Common fallbacks:
                    # 'mp4v' - standard MPEG-4 (software, widely supported)
                    # 'avc1' - H.264 (usually software libx264, highly compatible)
                    # 'hevc' - H.265 (better compression, heavier CPU)

                    # Using 'mp4v' as a safe cross-platform default for this demo.
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

                    # Initialize the writer
                    writer = cv2.VideoWriter(
                        str(filename), fourcc, fps, (width, height)
                    )

                    if not writer.isOpened():
                        logger.error(f"Failed to open video writer for {filename}!")
                        recording_state = "idle"
                        writer = None
                        depth_recorder = None
                        recording_fps = None
                    else:
                        depth_recorder = DepthNpzRecorder(
                            filename.with_name(f"{filename.stem}_depth.npz")
                        )
                        recording_fps = fps
                        frame_count = 0
                        recording_state = "recording"
                        logger.success(
                            f"Started recording to {filename} ({width}x{height})"
                        )
                        logger.info(
                            "Depth NPZ sidecar will be saved automatically "
                            "if depth frames are available."
                        )

                elif recording_state == "recording":
                    # STOP RECORDING
                    writer, depth_recorder, pending_depth_save = stop_recording(
                        writer=writer,
                        depth_recorder=depth_recorder,
                        frame_count=frame_count,
                        fps=recording_fps,
                    )
                    recording_fps = None
                    recording_state = (
                        "saving" if pending_depth_save is not None else "idle"
                    )
                else:
                    logger.info(
                        "Depth NPZ is still saving in the background. "
                        "Wait for the save to finish before starting a new recording."
                    )
            elif key == ord("m"):
                view_idx = (view_idx + 1) % len(view_cycle)
                logger.info(f"Switched view mode to: {view_cycle[view_idx]}")

            try:
                current_frame, current_meta = await anext(frame_iter)
            except StopAsyncIteration:
                break

    except asyncio.CancelledError:
        logger.info("Client task cancelled.")
    finally:
        # Cleanup
        if recording_state == "recording":
            writer, depth_recorder, pending_depth_save = stop_recording(
                writer=writer,
                depth_recorder=depth_recorder,
                frame_count=frame_count,
                fps=recording_fps,
            )
            recording_fps = None
            recording_state = "saving" if pending_depth_save is not None else "idle"
        if pending_depth_save is not None:
            logger.info("Waiting for background depth save to finish...")
            await finish_depth_save(pending_depth_save)
        cv2.destroyAllWindows()
        logger.info("Exiting.")


@click.command()
@click.argument("stream_name")
@click.option(
    "--out",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./recordings"),
    help="Output directory",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Force a specific FPS (otherwise estimated from timestamps)",
)
@click.option(
    "--view-mode",
    type=click.Choice(["left", "depth", "both"], case_sensitive=False),
    default="left",
    show_default=True,
    help="Preview/record mode",
)
@click.option(
    "--left-order",
    type=click.Choice(["auto", "bgr", "rgb", "bgra", "rgba"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Input channel order for the left image",
)
@click.option(
    "--first-frame-timeout",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds to wait for first frame before exiting with error",
)
def cli(
    stream_name: str,
    out: Path,
    fps: Optional[float],
    view_mode: ViewMode,
    left_order: LeftOrder,
    first_frame_timeout: float,
):
    """cv-mmap recording client"""
    try:
        # Run the async loop
        asyncio.run(
            record_client(
                stream_name,
                out,
                fps,
                view_mode,
                left_order,
                first_frame_timeout,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    runner = getattr(cli, "main", None)
    if callable(runner):
        runner(standalone_mode=True)
