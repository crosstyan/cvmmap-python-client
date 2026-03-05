import asyncio
import click
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
    scaled = ((clipped - low) * (255.0 / (high - low))).astype(np.uint8)
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
    recording = False
    writer = None
    frame_count = 0

    # FPS estimation using timestamps from metadata
    last_timestamps: list[int] = []
    MAX_FPS_SAMPLES = 60
    try:
        current_frame = first_frame
        current_meta = first_meta
        while True:
            frame = current_frame
            meta = current_meta
            # Update FPS estimation queue using metadata timestamps (nanoseconds)
            last_timestamps.append(meta.timestamp_ns)
            if len(last_timestamps) > MAX_FPS_SAMPLES:
                last_timestamps.pop(0)
            left_bgr = to_bgr(frame, meta.info.pixel_format, left_order)
            depth = client.depth_plane(meta)
            depth_bgr = None if depth is None else depth_to_colormap(depth)
            active_view_mode: ViewMode = view_cycle[view_idx]
            display_frame = compose_display(left_bgr, depth_bgr, active_view_mode)
            if recording:
                display_frame = display_frame.copy()

            # Add visual indicator if recording
            if recording:
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
            cv2.namedWindow(f"Preview: {stream_name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Preview: {stream_name}", display_frame)

            # --- RECORDING ---
            if recording and writer is not None:
                # writer.write blocks slightly, but NVENC/hardware encoders are fast.
                # If disk I/O blocks too much, this could cause ZeroMQ conflate to
                # drop frames (which is intended behavior for real-time systems to avoid memory explosion).
                writer.write(display_frame)
                frame_count += 1

            # --- INPUT HANDLING ---
            # cv2.waitKey(1) allows OpenCV GUI events to process
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break

            elif key == ord(" "):  # Spacebar toggles recording
                recording = not recording

                if recording:
                    # START RECORDING
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = out_path / f"record_{stream_name}_{timestamp}.mp4"

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
                        recording = False
                    else:
                        frame_count = 0
                        logger.success(
                            f"Started recording to {filename} ({width}x{height})"
                        )

                else:
                    # STOP RECORDING
                    if writer:
                        writer.release()
                        writer = None
                    logger.info(f"Stopped recording. Saved {frame_count} frames.")
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
        if writer:
            writer.release()
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
