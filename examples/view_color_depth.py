import asyncio
from typing import Literal, TypeAlias

import click
import cv2
import numpy as np

from cvmmap import CvMmapClient
from cvmmap.msg import (
    PIXEL_FORMAT_BGR,
    PIXEL_FORMAT_BGRA,
    PIXEL_FORMAT_RGB,
    PIXEL_FORMAT_RGBA,
)

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
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


async def run_viewer(
    stream_name: str,
    left_order: LeftOrder,
    first_frame_timeout: float,
) -> None:
    client = CvMmapClient(stream_name)
    window = f"{stream_name}: left|depth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_iter = client.__aiter__()
    try:
        async with asyncio.timeout(first_frame_timeout):
            current_frame, current_meta = await anext(frame_iter)
    except TimeoutError as exc:
        raise click.ClickException(
            f"No frame received within {first_frame_timeout:.2f}s from stream '{stream_name}'"
        ) from exc

    while True:
        frame = current_frame
        meta = current_meta
        left_bgr = to_bgr(frame, meta.info.pixel_format, left_order)
        depth = client.depth_plane(meta)
        if depth is None:
            depth_bgr = np.zeros_like(left_bgr)
        else:
            depth_bgr = depth_to_colormap(depth)
            if depth_bgr.shape[:2] != left_bgr.shape[:2]:
                depth_bgr = cv2.resize(
                    depth_bgr,
                    (left_bgr.shape[1], left_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        panel = np.hstack([left_bgr, depth_bgr])
        cv2.imshow(window, panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        try:
            current_frame, current_meta = await anext(frame_iter)
        except StopAsyncIteration:
            break

    cv2.destroyAllWindows()


@click.command()
@click.argument("stream_name")
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
def cli(stream_name: str, left_order: LeftOrder, first_frame_timeout: float) -> None:
    asyncio.run(run_viewer(stream_name, left_order, first_frame_timeout))


if __name__ == "__main__":
    runner = getattr(cli, "main", None)
    if callable(runner):
        runner(standalone_mode=True)
