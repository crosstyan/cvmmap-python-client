# cvmmap-client

Python client library for the `cv-mmap` producer protocol.

The transport split is:

- shared memory for image payloads
- ZeroMQ PUB/SUB for frame sync
- NATS for control, module status, and body tracking

## Install

Core library only:

```bash
pip install cvmmap-client
```

With demo/recording tooling (OpenCV, CLI, logging):

```bash
pip install "cvmmap-client[tools]"
```

For local development and tests:

```bash
pip install "cvmmap-client[tools,test]"
```

## Quick usage

```python
from cvmmap import CvMmapClient

client = CvMmapClient("default")
```

URI form is also supported:

```python
client = CvMmapClient("cvmmap://default")
client = CvMmapClient("cvmmap://cam0@/run/cvmmap?namespace=zed")
```

The default conventions are:

- shared memory name: `cvmmap_{name}`
- frame topic endpoint: `ipc:///tmp/cvmmap_{name}`
- NATS target key: `cvmmap_{name}`
- default NATS server URL: `nats://localhost:4222`

## CVMMAP URI scheme

Accepted target forms:

- Plain instance name: `<instance>`
- URI: `cvmmap://<instance>[@<prefix>][?namespace=<namespace>]`

Defaults:

- `prefix=/tmp`
- `namespace=cvmmap`

Mapping:

- `base_name = <namespace>_<instance>`
- `nats_target_key = <base_name>` with `.` normalized to `_`
- frame endpoint: `ipc://<prefix>/<base_name>`
- shared memory name: `<base_name>` (Linux path `/dev/shm/<base_name>`)

Examples:

- `default` -> `ipc:///tmp/cvmmap_default`, shm `cvmmap_default`
- `cvmmap://default` -> `ipc:///tmp/cvmmap_default`, shm `cvmmap_default`
- `cvmmap://camera0@/run/cvmmap?namespace=zed` -> `ipc:///run/cvmmap/zed_camera0`, shm `zed_camera0`

## Development notes

- `uv.lock` is optional for a library project. Keep it if you want reproducible contributor/test environments.
- Runtime dependencies are intentionally small (`numpy`, `pyzmq`, `protobuf`, `nats-py`).
- Tooling dependencies (OpenCV, click, loguru, anyio) are in optional extras.

## Local examples

- `examples/record_client.py`
- `examples/sync_monitor.py`

## Protocol compatibility

This client stays aligned with producer-side IPC structures.

### ABI v1/v2 support

| Component | Supported Versions | Notes |
|-----------|-------------------|-------|
| SHM Metadata | v1, v2 | Auto-detects from header magic and version fields |
| Control/Status | protobuf over NATS | Supports `RESET_FRAME_COUNT`, `GET_SOURCE_INFO`, `GET_CAPABILITIES`, `SEEK_TIMESTAMP_NS`, `START_RECORDING`, `STOP_RECORDING`, and `GET_RECORDING_STATUS` |
| Body Tracking | raw `cvmmap_body_tracking_v1` bytes over NATS | Same payload format as the producer body packet |
| Sync Wire | v1 over ZMQ | Frame notification only |

### Migration window notes

During the migration window, the producer may emit SHM v2 metadata while keeping the frame sync wire unchanged and moving control/body to NATS. This client follows that split directly.

Cross-repo compatibility tests can consume C++-generated fixtures directly from
`cv-mmap/core/fixtures/protocol`. Override the location with
`CVMMAP_CORE_PROTOCOL_FIXTURE_DIR=/path/to/cv-mmap/core/fixtures/protocol` when
the sibling repo is not at the default local path.

### Rollout sequencing (for deployments)

1. Deploy updated Python client and GUI consumer first (they parse v1/v2).
2. Then deploy producer-only-v2 builds.
3. Never deploy producer-v2 before consumers are ready.

### Frame iteration with depth and confidence

When consuming v2 streams with additional v2 planes:

```python
from cvmmap import CvMmapClient

client = CvMmapClient("default")
async for image, metadata in client:
    # image is always the left plane (backward compatible)
    # access optional extra planes via client helpers when available
    depth = client.depth_plane(metadata)
    confidence = client.confidence_plane(metadata)
    depth_unit = client.depth_unit(metadata)
    if depth is not None:
        # depth_unit is one of DEPTH_UNIT_UNKNOWN / DEPTH_UNIT_MILLIMETER / DEPTH_UNIT_METER
        # process depth plane
        pass
    if confidence is not None:
        # process confidence plane
        pass
```

See the producer's `docs/abi_v2_migration_guide.md` for full specifications.

The v2 header byte at offset `0x2C` is parsed as `depth_unit`. Legacy v2
packets with a zeroed byte remain valid and surface as `DEPTH_UNIT_UNKNOWN`.

### Body tracking substream

Body tracking is published on a NATS subject and does not change the existing
frame iterator:

```python
from cvmmap import CvMmapClient

client = CvMmapClient("default")
async for body_frame in client.body_stream():
    for body in body_frame.bodies:
        print(body.id, body.position)
```

### Control requests

```python
from cvmmap import (
    CvMmapRequestClient,
    RecordingRequest,
    SvoRecordingOptions,
    RECORDING_FORMAT_SVO,
)

client = CvMmapRequestClient("default")
info = await client.get_source_info()
print(info.source_kind, info.can_seek, info.can_record)

caps = await client.get_capabilities()
print(caps.can_seek, caps.available_recording_formats)

result = await client.seek_timestamp_ns(info.timeline_start_ns)
print(result.landed_timestamp_ns, result.exact_match)

status = await client.start_recording(
    RecordingRequest(
        recording_format=RECORDING_FORMAT_SVO,
        output_path="/tmp/example.svo2",
        svo_options=SvoRecordingOptions(compression_mode="h265"),
    )
)
print(status.is_recording, status.active_path)

status = await client.stop_recording()
print(status.is_recording, status.frames_encoded)
```
