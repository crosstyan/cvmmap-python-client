# cvmmap-client

Python client library for the `cv-mmap` producer protocol (shared memory + ZeroMQ signaling).

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

The default conventions are:

- shared memory name: `cvmmap_{name}`
- frame topic endpoint: `ipc:///tmp/cvmmap_{name}`
- control endpoint: `ipc:///tmp/cvmmap_{name}_control`

## Development notes

- `uv.lock` is optional for a library project. Keep it if you want reproducible contributor/test environments.
- Runtime dependencies are intentionally minimal (`numpy`, `pyzmq`).
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
| Control Wire | v1 | Request/response structs match producer v1 |
| Sync Wire | v1 | ZMQ pub/sub framing unchanged |

### Migration window notes

During the migration window, the producer may emit SHM v2 metadata while maintaining control/sync compatibility at v1. This is the intended mixed state.

### Rollout sequencing (for deployments)

1. Deploy updated Python client and GUI consumer first (they parse v1/v2).
2. Then deploy producer-only-v2 builds.
3. Never deploy producer-v2 before consumers are ready.

### Frame iteration with depth

When consuming v2 streams with depth planes:

```python
from cvmmap import CvMmapClient

client = CvMmapClient("default")
async for image, metadata in client:
    # image is always the left plane (backward compatible)
    # access depth via client helper when available
    depth = client.depth_plane(metadata)
    if depth is not None:
        # process depth plane
        pass
```

See the producer's `docs/abi_v2_migration_guide.md` for full specifications.
