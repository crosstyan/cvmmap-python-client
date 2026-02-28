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
