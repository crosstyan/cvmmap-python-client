import asyncio
import json
from importlib import import_module
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cvmmap = import_module("cvmmap")


class _FakeMessage:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _FakeNatsClient:
    def __init__(
        self,
        *,
        ping_payloads: list[bytes],
        info_payloads: dict[str, bytes],
    ) -> None:
        self._ping_payloads = ping_payloads
        self._info_payloads = info_payloads
        self._subscriptions: dict[str, object] = {}
        self.closed = False
        self.connected_servers: list[str] | None = None

    def new_inbox(self) -> str:
        return "_INBOX.discovery"

    async def subscribe(self, subject: str, cb) -> object:
        self._subscriptions[subject] = cb
        return object()

    async def flush(self) -> None:
        return None

    async def publish(self, subject: str, payload: bytes, reply: str | None = None) -> None:
        if subject == "$SRV.PING.cvmmap_producer" and reply is not None:
            callback = self._subscriptions[reply]
            for ping_payload in self._ping_payloads:
                await callback(_FakeMessage(ping_payload))

    async def request(self, subject: str, payload: bytes, timeout: float) -> _FakeMessage:
        if subject not in self._info_payloads:
            raise TimeoutError(subject)
        return _FakeMessage(self._info_payloads[subject])

    async def close(self) -> None:
        self.closed = True


class _FakeNatsModule:
    def __init__(self, client: _FakeNatsClient) -> None:
        self._client = client

    async def connect(self, *, servers: list[str]):
        self._client.connected_servers = servers
        return self._client


def _ping_payload(service_id: str) -> bytes:
    return json.dumps(
        {
            "name": "cvmmap_producer",
            "version": "0.1.0",
            "id": service_id,
            "type": "io.nats.micro.v1.ping_response",
        }
    ).encode("utf-8")


def _info_payload(
    service_id: str,
    *,
    instance_name: str,
    target_key: str,
    backend: str,
    shm_name: str,
    zmq_addr: str,
) -> bytes:
    return json.dumps(
        {
            "type": "io.nats.micro.v1.info_response",
            "name": "cvmmap_producer",
            "version": "0.1.0",
            "id": service_id,
            "description": "cv-mmap producer discovery and control service",
            "metadata": {
                "instance_name": instance_name,
                "namespace": "cvmmap",
                "ipc_prefix": "/tmp",
                "base_name": shm_name,
                "nats_target_key": target_key,
                "shm_name": shm_name,
                "zmq_addr": zmq_addr,
                "backend": backend,
            },
            "endpoints": [
                {"name": "source.info", "subject": f"cvmmap.{target_key}.control.source.info"},
                {"name": "source.reset", "subject": f"cvmmap.{target_key}.control.source.reset"},
            ],
        }
    ).encode("utf-8")


def test_discover_cvmmap_producers_filters_and_fallback_subjects(monkeypatch) -> None:
    fake_client = _FakeNatsClient(
        ping_payloads=[_ping_payload("svc-1"), _ping_payload("svc-2")],
        info_payloads={
            "$SRV.INFO.cvmmap_producer.svc-1": _info_payload(
                "svc-1",
                instance_name="zed4",
                target_key="cvmmap_zed4",
                backend="zed",
                shm_name="cvmmap_zed4",
                zmq_addr="ipc:///tmp/cvmmap_zed4",
            ),
            "$SRV.INFO.cvmmap_producer.svc-2": _info_payload(
                "svc-2",
                instance_name="dummy1",
                target_key="cvmmap_dummy1",
                backend="dummy",
                shm_name="cvmmap_dummy1",
                zmq_addr="ipc:///tmp/cvmmap_dummy1",
            ),
        },
    )
    monkeypatch.setattr(cvmmap, "_import_nats", lambda: _FakeNatsModule(fake_client))

    async def _run() -> None:
        producers = await cvmmap.discover_cvmmap_producers(
            instance_name="zed4",
            timeout_ms=50,
        )
        assert len(producers) == 1
        producer = producers[0]
        assert producer.service_id == "svc-1"
        assert producer.instance_name == "zed4"
        assert producer.nats_target_key == "cvmmap_zed4"
        assert producer.body_subject == "cvmmap.cvmmap_zed4.body"
        assert producer.status_subject == "cvmmap.cvmmap_zed4.status"
        assert producer.control_subject_prefix == "cvmmap.cvmmap_zed4.control"
        assert producer.control_subjects == (
            "cvmmap.cvmmap_zed4.control.source.info",
            "cvmmap.cvmmap_zed4.control.source.reset",
        )
        assert fake_client.connected_servers == [cvmmap.DEFAULT_NATS_URL]
        assert fake_client.closed is True

    asyncio.run(_run())


def test_discover_cvmmap_producer_exact_one_errors(monkeypatch) -> None:
    fake_client = _FakeNatsClient(
        ping_payloads=[_ping_payload("svc-1"), _ping_payload("svc-2")],
        info_payloads={
            "$SRV.INFO.cvmmap_producer.svc-1": _info_payload(
                "svc-1",
                instance_name="zed4",
                target_key="cvmmap_zed4",
                backend="zed",
                shm_name="cvmmap_zed4",
                zmq_addr="ipc:///tmp/cvmmap_zed4",
            ),
            "$SRV.INFO.cvmmap_producer.svc-2": _info_payload(
                "svc-2",
                instance_name="zed5",
                target_key="cvmmap_zed5",
                backend="zed",
                shm_name="cvmmap_zed5",
                zmq_addr="ipc:///tmp/cvmmap_zed5",
            ),
        },
    )
    monkeypatch.setattr(cvmmap, "_import_nats", lambda: _FakeNatsModule(fake_client))

    async def _run() -> None:
        try:
            await cvmmap.discover_cvmmap_producer(timeout_ms=50)
        except cvmmap.DiscoveryAmbiguousError as exc:
            assert "matched 2 producers" in str(exc)
        else:
            raise AssertionError("Expected ambiguous discovery to raise")

        try:
            await cvmmap.discover_cvmmap_producer(
                instance_name="does-not-exist",
                timeout_ms=50,
            )
        except cvmmap.DiscoveryNotFoundError as exc:
            assert "no cvmmap producer matched" in str(exc)
        else:
            raise AssertionError("Expected not-found discovery to raise")

    asyncio.run(_run())


def test_discovered_producer_constructors_preserve_connect_info() -> None:
    producer = cvmmap.DiscoveredProducer(
        service_id="svc-1",
        service_name="cvmmap_producer",
        service_version="0.1.0",
        instance_name="zed4",
        namespace="cvmmap",
        ipc_prefix="/tmp",
        base_name="cvmmap_zed4",
        nats_target_key="cvmmap_zed4",
        shm_name="cvmmap_zed4",
        zmq_addr="ipc:///tmp/cvmmap_zed4",
        body_subject="cvmmap.cvmmap_zed4.body",
        status_subject="cvmmap.cvmmap_zed4.status",
        control_subject_prefix="cvmmap.cvmmap_zed4.control",
        backend="zed",
        control_subjects=("cvmmap.cvmmap_zed4.control.source.info",),
    )

    frame_client = cvmmap.CvMmapClient(producer, nats_url=None)
    try:
        assert frame_client.shm_name == "cvmmap_zed4"
        assert frame_client.zmq_addr == "ipc:///tmp/cvmmap_zed4"
        assert frame_client.nats_target_key == "cvmmap_zed4"
    finally:
        frame_client.close()

    request_client = cvmmap.CvMmapRequestClient(producer)
    assert request_client.shm_name == "cvmmap_zed4"
    assert request_client.zmq_addr == "ipc:///tmp/cvmmap_zed4"
    assert request_client.nats_target_key == "cvmmap_zed4"

    body_stream = cvmmap.CvMmapBodyStream(producer)
    assert body_stream._target_key == "cvmmap_zed4"  # type: ignore[attr-defined]
