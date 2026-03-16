from __future__ import annotations

import asyncio
from contextlib import suppress
from logging import getLogger
import importlib
import re
import struct
from typing import Any, AsyncGenerator, NamedTuple, TypedDict, cast

import numpy as np

zmq = importlib.import_module("zmq")
_zmq_asyncio = importlib.import_module("zmq.asyncio")
Context = _zmq_asyncio.Context
Socket = _zmq_asyncio.Socket

from . import control_pb2
from .msg import (
    BODY_TRACKING_MAGIC,
    CONTROL_MSG_CMD_GENERIC,
    CONTROL_MSG_CMD_GET_RECORDING_STATUS,
    CONTROL_MSG_CMD_GET_SOURCE_INFO,
    CONTROL_MSG_CMD_RESET_FRAME_COUNT,
    CONTROL_MSG_CMD_SEEK_TIMESTAMP_NS,
    CONTROL_MSG_CMD_START_RECORDING,
    CONTROL_MSG_CMD_STOP_RECORDING,
    CONTROL_RESPONSE_ERROR,
    CONTROL_RESPONSE_INVALID_LABEL,
    CONTROL_RESPONSE_INVALID_MAGIC,
    CONTROL_RESPONSE_INVALID_MSG_SIZE,
    CONTROL_RESPONSE_INVALID_PAYLOAD,
    CONTROL_RESPONSE_INVALID_VERSION,
    CONTROL_RESPONSE_OK,
    CONTROL_RESPONSE_OUT_OF_RANGE,
    CONTROL_RESPONSE_TIMEOUT,
    CONTROL_RESPONSE_UNKNOWN_CMD,
    CONTROL_RESPONSE_UNSUPPORTED,
    CV_MMAP_MAGIC,
    CV_MMAP_MAGIC_LEN,
    DEPTH_UNIT_METER,
    DEPTH_UNIT_MILLIMETER,
    DEPTH_UNIT_UNKNOWN,
    FRAME_METADATA_REGION_SIZE,
    FRAME_TOPIC_MAGIC,
    McapRecordingOptions,
    BodyFrame,
    BodyTrack,
    BodyTrackingMessageHeader,
    ControlCapabilities,
    ControlMessageRequest,
    ControlMessageResponse,
    FrameInfo,
    FrameMetadata,
    FrameMetadataV2,
    FrameMetadataV2Header,
    FramePlaneDescriptorV2,
    ModuleStatusMessage,
    RecordingRequest,
    RecordingStartRequest,
    RecordingStatus,
    SeekResult,
    SeekTimestampRequest,
    SourceInfo,
    SyncMessage,
    SvoRecordingOptions,
    TIMESTAMP_DOMAIN_MEDIA_TIME_NS,
    TIMESTAMP_DOMAIN_UNIX_EPOCH_NS,
    TIMESTAMP_DOMAIN_UNKNOWN,
    SOURCE_INFO_FLAG_AUTO_LOOP,
    SOURCE_INFO_FLAG_CAN_RECORD,
    SOURCE_INFO_FLAG_CAN_SEEK,
    SOURCE_INFO_FLAG_HAS_BODY,
    SOURCE_INFO_FLAG_HAS_DEPTH,
    SOURCE_KIND_FINITE,
    SOURCE_KIND_LIVE,
    SOURCE_KIND_UNKNOWN,
    MODULE_STATUS_OFFLINE,
    MODULE_STATUS_ONLINE,
    MODULE_STATUS_STREAM_RESET,
    RECORDING_FORMAT_MCAP,
    RECORDING_FORMAT_SVO,
    RECORDING_FORMAT_UNKNOWN,
    RECORDING_STATUS_FLAG_CAN_RECORD,
    RECORDING_STATUS_FLAG_IS_PAUSED,
    RECORDING_STATUS_FLAG_IS_RECORDING,
    RECORDING_STATUS_FLAG_LAST_FRAME_OK,
    unmarshal_body_tracking_message,
    unmarshal_frame_metadata,
)
from .nats_subjects import (
    DEFAULT_NATS_URL,
    subject_body,
    subject_control_recorder_mcap_capabilities,
    subject_control_recorder_mcap_start,
    subject_control_recorder_mcap_status,
    subject_control_recorder_mcap_stop,
    subject_control_recorder_svo_capabilities,
    subject_control_recorder_svo_start,
    subject_control_recorder_svo_status,
    subject_control_recorder_svo_stop,
    subject_control_source_capabilities,
    subject_control_source_info,
    subject_control_source_reset,
    subject_control_source_seek,
    subject_status,
)
from .shm import SharedMemory

NDArray = np.ndarray
FrameMetadataAny = FrameMetadata | FrameMetadataV2

_INSTANCE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,22})$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,31})$")
_UNIX_PATH_MAX = 107
_FRAME_SUBSCRIPTION_PREFIX = bytes([FRAME_TOPIC_MAGIC])
_CONTROL_RESPONSE_LABELS = {
    CONTROL_RESPONSE_OK: "OK",
    CONTROL_RESPONSE_UNKNOWN_CMD: "UNKNOWN_CMD",
    CONTROL_RESPONSE_ERROR: "ERROR",
    CONTROL_RESPONSE_INVALID_MAGIC: "INVALID_MAGIC",
    CONTROL_RESPONSE_INVALID_LABEL: "INVALID_LABEL",
    CONTROL_RESPONSE_INVALID_VERSION: "INVALID_VERSION",
    CONTROL_RESPONSE_INVALID_MSG_SIZE: "INVALID_MSG_SIZE",
    CONTROL_RESPONSE_UNSUPPORTED: "UNSUPPORTED",
    CONTROL_RESPONSE_INVALID_PAYLOAD: "INVALID_PAYLOAD",
    CONTROL_RESPONSE_OUT_OF_RANGE: "OUT_OF_RANGE",
    CONTROL_RESPONSE_TIMEOUT: "TIMEOUT",
}
_PROTO_TO_CONTROL_ERROR = {
    control_pb2.ERROR_CODE_OK: CONTROL_RESPONSE_OK,
    control_pb2.ERROR_CODE_UNKNOWN_CMD: CONTROL_RESPONSE_UNKNOWN_CMD,
    control_pb2.ERROR_CODE_ERROR: CONTROL_RESPONSE_ERROR,
    control_pb2.ERROR_CODE_UNSUPPORTED: CONTROL_RESPONSE_UNSUPPORTED,
    control_pb2.ERROR_CODE_INVALID_PAYLOAD: CONTROL_RESPONSE_INVALID_PAYLOAD,
    control_pb2.ERROR_CODE_OUT_OF_RANGE: CONTROL_RESPONSE_OUT_OF_RANGE,
    control_pb2.ERROR_CODE_TIMEOUT: CONTROL_RESPONSE_TIMEOUT,
}
_PROTO_TO_MODULE_STATUS = {
    control_pb2.MODULE_STATUS_CODE_ONLINE: MODULE_STATUS_ONLINE,
    control_pb2.MODULE_STATUS_CODE_OFFLINE: MODULE_STATUS_OFFLINE,
    control_pb2.MODULE_STATUS_CODE_STREAM_RESET: MODULE_STATUS_STREAM_RESET,
}


class _ResolvedTarget(NamedTuple):
    instance: str
    namespace: str
    prefix: str
    base_name: str
    nats_target_key: str


def _validate_prefix(prefix: str) -> str:
    if not prefix.startswith("/"):
        raise ValueError("cvmmap uri prefix must be an absolute path")
    if "\\" in prefix or "\x00" in prefix:
        raise ValueError("cvmmap uri prefix contains invalid characters")
    if "//" in prefix:
        raise ValueError("cvmmap uri prefix must not contain empty path segments")

    normalized = prefix if prefix == "/" else prefix.rstrip("/")
    for segment in normalized.split("/"):
        if segment in {".", ".."}:
            raise ValueError("cvmmap uri prefix must not contain traversal segments")
    return normalized


def _resolve_target(name_or_uri: str) -> _ResolvedTarget:
    default_namespace = "cvmmap"
    default_prefix = "/tmp"

    if name_or_uri.startswith("cvmmap://"):
        body = name_or_uri[len("cvmmap://") :]
        authority, sep, query = body.partition("?")
        if sep and not query:
            raise ValueError("cvmmap uri query string is empty")
        if "@" in authority:
            if authority.count("@") != 1:
                raise ValueError("cvmmap uri authority must contain at most one '@'")
            instance, prefix = authority.split("@", 1)
            if not prefix:
                raise ValueError("cvmmap uri prefix is empty")
        else:
            instance = authority
            prefix = default_prefix

        namespace = default_namespace
        if query:
            parts = query.split("&")
            if any(not part for part in parts):
                raise ValueError("cvmmap uri query contains empty key/value")
            for part in parts:
                key, eq, value = part.partition("=")
                if not eq:
                    raise ValueError("cvmmap uri query must be key=value pairs")
                if key != "namespace":
                    raise ValueError(f"unsupported cvmmap uri query key: {key}")
                if not value:
                    raise ValueError("cvmmap uri namespace is empty")
                namespace = value
    else:
        instance = name_or_uri
        prefix = default_prefix
        namespace = default_namespace
        if any(ch in instance for ch in ("@", "?", "/")):
            raise ValueError(
                "plain cvmmap instance names must not contain '@', '?', or '/'"
            )

    if not _INSTANCE_RE.fullmatch(instance):
        raise ValueError(
            "invalid cvmmap instance; expected [A-Za-z0-9][A-Za-z0-9._-]{0,22}"
        )
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "invalid cvmmap namespace; expected [A-Za-z0-9][A-Za-z0-9._-]{0,31}"
        )

    normalized_prefix = _validate_prefix(prefix)
    base_name = f"{namespace}_{instance}"
    sync_path = f"{normalized_prefix}/{base_name}"
    if len(sync_path) > _UNIX_PATH_MAX:
        raise ValueError(
            f"cvmmap ipc path too long ({len(sync_path)}>{_UNIX_PATH_MAX})"
        )

    return _ResolvedTarget(
        instance=instance,
        namespace=namespace,
        prefix=normalized_prefix,
        base_name=base_name,
        nats_target_key=base_name.replace(".", "_"),
    )


def _import_nats() -> Any:
    return importlib.import_module("nats")


def _proto_error_to_control_code(error_code: int) -> int:
    return _PROTO_TO_CONTROL_ERROR.get(error_code, CONTROL_RESPONSE_ERROR)


def _format_control_failure(command_name: str, response_code: int) -> str:
    label = _CONTROL_RESPONSE_LABELS.get(response_code, "UNKNOWN")
    return f"{command_name} failed with {label} ({response_code})"


def _module_status_from_proto(event: control_pb2.ModuleStatusEvent) -> int:
    return _PROTO_TO_MODULE_STATUS.get(event.status, MODULE_STATUS_ONLINE)


def _recording_subject(
    recording_format: int,
    *,
    svo_subject: str,
    mcap_subject: str,
) -> str:
    if recording_format == RECORDING_FORMAT_SVO:
        return svo_subject
    if recording_format == RECORDING_FORMAT_MCAP:
        return mcap_subject
    raise ValueError("recording_format is required")


def _recording_flags_from_pb(response: control_pb2.RecordingStatusResponse) -> int:
    flags = 0
    if response.can_record:
        flags |= RECORDING_STATUS_FLAG_CAN_RECORD
    if response.is_recording:
        flags |= RECORDING_STATUS_FLAG_IS_RECORDING
    if response.is_paused:
        flags |= RECORDING_STATUS_FLAG_IS_PAUSED
    if response.last_frame_ok:
        flags |= RECORDING_STATUS_FLAG_LAST_FRAME_OK
    return flags


def _recording_status_from_pb(
    response: control_pb2.RecordingStatusResponse,
) -> RecordingStatus:
    return RecordingStatus(
        recording_format=response.format,
        flags=_recording_flags_from_pb(response),
        active_path=response.active_path,
        frames_ingested=response.frames_ingested,
        frames_encoded=response.frames_encoded,
        error_message=response.error_message,
    )


def _source_info_from_pb(response: control_pb2.GetSourceInfoResponse) -> SourceInfo:
    return SourceInfo(
        source_kind=response.source_kind,
        timestamp_domain=response.timestamp_domain,
        flags=response.flags,
        timeline_start_ns=response.timeline_start_ns,
        timeline_end_ns=response.timeline_end_ns,
        duration_ns=response.duration_ns,
        current_timestamp_ns=response.current_timestamp_ns,
        current_frame_count=response.current_frame_count,
    )


def _seek_result_from_pb(response: control_pb2.SeekTimestampResponse) -> SeekResult:
    return SeekResult(
        requested_timestamp_ns=response.requested_timestamp_ns,
        landed_timestamp_ns=response.landed_timestamp_ns,
        landed_frame_count=response.landed_frame_count,
        exact_match=response.exact_match,
    )


def _legacy_control_response(
    command_id: int,
    response_code: int,
    response_message: bytes = b"",
) -> ControlMessageResponse:
    return ControlMessageResponse(
        command_id=command_id,
        response_code=response_code,
        label="",
        response_message=response_message,
    )


def _marshal_source_info_payload(response: control_pb2.GetSourceInfoResponse) -> bytes:
    return struct.pack(
        SourceInfo.PACK_FMT,
        SourceInfo.size(),
        response.source_kind,
        response.timestamp_domain,
        response.flags,
        response.timeline_start_ns,
        response.timeline_end_ns,
        response.duration_ns,
        response.current_timestamp_ns,
        response.current_frame_count,
        0,
    )


def _marshal_seek_result_payload(
    response: control_pb2.SeekTimestampResponse,
) -> bytes:
    return struct.pack(
        SeekResult.PACK_FMT,
        SeekResult.size(),
        1 if response.exact_match else 0,
        0,
        response.requested_timestamp_ns,
        response.landed_timestamp_ns,
        response.landed_frame_count,
        0,
    )


def _marshal_recording_status_payload(
    response: control_pb2.RecordingStatusResponse,
) -> bytes:
    status = _recording_status_from_pb(response)
    encoded_path = status.active_path.encode("utf-8")
    return struct.pack(
        RecordingStatus.PACK_FMT,
        RecordingStatus.size(),
        status.recording_format,
        0,
        status.flags,
        len(encoded_path),
        status.frames_ingested,
        status.frames_encoded,
        0,
    ) + encoded_path


def _parse_legacy_seek_request(payload: bytes) -> int:
    if len(payload) < SeekTimestampRequest.size():
        raise ValueError(
            f"seek request payload too short: {len(payload)} < {SeekTimestampRequest.size()}"
        )
    struct_size, _reserved, target_timestamp_ns = struct.unpack(
        SeekTimestampRequest.PACK_FMT,
        payload[: SeekTimestampRequest.size()],
    )
    if struct_size < SeekTimestampRequest.size():
        raise ValueError(
            f"invalid seek request payload size: {struct_size} < {SeekTimestampRequest.size()}"
        )
    return int(target_timestamp_ns)


def _parse_legacy_recording_start_request(payload: bytes) -> str:
    if len(payload) < RecordingStartRequest.size():
        raise ValueError(
            "recording start payload too short: "
            f"{len(payload)} < {RecordingStartRequest.size()}"
        )
    struct_size, _flags, path_length, _reserved = struct.unpack(
        RecordingStartRequest.PACK_FMT,
        payload[: RecordingStartRequest.size()],
    )
    if struct_size < RecordingStartRequest.size():
        raise ValueError(
            "invalid recording start payload size: "
            f"{struct_size} < {RecordingStartRequest.size()}"
        )

    total_size = RecordingStartRequest.size() + path_length
    if len(payload) < total_size:
        raise ValueError(f"recording start path truncated: {len(payload)} < {total_size}")
    return payload[RecordingStartRequest.size() : total_size].decode("utf-8")


def _schedule_nats_close(client: Any | None) -> None:
    if client is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        with suppress(Exception):
            asyncio.run(client.close())
        return

    with suppress(RuntimeError):
        loop.create_task(client.close())


class _NatsMixin:
    _nats_url: str
    _nats: Any | None

    async def _ensure_nats(self) -> Any:
        if self._nats is None:
            nats = _import_nats()
            self._nats = await nats.connect(servers=[self._nats_url])
        return self._nats

    def _close_nats(self) -> None:
        client = getattr(self, "_nats", None)
        self._nats = None
        _schedule_nats_close(client)


class CvMmapClient(_NatsMixin):
    """Shared-memory frame client with NATS status/body control plane."""

    _SHM_PAYLOAD_OFFSET: int = 256

    def __init__(self, name_or_uri: str, *, nats_url: str = DEFAULT_NATS_URL):
        resolved = _resolve_target(name_or_uri)
        self._name = resolved.instance
        self._prefix = resolved.prefix
        self._namespace = resolved.namespace
        self._base_name = resolved.base_name
        self._target_key = resolved.nats_target_key
        self._nats_url = nats_url
        self._name_or_uri = name_or_uri

        self._ctx = Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(self.zmq_addr)
        self._sock.subscribe(_FRAME_SUBSCRIPTION_PREFIX)

        self._image_buffer = None
        self._shm = None
        self._nats = None
        self._status_queue: asyncio.Queue[int] = asyncio.Queue()
        self._status_subscription_ready = False

    @property
    def shm_name(self) -> str:
        return self._base_name

    @property
    def zmq_addr(self) -> str:
        return f"ipc://{self._prefix}/{self.shm_name}"

    @property
    def nats_target_key(self) -> str:
        return self._target_key

    async def _ensure_status_subscription(self) -> None:
        if self._status_subscription_ready:
            return

        nc = await self._ensure_nats()

        async def _on_status(message: Any) -> None:
            event = control_pb2.ModuleStatusEvent()
            try:
                event.ParseFromString(message.data)
            except Exception as exc:
                getLogger(__name__).warning(
                    "Discarding invalid NATS status payload for %s: %s",
                    self._name,
                    exc,
                )
                return
            await self._status_queue.put(_module_status_from_proto(event))

        await nc.subscribe(subject_status(self._target_key), cb=_on_status)
        self._status_subscription_ready = True

    def body_stream(self) -> "CvMmapBodyStream":
        return CvMmapBodyStream(self._name_or_uri, nats_url=self._nats_url)

    def _read_metadata(self) -> FrameMetadataAny:
        assert self._shm is not None, "Shared memory not attached"
        assert self._shm.buf is not None, "Shared memory buffer is None"
        metadata_raw = bytes(self._shm.buf[:FRAME_METADATA_REGION_SIZE])
        return unmarshal_frame_metadata(metadata_raw)

    def _read_metadata_unchecked(self) -> FrameMetadataAny:
        assert self._shm is not None, "Shared memory not attached"
        assert self._shm.buf is not None, "Shared memory buffer is None"
        metadata_raw = bytes(self._shm.buf[:FRAME_METADATA_REGION_SIZE])
        return unmarshal_frame_metadata(metadata_raw)

    def _payload_view(self, payload_size: int) -> memoryview:
        assert self._shm is not None, "Shared memory not attached"
        assert self._shm.buf is not None, "Shared memory buffer is None"

        if payload_size < 0:
            raise RuntimeError(f"Invalid payload size: {payload_size}")

        start = self._SHM_PAYLOAD_OFFSET
        end = start + payload_size
        shm_size = len(self._shm.buf)
        if end > shm_size:
            raise RuntimeError(
                f"Payload range out of shared memory bounds: [{start}, {end}) exceeds shm size {shm_size}"
            )
        return self._shm.buf[start:end]

    def left_plane(self, metadata: FrameMetadataAny) -> NDArray:
        if isinstance(metadata, FrameMetadataV2):
            payload = self._payload_view(metadata.header.payload_size_bytes)
            return metadata.left_plane(payload)

        payload = self._payload_view(metadata.info.buffer_size)
        shape = (metadata.info.height, metadata.info.width, metadata.info.channels)
        if self._image_buffer is not None and self._image_buffer.shape == shape:
            return self._image_buffer

        self._image_buffer = np.ndarray(shape, dtype=np.uint8, buffer=payload)
        return self._image_buffer

    def depth_plane(self, metadata: FrameMetadataAny) -> NDArray | None:
        if not isinstance(metadata, FrameMetadataV2):
            return None
        payload = self._payload_view(metadata.header.payload_size_bytes)
        return metadata.depth_plane(payload)

    def confidence_plane(self, metadata: FrameMetadataAny) -> NDArray | None:
        if not isinstance(metadata, FrameMetadataV2):
            return None
        payload = self._payload_view(metadata.header.payload_size_bytes)
        return metadata.confidence_plane(payload)

    def depth_unit(self, metadata: FrameMetadataAny) -> int:
        if not isinstance(metadata, FrameMetadataV2):
            return DEPTH_UNIT_UNKNOWN
        return metadata.depth_unit

    def _ensure_memory(self) -> None:
        if self._shm is not None:
            return
        self._shm = SharedMemory(
            name=self.shm_name,
            create=False,
            track=False,
            readonly=True,
        )
        self._read_metadata()

    async def __aiter__(self) -> AsyncGenerator[tuple[NDArray, FrameMetadataAny], None]:
        await self._ensure_status_subscription()

        while True:
            assert self._sock is not None, "Client socket is closed"
            recv_task = asyncio.create_task(self._sock.recv())
            status_task = asyncio.create_task(self._status_queue.get())
            done: set[asyncio.Task[Any]] = set()
            pending: set[asyncio.Task[Any]] = set()
            try:
                done, pending = await asyncio.wait(
                    {recv_task, status_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task

            if status_task in done:
                status = int(status_task.result())
                if status in (MODULE_STATUS_OFFLINE, MODULE_STATUS_STREAM_RESET):
                    return
                continue

            message = cast(bytes, recv_task.result())
            if len(message) < 1:
                raise RuntimeError("Received empty message")
            if message[0] != FRAME_TOPIC_MAGIC:
                raise RuntimeError(f"Unknown message magic: {message[0]:#x}")

            sync_message = SyncMessage.unmarshal(message)
            self._ensure_memory()

            if sync_message.label != self._name:
                raise RuntimeError(
                    f"Label mismatch: expected '{self._name}', got '{sync_message.label}'"
                )

            metadata = self._read_metadata_unchecked()
            yield self.left_plane(metadata), metadata

    def close(self) -> None:
        sock = getattr(self, "_sock", None)
        if sock is not None:
            sock.close()
            self._sock = None
        self._close_nats()

    def __del__(self) -> None:
        self.close()


class CvMmapBodyStream(_NatsMixin):
    """Async iterator over raw body-tracking payloads delivered via NATS."""

    def __init__(self, name_or_uri: str, *, nats_url: str = DEFAULT_NATS_URL):
        resolved = _resolve_target(name_or_uri)
        self._name = resolved.instance
        self._target_key = resolved.nats_target_key
        self._nats_url = nats_url
        self._nats = None
        self._body_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._status_queue: asyncio.Queue[int] = asyncio.Queue()
        self._subscriptions_ready = False

    async def _ensure_subscriptions(self) -> None:
        if self._subscriptions_ready:
            return

        nc = await self._ensure_nats()

        async def _on_body(message: Any) -> None:
            await self._body_queue.put(bytes(message.data))

        async def _on_status(message: Any) -> None:
            event = control_pb2.ModuleStatusEvent()
            try:
                event.ParseFromString(message.data)
            except Exception as exc:
                getLogger(__name__).warning(
                    "Discarding invalid NATS status payload for body stream %s: %s",
                    self._name,
                    exc,
                )
                return
            await self._status_queue.put(_module_status_from_proto(event))

        await nc.subscribe(subject_body(self._target_key), cb=_on_body)
        await nc.subscribe(subject_status(self._target_key), cb=_on_status)
        self._subscriptions_ready = True

    async def __aiter__(self) -> AsyncGenerator[BodyFrame, None]:
        await self._ensure_subscriptions()

        while True:
            body_task = asyncio.create_task(self._body_queue.get())
            status_task = asyncio.create_task(self._status_queue.get())
            done: set[asyncio.Task[Any]] = set()
            pending: set[asyncio.Task[Any]] = set()
            try:
                done, pending = await asyncio.wait(
                    {body_task, status_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task

            if status_task in done:
                status = int(status_task.result())
                if status in (MODULE_STATUS_OFFLINE, MODULE_STATUS_STREAM_RESET):
                    return
                continue

            message = body_task.result()
            if len(message) < 1:
                raise RuntimeError("Received empty body message")
            if message[0] != BODY_TRACKING_MAGIC:
                raise RuntimeError(
                    f"Unknown body stream message magic: {message[0]:#x}"
                )

            body_frame = unmarshal_body_tracking_message(message)
            if body_frame.label and body_frame.label != self._name:
                raise RuntimeError(
                    f"Body label mismatch: expected '{self._name}', got '{body_frame.label}'"
                )
            yield body_frame

    def close(self) -> None:
        self._close_nats()

    def __del__(self) -> None:
        self.close()


class CvMmapConfig(TypedDict):
    name: str
    shm_name: str | None
    zmq_addr: str | None
    nats_url: str | None


class CvMmapRequestClient(_NatsMixin):
    """NATS request/reply control client."""

    def __init__(self, name_or_uri: str, *, nats_url: str = DEFAULT_NATS_URL):
        resolved = _resolve_target(name_or_uri)
        self._name = resolved.instance
        self._prefix = resolved.prefix
        self._namespace = resolved.namespace
        self._base_name = resolved.base_name
        self._target_key = resolved.nats_target_key
        self._nats_url = nats_url
        self._nats = None

    @property
    def shm_name(self) -> str:
        return self._base_name

    @property
    def zmq_addr(self) -> str:
        return f"ipc://{self._prefix}/{self.shm_name}"

    @property
    def nats_target_key(self) -> str:
        return self._target_key

    async def _request_pb(
        self,
        subject: str,
        request_message: Any,
        response_type: type[Any],
        timeout_ms: int,
    ) -> Any:
        nc = await self._ensure_nats()
        try:
            message = await nc.request(
                subject,
                request_message.SerializeToString(),
                timeout=timeout_ms / 1000.0,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "TimeoutError":
                raise TimeoutError(f"No response received within {timeout_ms}ms") from exc
            raise RuntimeError(f"NATS request failed for '{subject}': {exc}") from exc

        response = response_type()
        response.ParseFromString(message.data)
        return response

    async def send_request(
        self,
        command_id: int,
        request_message: bytes = b"",
        timeout_ms: int = 5000,
    ) -> ControlMessageResponse:
        if command_id == CONTROL_MSG_CMD_RESET_FRAME_COUNT:
            response = await self._request_pb(
                subject_control_source_reset(self._target_key),
                control_pb2.ResetFrameCountRequest(),
                control_pb2.ResetFrameCountResponse,
                timeout_ms,
            )
            return _legacy_control_response(
                command_id,
                _proto_error_to_control_code(response.error),
            )

        if command_id == CONTROL_MSG_CMD_GET_SOURCE_INFO:
            response = await self._request_pb(
                subject_control_source_info(self._target_key),
                control_pb2.GetSourceInfoRequest(),
                control_pb2.GetSourceInfoResponse,
                timeout_ms,
            )
            response_code = _proto_error_to_control_code(response.error)
            payload = (
                _marshal_source_info_payload(response)
                if response_code == CONTROL_RESPONSE_OK
                else b""
            )
            return _legacy_control_response(command_id, response_code, payload)

        if command_id == CONTROL_MSG_CMD_SEEK_TIMESTAMP_NS:
            parsed = _parse_legacy_seek_request(request_message)
            request = control_pb2.SeekTimestampRequest()
            request.target_timestamp_ns = parsed
            response = await self._request_pb(
                subject_control_source_seek(self._target_key),
                request,
                control_pb2.SeekTimestampResponse,
                timeout_ms,
            )
            response_code = _proto_error_to_control_code(response.error)
            payload = (
                _marshal_seek_result_payload(response)
                if response_code == CONTROL_RESPONSE_OK
                else b""
            )
            return _legacy_control_response(command_id, response_code, payload)

        if command_id == CONTROL_MSG_CMD_START_RECORDING:
            parsed = _parse_legacy_recording_start_request(request_message)
            request = control_pb2.RecordingStartRequest()
            request.output_path = parsed
            response = await self._request_pb(
                subject_control_recorder_svo_start(self._target_key),
                request,
                control_pb2.RecordingStatusResponse,
                timeout_ms,
            )
            response_code = _proto_error_to_control_code(response.error)
            payload = (
                _marshal_recording_status_payload(response)
                if response_code == CONTROL_RESPONSE_OK
                else b""
            )
            return _legacy_control_response(command_id, response_code, payload)

        if command_id == CONTROL_MSG_CMD_STOP_RECORDING:
            response = await self._request_pb(
                subject_control_recorder_svo_stop(self._target_key),
                control_pb2.RecordingStopRequest(),
                control_pb2.RecordingStatusResponse,
                timeout_ms,
            )
            response_code = _proto_error_to_control_code(response.error)
            payload = (
                _marshal_recording_status_payload(response)
                if response_code == CONTROL_RESPONSE_OK
                else b""
            )
            return _legacy_control_response(command_id, response_code, payload)

        if command_id == CONTROL_MSG_CMD_GET_RECORDING_STATUS:
            response = await self._request_pb(
                subject_control_recorder_svo_status(self._target_key),
                control_pb2.RecordingStatusRequest(),
                control_pb2.RecordingStatusResponse,
                timeout_ms,
            )
            response_code = _proto_error_to_control_code(response.error)
            payload = (
                _marshal_recording_status_payload(response)
                if response_code == CONTROL_RESPONSE_OK
                else b""
            )
            return _legacy_control_response(command_id, response_code, payload)

        return _legacy_control_response(command_id, CONTROL_RESPONSE_UNKNOWN_CMD)

    async def reset_frame_count(self, timeout_ms: int = 5000) -> int:
        response = await self._request_pb(
            subject_control_source_reset(self._target_key),
            control_pb2.ResetFrameCountRequest(),
            control_pb2.ResetFrameCountResponse,
            timeout_ms,
        )
        return _proto_error_to_control_code(response.error)

    async def get_source_info(self, timeout_ms: int = 5000) -> SourceInfo:
        response = await self._request_pb(
            subject_control_source_info(self._target_key),
            control_pb2.GetSourceInfoRequest(),
            control_pb2.GetSourceInfoResponse,
            timeout_ms,
        )
        if response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "GET_SOURCE_INFO",
                    _proto_error_to_control_code(response.error),
                )
            )
        return _source_info_from_pb(response)

    async def seek_timestamp_ns(
        self,
        target_timestamp_ns: int,
        timeout_ms: int = 5000,
    ) -> SeekResult:
        request = control_pb2.SeekTimestampRequest()
        request.target_timestamp_ns = target_timestamp_ns
        response = await self._request_pb(
            subject_control_source_seek(self._target_key),
            request,
            control_pb2.SeekTimestampResponse,
            timeout_ms,
        )
        if response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "SEEK_TIMESTAMP_NS",
                    _proto_error_to_control_code(response.error),
                )
            )
        return _seek_result_from_pb(response)

    async def get_capabilities(self, timeout_ms: int = 5000) -> ControlCapabilities:
        source_response = await self._request_pb(
            subject_control_source_capabilities(self._target_key),
            control_pb2.CapabilitiesRequest(),
            control_pb2.CapabilitiesResponse,
            timeout_ms,
        )
        if source_response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "GET_CAPABILITIES",
                    _proto_error_to_control_code(source_response.error),
                )
            )

        capabilities = ControlCapabilities(
            can_seek=source_response.can_seek,
            available_recording_formats=list(source_response.available_recording_formats),
        )

        async def _merge_recorder(subject: str) -> None:
            try:
                response = await self._request_pb(
                    subject,
                    control_pb2.CapabilitiesRequest(),
                    control_pb2.CapabilitiesResponse,
                    timeout_ms,
                )
            except TimeoutError:
                return
            except RuntimeError:
                return

            if response.error != control_pb2.ERROR_CODE_OK:
                return

            for recording_format in response.available_recording_formats:
                if not capabilities.supports_recording_format(recording_format):
                    capabilities.available_recording_formats.append(recording_format)

        await _merge_recorder(
            subject_control_recorder_svo_capabilities(self._target_key)
        )
        await _merge_recorder(
            subject_control_recorder_mcap_capabilities(self._target_key)
        )
        return capabilities

    async def start_recording(
        self,
        request_or_output_path: RecordingRequest | str,
        timeout_ms: int = 5000,
    ) -> RecordingStatus:
        if isinstance(request_or_output_path, str):
            request_model = RecordingRequest(
                recording_format=RECORDING_FORMAT_SVO,
                output_path=request_or_output_path,
            )
        else:
            request_model = request_or_output_path

        if not request_model.output_path:
            raise ValueError("output_path must not be empty")

        request = control_pb2.RecordingStartRequest()
        request.output_path = request_model.output_path

        if request_model.recording_format == RECORDING_FORMAT_SVO:
            if request_model.mcap_options is not None:
                raise ValueError("MCAP options are invalid for SVO recording")
            if request_model.svo_options is not None:
                options = request.svo_options
                if request_model.svo_options.compression_mode is not None:
                    options.compression_mode = request_model.svo_options.compression_mode
                if request_model.svo_options.bitrate is not None:
                    options.bitrate = request_model.svo_options.bitrate
                if request_model.svo_options.target_framerate is not None:
                    options.target_framerate = (
                        request_model.svo_options.target_framerate
                    )
                if request_model.svo_options.transcode_streaming_input is not None:
                    options.transcode_streaming_input = (
                        request_model.svo_options.transcode_streaming_input
                    )
        elif request_model.recording_format == RECORDING_FORMAT_MCAP:
            if request_model.svo_options is not None:
                raise ValueError("SVO options are invalid for MCAP recording")
            if request_model.mcap_options is not None:
                options = request.mcap_options
                if request_model.mcap_options.compression is not None:
                    options.compression = request_model.mcap_options.compression
                if request_model.mcap_options.topic is not None:
                    options.topic = request_model.mcap_options.topic
                if request_model.mcap_options.depth_topic is not None:
                    options.depth_topic = request_model.mcap_options.depth_topic
                if request_model.mcap_options.body_topic is not None:
                    options.body_topic = request_model.mcap_options.body_topic
                if request_model.mcap_options.frame_id is not None:
                    options.frame_id = request_model.mcap_options.frame_id
        else:
            raise ValueError("recording_format is required")

        response = await self._request_pb(
            _recording_subject(
                request_model.recording_format,
                svo_subject=subject_control_recorder_svo_start(self._target_key),
                mcap_subject=subject_control_recorder_mcap_start(self._target_key),
            ),
            request,
            control_pb2.RecordingStatusResponse,
            timeout_ms,
        )
        if response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "START_RECORDING",
                    _proto_error_to_control_code(response.error),
                )
            )
        return _recording_status_from_pb(response)

    async def stop_recording(
        self,
        recording_format: int = RECORDING_FORMAT_SVO,
        timeout_ms: int = 5000,
    ) -> RecordingStatus:
        response = await self._request_pb(
            _recording_subject(
                recording_format,
                svo_subject=subject_control_recorder_svo_stop(self._target_key),
                mcap_subject=subject_control_recorder_mcap_stop(self._target_key),
            ),
            control_pb2.RecordingStopRequest(),
            control_pb2.RecordingStatusResponse,
            timeout_ms,
        )
        if response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "STOP_RECORDING",
                    _proto_error_to_control_code(response.error),
                )
            )
        return _recording_status_from_pb(response)

    async def get_recording_status(
        self,
        recording_format: int = RECORDING_FORMAT_SVO,
        timeout_ms: int = 5000,
    ) -> RecordingStatus:
        response = await self._request_pb(
            _recording_subject(
                recording_format,
                svo_subject=subject_control_recorder_svo_status(self._target_key),
                mcap_subject=subject_control_recorder_mcap_status(self._target_key),
            ),
            control_pb2.RecordingStatusRequest(),
            control_pb2.RecordingStatusResponse,
            timeout_ms,
        )
        if response.error != control_pb2.ERROR_CODE_OK:
            raise RuntimeError(
                _format_control_failure(
                    "GET_RECORDING_STATUS",
                    _proto_error_to_control_code(response.error),
                )
            )
        return _recording_status_from_pb(response)

    def close(self) -> None:
        self._close_nats()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "CvMmapBodyStream",
    "CvMmapClient",
    "CvMmapConfig",
    "CvMmapRequestClient",
    "BodyFrame",
    "BodyTrack",
    "BodyTrackingMessageHeader",
    "ControlCapabilities",
    "ControlMessageRequest",
    "ControlMessageResponse",
    "FrameInfo",
    "FrameMetadata",
    "FrameMetadataV2",
    "FrameMetadataV2Header",
    "FramePlaneDescriptorV2",
    "McapRecordingOptions",
    "ModuleStatusMessage",
    "RecordingRequest",
    "RecordingStartRequest",
    "RecordingStatus",
    "SeekResult",
    "SeekTimestampRequest",
    "SourceInfo",
    "SvoRecordingOptions",
    "SyncMessage",
    "BODY_TRACKING_MAGIC",
    "CONTROL_MSG_CMD_GENERIC",
    "CONTROL_MSG_CMD_GET_RECORDING_STATUS",
    "CONTROL_MSG_CMD_GET_SOURCE_INFO",
    "CONTROL_MSG_CMD_RESET_FRAME_COUNT",
    "CONTROL_MSG_CMD_SEEK_TIMESTAMP_NS",
    "CONTROL_MSG_CMD_START_RECORDING",
    "CONTROL_MSG_CMD_STOP_RECORDING",
    "CONTROL_RESPONSE_ERROR",
    "CONTROL_RESPONSE_INVALID_LABEL",
    "CONTROL_RESPONSE_INVALID_MAGIC",
    "CONTROL_RESPONSE_INVALID_MSG_SIZE",
    "CONTROL_RESPONSE_INVALID_PAYLOAD",
    "CONTROL_RESPONSE_INVALID_VERSION",
    "CONTROL_RESPONSE_OK",
    "CONTROL_RESPONSE_OUT_OF_RANGE",
    "CONTROL_RESPONSE_TIMEOUT",
    "CONTROL_RESPONSE_UNKNOWN_CMD",
    "CONTROL_RESPONSE_UNSUPPORTED",
    "CV_MMAP_MAGIC",
    "CV_MMAP_MAGIC_LEN",
    "DEFAULT_NATS_URL",
    "DEPTH_UNIT_METER",
    "DEPTH_UNIT_MILLIMETER",
    "DEPTH_UNIT_UNKNOWN",
    "FRAME_METADATA_REGION_SIZE",
    "FRAME_TOPIC_MAGIC",
    "MODULE_STATUS_OFFLINE",
    "MODULE_STATUS_ONLINE",
    "MODULE_STATUS_STREAM_RESET",
    "RECORDING_FORMAT_MCAP",
    "RECORDING_FORMAT_SVO",
    "RECORDING_FORMAT_UNKNOWN",
    "RECORDING_STATUS_FLAG_CAN_RECORD",
    "RECORDING_STATUS_FLAG_IS_PAUSED",
    "RECORDING_STATUS_FLAG_IS_RECORDING",
    "RECORDING_STATUS_FLAG_LAST_FRAME_OK",
    "SOURCE_INFO_FLAG_AUTO_LOOP",
    "SOURCE_INFO_FLAG_CAN_RECORD",
    "SOURCE_INFO_FLAG_CAN_SEEK",
    "SOURCE_INFO_FLAG_HAS_BODY",
    "SOURCE_INFO_FLAG_HAS_DEPTH",
    "SOURCE_KIND_FINITE",
    "SOURCE_KIND_LIVE",
    "SOURCE_KIND_UNKNOWN",
    "TIMESTAMP_DOMAIN_MEDIA_TIME_NS",
    "TIMESTAMP_DOMAIN_UNIX_EPOCH_NS",
    "TIMESTAMP_DOMAIN_UNKNOWN",
    "unmarshal_body_tracking_message",
    "unmarshal_frame_metadata",
]
