from logging import getLogger
import importlib
import re
from typing import (
    AsyncGenerator,
    cast,
    TypedDict,
    NamedTuple,
)

import numpy as np

zmq = importlib.import_module("zmq")
_zmq_asyncio = importlib.import_module("zmq.asyncio")
Context = _zmq_asyncio.Context
Poller = _zmq_asyncio.Poller
Socket = _zmq_asyncio.Socket

from .msg import (
    SyncMessage,
    BodyFrame,
    BodyTrack,
    BodyTrackingMessageHeader,
    FrameMetadata,
    FrameMetadataV2,
    FrameMetadataV2Header,
    FramePlaneDescriptorV2,
    FrameInfo,
    ModuleStatusMessage,
    ControlMessageRequest,
    ControlMessageResponse,
    DEPTH_UNIT_UNKNOWN,
    DEPTH_UNIT_MILLIMETER,
    DEPTH_UNIT_METER,
    FRAME_TOPIC_MAGIC,
    MODULE_STATUS_MAGIC,
    BODY_TRACKING_MAGIC,
    CV_MMAP_MAGIC,
    CV_MMAP_MAGIC_LEN,
    CONTROL_MSG_CMD_GENERIC,
    CONTROL_MSG_CMD_RESET_FRAME_COUNT,
    CONTROL_RESPONSE_OK,
    MODULE_STATUS_OFFLINE,
    MODULE_STATUS_STREAM_RESET,
    unmarshal_frame_metadata,
    unmarshal_body_tracking_message,
    FRAME_METADATA_REGION_SIZE,
)
from .shm import SharedMemory

NDArray = np.ndarray
FrameMetadataAny = FrameMetadata | FrameMetadataV2

_INSTANCE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,22})$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,31})$")
_UNIX_PATH_MAX = 107


class _ResolvedTarget(NamedTuple):
    instance: str
    namespace: str
    prefix: str
    base_name: str


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
    control_suffix = f"{normalized_prefix}/{base_name}_control"
    if len(control_suffix) > _UNIX_PATH_MAX:
        raise ValueError(
            f"cvmmap ipc path too long ({len(control_suffix)}>{_UNIX_PATH_MAX})"
        )

    return _ResolvedTarget(
        instance=instance,
        namespace=namespace,
        prefix=normalized_prefix,
        base_name=base_name,
    )


# Re-export message types for convenience
__all__ = [
    "CvMmapClient",
    "CvMmapBodyStream",
    "CvMmapRequestClient",
    "CvMmapConfig",
    "SyncMessage",
    "BodyFrame",
    "BodyTrack",
    "BodyTrackingMessageHeader",
    "FrameMetadata",
    "FrameMetadataV2",
    "FrameMetadataV2Header",
    "FramePlaneDescriptorV2",
    "FrameInfo",
    "DEPTH_UNIT_UNKNOWN",
    "DEPTH_UNIT_MILLIMETER",
    "DEPTH_UNIT_METER",
    "ModuleStatusMessage",
    "ControlMessageRequest",
    "ControlMessageResponse",
]


class CvMmapClient:
    """
    A client for the CvMmap protocol
    """

    _name: str
    _prefix: str
    _namespace: str
    _base_name: str

    _ctx: Context
    _sock: Socket
    _poller: Poller

    _image_buffer: NDArray | None = None
    _shm: SharedMemory | None = None

    @property
    def shm_name(self) -> str:
        """Get the shared memory name used by this client."""
        return self._base_name

    @property
    def zmq_addr(self) -> str:
        """Get the ZMQ address used by this client."""
        return f"ipc://{self._prefix}/{self.shm_name}"

    @property
    def zmq_body_addr(self) -> str:
        """Get the ZMQ body address used by this client."""
        return f"ipc://{self._prefix}/{self.shm_name}_body"

    def _subscribe(self):
        """
        Manually trigger the subscription to the topic.

        CRITICAL ARCHITECTURE NOTE (ZeroMQ Bug):
        We MUST subscribe to all topics (empty string `b""`) because we are using
        the ZMQ_CONFLATE socket option (keep only the latest message).

        In ZeroMQ, conflation happens before/alongside filtering, and it uses a
        hardcoded queue size of exactly 1 message TOTAL for the socket. If the
        server publishes a FRAME_TOPIC followed immediately by a MODULE_STATUS:
        1. The queue receives FRAME_TOPIC.
        2. The queue immediately overwrites it with MODULE_STATUS (conflate behavior).
        3. If we only subscribed to FRAME_TOPIC, ZMQ looks at the single item
           (MODULE_STATUS), says "this doesn't match", drops it, and we receive nothing.

        By subscribing to everything, we guarantee the `recv()` call gives us
        whatever was sent last, and we manually route it by checking `msg[0]`
        (the magic byte) in `__aiter__`.

        References:
        - https://github.com/zeromq/libzmq/issues/1688
        - https://stackoverflow.com/questions/57901180/only-keep-latest-multipart-message-in-subscriber-with-pyzmq-pub-sub-socket
        """
        self._sock.subscribe(b"")

    def _unsubscribe(self):
        """
        Manually trigger the un-subscription to the topic.
        Since we subscribe to everything (due to CONFLATE limitations), we
        must also unsubscribe from everything.
        """
        self._sock.unsubscribe(b"")

    def __init__(
        self,
        name_or_uri: str,
    ):
        """Create a CvMmapClient.

        Parameters
        ----------
        name_or_uri
            Either a plain instance name (e.g. ``default``) or URI form
            ``cvmmap://<instance>[@<prefix>][?namespace=<namespace>]``.
        """

        resolved = _resolve_target(name_or_uri)
        self._name = resolved.instance
        self._prefix = resolved.prefix
        self._namespace = resolved.namespace
        self._base_name = resolved.base_name

        self._ctx = Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        # In Python, you set the CONFLATE option before you connect to the socket
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(self.zmq_addr)
        self._subscribe()
        self._poller = Poller()
        self._poller.register(self._sock, zmq.POLLIN)

        self._image_buffer = None
        self._shm = None

    def body_stream(self) -> "CvMmapBodyStream":
        return CvMmapBodyStream(self._name, self.zmq_body_addr)

    _SHM_PAYLOAD_OFFSET: int = 256

    def _read_metadata(self) -> FrameMetadataAny:
        """Read and decode the `FrameMetadata` structure from shared memory.

        The memory layout written by the C++ producer is:

        ```
        0-255 : SHM metadata region (v1 or v2, auto-detected)
        256-… : payload region containing the packed active planes
        ```

        This function validates the magic prefix and dispatches to the
        matching Python struct definition for the metadata major version.
        """
        assert self._shm is not None, "Shared memory not attached"
        assert self._shm.buf is not None, "Shared memory buffer is None"

        metadata_raw = bytes(self._shm.buf[:FRAME_METADATA_REGION_SIZE])
        return unmarshal_frame_metadata(metadata_raw)

    def _read_metadata_unchecked(self) -> FrameMetadataAny:
        """
        Read and decode the `FrameMetadata` structure from shared memory directly without checking the magic
        """
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

        self._image_buffer = np.ndarray(
            shape,
            dtype=np.uint8,
            buffer=payload,
        )
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

    def _ensure_memory(self):
        """Attach to shared memory and initialize the numpy view if necessary."""
        if self._shm is not None:
            return

        if self._shm is None:
            self._shm = SharedMemory(  # pylint: disable=unexpected-keyword-arg
                name=self.shm_name, create=False, track=False, readonly=True
            )

        self._read_metadata()

    async def __aiter__(self) -> AsyncGenerator[tuple[NDArray, FrameMetadataAny], None]:
        """
        Asynchronous generator that yields numpy array of image.

        Raises
        ------
        StopAsyncIteration
            When module goes offline or stream is reset.
        RuntimeError
            When label mismatch or other errors occur.
        """
        while True:
            events = await self._poller.poll()
            for socket, event in events:
                if event & zmq.POLLIN:
                    message = await socket.recv()
                    message = cast(bytes, message)

                    # Check the magic byte to determine message type
                    if len(message) < 1:
                        raise RuntimeError("Received empty message")

                    magic = message[0]

                    if magic == MODULE_STATUS_MAGIC:
                        # Handle module status message
                        status_msg = ModuleStatusMessage.unmarshal(message)
                        if status_msg.module_status in (
                            MODULE_STATUS_OFFLINE,
                            MODULE_STATUS_STREAM_RESET,
                        ):
                            status_name = (
                                "OFFLINE"
                                if status_msg.module_status == MODULE_STATUS_OFFLINE
                                else "STREAM_RESET"
                            )
                            getLogger(__name__).info(
                                "Module '%s' status: %s, stopping generator",
                                status_msg.label,
                                status_name,
                            )
                            return
                        continue

                    if magic == FRAME_TOPIC_MAGIC:
                        # Handle sync message (frame notification)
                        sync_message = SyncMessage.unmarshal(message)
                        self._ensure_memory()

                        if sync_message.label != self._name:
                            raise RuntimeError(
                                f"Label mismatch: expected '{self._name}', got '{sync_message.label}'"
                            )

                        metadata = self._read_metadata_unchecked()
                        yield self.left_plane(metadata), metadata
                    else:
                        raise RuntimeError(f"Unknown message magic: {magic:#x}")


class CvMmapBodyStream:
    """Async iterator over the optional body tracking PUB substream."""

    def __init__(self, instance_name: str, zmq_body_addr: str):
        self._name = instance_name
        self._zmq_body_addr = zmq_body_addr
        self._ctx = Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(self._zmq_body_addr)
        self._sock.subscribe(b"")
        self._poller = Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    async def __aiter__(self) -> AsyncGenerator[BodyFrame, None]:
        while True:
            events = await self._poller.poll()
            for socket, event in events:
                if not (event & zmq.POLLIN):
                    continue
                message = cast(bytes, await socket.recv())
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
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __del__(self) -> None:
        self.close()


class CvMmapConfig(TypedDict):
    name: str
    # Optional overrides for non-standard setups
    shm_name: str | None
    zmq_addr: str | None


class CvMmapRequestClient:
    """
    A client for sending control requests to the CvMmap server.

    Uses ZMQ REQ/REP pattern to send control messages and receive responses.
    """

    _name: str
    _prefix: str
    _namespace: str
    _base_name: str

    _ctx: Context
    _sock: Socket

    @property
    def shm_name(self) -> str:
        """Get the shared memory name used by this client."""
        return self._base_name

    @property
    def zmq_addr(self) -> str:
        """Get the ZMQ address used by this client."""
        return f"ipc://{self._prefix}/{self.shm_name}_control"

    def __init__(self, name_or_uri: str):
        """Create a CvMmapRequestClient.

        Parameters
        ----------
        name_or_uri
            Either a plain instance name (e.g. ``default``) or URI form
            ``cvmmap://<instance>[@<prefix>][?namespace=<namespace>]``.
        """
        resolved = _resolve_target(name_or_uri)
        self._name = resolved.instance
        self._prefix = resolved.prefix
        self._namespace = resolved.namespace
        self._base_name = resolved.base_name

        self._ctx = Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(self.zmq_addr)

    async def send_request(
        self,
        command_id: int,
        request_message: bytes = b"",
        timeout_ms: int = 5000,
    ) -> ControlMessageResponse:
        """
        Send a control request and wait for response.

        Parameters
        ----------
        command_id
            The command ID to send.
        request_message
            Optional additional data to include in the request.
        timeout_ms
            Timeout in milliseconds to wait for response.

        Returns
        -------
        ControlMessageResponse
            The response from the server.

        Raises
        ------
        TimeoutError
            If no response is received within the timeout.
        """
        request = ControlMessageRequest(
            label=self._name,
            command_id=command_id,
            request_message=request_message,
        )

        await self._sock.send(request.marshal())

        # Poll with timeout
        poller = Poller()
        poller.register(self._sock, zmq.POLLIN)
        events = await poller.poll(timeout=timeout_ms)

        if not events:
            raise TimeoutError(f"No response received within {timeout_ms}ms")

        response_data = await self._sock.recv()
        return ControlMessageResponse.unmarshal(cast(bytes, response_data))

    async def reset_frame_count(self, timeout_ms: int = 5000) -> ControlMessageResponse:
        """
        Send a reset frame count command.

        Parameters
        ----------
        timeout_ms
            Timeout in milliseconds to wait for response.

        Returns
        -------
        ControlMessageResponse
            The response from the server.
        """
        return await self.send_request(
            command_id=CONTROL_MSG_CMD_RESET_FRAME_COUNT,
            timeout_ms=timeout_ms,
        )

    def close(self):
        """Close the ZMQ socket."""
        if self._sock is not None:
            self._sock.close()

    def __del__(self):
        self.close()
