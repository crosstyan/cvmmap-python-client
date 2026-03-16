import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Common constants from app_common_models.hpp
LABEL_LEN_MAX = 24

FRAME_TOPIC_MAGIC = 0x7D
MODULE_STATUS_MAGIC = 0x5A

CONTROL_MESSAGE_REQUEST_MAGIC = 0x3C
CONTROL_MESSAGE_RESPONSE_MAGIC = 0x3D
BODY_TRACKING_MAGIC = 0x62

CONTROL_MSG_CMD_GENERIC = 0
CONTROL_MSG_CMD_RESET_FRAME_COUNT = 0x1001
CONTROL_MSG_CMD_GET_SOURCE_INFO = 0x1002
CONTROL_MSG_CMD_SEEK_TIMESTAMP_NS = 0x1003
CONTROL_MSG_CMD_START_RECORDING = 0x1004
CONTROL_MSG_CMD_STOP_RECORDING = 0x1005
CONTROL_MSG_CMD_GET_RECORDING_STATUS = 0x1006

CONTROL_RESPONSE_OK = 0
CONTROL_RESPONSE_UNKNOWN_CMD = -1
CONTROL_RESPONSE_ERROR = -2
CONTROL_RESPONSE_INVALID_MAGIC = -3
CONTROL_RESPONSE_INVALID_LABEL = -4
CONTROL_RESPONSE_INVALID_VERSION = -5
CONTROL_RESPONSE_INVALID_MSG_SIZE = -6
CONTROL_RESPONSE_UNSUPPORTED = -7
CONTROL_RESPONSE_INVALID_PAYLOAD = -8
CONTROL_RESPONSE_OUT_OF_RANGE = -9
CONTROL_RESPONSE_TIMEOUT = -100
CONTROL_RESPONSE_TIMEOUT = -100

MODULE_STATUS_ONLINE = 0xA1
MODULE_STATUS_OFFLINE = 0xA0
MODULE_STATUS_STREAM_RESET = 0xB0

VERSION_MAJOR = 1
VERSION_MINOR = 0

FRAME_METADATA_V1_MAJOR = 1
FRAME_METADATA_V2_MAJOR = 2

SOURCE_KIND_UNKNOWN = 0
SOURCE_KIND_LIVE = 1
SOURCE_KIND_FINITE = 2

TIMESTAMP_DOMAIN_UNKNOWN = 0
TIMESTAMP_DOMAIN_UNIX_EPOCH_NS = 1
TIMESTAMP_DOMAIN_MEDIA_TIME_NS = 2

SOURCE_INFO_FLAG_CAN_SEEK = 0x00000001
SOURCE_INFO_FLAG_AUTO_LOOP = 0x00000002
SOURCE_INFO_FLAG_HAS_DEPTH = 0x00000004
SOURCE_INFO_FLAG_HAS_BODY = 0x00000008
SOURCE_INFO_FLAG_CAN_RECORD = 0x00000010

RECORDING_FORMAT_UNKNOWN = 0
RECORDING_FORMAT_SVO = 1
RECORDING_FORMAT_MCAP = 2
RECORDING_FORMAT_MCAP = 2

RECORDING_STATUS_FLAG_CAN_RECORD = 0x0001
RECORDING_STATUS_FLAG_IS_RECORDING = 0x0002
RECORDING_STATUS_FLAG_IS_PAUSED = 0x0004
RECORDING_STATUS_FLAG_LAST_FRAME_OK = 0x0008

FRAME_PLANE_TYPE_LEFT = 0
FRAME_PLANE_TYPE_DEPTH = 1
FRAME_PLANE_TYPE_CONFIDENCE = 2

PIXEL_FORMAT_RGB = 0
PIXEL_FORMAT_BGR = 1
PIXEL_FORMAT_RGBA = 2
PIXEL_FORMAT_BGRA = 3
PIXEL_FORMAT_GRAY = 4
PIXEL_FORMAT_YUV = 5
PIXEL_FORMAT_YUYV = 6

DEPTH_U8 = 0
DEPTH_S8 = 1
DEPTH_U16 = 2
DEPTH_S16 = 3
DEPTH_S32 = 4
DEPTH_F32 = 5
DEPTH_F64 = 6
DEPTH_F16 = 7

DEPTH_UNIT_UNKNOWN = 0
DEPTH_UNIT_MILLIMETER = 1
DEPTH_UNIT_METER = 2

# "CV-MMAP\0" exactly 8 bytes – see `frame_metadata_t::CV_MMAP_MAGIC` in C++
CV_MMAP_MAGIC: bytes = b"CV-MMAP\0"
CV_MMAP_MAGIC_LEN: int = len(CV_MMAP_MAGIC)

FRAME_METADATA_REGION_SIZE = 256
BODY_KEYPOINT_CAPACITY = 38
BODY_BOX2D_POINTS = 4
BODY_BOX3D_POINTS = 8
BODY_TRACKING_HEADER_SIZE = 64
BODY_TRACKING_BODY_RECORD_SIZE = 3248

BODY_FORMAT_BODY_18 = 0
BODY_FORMAT_BODY_34 = 1
BODY_FORMAT_BODY_38 = 2

BODY_KEYPOINT_SELECTION_FULL = 0
BODY_KEYPOINT_SELECTION_UPPER_BODY = 1

BODY_TRACKING_MODEL_HUMAN_BODY_FAST = 0
BODY_TRACKING_MODEL_HUMAN_BODY_MEDIUM = 1
BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE = 2

INFERENCE_PRECISION_FP32 = 0
INFERENCE_PRECISION_FP16 = 1
INFERENCE_PRECISION_INT8 = 2

OBJECT_TRACKING_STATE_OFF = 0
OBJECT_TRACKING_STATE_OK = 1
OBJECT_TRACKING_STATE_SEARCHING = 2
OBJECT_TRACKING_STATE_TERMINATE = 3

OBJECT_ACTION_STATE_IDLE = 0
OBJECT_ACTION_STATE_MOVING = 1

BODY_TRACKING_FLAG_IS_NEW = 0x0001
BODY_TRACKING_FLAG_IS_TRACKED = 0x0002
BODY_TRACKING_FLAG_BODY_FITTING_ENABLED = 0x0004
BODY_TRACKING_FLAG_REDUCED_PRECISION_REQUESTED = 0x0008
BODY_TRACKING_FLAG_FLOOR_AS_ORIGIN = 0x0010

BODY_TRACKING_BODY_FLAG_HAS_LOCAL_JOINTS = 0x0001
BODY_TRACKING_BODY_FLAG_HAS_ROOT_ORIENTATION = 0x0002

_PIXEL_FORMAT_CHANNELS: dict[int, int] = {
    PIXEL_FORMAT_RGB: 3,
    PIXEL_FORMAT_BGR: 3,
    PIXEL_FORMAT_RGBA: 4,
    PIXEL_FORMAT_BGRA: 4,
    PIXEL_FORMAT_GRAY: 1,
    PIXEL_FORMAT_YUV: 3,
    PIXEL_FORMAT_YUYV: 2,
}

_DEPTH_TO_DTYPE: dict[int, np.dtype] = {
    DEPTH_U8: np.dtype(np.uint8),
    DEPTH_S8: np.dtype(np.int8),
    DEPTH_U16: np.dtype(np.uint16),
    DEPTH_S16: np.dtype(np.int16),
    DEPTH_S32: np.dtype(np.int32),
    DEPTH_F32: np.dtype(np.float32),
    DEPTH_F64: np.dtype(np.float64),
    DEPTH_F16: np.dtype(np.float16),
}


# SyncMessage - matches C++ sync_message_t
@dataclass
class SyncMessage:
    """
    Sync message sent over ZMQ to notify subscribers of a new frame.
    """

    frame_count: int
    timestamp_ns: int
    label: str

    # C++ struct layout:
    # uint8_t _magic{FRAME_TOPIC_MAGIC};        // offset 0
    # uint8_t _reserved_0[1];                   // offset 1
    # uint8_t versions_major;                   // offset 2
    # uint8_t versions_minor;                   // offset 3
    # uint32_t frame_count;                     // offset 4
    # uint8_t _reserved_1[4];                   // offset 8
    # 4 bytes implicit padding                  // offset 12
    # uint64_t timestamp_ns;                    // offset 16
    # uint8_t _label[LABEL_LEN_MAX];            // offset 24
    # (Total: 48 bytes due to 8-byte alignment)
    PACK_FMT = f"=BxBBI4s4xQ{LABEL_LEN_MAX}s"

    @staticmethod
    def size() -> int:
        return struct.calcsize(SyncMessage.PACK_FMT)

    def marshal(self) -> bytes:
        """Marshal the SyncMessage to bytes matching the C++ format"""
        encoded_label = self.label.encode("utf-8")[:LABEL_LEN_MAX].ljust(
            LABEL_LEN_MAX, b"\0"
        )
        return struct.pack(
            self.PACK_FMT,
            FRAME_TOPIC_MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            self.frame_count,
            b"\0" * 4,
            self.timestamp_ns,
            encoded_label,
        )

    @staticmethod
    def unmarshal(data: bytes) -> "SyncMessage":
        """Unmarshal bytes to SyncMessage"""
        if len(data) < SyncMessage.size():
            raise ValueError(f"Data too short: {len(data)} < {SyncMessage.size()}")

        (
            magic,
            v_major,
            v_minor,
            frame_count,
            _reserved_1,
            timestamp_ns,
            label_bytes,
        ) = struct.unpack(SyncMessage.PACK_FMT, data[: SyncMessage.size()])

        if magic != FRAME_TOPIC_MAGIC:
            raise ValueError(
                f"Invalid topic magic: expected {FRAME_TOPIC_MAGIC:#x}, got {magic:#x}"
            )

        # Migration policy: SHM metadata may be v2 while sync/control wire messages stay v1.
        # Keep minor-version leniency, but reject unsupported sync majors deterministically.
        if v_major != VERSION_MAJOR:
            raise ValueError(
                f"Unsupported sync major version: expected {VERSION_MAJOR}, got {v_major}"
            )

        label = label_bytes.split(b"\0", 1)[0].decode("utf-8")

        return SyncMessage(
            frame_count=frame_count, timestamp_ns=timestamp_ns, label=label
        )


@dataclass
class BodyTrackingMessageHeader:
    frame_count: int
    timestamp_ns: int
    sdk_timestamp_ns: int
    body_count: int
    body_record_size: int
    body_format: int
    body_selection: int
    detection_model: int
    inference_precision: int
    flags: int
    coordinate_system: int
    reference_frame: int
    floor_as_origin: bool
    payload_size_bytes: int
    label: str

    PACK_FMT = f"<BBBBIQQHHBBBBHHI{LABEL_LEN_MAX}s"

    @staticmethod
    def size() -> int:
        return struct.calcsize(BodyTrackingMessageHeader.PACK_FMT)

    @staticmethod
    def unmarshal(data: bytes) -> "BodyTrackingMessageHeader":
        if len(data) < BodyTrackingMessageHeader.size():
            raise ValueError(
                "Body tracking header too short: "
                f"{len(data)} < {BodyTrackingMessageHeader.size()}"
            )

        (
            magic,
            coordinate_system_code,
            versions_major,
            versions_minor,
            frame_count,
            timestamp_ns,
            sdk_timestamp_ns,
            body_count,
            body_record_size,
            body_format,
            body_selection,
            detection_model,
            inference_precision,
            flags,
            reference_frame_code,
            payload_size_bytes,
            label_bytes,
        ) = struct.unpack(
            BodyTrackingMessageHeader.PACK_FMT,
            data[: BodyTrackingMessageHeader.size()],
        )

        if magic != BODY_TRACKING_MAGIC:
            raise ValueError(
                "Invalid body tracking magic: "
                f"expected {BODY_TRACKING_MAGIC:#x}, got {magic:#x}"
            )
        if versions_major != VERSION_MAJOR:
            raise ValueError(
                "Unsupported body tracking major version: "
                f"expected {VERSION_MAJOR}, got {versions_major}"
            )
        if body_record_size != BODY_TRACKING_BODY_RECORD_SIZE:
            raise ValueError(
                "Invalid body_record_size: "
                f"expected {BODY_TRACKING_BODY_RECORD_SIZE}, got {body_record_size}"
            )

        expected_payload_size = body_count * BODY_TRACKING_BODY_RECORD_SIZE
        if payload_size_bytes != expected_payload_size:
            raise ValueError(
                "Invalid payload_size_bytes: "
                f"expected {expected_payload_size}, got {payload_size_bytes}"
            )
        if body_format > BODY_FORMAT_BODY_38:
            raise ValueError(f"Unsupported body_format={body_format}")
        if body_selection > BODY_KEYPOINT_SELECTION_UPPER_BODY:
            raise ValueError(f"Unsupported body_selection={body_selection}")
        if detection_model > BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE:
            raise ValueError(f"Unsupported detection_model={detection_model}")
        if inference_precision > INFERENCE_PRECISION_INT8:
            raise ValueError(
                f"Unsupported inference_precision={inference_precision}"
            )

        return BodyTrackingMessageHeader(
            frame_count=frame_count,
            timestamp_ns=timestamp_ns,
            sdk_timestamp_ns=sdk_timestamp_ns,
            body_count=body_count,
            body_record_size=body_record_size,
            body_format=body_format,
            body_selection=body_selection,
            detection_model=detection_model,
            inference_precision=inference_precision,
            flags=flags,
            coordinate_system=coordinate_system_code,
            reference_frame=reference_frame_code,
            floor_as_origin=(flags & BODY_TRACKING_FLAG_FLOOR_AS_ORIGIN) != 0,
            payload_size_bytes=payload_size_bytes,
            label=label_bytes.split(b"\0", 1)[0].decode("utf-8"),
        )


@dataclass
class BodyTrack:
    id: int
    tracking_state: int
    action_state: int
    confidence: float
    position: np.ndarray
    velocity: np.ndarray
    position_covariance: np.ndarray
    bounding_box_2d: np.ndarray
    bounding_box_3d: np.ndarray
    dimensions: np.ndarray
    keypoint_2d: np.ndarray
    keypoint_3d: np.ndarray
    keypoint_confidence: np.ndarray
    keypoint_covariance: np.ndarray
    head_bounding_box_2d: np.ndarray
    head_bounding_box_3d: np.ndarray
    head_position: np.ndarray
    local_position_per_joint: np.ndarray
    local_orientation_per_joint: np.ndarray
    global_root_orientation: np.ndarray
    keypoint_count: int
    flags: int

    @staticmethod
    def size() -> int:
        return BODY_TRACKING_BODY_RECORD_SIZE

    @staticmethod
    def unmarshal(data: memoryview | bytes | bytearray) -> "BodyTrack":
        chunk = memoryview(data)
        if len(chunk) < BodyTrack.size():
            raise ValueError(
                f"Body track record too short: {len(chunk)} < {BodyTrack.size()}"
            )

        id_, tracking_state, action_state = struct.unpack_from("<iBB", chunk, 0)
        confidence = struct.unpack_from("<f", chunk, 8)[0]

        cursor = 12

        def take_f32(count: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            size = count * 4
            arr = np.frombuffer(chunk[cursor : cursor + size], dtype="<f4").copy()
            cursor += size
            return arr.reshape(shape)

        position = take_f32(3, (3,))
        velocity = take_f32(3, (3,))
        position_covariance = take_f32(6, (6,))
        bounding_box_2d = take_f32(8, (BODY_BOX2D_POINTS, 2))
        bounding_box_3d = take_f32(24, (BODY_BOX3D_POINTS, 3))
        dimensions = take_f32(3, (3,))
        keypoint_2d = take_f32(BODY_KEYPOINT_CAPACITY * 2, (BODY_KEYPOINT_CAPACITY, 2))
        keypoint_3d = take_f32(BODY_KEYPOINT_CAPACITY * 3, (BODY_KEYPOINT_CAPACITY, 3))
        keypoint_confidence = take_f32(BODY_KEYPOINT_CAPACITY, (BODY_KEYPOINT_CAPACITY,))
        keypoint_covariance = take_f32(
            BODY_KEYPOINT_CAPACITY * 6, (BODY_KEYPOINT_CAPACITY, 6)
        )
        head_bounding_box_2d = take_f32(8, (BODY_BOX2D_POINTS, 2))
        head_bounding_box_3d = take_f32(24, (BODY_BOX3D_POINTS, 3))
        head_position = take_f32(3, (3,))
        local_position_per_joint = take_f32(
            BODY_KEYPOINT_CAPACITY * 3, (BODY_KEYPOINT_CAPACITY, 3)
        )
        local_orientation_per_joint = take_f32(
            BODY_KEYPOINT_CAPACITY * 4, (BODY_KEYPOINT_CAPACITY, 4)
        )
        global_root_orientation = take_f32(4, (4,))
        keypoint_count, body_flags = struct.unpack_from("<HH", chunk, cursor)

        return BodyTrack(
            id=id_,
            tracking_state=tracking_state,
            action_state=action_state,
            confidence=confidence,
            position=position,
            velocity=velocity,
            position_covariance=position_covariance,
            bounding_box_2d=bounding_box_2d,
            bounding_box_3d=bounding_box_3d,
            dimensions=dimensions,
            keypoint_2d=keypoint_2d,
            keypoint_3d=keypoint_3d,
            keypoint_confidence=keypoint_confidence,
            keypoint_covariance=keypoint_covariance,
            head_bounding_box_2d=head_bounding_box_2d,
            head_bounding_box_3d=head_bounding_box_3d,
            head_position=head_position,
            local_position_per_joint=local_position_per_joint,
            local_orientation_per_joint=local_orientation_per_joint,
            global_root_orientation=global_root_orientation,
            keypoint_count=keypoint_count,
            flags=body_flags,
        )


@dataclass
class BodyFrame:
    header: BodyTrackingMessageHeader
    bodies: list[BodyTrack]

    @property
    def frame_count(self) -> int:
        return self.header.frame_count

    @property
    def timestamp_ns(self) -> int:
        return self.header.timestamp_ns

    @property
    def sdk_timestamp_ns(self) -> int:
        return self.header.sdk_timestamp_ns

    @property
    def body_count(self) -> int:
        return self.header.body_count

    @property
    def body_format(self) -> int:
        return self.header.body_format

    @property
    def body_selection(self) -> int:
        return self.header.body_selection

    @property
    def detection_model(self) -> int:
        return self.header.detection_model

    @property
    def inference_precision(self) -> int:
        return self.header.inference_precision

    @property
    def flags(self) -> int:
        return self.header.flags

    @property
    def coordinate_system(self) -> int:
        return self.header.coordinate_system

    @property
    def reference_frame(self) -> int:
        return self.header.reference_frame

    @property
    def floor_as_origin(self) -> bool:
        return self.header.floor_as_origin

    @property
    def label(self) -> str:
        return self.header.label


def unmarshal_body_tracking_message(data: bytes) -> BodyFrame:
    header = BodyTrackingMessageHeader.unmarshal(data)
    header_size = BodyTrackingMessageHeader.size()
    total_size = header_size + header.payload_size_bytes
    if len(data) < total_size:
        raise ValueError(f"Body tracking message truncated: {len(data)} < {total_size}")

    bodies: list[BodyTrack] = []
    offset = header_size

    for _ in range(header.body_count):
        bodies.append(
            BodyTrack.unmarshal(
                memoryview(data)[offset : offset + BODY_TRACKING_BODY_RECORD_SIZE]
            )
        )
        offset += BODY_TRACKING_BODY_RECORD_SIZE

    return BodyFrame(header=header, bodies=bodies)


@dataclass
class ModuleStatusMessage:
    """
    Module status message sent over ZMQ
    """

    module_status: int
    label: str

    # C++ struct layout:
    # uint8_t _magic{MODULE_STATUS_MAGIC};      // offset 0
    # uint8_t _reserved_0[1];                   // offset 1
    # uint8_t versions_major{VERSION_MAJOR};    // offset 2
    # uint8_t versions_minor{VERSION_MINOR};    // offset 3
    # int32_t module_status;                    // offset 4
    # uint8_t _label[LABEL_LEN_MAX];            // offset 8
    # (Total: 32 bytes)
    PACK_FMT = f"=BxBBi{LABEL_LEN_MAX}s"

    @staticmethod
    def size() -> int:
        return struct.calcsize(ModuleStatusMessage.PACK_FMT)

    def marshal(self) -> bytes:
        encoded_label = self.label.encode("utf-8")[:LABEL_LEN_MAX].ljust(
            LABEL_LEN_MAX, b"\0"
        )
        return struct.pack(
            self.PACK_FMT,
            MODULE_STATUS_MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            self.module_status,
            encoded_label,
        )

    @staticmethod
    def unmarshal(data: bytes) -> "ModuleStatusMessage":
        if len(data) < ModuleStatusMessage.size():
            raise ValueError(
                f"Data too short: {len(data)} < {ModuleStatusMessage.size()}"
            )

        magic, v_major, v_minor, status, label_bytes = struct.unpack(
            ModuleStatusMessage.PACK_FMT, data[: ModuleStatusMessage.size()]
        )

        if magic != MODULE_STATUS_MAGIC:
            raise ValueError(
                f"Invalid module status magic: expected {MODULE_STATUS_MAGIC:#x}, got {magic:#x}"
            )

        if v_major != VERSION_MAJOR:
            raise ValueError(
                f"Unsupported module status major version: expected {VERSION_MAJOR}, got {v_major}"
            )

        label = label_bytes.split(b"\0", 1)[0].decode("utf-8")
        return ModuleStatusMessage(module_status=status, label=label)


@dataclass
class FrameInfo:
    """
    Matches C++ frame_info_t

    struct frame_info_t {
        uint16_t width;
        uint16_t height;
        uint8_t channels;
        Depth depth;  // uint8_t
        PixelFormat pixel_format; // uint8_t
        uint8_t _reserved_0[1]; // padding
        uint32_t buffer_size;
    }
    (Total: 12 bytes, 4-byte aligned)
    """

    width: int
    height: int
    channels: int
    depth: int
    pixel_format: int
    buffer_size: int

    # struct mapping: H = uint16, B = uint8, I = uint32, x = padding byte
    PACK_FMT = "=HHBBBxI"

    @staticmethod
    def size() -> int:
        return struct.calcsize(FrameInfo.PACK_FMT)

    def marshal(self) -> bytes:
        return struct.pack(
            self.PACK_FMT,
            self.width,
            self.height,
            self.channels,
            self.depth,
            self.pixel_format,
            self.buffer_size,
        )

    @staticmethod
    def unmarshal(data: bytes) -> "FrameInfo":
        if len(data) < FrameInfo.size():
            raise ValueError(f"Data too short: {len(data)} < {FrameInfo.size()}")

        (
            width,
            height,
            channels,
            depth,
            pixel_format,
            buffer_size,
        ) = struct.unpack(FrameInfo.PACK_FMT, data[: FrameInfo.size()])

        return FrameInfo(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            pixel_format=pixel_format,
            buffer_size=buffer_size,
        )


@dataclass
class FrameMetadata:
    """
    Matches C++ frame_metadata_t
    """

    frame_count: int
    timestamp_ns: int
    info: FrameInfo

    @staticmethod
    def size() -> int:
        return CV_MMAP_MAGIC_LEN + struct.calcsize("=BBxxIQ") + FrameInfo.size()

    @staticmethod
    def unmarshal(data: bytes) -> "FrameMetadata":
        """
        Note: This expects data starting AFTER the magic bytes (offset from CV_MMAP_MAGIC_LEN).
        """
        # C++ layout (after magic bytes):
        # uint8_t versions_major;                   // offset 0
        # uint8_t versions_minor;                   // offset 1
        # uint8_t _reserved_0[2];                   // offset 2
        # uint32_t frame_count;                     // offset 4
        # uint64_t timestamp_ns;                    // offset 8
        # frame_info_t info;                        // offset 16
        # (Header without magic + info is 16 + 12 = 28 bytes)
        # Total with 8-byte magic + 4 byte trailing padding (for 8-byte alignment) = 40 bytes

        fmt = "=BBxxIQ"
        header_size = struct.calcsize(fmt)

        if len(data) < header_size + FrameInfo.size():
            raise ValueError("Data too short for FrameMetadata")

        v_major, v_minor, frame_count, timestamp_ns = struct.unpack(
            fmt, data[:header_size]
        )

        info = FrameInfo.unmarshal(data[header_size:])

        return FrameMetadata(
            frame_count=frame_count,
            timestamp_ns=timestamp_ns,
            info=info,
        )


@dataclass
class FramePlaneDescriptorV2:
    plane_type: int
    pixel_format: int
    depth: int
    width: int
    height: int
    stride_bytes: int
    offset_bytes: int
    size_bytes: int

    PACK_FMT = "=BBBBIIIII"

    @staticmethod
    def size() -> int:
        return struct.calcsize(FramePlaneDescriptorV2.PACK_FMT)

    @staticmethod
    def unmarshal(data: bytes) -> "FramePlaneDescriptorV2":
        if len(data) < FramePlaneDescriptorV2.size():
            raise ValueError(
                f"Data too short for FramePlaneDescriptorV2: {len(data)} < {FramePlaneDescriptorV2.size()}"
            )

        (
            plane_type,
            pixel_format,
            depth,
            _reserved_0,
            width,
            height,
            stride_bytes,
            offset_bytes,
            size_bytes,
        ) = struct.unpack(
            FramePlaneDescriptorV2.PACK_FMT, data[: FramePlaneDescriptorV2.size()]
        )

        return FramePlaneDescriptorV2(
            plane_type=plane_type,
            pixel_format=pixel_format,
            depth=depth,
            width=width,
            height=height,
            stride_bytes=stride_bytes,
            offset_bytes=offset_bytes,
            size_bytes=size_bytes,
        )

    @property
    def is_empty_descriptor(self) -> bool:
        return (
            self.width == 0
            and self.height == 0
            and self.stride_bytes == 0
            and self.offset_bytes == 0
            and self.size_bytes == 0
        )

    @property
    def channels(self) -> int:
        channels = _PIXEL_FORMAT_CHANNELS.get(self.pixel_format)
        if channels is None:
            raise ValueError(
                f"Unsupported pixel_format in descriptor: {self.pixel_format}"
            )
        return channels

    @property
    def dtype(self) -> np.dtype:
        np_dtype = _DEPTH_TO_DTYPE.get(self.depth)
        if np_dtype is None:
            raise ValueError(f"Unsupported depth in descriptor: {self.depth}")
        return np_dtype

    def validate_bounds(self, payload_size_bytes: int, slot: int) -> None:
        if payload_size_bytes < 0:
            raise ValueError(f"Invalid payload_size_bytes: {payload_size_bytes}")
        if self.offset_bytes > payload_size_bytes:
            raise ValueError(
                f"v2 descriptor slot {slot} offset out of bounds: offset={self.offset_bytes}, payload_size={payload_size_bytes}"
            )
        remaining = payload_size_bytes - self.offset_bytes
        if self.size_bytes > remaining:
            raise ValueError(
                f"v2 descriptor slot {slot} size out of bounds: size={self.size_bytes}, offset={self.offset_bytes}, payload_size={payload_size_bytes}"
            )

    def validate_active(self, slot: int, payload_size_bytes: int) -> None:
        self.validate_bounds(payload_size_bytes=payload_size_bytes, slot=slot)

        if (
            self.width == 0
            or self.height == 0
            or self.stride_bytes == 0
            or self.size_bytes == 0
        ):
            raise ValueError(
                f"v2 active descriptor slot {slot} must have non-zero width/height/stride/size"
            )

        itemsize = self.dtype.itemsize
        min_stride = self.width * self.channels * itemsize
        if self.stride_bytes < min_stride:
            raise ValueError(
                f"v2 descriptor slot {slot} has stride smaller than minimum: stride={self.stride_bytes}, min_stride={min_stride}"
            )

        needed_size = self.stride_bytes * self.height
        if self.size_bytes < needed_size:
            raise ValueError(
                f"v2 descriptor slot {slot} has insufficient size for stride/height: size={self.size_bytes}, needed={needed_size}"
            )

    def as_ndarray(
        self, payload: memoryview | bytes | bytearray, payload_size_bytes: int
    ) -> np.ndarray:
        self.validate_active(slot=-1, payload_size_bytes=payload_size_bytes)

        payload_view = memoryview(payload)
        if len(payload_view) < payload_size_bytes:
            raise ValueError(
                f"Payload shorter than declared payload_size_bytes: {len(payload_view)} < {payload_size_bytes}"
            )

        start = self.offset_bytes
        end = start + self.size_bytes
        plane_view = payload_view[start:end]

        dtype = self.dtype
        itemsize = dtype.itemsize
        channels = self.channels

        if channels == 1:
            shape = (self.height, self.width)
            strides = (self.stride_bytes, itemsize)
        else:
            shape = (self.height, self.width, channels)
            strides = (self.stride_bytes, channels * itemsize, itemsize)

        return np.ndarray(shape=shape, dtype=dtype, buffer=plane_view, strides=strides)


@dataclass
class FrameMetadataV2Header:
    versions_minor: int
    flags: int
    frame_id: int
    capture_ts_ns: int
    publish_seq: int
    plane_count: int
    plane_presence_mask: int
    plane_descriptors_offset: int
    plane_descriptor_size: int
    plane_descriptor_capacity: int
    payload_size_bytes: int
    depth_unit: int

    PACK_FMT = "=8sBBHIQQBBHHHIB19s"

    @staticmethod
    def size() -> int:
        return struct.calcsize(FrameMetadataV2Header.PACK_FMT)

    @staticmethod
    def unmarshal(data: bytes) -> "FrameMetadataV2Header":
        if len(data) < FrameMetadataV2Header.size():
            raise ValueError(
                f"Data too short for FrameMetadataV2Header: {len(data)} < {FrameMetadataV2Header.size()}"
            )

        (
            magic,
            versions_major,
            versions_minor,
            flags,
            frame_id,
            capture_ts_ns,
            publish_seq,
            plane_count,
            plane_presence_mask,
            plane_descriptors_offset,
            plane_descriptor_size,
            plane_descriptor_capacity,
            payload_size_bytes,
            depth_unit,
            _reserved_0,
        ) = struct.unpack(
            FrameMetadataV2Header.PACK_FMT, data[: FrameMetadataV2Header.size()]
        )

        if magic != CV_MMAP_MAGIC:
            raise ValueError(
                f"Invalid CV_MMAP magic in v2 header: {magic!r} (expected {CV_MMAP_MAGIC!r})"
            )
        if versions_major != FRAME_METADATA_V2_MAJOR:
            raise ValueError(
                f"Invalid v2 major version: expected {FRAME_METADATA_V2_MAJOR}, got {versions_major}"
            )
        if plane_count < 1 or plane_count > 4:
            raise ValueError(f"Invalid v2 plane_count: {plane_count}, expected 1..4")
        if (plane_presence_mask & 0xF0) != 0:
            raise ValueError(
                f"Invalid v2 plane_presence_mask: upper nibble must be zero, got {plane_presence_mask:#04x}"
            )
        contiguous_mask_expected = (1 << plane_count) - 1
        if plane_presence_mask != contiguous_mask_expected:
            raise ValueError(
                "Invalid v2 plane_presence_mask: "
                f"expected contiguous mask {contiguous_mask_expected:#04x}, got {plane_presence_mask:#04x}"
            )
        if plane_descriptors_offset != 64:
            raise ValueError(
                f"Invalid v2 plane_descriptors_offset: expected 64, got {plane_descriptors_offset}"
            )
        if plane_descriptor_size != FramePlaneDescriptorV2.size():
            raise ValueError(
                "Invalid v2 plane_descriptor_size: "
                f"expected {FramePlaneDescriptorV2.size()}, got {plane_descriptor_size}"
            )
        if plane_descriptor_capacity != 4:
            raise ValueError(
                f"Invalid v2 plane_descriptor_capacity: expected 4, got {plane_descriptor_capacity}"
            )
        if payload_size_bytes <= 0:
            raise ValueError(
                f"Invalid v2 payload_size_bytes: expected >0, got {payload_size_bytes}"
            )
        if depth_unit not in (
            DEPTH_UNIT_UNKNOWN,
            DEPTH_UNIT_MILLIMETER,
            DEPTH_UNIT_METER,
        ):
            raise ValueError(
                f"Invalid v2 depth_unit: expected 0, 1, or 2, got {depth_unit}"
            )

        return FrameMetadataV2Header(
            versions_minor=versions_minor,
            flags=flags,
            frame_id=frame_id,
            capture_ts_ns=capture_ts_ns,
            publish_seq=publish_seq,
            plane_count=plane_count,
            plane_presence_mask=plane_presence_mask,
            plane_descriptors_offset=plane_descriptors_offset,
            plane_descriptor_size=plane_descriptor_size,
            plane_descriptor_capacity=plane_descriptor_capacity,
            payload_size_bytes=payload_size_bytes,
            depth_unit=depth_unit,
        )


@dataclass
class FrameMetadataV2:
    header: FrameMetadataV2Header
    descriptors: tuple[
        FramePlaneDescriptorV2,
        FramePlaneDescriptorV2,
        FramePlaneDescriptorV2,
        FramePlaneDescriptorV2,
    ]

    @staticmethod
    def size() -> int:
        return FRAME_METADATA_REGION_SIZE

    @property
    def versions_major(self) -> int:
        return FRAME_METADATA_V2_MAJOR

    @property
    def versions_minor(self) -> int:
        return self.header.versions_minor

    @property
    def frame_count(self) -> int:
        return self.header.frame_id

    @property
    def timestamp_ns(self) -> int:
        return self.header.capture_ts_ns

    @property
    def info(self) -> FrameInfo:
        left = self.left_descriptor
        return FrameInfo(
            width=left.width,
            height=left.height,
            channels=left.channels,
            depth=left.depth,
            pixel_format=left.pixel_format,
            buffer_size=self.header.payload_size_bytes,
        )

    @property
    def depth_unit(self) -> int:
        return self.header.depth_unit

    @property
    def left_descriptor(self) -> FramePlaneDescriptorV2:
        return self.descriptors[0]

    @property
    def depth_descriptor(self) -> Optional[FramePlaneDescriptorV2]:
        if self.header.plane_count < 2:
            return None
        return self.descriptors[1]

    @property
    def confidence_descriptor(self) -> Optional[FramePlaneDescriptorV2]:
        if self.header.plane_count < 3:
            return None
        return self.descriptors[2]

    @property
    def active_descriptors(self) -> tuple[FramePlaneDescriptorV2, ...]:
        return self.descriptors[: self.header.plane_count]

    def _validate_descriptors(self) -> None:
        payload_size = self.header.payload_size_bytes

        for slot, descriptor in enumerate(self.descriptors):
            descriptor.validate_bounds(payload_size_bytes=payload_size, slot=slot)

        for slot, descriptor in enumerate(self.descriptors):
            if slot < self.header.plane_count:
                descriptor.validate_active(slot=slot, payload_size_bytes=payload_size)
            elif not descriptor.is_empty_descriptor:
                raise ValueError(
                    f"v2 inactive descriptor slot {slot} must be empty (width/height/stride/offset/size must all be zero)"
                )

        if self.descriptors[0].plane_type != FRAME_PLANE_TYPE_LEFT:
            raise ValueError(
                f"v2 descriptor slot 0 must be LEFT plane ({FRAME_PLANE_TYPE_LEFT}), got {self.descriptors[0].plane_type}"
            )
        if self.descriptors[0].offset_bytes != 0:
            raise ValueError(
                f"v2 descriptor slot 0 offset must be 0, got {self.descriptors[0].offset_bytes}"
            )

        if self.header.plane_count >= 2:
            if self.descriptors[1].plane_type != FRAME_PLANE_TYPE_DEPTH:
                raise ValueError(
                    f"v2 descriptor slot 1 must be DEPTH plane ({FRAME_PLANE_TYPE_DEPTH}), got {self.descriptors[1].plane_type}"
                )

        if self.header.plane_count >= 3:
            if self.descriptors[2].plane_type != FRAME_PLANE_TYPE_CONFIDENCE:
                raise ValueError(
                    "v2 descriptor slot 2 must be CONFIDENCE plane "
                    f"({FRAME_PLANE_TYPE_CONFIDENCE}), got {self.descriptors[2].plane_type}"
                )

        for slot in range(1, self.header.plane_count):
            prev = self.descriptors[slot - 1]
            curr = self.descriptors[slot]
            expected_offset = prev.offset_bytes + prev.size_bytes
            if curr.offset_bytes != expected_offset:
                raise ValueError(
                    f"v2 descriptor slot {slot} offset must be packed/contiguous: expected {expected_offset}, got {curr.offset_bytes}"
                )

    def plane(self, slot: int, payload: memoryview | bytes | bytearray) -> np.ndarray:
        if slot < 0 or slot >= self.header.plane_count:
            raise ValueError(
                f"Requested plane slot {slot} is not active (plane_count={self.header.plane_count})"
            )
        return self.descriptors[slot].as_ndarray(
            payload=payload,
            payload_size_bytes=self.header.payload_size_bytes,
        )

    def left_plane(self, payload: memoryview | bytes | bytearray) -> np.ndarray:
        return self.plane(0, payload)

    def depth_plane(
        self, payload: memoryview | bytes | bytearray
    ) -> Optional[np.ndarray]:
        if self.header.plane_count < 2:
            return None
        return self.plane(1, payload)

    def confidence_plane(
        self, payload: memoryview | bytes | bytearray
    ) -> Optional[np.ndarray]:
        if self.header.plane_count < 3:
            return None
        return self.plane(2, payload)

    @staticmethod
    def unmarshal(data: bytes) -> "FrameMetadataV2":
        if len(data) < FrameMetadataV2.size():
            raise ValueError(
                f"Data too short for FrameMetadataV2: {len(data)} < {FrameMetadataV2.size()}"
            )

        header = FrameMetadataV2Header.unmarshal(data[: FrameMetadataV2Header.size()])

        descriptors: list[FramePlaneDescriptorV2] = []
        descriptor_offset = header.plane_descriptors_offset
        descriptor_size = header.plane_descriptor_size
        for slot in range(header.plane_descriptor_capacity):
            start = descriptor_offset + (slot * descriptor_size)
            end = start + descriptor_size
            descriptors.append(FramePlaneDescriptorV2.unmarshal(data[start:end]))

        metadata = FrameMetadataV2(
            header=header,
            descriptors=(
                descriptors[0],
                descriptors[1],
                descriptors[2],
                descriptors[3],
            ),
        )
        metadata._validate_descriptors()
        return metadata


def unmarshal_frame_metadata(
    data: bytes,
) -> FrameMetadata | FrameMetadataV2:
    if len(data) < CV_MMAP_MAGIC_LEN + 2:
        raise ValueError(
            f"Data too short for frame metadata dispatch: {len(data)} < {CV_MMAP_MAGIC_LEN + 2}"
        )
    if data[:CV_MMAP_MAGIC_LEN] != CV_MMAP_MAGIC:
        raise ValueError(
            f"Invalid CV_MMAP magic prefix in metadata: {data[:CV_MMAP_MAGIC_LEN]!r}"
        )

    versions_major = data[CV_MMAP_MAGIC_LEN]
    if versions_major == FRAME_METADATA_V1_MAJOR:
        return FrameMetadata.unmarshal(data[CV_MMAP_MAGIC_LEN:])
    if versions_major == FRAME_METADATA_V2_MAJOR:
        return FrameMetadataV2.unmarshal(data[:FRAME_METADATA_REGION_SIZE])

    raise ValueError(f"Unsupported frame metadata major version: {versions_major}")


@dataclass
class SourceInfo:
    source_kind: int
    timestamp_domain: int
    flags: int
    timeline_start_ns: int
    timeline_end_ns: int
    duration_ns: int
    current_timestamp_ns: int
    current_frame_count: int

    PACK_FMT = "<HBBIQQQQII"

    @staticmethod
    def size() -> int:
        return struct.calcsize(SourceInfo.PACK_FMT)

    @property
    def can_seek(self) -> bool:
        return (self.flags & SOURCE_INFO_FLAG_CAN_SEEK) != 0

    @property
    def auto_loop(self) -> bool:
        return (self.flags & SOURCE_INFO_FLAG_AUTO_LOOP) != 0

    @property
    def has_depth(self) -> bool:
        return (self.flags & SOURCE_INFO_FLAG_HAS_DEPTH) != 0

    @property
    def has_body(self) -> bool:
        return (self.flags & SOURCE_INFO_FLAG_HAS_BODY) != 0

    @property
    def can_record(self) -> bool:
        return (self.flags & SOURCE_INFO_FLAG_CAN_RECORD) != 0

    @staticmethod
    def unmarshal(data: bytes) -> "SourceInfo":
        if len(data) < SourceInfo.size():
            raise ValueError(f"Data too short: {len(data)} < {SourceInfo.size()}")

        (
            struct_size,
            source_kind,
            timestamp_domain,
            flags,
            timeline_start_ns,
            timeline_end_ns,
            duration_ns,
            current_timestamp_ns,
            current_frame_count,
            _reserved_0,
        ) = struct.unpack(SourceInfo.PACK_FMT, data[: SourceInfo.size()])

        if struct_size < SourceInfo.size():
            raise ValueError(
                f"Invalid source info payload size: {struct_size} < {SourceInfo.size()}"
            )

        return SourceInfo(
            source_kind=source_kind,
            timestamp_domain=timestamp_domain,
            flags=flags,
            timeline_start_ns=timeline_start_ns,
            timeline_end_ns=timeline_end_ns,
            duration_ns=duration_ns,
            current_timestamp_ns=current_timestamp_ns,
            current_frame_count=current_frame_count,
        )


@dataclass
class SeekTimestampRequest:
    target_timestamp_ns: int

    PACK_FMT = "<HHQ"

    @staticmethod
    def size() -> int:
        return struct.calcsize(SeekTimestampRequest.PACK_FMT)

    def marshal(self) -> bytes:
        return struct.pack(
            self.PACK_FMT,
            self.size(),
            0,
            self.target_timestamp_ns,
        )


@dataclass
class SeekResult:
    requested_timestamp_ns: int
    landed_timestamp_ns: int
    landed_frame_count: int
    exact_match: bool

    PACK_FMT = "<HBBQQII"

    @staticmethod
    def size() -> int:
        return struct.calcsize(SeekResult.PACK_FMT)

    @staticmethod
    def unmarshal(data: bytes) -> "SeekResult":
        if len(data) < SeekResult.size():
            raise ValueError(f"Data too short: {len(data)} < {SeekResult.size()}")

        (
            struct_size,
            exact_match,
            _reserved_0,
            requested_timestamp_ns,
            landed_timestamp_ns,
            landed_frame_count,
            _reserved_1,
        ) = struct.unpack(SeekResult.PACK_FMT, data[: SeekResult.size()])

        if struct_size < SeekResult.size():
            raise ValueError(
                f"Invalid seek result payload size: {struct_size} < {SeekResult.size()}"
            )

        return SeekResult(
            requested_timestamp_ns=requested_timestamp_ns,
            landed_timestamp_ns=landed_timestamp_ns,
            landed_frame_count=landed_frame_count,
            exact_match=exact_match != 0,
        )


@dataclass
class ControlCapabilities:
    can_seek: bool
    available_recording_formats: list[int]

    def supports_recording_format(self, recording_format: int) -> bool:
        return recording_format in self.available_recording_formats


@dataclass
class SvoRecordingOptions:
    compression_mode: str | None = None
    bitrate: int | None = None
    target_framerate: int | None = None
    transcode_streaming_input: bool | None = None


@dataclass
class McapRecordingOptions:
    compression: str | None = None
    topic: str | None = None
    depth_topic: str | None = None
    body_topic: str | None = None
    frame_id: str | None = None


@dataclass
class RecordingRequest:
    recording_format: int
    output_path: str
    svo_options: SvoRecordingOptions | None = None
    mcap_options: McapRecordingOptions | None = None


@dataclass
class RecordingStartRequest:
    output_path: str
    flags: int = 0

    PACK_FMT = "<HHHH"

    @staticmethod
    def size() -> int:
        return struct.calcsize(RecordingStartRequest.PACK_FMT)

    def marshal(self) -> bytes:
        encoded_path = self.output_path.encode("utf-8")
        if not encoded_path:
            raise ValueError("output_path must not be empty")
        if b"\0" in encoded_path:
            raise ValueError("output_path must not contain NUL bytes")
        if len(encoded_path) > 0xFFFF:
            raise ValueError(
                f"output_path too long for protocol payload: {len(encoded_path)} > 65535"
            )
        return struct.pack(
            self.PACK_FMT,
            self.size(),
            self.flags,
            len(encoded_path),
            0,
        ) + encoded_path


@dataclass
class RecordingStatus:
    recording_format: int
    flags: int
    active_path: str
    frames_ingested: int
    frames_encoded: int
    error_message: str = ""
    error_message: str = ""

    PACK_FMT = "<HBBHHIII"

    @staticmethod
    def size() -> int:
        return struct.calcsize(RecordingStatus.PACK_FMT)

    @property
    def can_record(self) -> bool:
        return (self.flags & RECORDING_STATUS_FLAG_CAN_RECORD) != 0

    @property
    def is_recording(self) -> bool:
        return (self.flags & RECORDING_STATUS_FLAG_IS_RECORDING) != 0

    @property
    def is_paused(self) -> bool:
        return (self.flags & RECORDING_STATUS_FLAG_IS_PAUSED) != 0

    @property
    def last_frame_ok(self) -> bool:
        return (self.flags & RECORDING_STATUS_FLAG_LAST_FRAME_OK) != 0

    @staticmethod
    def unmarshal(data: bytes) -> "RecordingStatus":
        if len(data) < RecordingStatus.size():
            raise ValueError(
                f"Data too short: {len(data)} < {RecordingStatus.size()}"
            )

        (
            struct_size,
            recording_format,
            _reserved_0,
            flags,
            path_length,
            frames_ingested,
            frames_encoded,
            _reserved_1,
        ) = struct.unpack(RecordingStatus.PACK_FMT, data[: RecordingStatus.size()])

        if struct_size < RecordingStatus.size():
            raise ValueError(
                "Invalid recording status payload size: "
                f"{struct_size} < {RecordingStatus.size()}"
            )

        total_size = RecordingStatus.size() + path_length
        if len(data) < total_size:
            raise ValueError(
                f"Data too short for recording status path: {len(data)} < {total_size}"
            )

        active_path = data[RecordingStatus.size() : total_size].decode("utf-8")
        return RecordingStatus(
            recording_format=recording_format,
            flags=flags,
            active_path=active_path,
            frames_ingested=frames_ingested,
            frames_encoded=frames_encoded,
        )


@dataclass
class ControlMessageRequest:
    """
    Request message sent over ZMQ REQ socket
    """

    label: str
    command_id: int
    request_message: bytes

    @staticmethod
    def marshal_format() -> str:
        # uint8_t _magic                            // offset 0
        # uint8_t _reserved_0[1];                   // offset 1
        # uint8_t versions_major                    // offset 2
        # uint8_t versions_minor                    // offset 3
        # int32_t command_id                        // offset 4
        # uint8_t _label[LABEL_LEN_MAX]             // offset 8
        # uint16_t request_message_length;          // offset 32
        # (followed by variable length request message)
        # Total wire header size: 34 bytes. The C++ struct is sizeof(...) == 36
        # because of unsent trailing padding after the flexible-array length field.
        return f"<BxBBi{LABEL_LEN_MAX}sH"

    def marshal(self) -> bytes:
        encoded_label = self.label.encode("utf-8")[:LABEL_LEN_MAX].ljust(
            LABEL_LEN_MAX, b"\0"
        )
        msg_len = len(self.request_message)
        header = struct.pack(
            self.marshal_format(),
            CONTROL_MESSAGE_REQUEST_MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            self.command_id,
            encoded_label,
            msg_len,
        )
        return header + self.request_message

    @staticmethod
    def header_size() -> int:
        return struct.calcsize(ControlMessageRequest.marshal_format())


@dataclass
class ControlMessageResponse:
    """
    Response message received over ZMQ REP socket
    """

    command_id: int
    response_code: int
    label: str
    response_message: bytes

    @staticmethod
    def marshal_format() -> str:
        # uint8_t _magic                            // offset 0
        # uint8_t _reserved_0[1];                   // offset 1
        # uint8_t versions_major                    // offset 2
        # uint8_t versions_minor                    // offset 3
        # int32_t command_id                        // offset 4
        # int32_t response_code                     // offset 8
        # uint8_t _label[LABEL_LEN_MAX];            // offset 12
        # uint16_t response_message_length;         // offset 36
        # Total wire header size: 38 bytes. The C++ struct is sizeof(...) == 40
        # because of unsent trailing padding after the flexible-array length field.
        return f"<BxBBii{LABEL_LEN_MAX}sH"

    @staticmethod
    def header_size() -> int:
        return struct.calcsize(ControlMessageResponse.marshal_format())

    @staticmethod
    def unmarshal(data: bytes) -> "ControlMessageResponse":
        if len(data) < ControlMessageResponse.header_size():
            raise ValueError(
                f"Data too short: {len(data)} < {ControlMessageResponse.header_size()}"
            )

        (magic, v_major, v_minor, command_id, response_code, label_bytes, msg_len) = (
            struct.unpack(
                ControlMessageResponse.marshal_format(),
                data[: ControlMessageResponse.header_size()],
            )
        )

        if magic != CONTROL_MESSAGE_RESPONSE_MAGIC:
            raise ValueError(
                f"Invalid response magic: expected {CONTROL_MESSAGE_RESPONSE_MAGIC:#x}, got {magic:#x}"
            )

        # Migration policy: control wire major remains v1 even when SHM metadata is v2.
        # Keep minor-version leniency, but fail unsupported control majors deterministically.
        if v_major != VERSION_MAJOR:
            raise ValueError(
                f"Unsupported control response major version: expected {VERSION_MAJOR}, got {v_major}"
            )

        label = label_bytes.split(b"\0", 1)[0].decode("utf-8")
        total_size = ControlMessageResponse.header_size() + msg_len
        if len(data) < total_size:
            raise ValueError(f"Data too short for response payload: {len(data)} < {total_size}")
        response_msg = data[
            ControlMessageResponse.header_size() : ControlMessageResponse.header_size()
            + msg_len
        ]

        return ControlMessageResponse(
            command_id=command_id,
            response_code=response_code,
            label=label,
            response_message=response_msg,
        )
