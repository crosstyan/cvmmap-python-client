# ABI v2 Contract Checklist

**Source of Truth:** `cv-mmap/docs/cvmmap_shm_metadata_v1_v2.ksy`  
**Contract Version:** v2 (major=2)  
**Generated:** 2026-03-04

This document maps ALL v2 header/descriptor invariants to exact target code locations in the Python client.

---

## Normative References

| Document | Line Range | Description |
|----------|------------|-------------|
| `cv-mmap/docs/cvmmap.ksy` | 314-369 | `frame_metadata_v2_header` struct definition |
| `cv-mmap/docs/cvmmap.ksy` | 375-433 | `frame_metadata_v2` struct definition (full metadata) |
| `cv-mmap/docs/cvmmap.ksy` | 276-312 | `frame_plane_descriptor_v2` struct definition |
| `cv-mmap/docs/cvmmap.ksy` | 384-390 | Deterministic plane ordering rules |

---

## v2 Header Invariants (64 bytes, packed)

### Byte Layout (Normative)

| Offset | Field | Type | KSY Line | Valid/Constraint |
|--------|-------|------|----------|------------------|
| 0x00 | `magic[8]` | bytes | 334-335 | Must be `[67, 86, 45, 77, 77, 65, 80, 0]` ("CV-MMAP\0") |
| 0x08 | `versions_major` | u1 | 336-338 | Must equal `2` |
| 0x09 | `versions_minor` | u1 | 339-340 | Any u8 value |
| 0x0A | `flags` | u2 | 341-342 | Currently reserved (u16) |
| 0x0C | `frame_id` | u4 | 343-344 | Any u32 value |
| 0x10 | `capture_ts_ns` | u8 | 345-346 | Any u64 value |
| 0x18 | `publish_seq` | u8 | 347-348 | Any u64 value |
| 0x20 | `plane_count` | u1 | 349-351 | Must be in range `[1, 4]` |
| 0x21 | `plane_presence_mask` | u1 | 352-354 | Upper 4 bits must be 0: `(_ & 0xF0) == 0` |
| 0x22 | `plane_descriptors_offset` | u2 | 355-357 | Must equal `64` |
| 0x24 | `plane_descriptor_size` | u2 | 358-360 | Must equal `24` |
| 0x26 | `plane_descriptor_capacity` | u2 | 361-363 | Must equal `4` |
| 0x28 | `payload_size_bytes` | u4 | 364-366 | Must be greater than `0` |
| 0x2C | `reserved_0[20]` | bytes | 367-368 | 20 reserved bytes |

### KSY Instance Constraints

| Instance | Expression | KSY Line | Meaning |
|----------|------------|----------|---------|
| `contiguous_mask_expected` | `(1 << plane_count) - 1` | 370-371 | Expected bit pattern for contiguous plane mask |
| `contiguous_mask_valid` | `plane_presence_mask == contiguous_mask_expected` | 372-373 | Presence mask must match expected contiguous pattern |

---

## v2 Plane Descriptor Invariants (24 bytes, packed)

### Byte Layout (Normative)

| Offset | Field | Type | KSY Line | Description |
|--------|-------|------|----------|-------------|
| 0x00 | `plane_type` | u1 | 288-289 | Enum: `frame_plane_type` |
| 0x01 | `pixel_format` | u1 | 291-293 | Enum: `pixel_format` |
| 0x02 | `depth` | u1 | 294-296 | Enum: `depth` |
| 0x03 | `reserved_0` | u1 | 297-299 | Explicit reserved byte |
| 0x04 | `width` | u4 | 300-301 | Width in pixels |
| 0x08 | `height` | u4 | 302-303 | Height in pixels |
| 0x0C | `stride_bytes` | u4 | 304-305 | Stride in bytes |
| 0x10 | `offset_bytes` | u4 | 306-307 | Offset into payload |
| 0x14 | `size_bytes` | u4 | 308-309 | Size of plane data in bytes |

### Descriptor Invariants (KSY Lines 280-287)

1. `offset_bytes <= payload_size_bytes`
2. `size_bytes <= payload_size_bytes - offset_bytes`
3. Active descriptors must have non-zero width/height/stride/size
4. Inactive descriptors must be all-zero for geometric + byte-range fields

### KSY Instance Helpers

| Instance | Expression | KSY Line | Meaning |
|----------|------------|----------|---------|
| `is_empty_descriptor` | `width == 0 and height == 0 and stride_bytes == 0 and offset_bytes == 0 and size_bytes == 0` | 311-312 | True if descriptor is inactive/empty |

---

## v2 Full Metadata Layout (256 bytes)

| Region | Offset Range | Content | KSY Lines |
|--------|--------------|---------|-----------|
| Header | [0..63] | `frame_metadata_v2_header` | 392-393 |
| Descriptors | [64..159] | 4 x `frame_plane_descriptor_v2` (fixed slots) | 394-401 |
| Reserved | [160..255] | Reserved/padding | 402-404 |
| **Payload** | **256+** | **Frame data begins here** | - |

---

## Deterministic Plane Ordering Rules (KSY Lines 384-390)

**Rule 1:** Active descriptors are contiguous from slot 0.  
**Rule 2:** Slot 0 is always LEFT plane.  
**Rule 3:** Slot 1 is DEPTH plane when `plane_count >= 2`.  
**Rule 4:** Slot 2 is CONFIDENCE plane when `plane_count >= 3`.  
**Rule 5:** Slots >= plane_count are inactive and must be empty descriptors.  
**Rule 6:** Active planes are packed in payload order: each next offset equals the previous offset + previous size.

### Plane Slot Semantics

| Slot | Plane Type | Condition |
|------|------------|-----------|
| 0 | `left` | Always active (plane_count >= 1) |
| 1 | `depth` | Active when plane_count >= 2 |
| 2 | `confidence` | Active when plane_count >= 3 |
| 3 | Reserved | Active when plane_count == 4 |

---

## Implementation Target Mapping: Python Client

### Current v1 Implementation (Legacy)

| KSY v2 Field | Current Python Location | Status | Notes |
|--------------|------------------------|--------|-------|
| `magic[8]` | `src/cvmmap/msg.py:33` | EXISTS | `CV_MMAP_MAGIC: bytes = b"CV-MMAP\0"` |
| `versions_major` | `src/cvmmap/msg.py:29` | EXISTS v1 | `VERSION_MAJOR = 1` |
| `versions_minor` | `src/cvmmap/msg.py:30` | EXISTS v1 | `VERSION_MINOR = 0` |
| `flags` | - | **TODO** | Add u16 flags parsing |
| `frame_id` | - | **TODO** | Add u32 frame_id parsing |
| `capture_ts_ns` | `src/cvmmap/msg.py:259` | EXISTS | `timestamp_ns` in FrameMetadata |
| `publish_seq` | - | **TODO** | Add u64 publish sequence parsing |
| `plane_count` | - | **TODO** | Add u8 plane count parsing |
| `plane_presence_mask` | - | **TODO** | Add u8 presence mask parsing |
| `plane_descriptors_offset` | - | **TODO** | Validate u16 descriptor offset == 64 |
| `plane_descriptor_size` | - | **TODO** | Validate u16 descriptor size == 24 |
| `plane_descriptor_capacity` | - | **TODO** | Validate u16 descriptor capacity == 4 |
| `payload_size_bytes` | - | **TODO** | Add u32 payload size parsing |
| `reserved_0[20]` | - | **TODO** | Skip 20 reserved bytes |

### Current v1 `FrameInfo` (12 bytes) - `src/cvmmap/msg.py:169-232`

| Field | Python Location | Size | Pack Format | Notes |
|-------|-----------------|------|-------------|-------|
| `width` | `msg.py:186` | u16 | `H` | Same as v2 |
| `height` | `msg.py:187` | u16 | `H` | Same as v2 |
| `channels` | `msg.py:188` | u8 | `B` | Replaced by `pixel_format` in v2 |
| `depth` | `msg.py:189` | u8 | `B` | Same as v2 |
| `pixel_format` | `msg.py:190` | u8 | `B` | Same as v2 |
| `_reserved_0[1]` | `msg.py:191` | u8 | `x` | Padding |
| `buffer_size` | `msg.py:191` | u32 | `I` | Replaced by per-plane `size_bytes` in v2 |

**Current Pack Format:** `"=HHBBBxI"` (native, standard size)

### Current v1 `FrameMetadata` - `src/cvmmap/msg.py:235-280`

| Field | Python Location | Offset | Format | Notes |
|-------|-----------------|--------|--------|-------|
| `versions_major` | `msg.py:270` | 0 | `B` | After magic |
| `versions_minor` | `msg.py:270` | 1 | `B` | After magic |
| `_reserved_0[2]` | `msg.py:264` | 2 | `xx` | Padding |
| `frame_count` | `msg.py:270` | 4 | `I` | v1 field, becomes `frame_id` in v2 |
| `timestamp_ns` | `msg.py:270` | 8 | `Q` | Same as v2 `capture_ts_ns` |
| `info` (FrameInfo) | `msg.py:274` | 16 | 12 bytes | v1 info struct |

**Current Format:** `"=BBxxIQ"` + FrameInfo

### Reading from SHM - `src/cvmmap/__init__.py`

| Method | Location | Status | Notes |
|--------|----------|--------|-------|
| `_read_metadata()` | `__init__.py:138-163` | EXISTS v1 | Reads FrameMetadata from SHM |
| `_read_metadata_unchecked()` | `__init__.py:165-174` | EXISTS v1 | Same without magic check |
| Magic validation | `__init__.py:155-159` | EXISTS | Checks `CV_MMAP_MAGIC` |
| `_SHM_PAYLOAD_OFFSET` | `__init__.py:136` | EXISTS | Currently 256, matches v2 |

---

## Migration Checklist: Python Client

### Phase 1: New v2 Classes

- [ ] Create `FramePlaneDescriptorV2` dataclass (24 bytes)
  - Location: `src/cvmmap/msg.py`
  - Pack format: `"=BBBBIIIII"` (1+1+1+1+4+4+4+4+4 = 24 bytes)
- [ ] Create `FrameMetadataV2Header` dataclass (64 bytes)
  - Location: `src/cvmmap/msg.py`
  - Pack format: `"=8sBBBHQQBBHHHHI20s"`
- [ ] Create `FrameMetadataV2` dataclass (256 bytes)
  - Location: `src/cvmmap/msg.py`
  - Contains: header + 4 descriptors + 96 reserved bytes

### Phase 2: Validation Logic

- [ ] Implement `plane_presence_mask` validation: `(mask & 0xF0) == 0`
- [ ] Implement `plane_count` validation: `1 <= count <= 4`
- [ ] Implement `contiguous_mask_valid` check
- [ ] Implement `is_empty_descriptor` check for inactive slots
- [ ] Implement plane bounds validation
- [ ] Update `VERSION_MAJOR` constant to `2`

### Phase 3: Reader Updates

- [ ] Add `FrameMetadataV2` import to `__init__.py`
- [ ] Update `_read_metadata()` to support v2 parsing
- [ ] Maintain backward compatibility with v1 (version negotiation)

### Phase 4: Tests

- [ ] Add tests for v2 header size: `FrameMetadataV2Header.size() == 64`
- [ ] Add tests for v2 descriptor size: `FramePlaneDescriptorV2.size() == 24`
- [ ] Add tests for v2 full metadata: `FrameMetadataV2.size() == 256`
- [ ] Add tests for plane ordering invariants
- [ ] Add roundtrip marshal/unmarshal tests

---

## Reference: KSY Type Enums

### `frame_plane_type` (u1)

| Value | Name |
|-------|------|
| 0 | `left` |
| 1 | `depth` |
| 2 | `confidence` |

### `pixel_format` (u1)

| Value | Name |
|-------|------|
| 0 | `rgb` |
| 1 | `bgr` |
| 2 | `rgba` |
| 3 | `bgra` |
| 4 | `gray` |
| 5 | `yuv` |
| 6 | `yuyv` |

### `depth` (u1)

| Value | Name |
|-------|------|
| 0 | `u8` |
| 1 | `s8` |
| 2 | `u16` |
| 3 | `s16` |
| 4 | `s32` |
| 5 | `f32` |
| 6 | `f64` |
| 7 | `f16` |

---

## Version Constants

| Constant | Current Value | Location | Target v2 |
|----------|---------------|----------|-----------|
| `VERSION_MAJOR` | `1` | `src/cvmmap/msg.py:29` | `2` |
| `VERSION_MINOR` | `0` | `src/cvmmap/msg.py:30` | `0` |

---

## Verification

Run this grep to verify all normative fields are present in upstream ksy:

```bash
cd /workspaces/zed-playground/cv-mmap && grep -n "plane_descriptor_size\|plane_descriptor_capacity\|plane_presence_mask" docs/cvmmap_shm_metadata_v1_v2.ksy
```

Expected output should include:
- Line 352: `plane_presence_mask`
- Line 358-360: `plane_descriptor_size`
- Line 361-363: `plane_descriptor_capacity`
