Deterministic golden fixtures for cv-mmap metadata parsing.

Cases:
- `v1_valid`: v1 metadata + single payload plane
- `v2_left_only_valid`: v2 metadata with one active LEFT descriptor
- `v2_left_depth_valid`: v2 metadata with active LEFT + DEPTH descriptors
- `v2_left_depth_confidence_valid`: v2 metadata with active LEFT + DEPTH + CONFIDENCE descriptors
- `v2_malformed_descriptor`: v2 metadata where descriptor size exceeds payload bounds

All `*.hex` files are lowercase hex bytes (whitespace-insensitive).
