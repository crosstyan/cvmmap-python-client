DEFAULT_NATS_URL = "nats://localhost:4222"


def _make_prefix(target_key: str) -> str:
    return f"cvmmap.{target_key}"


def subject_producer_prefix(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.producer"


def subject_producer_source_reset(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.source.reset"


def subject_producer_source_info(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.source.info"


def subject_producer_source_seek(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.source.seek"


def subject_producer_source_capabilities(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.source.capabilities"


def subject_producer_svo_recorder_capabilities(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.recorder.svo.capabilities"


def subject_producer_svo_recorder_start(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.recorder.svo.start"


def subject_producer_svo_recorder_stop(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.recorder.svo.stop"


def subject_producer_svo_recorder_status(target_key: str) -> str:
    return f"{subject_producer_prefix(target_key)}.recorder.svo.status"


def subject_body(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.body"


def subject_status(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.status"
