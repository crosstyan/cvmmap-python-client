DEFAULT_NATS_URL = "nats://localhost:4222"


def _make_prefix(target_key: str) -> str:
    return f"cvmmap.{target_key}"


def subject_control_source_reset(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.source.reset"


def subject_control_source_info(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.source.info"


def subject_control_source_seek(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.source.seek"


def subject_control_source_capabilities(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.source.capabilities"


def subject_control_recorder_svo_capabilities(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.svo.capabilities"


def subject_control_recorder_svo_start(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.svo.start"


def subject_control_recorder_svo_stop(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.svo.stop"


def subject_control_recorder_svo_status(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.svo.status"


def subject_control_recorder_mcap_capabilities(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.mcap.capabilities"


def subject_control_recorder_mcap_start(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.mcap.start"


def subject_control_recorder_mcap_stop(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.mcap.stop"


def subject_control_recorder_mcap_status(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.control.recorder.mcap.status"


def subject_body(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.body"


def subject_status(target_key: str) -> str:
    return f"{_make_prefix(target_key)}.status"
