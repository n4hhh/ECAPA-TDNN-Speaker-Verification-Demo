"""Local Streamlit UI for 1:1 speaker verification."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import soundfile as sf
import torch

from src.audio import (
    AudioLoadError,
    AudioValidationError,
    InsufficientSpeechError,
    SpeechWindowMetadata,
)
from src.inference import (
    PROVISIONAL_ADP_AUG_CROSS_DOMAIN_EER_THRESHOLD,
    RealtimeEmbedding,
    SpeakerVerifier,
    score_embeddings,
)
from src.model import (
    CheckpointCompatibilityError,
    load_checkpoint,
    validate_checkpoint,
)

LOGGER = logging.getLogger("speaker_verification_demo")
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = PROJECT_ROOT / "checkpoints"
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
SUPPORTED_UPLOAD_SUFFIXES = {".wav", ".mp3"}
MP3_DECODE_ERROR_MESSAGE = (
    "Unable to decode this MP3 file. Please try another file or upload WAV audio."
)

SESSION_DEFAULTS: dict[str, Any] = {
    "active_model_path": None,
    "enrollment_embedding": None,
    "enrollment_model_path": None,
    "enrollment_metadata": None,
    "enrollment_audio_bytes": None,
    "enrollment_audio_mime": None,
    "enrollment_saved": False,
    "latest_verification_result": None,
    "enrollment_input_version": 0,
    "verification_input_version": 0,
}


class Mp3DecodeError(AudioLoadError):
    """Raised when the local decoder cannot read an uploaded MP3."""


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Add missing keys without replacing values retained across reruns."""

    for key, value in SESSION_DEFAULTS.items():
        state.setdefault(key, value)


def clear_enrollment_state(
    state: MutableMapping[str, Any],
    *,
    reset_inputs: bool = True,
) -> None:
    """Remove all model-dependent enrollment and verification state."""

    state["enrollment_embedding"] = None
    state["enrollment_model_path"] = None
    state["enrollment_metadata"] = None
    state["enrollment_audio_bytes"] = None
    state["enrollment_audio_mime"] = None
    state["enrollment_saved"] = False
    state["latest_verification_result"] = None
    if reset_inputs:
        state["enrollment_input_version"] = state.get("enrollment_input_version", 0) + 1
        state["verification_input_version"] = state.get("verification_input_version", 0) + 1


def activate_model(state: MutableMapping[str, Any], model_path: str) -> bool:
    """Activate a model and invalidate enrollment if its identity changed.

    Returns True only when an existing enrollment was invalidated.
    """

    normalized_path = str(Path(model_path).resolve())
    previous_path = state.get("active_model_path")
    if previous_path == normalized_path:
        return False

    had_enrollment = bool(state.get("enrollment_saved"))
    clear_enrollment_state(state)
    state["active_model_path"] = normalized_path
    return had_enrollment


def enrollment_is_ready(state: MutableMapping[str, Any], model_path: str) -> bool:
    """Return whether a normalized enrollment belongs to the active model."""

    normalized_path = str(Path(model_path).resolve())
    embedding = state.get("enrollment_embedding")
    return bool(
        state.get("enrollment_saved")
        and isinstance(embedding, torch.Tensor)
        and state.get("enrollment_model_path") == normalized_path
    )


def save_enrollment_state(
    state: MutableMapping[str, Any],
    result: RealtimeEmbedding,
    model_path: str,
    audio_bytes: bytes,
    audio_mime: str,
) -> None:
    """Store a session-only enrollment and invalidate the previous result."""

    state["enrollment_embedding"] = result.embedding.detach().cpu()
    state["enrollment_model_path"] = str(Path(model_path).resolve())
    state["enrollment_metadata"] = result.metadata
    state["enrollment_audio_bytes"] = audio_bytes
    state["enrollment_audio_mime"] = audio_mime
    state["enrollment_saved"] = True
    state["latest_verification_result"] = None


def discover_checkpoint_files(directory: Path) -> list[Path]:
    """Find checkpoint candidates deterministically, including subdirectories."""

    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )


def checkpoint_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


@st.cache_data(show_spinner=False)
def inspect_checkpoint_candidate(
    checkpoint_path: str,
    file_size: int,
    modified_time_ns: int,
) -> tuple[bool, Optional[str]]:
    """Cache lightweight checkpoint schema validation by immutable file signature."""

    del file_size, modified_time_ns
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint(checkpoint)
    except Exception as exc:
        return False, str(exc)
    return True, None


def discover_compatible_checkpoints(
    directory: Path,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    compatible: list[Path] = []
    rejected: list[tuple[Path, str]] = []
    for path in discover_checkpoint_files(directory):
        signature = checkpoint_signature(path)
        valid, error = inspect_checkpoint_candidate(*signature)
        if valid:
            compatible.append(path)
        else:
            rejected.append((path, error or "Unknown compatibility error."))
    return compatible, rejected


@st.cache_resource(show_spinner=False)
def get_cached_verifier(
    checkpoint_path: str,
    file_size: int,
    modified_time_ns: int,
) -> SpeakerVerifier:
    """Strictly load and reuse one verifier per checkpoint file signature."""

    del file_size, modified_time_ns
    return SpeakerVerifier(checkpoint_path)


def mp3_decoder_available() -> bool:
    """Detect whether the local libsndfile build advertises MPEG decoding."""

    return "MP3" in sf.available_formats() and any(
        "MPEG" in subtype for subtype in sf.available_subtypes()
    )


def process_audio_bytes(
    verifier: SpeakerVerifier,
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
) -> RealtimeEmbedding:
    """Route browser/upload bytes through the existing realtime file pipeline."""

    if not audio_bytes:
        raise AudioValidationError("The selected audio file is empty.")
    normalized_suffix = suffix.lower()
    if normalized_suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise AudioValidationError("Only WAV and MP3 audio are supported by this demo.")
    if normalized_suffix == ".mp3" and not mp3_decoder_available():
        raise Mp3DecodeError(MP3_DECODE_ERROR_MESSAGE)

    with tempfile.TemporaryDirectory(prefix="speaker_verification_ui_") as temp_name:
        audio_path = Path(temp_name) / f"input{normalized_suffix}"
        audio_path.write_bytes(audio_bytes)
        try:
            return verifier.extract_embedding_realtime(audio_path)
        except AudioLoadError as exc:
            if normalized_suffix == ".mp3":
                raise Mp3DecodeError(MP3_DECODE_ERROR_MESSAGE) from exc
            raise


def render_metadata(metadata: SpeechWindowMetadata) -> None:
    st.write(f"Original duration: {metadata.original_duration_seconds:.2f} s")
    st.write(f"Detected speech: {metadata.total_speech_seconds:.2f} s")
    st.write(
        "Selected segment: "
        f"{metadata.selected_start_seconds:.2f} - {metadata.selected_end_seconds:.2f} s"
    )
    st.write(
        "Speech in segment: "
        f"{metadata.speech_in_selected_window_seconds:.2f} s"
    )


def render_audio_input(
    section: str,
    version: int,
) -> tuple[Optional[bytes], str, str]:
    """Render one unambiguous microphone-or-upload source selector."""

    source = st.radio(
        "Audio source",
        ("Record microphone", "Upload audio"),
        horizontal=True,
        key=f"{section}_source_{version}",
    )
    if source == "Record microphone":
        value = st.audio_input(
            "Record a voice utterance",
            sample_rate=16_000,
            key=f"{section}_microphone_{version}",
            help="Speak naturally for several seconds.",
        )
    else:
        value = st.file_uploader(
            "Upload a WAV or MP3 file",
            type=["wav", "mp3"],
            accept_multiple_files=False,
            key=f"{section}_upload_{version}",
        )
        if not mp3_decoder_available():
            st.warning(
                "Local MP3 decoding is unavailable in this environment. "
                "WAV upload remains available."
            )

    if value is None:
        return None, "audio/wav", ".wav"
    suffix = ".wav" if source == "Record microphone" else Path(value.name).suffix.lower()
    mime = value.type or ("audio/mpeg" if suffix == ".mp3" else "audio/wav")
    return value.getvalue(), mime, suffix


def show_processing_error(context: str, exc: Exception) -> None:
    """Show a concise error while retaining diagnostic detail in terminal logs."""

    LOGGER.warning("%s failed", context, exc_info=True)
    if isinstance(exc, Mp3DecodeError):
        st.error(MP3_DECODE_ERROR_MESSAGE)
    elif isinstance(exc, InsufficientSpeechError):
        st.error(str(exc))
    elif isinstance(exc, FileNotFoundError):
        st.error("The selected audio file could not be found. Please choose it again.")
    elif isinstance(exc, (AudioLoadError, AudioValidationError)):
        st.error(f"The audio could not be processed: {exc}")
    else:
        st.error(f"{context} failed. See the terminal log for details.")


def _checkpoint_label(path_string: str) -> str:
    path = Path(path_string)
    try:
        return str(path.relative_to(CHECKPOINT_DIRECTORY.resolve()))
    except ValueError:
        return path.name


def main() -> None:
    st.set_page_config(
        page_title="Vietnamese Speaker Verification Demo",
        page_icon="🎙️",
        layout="centered",
    )
    initialize_session_state(st.session_state)

    st.title("Vietnamese Speaker Verification Demo")
    st.caption(
        "Enrollment and verification may contain different spoken content. "
        "Speak naturally for several seconds."
    )

    compatible, rejected = discover_compatible_checkpoints(CHECKPOINT_DIRECTORY)
    if rejected:
        with st.expander("Skipped incompatible checkpoints"):
            for path, error in rejected:
                st.write(f"{path.name}: {error}")
    if not compatible:
        st.error("No compatible .pt or .pth checkpoints were found in checkpoints/.")
        st.stop()

    model_options = [str(path) for path in compatible]
    active_path = st.session_state.get("active_model_path")
    selected_index = model_options.index(active_path) if active_path in model_options else 0
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=selected_index,
        format_func=_checkpoint_label,
        key="model_selector",
    )
    enrollment_invalidated = activate_model(st.session_state, selected_model)
    if enrollment_invalidated:
        st.info("Model changed. Please save a new enrollment utterance.")

    try:
        verifier = get_cached_verifier(*checkpoint_signature(Path(selected_model)))
    except (CheckpointCompatibilityError, FileNotFoundError) as exc:
        LOGGER.exception("Model loading failed")
        st.error(f"The selected checkpoint is incompatible: {exc}")
        st.stop()
    except Exception:
        LOGGER.exception("Unexpected model loading failure")
        st.error("The selected model could not be loaded. See the terminal log for details.")
        st.stop()
    st.success("Model ready")

    st.divider()
    st.header("1. Enrollment Utterance")
    enrollment_bytes, enrollment_mime, enrollment_suffix = render_audio_input(
        "enrollment",
        st.session_state["enrollment_input_version"],
    )
    if st.button("SAVE ENROLLMENT", type="primary", use_container_width=True):
        if enrollment_bytes is None:
            st.warning("Record or upload enrollment audio first.")
        else:
            try:
                with st.spinner("Processing enrollment..."):
                    enrollment_result = process_audio_bytes(
                        verifier,
                        enrollment_bytes,
                        suffix=enrollment_suffix,
                    )
                save_enrollment_state(
                    st.session_state,
                    enrollment_result,
                    selected_model,
                    enrollment_bytes,
                    enrollment_mime,
                )
            except Exception as exc:
                show_processing_error("Enrollment", exc)

    if enrollment_is_ready(st.session_state, selected_model):
        st.success("✓ Enrollment saved")
        render_metadata(st.session_state["enrollment_metadata"])
        if st.session_state["enrollment_audio_bytes"]:
            st.audio(
                st.session_state["enrollment_audio_bytes"],
                format=st.session_state["enrollment_audio_mime"] or "audio/wav",
            )

    st.divider()
    st.header("2. Verification Utterance")
    verification_bytes, _, verification_suffix = render_audio_input(
        "verification",
        st.session_state["verification_input_version"],
    )
    ready_to_verify = enrollment_is_ready(st.session_state, selected_model)
    if not ready_to_verify:
        st.info("Save an enrollment utterance before verification.")

    if st.button(
        "VERIFY SPEAKER",
        type="primary",
        use_container_width=True,
        disabled=not ready_to_verify,
    ):
        if verification_bytes is None:
            st.warning("Record or upload verification audio first.")
        else:
            try:
                with st.spinner("Verifying speaker..."):
                    verification_result = process_audio_bytes(
                        verifier,
                        verification_bytes,
                        suffix=verification_suffix,
                    )
                    similarity = score_embeddings(
                        st.session_state["enrollment_embedding"],
                        verification_result.embedding,
                    )
                threshold = PROVISIONAL_ADP_AUG_CROSS_DOMAIN_EER_THRESHOLD
                st.session_state["latest_verification_result"] = {
                    "similarity": similarity,
                    "threshold": threshold,
                    "same_speaker": similarity >= threshold,
                    "metadata": verification_result.metadata,
                }
            except Exception as exc:
                show_processing_error("Verification", exc)

    latest_result = st.session_state.get("latest_verification_result")
    if latest_result is not None and ready_to_verify:
        st.subheader("Verification Result")
        st.write(f"Similarity Score: `{latest_result['similarity']:.4f}`")
        st.write(f"Decision Threshold: `{latest_result['threshold']:.4f}`")
        if latest_result["same_speaker"]:
            st.success("SAME SPEAKER")
        else:
            st.error("DIFFERENT SPEAKER")
        render_metadata(latest_result["metadata"])

    st.divider()
    if st.button(
        "CLEAR ENROLLMENT",
        use_container_width=True,
        disabled=not bool(st.session_state.get("enrollment_saved")),
    ):
        clear_enrollment_state(st.session_state)
        st.rerun()

    with st.expander("Technical Information"):
        st.write("Model: ECAPA-TDNN")
        st.write("Embedding dimension: 192")
        st.write("Input: 16 kHz mono")
        st.write("Model segment: 3.0 seconds / 48,000 samples")
        st.write("Preprocessing: WebRTC VAD + speech-rich window selection")
        st.write("Scoring: Cosine similarity")
        st.write(
            "Current threshold: "
            f"{PROVISIONAL_ADP_AUG_CROSS_DOMAIN_EER_THRESHOLD:.4f} "
            "(provisional operating point)"
        )
        st.write(
            "The spoken content of enrollment and verification utterances "
            "does not need to be the same."
        )


if __name__ == "__main__":
    main()
