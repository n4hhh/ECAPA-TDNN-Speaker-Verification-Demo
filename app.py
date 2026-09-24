"""Presentation-ready Streamlit UI for 1:1 speaker verification."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any, Optional, TypeVar

import soundfile as sf
import streamlit as st
import torch

from src.audio import (
    AudioLoadError,
    AudioValidationError,
    InsufficientSpeechError,
    MultiSegmentMetadata,
    MultiSegmentSpeechError,
    RecordingTooLongError,
)
from src.inference import (
    MultiSegmentEmbedding,
    RealtimeEmbedding,
    SpeakerVerifier,
    decision_from_threshold,
    score_embeddings,
)
from src.model import CheckpointCompatibilityError, load_checkpoint, validate_checkpoint
from src.model_thresholds import multisegment_threshold_for_model
from ui.components import (
    load_styles,
    render_calibration_note,
    render_enrollment_ready,
    render_header,
    render_locked_verification,
    render_privacy_note,
    render_result_card,
    render_sample_details,
    render_score_visualization,
    render_section_heading,
    render_step_indicator,
)

LOGGER = logging.getLogger("speaker_verification_demo")
EmbeddingResult = TypeVar("EmbeddingResult")
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = PROJECT_ROOT / "checkpoints"
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
PRIMARY_MODEL_FILES = {
    "RAW": "best_raw.pt",
    "RANDOM": "best_random.pt",
    "ADAPTIVE": "best_adp.pt",
}
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

    # An enrollment embedding is meaningful only for the checkpoint that produced it;
    # clear every derived value together to prevent cross-model verification.
    state["enrollment_embedding"] = None
    state["enrollment_model_path"] = None
    state["enrollment_metadata"] = None
    state["enrollment_audio_bytes"] = None
    state["enrollment_audio_mime"] = None
    state["enrollment_saved"] = False
    state["latest_verification_result"] = None
    if reset_inputs:
        # Streamlit widget keys are immutable during a run. Advancing the version gives
        # the recorder and uploader fresh keys on the next rerun, which clears their UI.
        state["enrollment_input_version"] = state.get("enrollment_input_version", 0) + 1
        state["verification_input_version"] = state.get("verification_input_version", 0) + 1


def reset_verification_state(state: MutableMapping[str, Any]) -> None:
    """Clear the latest comparison while retaining the enrolled voice profile."""

    state["latest_verification_result"] = None
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


def multisegment_enrollment_is_ready(
    state: MutableMapping[str, Any],
    model_path: str,
) -> bool:
    """Reject stale session enrollments created by the historical protocol."""

    metadata = state.get("enrollment_metadata")
    return bool(
        enrollment_is_ready(state, model_path)
        and isinstance(metadata, MultiSegmentMetadata)
        and metadata.multisegment_used
    )


def save_enrollment_state(
    state: MutableMapping[str, Any],
    result: RealtimeEmbedding | MultiSegmentEmbedding,
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
    # Validate every candidate up front so the model selector never offers a file that
    # is known to violate the frozen checkpoint contract.
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


def _process_audio_bytes_with_extractor(
    extractor: Callable[[Path], EmbeddingResult],
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
) -> EmbeddingResult:
    """Bridge browser bytes to one of the explicit file-based inference protocols."""

    if not audio_bytes:
        raise AudioValidationError("The selected audio file is empty.")
    normalized_suffix = suffix.lower()
    if normalized_suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise AudioValidationError("Only WAV and MP3 audio are supported by this demo.")
    if normalized_suffix == ".mp3" and not mp3_decoder_available():
        raise Mp3DecodeError(MP3_DECODE_ERROR_MESSAGE)

    # The core audio pipeline is file-based. A private temporary directory bridges
    # Streamlit's in-memory upload without leaving recordings on disk after inference.
    with tempfile.TemporaryDirectory(prefix="speaker_verification_ui_") as temp_name:
        audio_path = Path(temp_name) / f"input{normalized_suffix}"
        audio_path.write_bytes(audio_bytes)
        try:
            return extractor(audio_path)
        except AudioLoadError as exc:
            if normalized_suffix == ".mp3":
                raise Mp3DecodeError(MP3_DECODE_ERROR_MESSAGE) from exc
            raise


def process_audio_bytes(
    verifier: SpeakerVerifier,
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
) -> RealtimeEmbedding:
    """Run the historical single-window realtime protocol."""

    return _process_audio_bytes_with_extractor(
        verifier.extract_embedding_realtime,
        audio_bytes,
        suffix=suffix,
    )


def process_audio_bytes_multisegment(
    verifier: SpeakerVerifier,
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
) -> MultiSegmentEmbedding:
    """Run the Streamlit demo's multi-segment realtime protocol."""

    return _process_audio_bytes_with_extractor(
        verifier.extract_embedding_multisegment,
        audio_bytes,
        suffix=suffix,
    )


def build_multisegment_verification_result(
    enrollment_embedding: torch.Tensor,
    verification: MultiSegmentEmbedding,
    model_label: str,
) -> dict[str, Any]:
    """Score aggregated vectors at the model's multi-segment operating point."""

    similarity = score_embeddings(enrollment_embedding, verification.embedding)
    threshold = multisegment_threshold_for_model(model_label)
    return {
        "similarity": similarity,
        "threshold": threshold,
        "same_speaker": decision_from_threshold(similarity, threshold),
        "model_label": model_label,
        "metadata": verification.metadata,
    }


def render_audio_input(
    section: str,
    version: int,
) -> tuple[Optional[bytes], str, str]:
    """Render one unambiguous microphone-or-upload source selector."""

    source = st.radio(
        "Choose input method",
        ("Record microphone", "Upload audio"),
        horizontal=True,
        key=f"{section}_source_{version}",
    )
    if source == "Record microphone":
        value = st.audio_input(
            "Record a voice sample",
            sample_rate=16_000,
            key=f"{section}_microphone_{version}",
            help="For best coverage, speak naturally for approximately 8–10 seconds.",
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
    elif isinstance(exc, (RecordingTooLongError, MultiSegmentSpeechError)):
        st.error(str(exc))
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
    for label, filename in PRIMARY_MODEL_FILES.items():
        if path.name.lower() == filename.lower():
            return label
    try:
        return str(path.relative_to(CHECKPOINT_DIRECTORY.resolve()))
    except ValueError:
        return path.name


def _model_label(path_string: str) -> str:
    return _checkpoint_label(path_string).upper()


def _ordered_model_options(paths: list[Path]) -> list[str]:
    # Keep thesis conditions in a stable, recognizable order; append any additional
    # compatible checkpoints without hiding them from the user.
    by_name = {path.name.lower(): str(path) for path in paths}
    ordered: list[str] = []
    for filename in PRIMARY_MODEL_FILES.values():
        value = by_name.get(filename.lower())
        if value is not None:
            ordered.append(value)
    return ordered + [str(path) for path in paths if str(path) not in ordered]


def _render_sidebar(
    model_options: list[str],
    selected_index: int,
    rejected: list[tuple[Path, str]],
) -> str:
    """Render research configuration outside the primary user journey."""

    with st.sidebar:
        st.header("Demo configuration")
        st.caption(
            "Select the thesis training condition used for both enrollment and verification."
        )
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=selected_index,
            format_func=_checkpoint_label,
            key="model_selector",
        )
        selected_label = _model_label(selected_model)
        operating_threshold = multisegment_threshold_for_model(selected_label)
        threshold_text = (
            "not configured" if operating_threshold is None else f"{operating_threshold:.4f}"
        )
        st.caption(f"Checkpoint: `{Path(selected_model).name}`")
        with st.expander("Advanced / Technical details"):
            st.markdown(
                f"""
                - **Architecture:** ECAPA-TDNN
                - **Base:** speechbrain/spkrec-ecapa-voxceleb
                - **Embedding:** 192-D, L2 normalized
                - **Audio input:** mono, 16 kHz
                - **Model input:** overlapping 3.0-second windows
                - **Window hop:** 1.5 seconds
                - **Features:** SpeechBrain FBank, 80 Mel bins
                - **Realtime selection:** all qualifying contiguous VAD windows
                - **Aggregation:** arithmetic mean, then L2 normalization
                - **Scoring:** cosine similarity
                - **Multi-segment operating threshold:** {threshold_text}
                - **Training condition:** {selected_label}
                """
            )
        if rejected:
            with st.expander("Skipped incompatible checkpoints"):
                for path, error in rejected:
                    st.write(f"**{path.name}:** {error}")
    return selected_model


def _selected_model_index(model_options: list[str], active_path: Optional[str]) -> int:
    if active_path in model_options:
        return model_options.index(active_path)
    adaptive_path = next(
        (
            option
            for option in model_options
            if Path(option).name.lower() == PRIMARY_MODEL_FILES["ADAPTIVE"]
        ),
        None,
    )
    return model_options.index(adaptive_path) if adaptive_path is not None else 0


def main() -> None:
    st.set_page_config(
        page_title="Voice ID · Vietnamese Speaker Verification",
        page_icon="◉",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_session_state(st.session_state)
    load_styles()
    render_header()

    compatible, rejected = discover_compatible_checkpoints(CHECKPOINT_DIRECTORY)
    if not compatible:
        st.error("No compatible .pt or .pth checkpoints were found in checkpoints/.")
        st.stop()

    model_options = _ordered_model_options(compatible)
    selected_index = _selected_model_index(
        model_options,
        st.session_state.get("active_model_path"),
    )
    selected_model = _render_sidebar(model_options, selected_index, rejected)
    selected_label = _model_label(selected_model)
    enrollment_invalidated = activate_model(st.session_state, selected_model)

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
    st.sidebar.success("Model ready")

    if enrollment_invalidated:
        st.info("Model changed. Create a new voice profile for this model.")

    ready_to_verify = multisegment_enrollment_is_ready(st.session_state, selected_model)
    latest_result = st.session_state.get("latest_verification_result")
    current_step = (
        3 if latest_result is not None and ready_to_verify else 2 if ready_to_verify else 1
    )
    render_step_indicator(current_step)

    render_section_heading(
        1,
        "Create Voice Profile",
        "Speak naturally for approximately 8–10 seconds. Recordings must be no longer than 20 seconds.",
        eyebrow="Enrollment",
    )
    if not ready_to_verify:
        enrollment_bytes, enrollment_mime, enrollment_suffix = render_audio_input(
            "enrollment",
            st.session_state["enrollment_input_version"],
        )
        render_privacy_note()
        if st.button("CREATE VOICE PROFILE", type="primary", use_container_width=True):
            if enrollment_bytes is None:
                st.warning("Record or upload enrollment audio first.")
            else:
                try:
                    with st.spinner("Creating voice profile..."):
                        enrollment_result = process_audio_bytes_multisegment(
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
                    st.rerun()
                except Exception as exc:
                    show_processing_error("Enrollment", exc)
    else:
        render_enrollment_ready(
            st.session_state["enrollment_metadata"],
            selected_label,
        )
        if st.session_state["enrollment_audio_bytes"]:
            with st.expander("Play enrollment recording"):
                st.audio(
                    st.session_state["enrollment_audio_bytes"],
                    format=st.session_state["enrollment_audio_mime"] or "audio/wav",
                )
        with st.expander("Enrollment segment details"):
            render_sample_details(st.session_state["enrollment_metadata"])

    render_section_heading(
        2,
        "Verify Speaker",
        "Record 8–10 seconds of natural speech. The sentence does not need to match enrollment.",
        eyebrow="Verification",
        muted=not ready_to_verify,
    )
    if not ready_to_verify:
        render_locked_verification()
        st.info("Create a voice profile before verification.")
        st.button("VERIFY SPEAKER", type="primary", use_container_width=True, disabled=True)
    else:
        verification_bytes, _, verification_suffix = render_audio_input(
            "verification",
            st.session_state["verification_input_version"],
        )
        if st.button("VERIFY SPEAKER", type="primary", use_container_width=True):
            if verification_bytes is None:
                st.warning("Record or upload verification audio first.")
            else:
                try:
                    with st.spinner("Comparing voice samples..."):
                        verification_result = process_audio_bytes_multisegment(
                            verifier,
                            verification_bytes,
                            suffix=verification_suffix,
                        )
                    st.session_state["latest_verification_result"] = (
                        build_multisegment_verification_result(
                            st.session_state["enrollment_embedding"],
                            verification_result,
                            selected_label,
                        )
                    )
                    st.rerun()
                except Exception as exc:
                    show_processing_error("Verification", exc)

    if latest_result is not None and ready_to_verify:
        render_section_heading(
            3,
            "Verification Result",
            "Aggregated cosine similarity is compared with the model's multi-segment operating threshold.",
            eyebrow="Result",
        )
        render_result_card(
            latest_result["similarity"],
            latest_result["threshold"],
            latest_result["same_speaker"],
            latest_result.get("model_label", selected_label),
        )
        render_score_visualization(
            latest_result["similarity"],
            latest_result["threshold"],
        )
        render_calibration_note()
        with st.expander("Verification sample details"):
            render_sample_details(latest_result["metadata"])

    if ready_to_verify:
        st.divider()
        action_columns = st.columns(2)
        with action_columns[0]:
            if st.button("Clear enrollment", use_container_width=True):
                clear_enrollment_state(st.session_state)
                st.rerun()
        with action_columns[1]:
            if st.button(
                "Verify again",
                use_container_width=True,
                disabled=latest_result is None,
            ):
                reset_verification_state(st.session_state)
                st.rerun()


if __name__ == "__main__":
    main()
