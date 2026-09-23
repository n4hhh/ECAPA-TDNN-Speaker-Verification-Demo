"""Inference-only speaker verification package."""

from .audio import (
    MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS,
    MIN_TOTAL_SPEECH_SECONDS,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE,
    InsufficientSpeechError,
    PreparedAudio,
    SpeechWindowMetadata,
    load_audio,
    load_audio_realtime,
    prepare_audio_baseline,
    prepare_audio_realtime,
    prepare_waveform,
    select_speech_rich_window,
)
from .inference import (
    RealtimeEmbedding,
    RealtimeVerificationResult,
    SpeakerVerifier,
    VerificationResult,
    decision_from_threshold,
    score_embeddings,
)
from .model import EMBEDDING_DIM, FrozenHandoffECAPAModel, load_model
from .model_thresholds import threshold_for_model
from .vad import VAD_FRAME_DURATION_MS, VAD_MODE, SpeechActivity, detect_speech_activity

__all__ = [
    "EMBEDDING_DIM",
    "FrozenHandoffECAPAModel",
    "InsufficientSpeechError",
    "MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS",
    "MIN_TOTAL_SPEECH_SECONDS",
    "PreparedAudio",
    "RealtimeEmbedding",
    "RealtimeVerificationResult",
    "SEGMENT_SAMPLES",
    "SpeakerVerifier",
    "SpeechActivity",
    "SpeechWindowMetadata",
    "TARGET_SAMPLE_RATE",
    "VAD_FRAME_DURATION_MS",
    "VAD_MODE",
    "VerificationResult",
    "decision_from_threshold",
    "detect_speech_activity",
    "load_audio",
    "load_audio_realtime",
    "load_model",
    "prepare_audio_baseline",
    "prepare_audio_realtime",
    "prepare_waveform",
    "score_embeddings",
    "select_speech_rich_window",
    "threshold_for_model",
]
