"""Baseline and realtime-oriented audio preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchaudio

from .vad import SpeechActivity, detect_speech_activity, speech_sample_mask

TARGET_SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 3.0
SEGMENT_SAMPLES = int(TARGET_SAMPLE_RATE * SEGMENT_SECONDS)

MIN_TOTAL_SPEECH_SECONDS = 2.0
MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS = 1.5
INSUFFICIENT_SPEECH_MESSAGE = (
    "Not enough speech detected. Please speak for a few seconds and try again."
)

PathLike = Union[str, Path]


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded."""


class AudioValidationError(ValueError):
    """Raised when decoded or supplied audio is invalid."""


@dataclass(frozen=True)
class SpeechWindowMetadata:
    """Speech and selection timings measured against the original recording."""

    original_duration_seconds: float
    selected_start_seconds: float
    selected_end_seconds: float
    speech_in_selected_window_seconds: float
    total_speech_seconds: float
    sufficient_speech: bool


@dataclass(frozen=True)
class PreparedAudio:
    waveform: torch.Tensor
    metadata: SpeechWindowMetadata


class InsufficientSpeechError(AudioValidationError):
    """Raised when realtime quality requirements are not met."""

    def __init__(self, metadata: SpeechWindowMetadata) -> None:
        super().__init__(INSUFFICIENT_SPEECH_MESSAGE)
        self.metadata = metadata


def _validate_waveform(waveform: torch.Tensor, *, context: str) -> None:
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor.")
    if waveform.ndim not in (1, 2):
        raise AudioValidationError(
            f"{context} must have shape [time] or [channels, time], "
            f"but received {tuple(waveform.shape)}."
        )
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise AudioValidationError(f"{context} is empty.")
    if not waveform.is_floating_point():
        raise AudioValidationError(
            f"{context} must contain floating-point samples; integer PCM must be decoded first."
        )
    if not torch.isfinite(waveform).all():
        raise AudioValidationError(f"{context} contains NaN or infinite samples.")


def _validate_sample_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError(
            f"sample_rate must be a positive integer, but received {sample_rate!r}."
        )


def decode_audio(path: PathLike) -> tuple[torch.Tensor, int]:
    """Decode a file without cropping, padding, resampling, or mixing channels."""

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioValidationError(f"Audio path is not a file: {audio_path}")

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception as exc:
        raise AudioLoadError(f"Could not decode audio file: {audio_path}") from exc

    _validate_waveform(waveform, context="decoded waveform")
    _validate_sample_rate(sample_rate)
    return waveform, sample_rate


def convert_to_mono_16k(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample channel-first audio to 16 kHz, then average channels to mono."""

    _validate_sample_rate(sample_rate)
    _validate_waveform(waveform, context="waveform")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mono = waveform.squeeze(0).contiguous()
    _validate_waveform(mono, context="converted waveform")
    return mono


def _crop_or_right_pad_baseline(mono: torch.Tensor) -> torch.Tensor:
    num_samples = mono.shape[0]
    if num_samples >= SEGMENT_SAMPLES:
        start = (num_samples - SEGMENT_SAMPLES) // 2
        mono = mono[start : start + SEGMENT_SAMPLES]
    else:
        mono = F.pad(mono, (0, SEGMENT_SAMPLES - num_samples))

    if mono.shape != (SEGMENT_SAMPLES,):
        raise AudioValidationError(
            f"Prepared waveform has unexpected shape {tuple(mono.shape)}."
        )
    if not torch.isfinite(mono).all():
        raise AudioValidationError("Prepared waveform contains NaN or infinite samples.")
    return mono.contiguous()


def prepare_audio_baseline(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Reproduce the notebook-compatible center-crop/right-pad preparation."""

    return _crop_or_right_pad_baseline(convert_to_mono_16k(waveform, sample_rate))


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Backward-compatible alias for :func:`prepare_audio_baseline`."""

    return prepare_audio_baseline(waveform, sample_rate)


def load_audio_baseline(path: PathLike) -> torch.Tensor:
    waveform, sample_rate = decode_audio(path)
    return prepare_audio_baseline(waveform, sample_rate)


def load_audio(path: PathLike) -> torch.Tensor:
    """Backward-compatible baseline file loader."""

    return load_audio_baseline(path)


def _validate_activity(activity: SpeechActivity, num_samples: int) -> None:
    if activity.sample_rate != TARGET_SAMPLE_RATE:
        raise AudioValidationError(
            f"Speech activity must use {TARGET_SAMPLE_RATE} Hz audio."
        )
    if activity.num_samples != num_samples:
        raise AudioValidationError(
            "Speech activity sample count does not match the converted waveform."
        )


def select_speech_rich_window(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    speech_activity: Optional[SpeechActivity] = None,
    original_duration_seconds: Optional[float] = None,
) -> PreparedAudio:
    """Select the contiguous three-second window with the most detected speech."""

    _validate_sample_rate(sample_rate)
    _validate_waveform(waveform, context="speech-window waveform")
    if waveform.ndim != 1:
        raise AudioValidationError("Speech-window selection requires mono audio.")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioValidationError(
            f"Speech-window selection requires {TARGET_SAMPLE_RATE} Hz audio."
        )

    activity = speech_activity or detect_speech_activity(waveform, sample_rate)
    _validate_activity(activity, waveform.numel())
    activity_mask = speech_sample_mask(activity)
    total_speech_samples = int(activity_mask.sum().item())

    if waveform.numel() > SEGMENT_SAMPLES:
        cumulative = F.pad(activity_mask.to(torch.int64).cumsum(dim=0), (1, 0))
        window_scores = cumulative[SEGMENT_SAMPLES:] - cumulative[:-SEGMENT_SAMPLES]
        # torch.argmax returns the first maximum, providing the documented tie-break.
        selected_start = int(torch.argmax(window_scores).item())
        selected_stop = selected_start + SEGMENT_SAMPLES
        selected_speech_samples = int(window_scores[selected_start].item())
        selected_waveform = waveform[selected_start:selected_stop].contiguous()
        selected_end_seconds = selected_stop / sample_rate
    else:
        selected_start = 0
        selected_stop = waveform.numel()
        selected_speech_samples = total_speech_samples
        selected_waveform = F.pad(
            waveform,
            (0, SEGMENT_SAMPLES - waveform.numel()),
        ).contiguous()
        selected_end_seconds = selected_stop / sample_rate

    total_speech_seconds = total_speech_samples / sample_rate
    selected_speech_seconds = selected_speech_samples / sample_rate
    sufficient = (
        total_speech_seconds >= MIN_TOTAL_SPEECH_SECONDS
        and selected_speech_seconds >= MIN_SPEECH_IN_SELECTED_WINDOW_SECONDS
    )
    metadata = SpeechWindowMetadata(
        original_duration_seconds=(
            waveform.numel() / sample_rate
            if original_duration_seconds is None
            else original_duration_seconds
        ),
        selected_start_seconds=selected_start / sample_rate,
        selected_end_seconds=selected_end_seconds,
        speech_in_selected_window_seconds=selected_speech_seconds,
        total_speech_seconds=total_speech_seconds,
        sufficient_speech=sufficient,
    )
    if not sufficient:
        raise InsufficientSpeechError(metadata)
    if selected_waveform.shape != (SEGMENT_SAMPLES,):
        raise AudioValidationError(
            f"Realtime waveform has unexpected shape {tuple(selected_waveform.shape)}."
        )
    if not torch.isfinite(selected_waveform).all():
        raise AudioValidationError("Realtime waveform contains NaN or infinite samples.")
    return PreparedAudio(waveform=selected_waveform, metadata=metadata)


def prepare_audio_realtime(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    speech_activity: Optional[SpeechActivity] = None,
) -> PreparedAudio:
    """Convert arbitrary microphone/file audio and apply realtime preparation."""

    _validate_sample_rate(sample_rate)
    _validate_waveform(waveform, context="waveform")
    original_duration_seconds = waveform.shape[-1] / sample_rate
    mono = convert_to_mono_16k(waveform, sample_rate)
    return select_speech_rich_window(
        mono,
        TARGET_SAMPLE_RATE,
        speech_activity=speech_activity,
        original_duration_seconds=original_duration_seconds,
    )


def load_audio_realtime(path: PathLike) -> PreparedAudio:
    waveform, sample_rate = decode_audio(path)
    return prepare_audio_realtime(waveform, sample_rate)
