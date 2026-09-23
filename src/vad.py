"""Lightweight local speech activity analysis using WebRTC VAD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import webrtcvad

VAD_MODE = 2
VAD_FRAME_DURATION_MS = 20
WEBRTC_SAMPLE_RATES = {8_000, 16_000, 32_000, 48_000}
WEBRTC_FRAME_DURATIONS_MS = {10, 20, 30}


@dataclass(frozen=True)
class SpeechActivity:
    """Frame decisions and timing information produced by WebRTC VAD."""

    sample_rate: int
    frame_duration_ms: int
    frame_samples: int
    num_samples: int
    frame_is_speech: tuple[bool, ...]

    @property
    def total_speech_samples(self) -> int:
        total = 0
        for index, is_speech in enumerate(self.frame_is_speech):
            if is_speech:
                start = index * self.frame_samples
                total += max(0, min(self.frame_samples, self.num_samples - start))
        return total

    @property
    def total_speech_seconds(self) -> float:
        return self.total_speech_samples / self.sample_rate


def detect_speech_activity(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    mode: int = VAD_MODE,
    frame_duration_ms: int = VAD_FRAME_DURATION_MS,
) -> SpeechActivity:
    """Return sequential WebRTC VAD decisions for a mono floating waveform.

    A clipped 16-bit PCM copy is created only for WebRTC analysis. The original
    floating waveform is never modified and remains the source for model input.
    """

    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("WebRTC VAD requires a non-empty mono waveform.")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError("WebRTC VAD requires finite floating-point samples.")
    if sample_rate not in WEBRTC_SAMPLE_RATES:
        raise ValueError(
            f"WebRTC VAD sample rate must be one of {sorted(WEBRTC_SAMPLE_RATES)}."
        )
    if frame_duration_ms not in WEBRTC_FRAME_DURATIONS_MS:
        raise ValueError(
            "WebRTC VAD frame duration must be 10, 20, or 30 milliseconds."
        )
    if mode not in (0, 1, 2, 3):
        raise ValueError("WebRTC VAD mode must be between 0 and 3.")

    frame_samples = sample_rate * frame_duration_ms // 1_000
    # WebRTC VAD accepts 16-bit PCM bytes rather than normalized float samples.
    # Clipping protects the conversion from wraparound without altering model input.
    pcm = (
        waveform.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .mul(32_767.0)
        .round()
        .to(torch.int16)
        .contiguous()
    )
    detector = webrtcvad.Vad(mode)
    decisions: list[bool] = []

    for start in range(0, pcm.numel(), frame_samples):
        frame = pcm[start : start + frame_samples]
        # WebRTC requires a complete 10/20/30 ms frame. Padding affects only the
        # analysis copy; SpeechActivity.num_samples keeps the true input boundary.
        if frame.numel() < frame_samples:
            frame = F.pad(frame, (0, frame_samples - frame.numel()))
        decisions.append(detector.is_speech(frame.numpy().tobytes(), sample_rate))

    return SpeechActivity(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        frame_samples=frame_samples,
        num_samples=waveform.numel(),
        frame_is_speech=tuple(decisions),
    )


def speech_sample_mask(activity: SpeechActivity) -> torch.Tensor:
    """Expand frame decisions into a boolean mask over the real input samples."""

    expected_frames = (
        activity.num_samples + activity.frame_samples - 1
    ) // activity.frame_samples
    if len(activity.frame_is_speech) != expected_frames:
        raise ValueError(
            "SpeechActivity frame count does not match its sample count and frame size."
        )

    # Expand coarse frame decisions back to sample resolution for efficient sliding-
    # window scoring. The final padded analysis frame is clipped to the real length.
    mask = torch.zeros(activity.num_samples, dtype=torch.bool)
    for index, is_speech in enumerate(activity.frame_is_speech):
        if is_speech:
            start = index * activity.frame_samples
            end = min(start + activity.frame_samples, activity.num_samples)
            mask[start:end] = True
    return mask
