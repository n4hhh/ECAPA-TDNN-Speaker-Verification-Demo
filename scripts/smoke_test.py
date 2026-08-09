"""End-to-end execution-integrity checks for the inference core."""

from __future__ import annotations

import math
import platform
import tempfile
from pathlib import Path

import speechbrain
import soundfile
import torch
import torchaudio

from src.audio import (
    AudioLoadError,
    AudioValidationError,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE,
    load_audio,
    prepare_waveform,
)
from src.inference import (
    DEFAULT_CHECKPOINT_PATH,
    PROVISIONAL_ADP_AUG_CROSS_DOMAIN_EER_THRESHOLD,
    SpeakerVerifier,
    score_embeddings,
)
from src.model import EMBEDDING_DIM, load_model


def _assert_raises(expected_exception: type[Exception], call) -> None:
    try:
        call()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised.")


def _exercise_audio_pipeline(temp_dir: Path) -> Path:
    source_rate = 44_100
    duration_seconds = 4
    time_axis = torch.arange(source_rate * duration_seconds, dtype=torch.float32) / source_rate
    stereo = torch.stack(
        (
            0.05 * torch.sin(2 * torch.pi * 220 * time_axis),
            0.04 * torch.sin(2 * torch.pi * 330 * time_axis),
        )
    )
    stereo_path = temp_dir / "stereo_44100.wav"
    torchaudio.save(str(stereo_path), stereo, source_rate)

    prepared = load_audio(stereo_path)
    assert prepared.shape == (SEGMENT_SAMPLES,)
    assert prepared.dtype == torch.float32
    assert prepared.is_contiguous()
    assert torch.isfinite(prepared).all()

    short = torch.linspace(-0.05, 0.05, TARGET_SAMPLE_RATE, dtype=torch.float32)
    padded = prepare_waveform(short, TARGET_SAMPLE_RATE)
    assert padded.shape == (SEGMENT_SAMPLES,)
    assert torch.equal(padded[:TARGET_SAMPLE_RATE], short)
    assert torch.count_nonzero(padded[TARGET_SAMPLE_RATE:]) == 0

    _assert_raises(
        AudioValidationError,
        lambda: prepare_waveform(torch.empty(1, 0), TARGET_SAMPLE_RATE),
    )
    _assert_raises(
        AudioValidationError,
        lambda: prepare_waveform(torch.tensor([0.0, float("nan")]), TARGET_SAMPLE_RATE),
    )

    corrupted_path = temp_dir / "corrupted.wav"
    corrupted_path.write_bytes(b"not an audio file")
    _assert_raises(AudioLoadError, lambda: load_audio(corrupted_path))
    return stereo_path


def main() -> None:
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory(prefix="speaker_verification_smoke_") as temp_name:
        stereo_path = _exercise_audio_pipeline(Path(temp_name))

        model = load_model(DEFAULT_CHECKPOINT_PATH, device="cpu")
        assert not model.training

        dummy_waveform = torch.randn(1, SEGMENT_SAMPLES, dtype=torch.float32) * 0.01
        with torch.inference_mode():
            assert not torch.is_grad_enabled()
            embedding = model.extract_embedding(dummy_waveform)

        assert embedding.shape == (1, EMBEDDING_DIM)
        assert not embedding.requires_grad
        assert torch.isfinite(embedding).all()
        embedding_norm = float(torch.linalg.vector_norm(embedding, dim=1).item())
        assert math.isclose(embedding_norm, 1.0, rel_tol=0.0, abs_tol=1e-5)

        self_similarity = score_embeddings(embedding, embedding)
        assert math.isclose(self_similarity, 1.0, rel_tol=0.0, abs_tol=1e-5)

        verifier = SpeakerVerifier(DEFAULT_CHECKPOINT_PATH, device="cpu")
        file_embedding = verifier.extract_embedding(stereo_path)
        assert file_embedding.shape == (EMBEDDING_DIM,)
        assert torch.isfinite(file_embedding).all()
        result = verifier.verify(
            stereo_path,
            stereo_path,
            PROVISIONAL_ADP_AUG_CROSS_DOMAIN_EER_THRESHOLD,
        )
        assert result["same_speaker"]
        assert math.isclose(result["similarity"], 1.0, rel_tol=0.0, abs_tol=1e-5)

    print("Speaker verification smoke test: PASS")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"SpeechBrain: {speechbrain.__version__}")
    print(f"soundfile: {soundfile.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model mode: {'eval' if not model.training else 'train'}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Embedding norm: {embedding_norm:.8f}")
    print(f"Self-similarity: {self_similarity:.8f}")
    print(
        "Synthetic audio validates execution only; it is not a meaningful "
        "speaker-verification quality test."
    )


if __name__ == "__main__":
    main()
