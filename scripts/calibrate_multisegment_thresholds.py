"""Audit local audio and calibrate multi-segment demo operating thresholds.

Run from the repository root with:
    .\.venv\Scripts\python.exe scripts\calibrate_multisegment_thresholds.py

Reports are for manual review. This script never edits application thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

# Direct ``python scripts\...py`` execution places scripts/, not the repository,
# on sys.path. Resolve the imports without depending on the caller's PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.multisegment_calibration import (  # noqa: E402
    CALIBRATION_DURATION_SECONDS,
    CALIBRATION_SAMPLES,
    EXPECTED_RECORDINGS_PER_SPEAKER,
    EXPECTED_SPEAKERS,
    EXPECTED_TOTAL_RECORDINGS,
    Recording,
    Trial,
    center_crop_10_seconds,
    empirical_eer_operating_point,
    make_trials,
    score_statistics,
)
from src.audio import (  # noqa: E402
    MULTISEGMENT_HOP_SAMPLES,
    MULTISEGMENT_MAX_RECORDING_SECONDS,
    MULTISEGMENT_MIN_SPEECH_PER_WINDOW_SECONDS,
    MULTISEGMENT_MIN_TOTAL_SPEECH_SECONDS,
    MULTISEGMENT_MIN_VALID_WINDOWS,
    MULTISEGMENT_WINDOW_SAMPLES,
    TARGET_SAMPLE_RATE,
    MultiSegmentSpeechError,
    PreparedMultiSegmentAudio,
    convert_to_mono_16k,
    decode_audio,
    select_multisegment_windows,
)
from src.inference import embedding_from_prepared_multisegment, score_embeddings  # noqa: E402
from src.model import EMBEDDING_DIM, load_model  # noqa: E402

CHECKPOINTS = {
    "RAW": PROJECT_ROOT / "checkpoints" / "best_raw.pt",
    "RANDOM": PROJECT_ROOT / "checkpoints" / "best_random.pt",
    "ADAPTIVE": PROJECT_ROOT / "checkpoints" / "best_adp.pt",
}

PROVENANCE_FIELDS = (
    "speaker_id", "recording_id", "relative_path", "source_sha256",
    "source_sample_rate", "original_duration_seconds", "converted_num_samples",
    "crop_start_sample", "crop_end_sample", "crop_start_seconds", "crop_end_seconds",
    "crop_duration_seconds", "total_detected_speech_seconds",
    "candidate_window_count", "valid_window_count", "valid_segment_ranges",
    "status", "failure_reason",
)
TRIAL_FIELDS = (
    "trial_id", "label", "speaker_a", "recording_a", "path_a",
    "speaker_b", "recording_b", "path_b",
)
SCORE_FIELDS = ("trial_id", "label", "similarity")


@dataclass
class DatasetAudit:
    summary: dict[str, Any]
    provenance: list[dict[str, Any]]
    recordings: tuple[Recording, ...]
    prepared_by_path: dict[str, PreparedMultiSegmentAudio]


def _repo_relative(path: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), PROJECT_ROOT)).as_posix()
    except ValueError:
        # A test or explicit --audio-root can live on another Windows drive.
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _new_provenance_row(path: Path, speaker_id: str) -> dict[str, Any]:
    row = {field: "" for field in PROVENANCE_FIELDS}
    row.update(
        speaker_id=speaker_id,
        recording_id=path.stem,
        relative_path=_repo_relative(path),
    )
    return row


def recording_from_path(path: Path) -> Recording:
    """Use the parent directory, never the filename, as the speaker label."""

    return Recording(path.parent.name, path.stem, _repo_relative(path))


def filename_matches_parent(path: Path) -> bool:
    speaker_id = path.parent.name
    return path.name in {f"{speaker_id}{number}.wav" for number in (1, 2, 3)}


def _audit_recording(path: Path, speaker_id: str) -> tuple[
    dict[str, Any], PreparedMultiSegmentAudio | None, float | None
]:
    row = _new_provenance_row(path, speaker_id)
    failures: list[str] = []
    if not filename_matches_parent(path):
        failures.append(
            f"Filename {path.name!r} does not match parent speaker {speaker_id!r}."
        )
    if path.suffix.lower() != ".wav" or path.suffix != ".wav":
        failures.append("Only lowercase .wav files are allowed.")

    prepared = None
    duration = None
    try:
        row["source_sha256"] = _sha256(path)
    except OSError as exc:
        failures.append(f"Cannot hash source: {exc}")

    if path.suffix.lower() == ".wav":
        try:
            waveform, source_rate = decode_audio(path)
            duration = waveform.shape[-1] / source_rate
            row["source_sample_rate"] = source_rate
            row["original_duration_seconds"] = duration
            if duration < CALIBRATION_DURATION_SECONDS:
                raise ValueError("Source recording is shorter than 10.0 seconds.")
            converted = convert_to_mono_16k(waveform, source_rate)
            row["converted_num_samples"] = converted.numel()
            crop, start, end = center_crop_10_seconds(converted)
            row["crop_start_sample"] = start
            row["crop_end_sample"] = end
            row["crop_start_seconds"] = start / TARGET_SAMPLE_RATE
            row["crop_end_seconds"] = end / TARGET_SAMPLE_RATE
            row["crop_duration_seconds"] = crop.numel() / TARGET_SAMPLE_RATE
            try:
                prepared = select_multisegment_windows(crop, TARGET_SAMPLE_RATE)
            except MultiSegmentSpeechError as exc:
                metadata = exc.metadata
                failures.append(str(exc))
            else:
                metadata = prepared.metadata
            row["total_detected_speech_seconds"] = metadata.total_speech_seconds
            row["candidate_window_count"] = metadata.candidate_window_count
            row["valid_window_count"] = metadata.valid_window_count
            row["valid_segment_ranges"] = json.dumps(
                [asdict(segment) for segment in metadata.valid_segments],
                separators=(",", ":"),
            )
            if metadata.candidate_window_count != 5:
                failures.append("A 10-second crop did not produce five candidate windows.")
        except Exception as exc:
            failures.append(f"Audio/crop validation failed: {exc}")
    row["status"] = "passed" if not failures else "failed"
    row["failure_reason"] = " | ".join(failures)
    return row, prepared, duration


def audit_dataset(audio_root: Path) -> DatasetAudit:
    """Inspect all sources and crops before permitting any checkpoint inference."""

    global_failures: list[str] = []
    speaker_dirs: list[Path] = []
    if not audio_root.is_dir():
        global_failures.append(f"Dataset directory not found: {_repo_relative(audio_root)}")
    else:
        for entry in sorted(audio_root.iterdir(), key=lambda path: path.name):
            if entry.is_dir():
                speaker_dirs.append(entry)
            else:
                global_failures.append(f"Unexpected item in dataset root: {_repo_relative(entry)}")

    if len(speaker_dirs) != EXPECTED_SPEAKERS:
        global_failures.append(
            f"Expected {EXPECTED_SPEAKERS} speaker directories, found {len(speaker_dirs)}."
        )
    speaker_ids = [path.name for path in speaker_dirs]
    if len({speaker.casefold() for speaker in speaker_ids}) != len(speaker_ids):
        global_failures.append("Speaker IDs are not unique ignoring letter case.")
    for speaker in speaker_ids:
        if speaker != speaker.lower():
            global_failures.append(f"Speaker directory is not lowercase: {speaker}")

    rows: list[dict[str, Any]] = []
    prepared_by_path: dict[str, PreparedMultiSegmentAudio] = {}
    recordings: list[Recording] = []
    durations: list[float] = []
    counts: dict[str, int] = {}
    for directory in speaker_dirs:
        files: list[Path] = []
        for entry in sorted(directory.iterdir(), key=lambda path: path.name):
            if entry.is_file():
                files.append(entry)
            else:
                global_failures.append(f"Unexpected nested item: {_repo_relative(entry)}")
        counts[directory.name] = len(files)
        if len(files) != EXPECTED_RECORDINGS_PER_SPEAKER:
            global_failures.append(
                f"{directory.name}: expected {EXPECTED_RECORDINGS_PER_SPEAKER} files, found {len(files)}."
            )
        for file_path in files:
            row, prepared, duration = _audit_recording(file_path, directory.name)
            rows.append(row)
            if duration is not None:
                durations.append(duration)
            if prepared is not None:
                prepared_by_path[row["relative_path"]] = prepared
            recordings.append(recording_from_path(file_path))

    if len(rows) != EXPECTED_TOTAL_RECORDINGS:
        global_failures.append(
            f"Expected {EXPECTED_TOTAL_RECORDINGS} recordings, found {len(rows)}."
        )
    file_failures = [
        {"relative_path": row["relative_path"], "reason": row["failure_reason"]}
        for row in rows if row["status"] != "passed"
    ]
    all_naming = not any(
        "lowercase" in reason.lower()
        or "filename" in reason.lower()
        or "speaker ids" in reason.lower()
        for reason in global_failures + [row["failure_reason"] for row in rows]
    )
    all_decode = len(durations) == len(rows)
    all_multisegment = len(prepared_by_path) == len(rows)
    ready = bool(
        not global_failures
        and not file_failures
        and len(speaker_dirs) == EXPECTED_SPEAKERS
        and len(rows) == EXPECTED_TOTAL_RECORDINGS
        and all_naming and all_decode and all_multisegment
    )
    summary = {
        "expected_speakers": EXPECTED_SPEAKERS,
        "observed_speakers": len(speaker_dirs),
        "expected_recordings_per_speaker": EXPECTED_RECORDINGS_PER_SPEAKER,
        "expected_total_recordings": EXPECTED_TOTAL_RECORDINGS,
        "observed_total_recordings": len(rows),
        "speaker_ids": speaker_ids,
        "recording_counts_by_speaker": counts,
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "duration_mean": sum(durations) / len(durations) if durations else None,
        "all_naming_checks_passed": all_naming,
        "all_decode_checks_passed": all_decode,
        "all_multisegment_checks_passed": all_multisegment,
        "calibration_ready": ready,
        "global_failures": global_failures,
        "recording_failures": file_failures,
    }
    return DatasetAudit(summary, rows, tuple(recordings), prepared_by_path)


def _score_model(
    model_name: str,
    checkpoint: Path,
    audit: DatasetAudit,
    trials: tuple[Trial, ...],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    model = load_model(checkpoint, device=device)
    embeddings: dict[str, torch.Tensor] = {}
    for recording in audit.recordings:
        prepared = audit.prepared_by_path[recording.relative_path]
        result = embedding_from_prepared_multisegment(model, prepared)
        embedding = result.embedding
        if (
            embedding.shape != (EMBEDDING_DIM,)
            or embedding.dtype != torch.float32
            or not bool(torch.isfinite(embedding).all().item())
            or not math.isclose(
                float(torch.linalg.vector_norm(embedding).item()),
                1.0,
                abs_tol=1e-5,
                rel_tol=0.0,
            )
        ):
            raise RuntimeError(f"Invalid aggregated embedding: {recording.relative_path}")
        embeddings[recording.relative_path] = embedding

    score_rows = [
        {
            "trial_id": trial.trial_id,
            "label": trial.label,
            "similarity": score_embeddings(
                embeddings[trial.path_a], embeddings[trial.path_b]
            ),
        }
        for trial in trials
    ]
    if len(score_rows) != 351 or not all(math.isfinite(row["similarity"]) for row in score_rows):
        raise RuntimeError(f"Incomplete or non-finite score list for {model_name}.")
    _write_csv(output_dir / f"{model_name.lower()}_scores.csv", SCORE_FIELDS, score_rows)

    labels = [row["label"] for row in score_rows]
    scores = [row["similarity"] for row in score_rows]
    point = empirical_eer_operating_point(labels, scores)
    genuine = score_statistics([score for label, score in zip(labels, scores) if label == 1])
    impostor = score_statistics([score for label, score in zip(labels, scores) if label == 0])
    summary = {
        "model": model_name,
        "checkpoint": _repo_relative(checkpoint),
        "protocol": "multi_segment_10s_center_crop_demo_calibration",
        "num_speakers": EXPECTED_SPEAKERS,
        "num_recordings": EXPECTED_TOTAL_RECORDINGS,
        "num_genuine_trials": labels.count(1),
        "num_impostor_trials": labels.count(0),
        "num_total_trials": len(trials),
        "eer": point.eer,
        "eer_threshold": point.eer_threshold,
        "far_at_operating_point": point.far_at_operating_point,
        "frr_at_operating_point": point.frr_at_operating_point,
        "eer_method": "nearest empirical FAR/FRR gap; EER=(FAR+FRR)/2; no interpolation",
        "eer_units": "fraction",
        **{f"genuine_score_{key}": value for key, value in genuine.items()},
        **{f"impostor_score_{key}": value for key, value in impostor.items()},
    }
    _write_json(output_dir / f"{model_name.lower()}_summary.json", summary)
    return summary


def run_calibration(
    audio_root: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    audit_only: bool = False,
) -> int:
    """Write an audit first; calibrate only if every required check passes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(audio_root)
    _write_csv(output_dir / "crop_provenance.csv", PROVENANCE_FIELDS, audit.provenance)
    _write_json(output_dir / "dataset_audit.json", audit.summary)
    print(
        f"Dataset audit: {audit.summary['observed_speakers']} speakers, "
        f"{audit.summary['observed_total_recordings']} recordings; "
        f"ready={audit.summary['calibration_ready']}"
    )
    if not audit.summary["calibration_ready"]:
        for failure in audit.summary["global_failures"]:
            print(f"AUDIT FAILURE: {failure}", file=sys.stderr)
        for failure in audit.summary["recording_failures"]:
            print(
                f"AUDIT FAILURE: {failure['relative_path']}: {failure['reason']}",
                file=sys.stderr,
            )
        return 2
    if audit_only:
        return 0

    trials = make_trials(audit.recordings)
    counts = Counter(trial.label for trial in trials)
    if counts[1] != 27 or counts[0] != 324 or len(trials) != 351:
        raise RuntimeError(f"Unexpected trial counts: {dict(counts)}, total={len(trials)}")
    _write_csv(output_dir / "trials.csv", TRIAL_FIELDS, [asdict(trial) for trial in trials])

    summaries: dict[str, dict[str, Any]] = {}
    for model_name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {_repo_relative(checkpoint)}")
        print(f"Calibrating {model_name}: {checkpoint.name}", flush=True)
        summaries[model_name] = _score_model(
            model_name, checkpoint, audit, trials, device, output_dir
        )
    _write_json(
        output_dir / "calibration_summary.json",
        {
            "purpose": "operational calibration of the multi-segment demo",
            "interpretation": "not a thesis model comparison or final-test evaluation",
            "protocol": {
                "source_crop_seconds": CALIBRATION_DURATION_SECONDS,
                "source_crop_samples": CALIBRATION_SAMPLES,
                "crop_method": "deterministic integer center crop after mono 16 kHz conversion",
                "sample_rate": TARGET_SAMPLE_RATE,
                "window_samples": MULTISEGMENT_WINDOW_SAMPLES,
                "hop_samples": MULTISEGMENT_HOP_SAMPLES,
                "minimum_speech_per_window_seconds": MULTISEGMENT_MIN_SPEECH_PER_WINDOW_SECONDS,
                "minimum_total_speech_seconds": MULTISEGMENT_MIN_TOTAL_SPEECH_SECONDS,
                "minimum_valid_windows": MULTISEGMENT_MIN_VALID_WINDOWS,
                "deployment_maximum_recording_seconds": MULTISEGMENT_MAX_RECORDING_SECONDS,
                "tail_policy": "discard incomplete tail",
                "aggregation": "arithmetic mean then L2 normalization",
                "decision_semantics": "similarity >= threshold",
                "eer_method": "nearest empirical FAR/FRR gap; EER=(FAR+FRR)/2; no interpolation",
            },
            "models": summaries,
        },
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("audio_val"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/multisegment_calibration")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_calibration(
            args.audio_root,
            args.output_dir,
            device=args.device,
            audit_only=args.audit_only,
        )
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
