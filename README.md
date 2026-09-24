# Vietnamese Speaker Verification Demo

## Project Overview

This local Streamlit demo performs Vietnamese, text-independent, one-to-one speaker verification with ECAPA-TDNN. It compares a new recording with a voice profile enrolled in the current browser session. Three thesis checkpoints represent different training conditions: `RAW`, `RANDOM`, and `ADAPTIVE`. The small demo calibration set does not establish a ranking among them.

For implementation details, troubleshooting, and thesis defense notes, see the [Demo Guide](docs/DEMO_GUIDE.md).

## What the Demo Does

```text
Enrollment -> Verification -> Result
```

Choose a model, record or upload enrollment speech, then record or upload verification speech. The result shows cosine similarity, that model's multi-segment operating threshold, and **SAME SPEAKER** or **DIFFERENT SPEAKER**. The two recordings may contain different sentences. Cosine similarity is a score, not a probability or confidence percentage. Enrollment exists only in the active Streamlit session.

## Quick Start

The pinned dependencies in `requirements.txt` and the existing setup target Python 3.10. From the repository root in Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Place the supplied checkpoints at these exact paths before launching:

| Model | Checkpoint |
| --- | --- |
| RAW | `checkpoints/best_raw.pt` |
| RANDOM | `checkpoints/best_random.pt` |
| ADAPTIVE | `checkpoints/best_adp.pt` |

The app selects `ADAPTIVE` by default when its checkpoint is available. It accepts microphone input, WAV files, and MP3 files when the local decoder supports MP3. Checkpoints are supplied separately; `.gitignore` excludes them.

## Recommended Demo Usage

Use approximately **8-10 seconds of natural speech** for each recording; approximately **10 seconds** most closely matches the calibration crop duration. The sentences may differ. The main target is natural speech rather than singing. Each interactive recording must be **20.0 seconds or shorter**. Speak with limited silence so enough windows pass speech detection.

## Current Multi-Segment Inference Pipeline

```text
recording -> mono 16 kHz -> WebRTC VAD
          -> overlapping 3.0 s windows at a 1.5 s hop
          -> retain windows with enough detected speech
          -> one batched ECAPA-TDNN inference call
          -> one normalized 192-D embedding per valid window
          -> arithmetic mean -> L2 normalization
          -> one aggregated recording embedding
          -> cosine similarity with the enrolled embedding
          -> selected model's multi-segment threshold
          -> SAME SPEAKER / DIFFERENT SPEAKER
```

Enrollment and verification both use this path. The retained windows are passed to ECAPA together as an `[N, 48000]` batch.

## Why 10 Seconds if ECAPA Uses 3 Seconds?

ECAPA still receives **exactly 3.0 seconds / 48,000 samples per input**. A 10-second recording supplies several overlapping observations; it is never passed to ECAPA as one 10-second tensor. For example, its complete candidate windows are:

```text
0.0-3.0 s   1.5-4.5 s   3.0-6.0 s   4.5-7.5 s   6.0-9.0 s
    -> one embedding per speech-valid window -> mean -> L2 normalize
    -> one recording-level embedding
```

The incomplete 9.0-10.0 second tail is discarded. WebRTC VAD may reject some candidates before inference.

## Multi-Segment Audio Rules

| Rule | Current value |
| --- | ---: |
| Model audio | Mono, 16 kHz |
| Window | 3.0 s / 48,000 samples |
| Hop | 1.5 s / 24,000 samples |
| Detected speech per retained window | At least 1.5 s |
| Total detected speech | At least 3.0 s |
| Retained windows | At least 2 |
| Interactive recording duration | At most 20.0 s |

Only complete windows are used. VAD decides which candidates contain enough speech; separated speech regions are **never concatenated** into artificial model inputs.

## Thresholds

The two protocols have separate threshold mappings in `src/model_thresholds.py`. **Never interchange them.**

| Model | Historical single-window | Current multi-segment demo |
| --- | ---: | ---: |
| RAW | `0.16837078332901` | `0.20708201825618744` |
| RANDOM | `0.16918502748012543` | `0.2140672206878662` |
| ADAPTIVE | `0.172615185379982` | `0.21084168553352356` |

The Streamlit demo uses only the multi-segment column. The decision rule is inclusive: `similarity >= threshold` means **SAME SPEAKER**; `similarity < threshold` means **DIFFERENT SPEAKER**. A model without a configured multi-segment threshold shows its similarity with the decision unavailable.

## Why the Threshold Changed

The historical path scores **one selected 3-second window -> one embedding -> cosine similarity**. The current demo scores **multiple 3-second windows -> mean and L2 normalization -> one aggregated embedding -> cosine similarity**. Aggregation changes the score distribution, so the historical operating threshold cannot be assumed valid for the demo. A separate operational calibration supplied the multi-segment thresholds; a numerically higher threshold is not inherently better.

## Multi-Segment Calibration

The [calibration summary](reports/multisegment_calibration/calibration_summary.json) records 9 speakers, 3 independent recordings per speaker, and 27 recordings total. The [dataset audit](reports/multisegment_calibration/dataset_audit.json) reports raw source durations from about **20.1 to 34.9 seconds**. Each source supplied exactly one deterministic 10-second center crop after conversion to mono 16 kHz. Raw sources may exceed the interactive 20-second limit because cropping happens first.

The same frozen list of **27 genuine** and **324 impostor** unique unordered pairs (**351 total**) was scored by all three checkpoints with cosine similarity. The threshold is the nearest empirical FAR/FRR operating point; EER is `(FAR + FRR) / 2` there, without interpolation. No final-test data was used. **This is operational calibration for the interactive demo, not the thesis final-test evaluation or evidence for model ranking.**

The report already exists. These commands audit inputs or rerun the full calibration from the repository root:

```powershell
python scripts/calibrate_multisegment_thresholds.py --audit-only
python scripts/calibrate_multisegment_thresholds.py
```

Both commands write report artifacts; normal demo use does not require either. The second command also scores all checkpoints and overwrites generated calibration reports.

## Historical vs Demo Protocol

| Item | Historical single-window | Current multi-segment demo |
| --- | --- | --- |
| Input evidence | One 3-second window | Multiple qualifying 3-second windows |
| Aggregation | None | Arithmetic mean, then L2 normalization |
| Threshold mapping | Single-window | Multi-segment |
| Purpose | Preserve historical inference behavior | Interactive Streamlit enrollment and verification |
| Thresholds interchangeable? | No | No |

## Testing

From the repository root with the environment active:

```powershell
python -m compileall -q app.py src ui scripts tests
python -m unittest discover -s tests -v
```

For a manual smoke check, try same-speaker and different-speaker pairs, then a same-speaker pair with different sentences. Check **Verify again** (retains enrollment), **Clear enrollment**, and switching models (clears enrollment). Try speech that is too short, a recording with too much silence, a recording at the 20-second limit, and one above it. These checks exercise behavior; a live score is not guaranteed to classify every pair correctly.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit flow, checkpoint choice, session state, decisions |
| `src/audio.py` | Decoding, conversion, windows, speech requirements |
| `src/vad.py` | WebRTC VAD frame decisions |
| `src/model.py` | Checkpoint validation, FBank and ECAPA inference |
| `src/inference.py` | Batched extraction, aggregation, cosine scoring |
| `src/model_thresholds.py` | Separate protocol-specific threshold mappings |
| `ui/components.py` | Result cards, metadata and score visualization |
| `ui/styles.css` | Streamlit visual styling |
| `scripts/calibrate_multisegment_thresholds.py` | Dataset audit and calibration runner |
| `scripts/multisegment_calibration.py` | Trial construction and empirical operating point |
| `reports/multisegment_calibration/` | Calibration summaries, scores and provenance |
| `tests/` | Automated behavior and calibration checks |

## Important Scientific Boundaries

This is speaker verification, not identification or proof of legal identity. It has no anti-spoofing, replay-attack detection, ASR challenge phrase verification, or liveness detection. Cosine similarity is not a probability. Singing is outside the main natural-speech target, and the small operational calibration set cannot support comparative thesis claims about RAW, RANDOM, and ADAPTIVE.

See the [Demo Guide](docs/DEMO_GUIDE.md) for the detailed handoff and defense Q&A.
