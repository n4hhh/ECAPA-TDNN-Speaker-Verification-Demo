# Vietnamese Speaker Verification Demo

A local Streamlit demonstration of 1:1, text-independent speaker verification
using three thesis ECAPA-TDNN checkpoints. Enroll one utterance, then compare a
verification utterance against it using cosine similarity between normalized
192-dimensional speaker embeddings.

Enrollment and verification utterances do not need to contain the same sentence.
Enrollment exists only in the current Streamlit session; the project does not
create accounts, databases, or persistent speaker profiles.

## Demo Flow

The presentation UI follows three explicit stages:

```text
Enrollment -> Verification -> Result
```

1. Select `RAW`, `RANDOM`, or `ADAPTIVE` in the sidebar. `ADAPTIVE` remains the
   default when its checkpoint is available.
2. Create a session-only voice profile from the microphone or a WAV/MP3 upload.
3. Record or upload a second, text-independent verification sample.
4. Review the aggregated cosine similarity, model-specific multi-segment
   operating threshold, SAME/DIFFERENT SPEAKER decision, and preprocessing timings.

Changing the model clears the model-specific enrollment. **Verify again** clears
only the latest verification result and input, while **Clear enrollment** removes
the current voice profile and result. Research configuration and backend details
are kept in the sidebar so the main page remains focused on the live demo.

## Thesis Checkpoints

Place the supplied checkpoints in `checkpoints/`:

```text
checkpoints/
|-- best_raw.pt
|-- best_random.pt
`-- best_adp.pt
```

The app presents these as:

- `RAW` -> `best_raw.pt`
- `RANDOM` -> `best_random.pt`
- `ADAPTIVE` -> `best_adp.pt`

All three share the same ECAPA architecture and differ by training condition and
learned weights. `ADAPTIVE` is selected by default when available. Enrollment
embeddings are model-specific, so changing the selected model clears the saved
enrollment and previous verification result.

## Checkpoint Contract

The supported checkpoint schema is:

```text
frozen_handoff_ecapa_aam_training_v1
```

Required inference fields are `runtime_config`, `embedding_model_state_dict`,
`mean_var_norm_state_dict`, and `cursor`. The loader validates
`runtime_config.model.source == "speechbrain/spkrec-ecapa-voxceleb"` and
`embedding_dim == 192`, then strictly loads the SpeechBrain ECAPA embedding
model and mean-var-normalization state.

The checkpoints may also contain `aam_state_dict`, optimizer, scheduler, scaler,
and RNG state. AAM-Softmax is training-only; verification does not construct or
use the AAM head.

## Inference Protocols

Both protocols share the frozen model frontend:

```text
audio
-> mono
-> 16 kHz
-> exact 3.0 s / 48,000 samples
-> SpeechBrain FBank, 80 mel bins
-> mean_var_norm
-> fine-tuned ECAPA embedding_model
-> float32 192-D embedding
-> L2 normalization
-> cosine similarity
```

### Historical / Single-Window Realtime

The historical public API remains available unchanged:

```text
long recording
-> mono 16 kHz
-> WebRTC VAD
-> best speech-rich contiguous 3-second window
-> ECAPA embedding
-> single-window calibrated threshold
```

Exact 48,000-sample mono/16 kHz historical inputs still bypass VAD and are used
directly. `extract_embedding_realtime()` retains this frozen behavior.

### Current Streamlit Multi-Segment Demo

Both enrollment and verification use:

```text
long recording
-> mono 16 kHz
-> WebRTC VAD
-> overlapping contiguous 3-second windows, 1.5-second hop
-> retain every window with >=1.5 seconds detected speech
-> one batched ECAPA inference call
-> arithmetic mean of segment embeddings
-> L2 normalization
-> one aggregated 192-D embedding
-> cosine similarity
```

The hard input rules are:

- no more than 20.0 seconds recording duration;
- at least 3.0 seconds total detected speech;
- at least two valid windows;
- only complete 48,000-sample windows are used;
- an incomplete tail shorter than three seconds is discarded;
- separated speech regions are never concatenated or padded together.

The UI recommends approximately 8–10 seconds of natural speech to make it easy
to satisfy these rules. Candidate windows are supplied to ECAPA as a single
`[N, 48000]` batch. Each returned 192-D embedding is already normalized; their
arithmetic mean is L2-normalized again to create the session embedding.

Cosine similarity is not a probability or confidence percentage.

## Thresholds And Decisions

### Historical single-window operating points

`src/model_thresholds.py` retains these validation-calibrated empirical EER
operating points for the historical single-window protocol:

| Model | Decision threshold |
| --- | ---: |
| RAW | `0.16837078332901` |
| RANDOM | `0.16918502748012543` |
| ADAPTIVE | `0.172615185379982` |

Historical callers apply the selected model's threshold using
`similarity >= threshold`. These values do not apply to aggregated recordings.

### Multi-segment operational demo thresholds

| Model | Decision threshold |
| --- | ---: |
| RAW | `0.20708201825618744` |
| RANDOM | `0.2140672206878662` |
| ADAPTIVE | `0.21084168553352356` |

The Streamlit demo compares aggregated enrollment and verification embeddings
using the selected model's multi-segment threshold. Equality means **SAME
SPEAKER**; a lower similarity means **DIFFERENT SPEAKER**. An unconfigured model
still shows its similarity with the decision unavailable. Cosine similarity is
not a probability.

These operating points were calibrated on a separate set of 9 speakers with
3 independent recordings each (27 recordings). Each source supplied one
deterministic 10-second center crop after mono 16 kHz conversion. The frozen
trial list contains 27 genuine and 324 impostor pairs, 351 total. The operating
point is the nearest empirical FAR/FRR gap, with EER `(FAR + FRR) / 2` and no
interpolation. No final-test data was used. The three reported calibration EERs
were each `0.14814814814814814`; this small demo set and its 27 genuine trials
do not establish equal or comparative thesis model performance.

## Multi-Segment Demo Calibration

The local `audio_val/` set is intended only for operational calibration of the
multi-segment demo. It is not training or final-test data, and its small size
does not establish comparative performance among RAW, RANDOM, and ADAPTIVE.
The script requires nine lowercase speaker folders with three correctly named
WAV recordings each. The parent folder supplies the speaker label.

```text
raw recording
-> deterministic 10.0-second center crop after mono 16 kHz conversion
-> production multi-segment window selection and batched ECAPA inference
-> one aggregated embedding per original recording
-> one frozen list of all genuine and impostor pairs
-> cosine scores
-> nearest empirical FAR/FRR operating point
```

Raw sources may exceed the demo's 20-second input limit because cropping occurs
first; the 20-second deployment limit is unchanged. Each source contributes
exactly one crop, so 27 files remain 27 independent recording samples. The
script writes dataset and crop provenance before inference and exits nonzero if
any file fails naming, decoding, duration, or multi-segment speech validation.
It never calibrates on a reduced subset.

Run the audit or full pipeline from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_multisegment_thresholds.py --audit-only
.\.venv\Scripts\python.exe scripts\calibrate_multisegment_thresholds.py
```

Outputs go to `reports/multisegment_calibration/`. The empirical EER method
examines all distinct observed scores as thresholds, uses acceptance when
`similarity >= threshold`, and selects the first (highest) threshold minimizing
`abs(FAR - FRR)`. It reports `(FAR + FRR) / 2` at that attainable operating
point, matching the thesis notebook's nearest empirical point convention; no
interpolation or hand selection is used. The reviewed report values are
explicit constants in `src/model_thresholds.py`; the app does not read report
files at runtime. The reports remain audit provenance.

## Setup

Python 3.10 is recommended. The project is pinned for:

- torch 2.2.0
- torchaudio 2.2.0
- numpy 1.26.4
- speechbrain 1.0.3
- soundfile 0.12.1
- webrtcvad-wheels 2.0.14
- streamlit 1.60.0

Install:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Demo

```powershell
streamlit run app.py
```

The app supports browser microphone recording, WAV upload, and MP3 upload when
the local audio stack can decode MP3. Record approximately 8–10 seconds of
natural speech for both enrollment and verification.

## Diagnostics

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Inspect checkpoint compatibility:

```powershell
python -m scripts.inspect_checkpoint checkpoints/best_raw.pt
python -m scripts.inspect_checkpoint checkpoints/best_random.pt
python -m scripts.inspect_checkpoint checkpoints/best_adp.pt
```

Run the inference smoke test:

```powershell
python -m scripts.smoke_test
```

Verify a local audio pair without producing a decision:

```powershell
python -m scripts.verify_pair enrollment.wav verification.wav
```

For development only, a manual threshold can be supplied:

```powershell
python -m scripts.verify_pair enrollment.wav verification.wav --threshold 0.5
```

## Project Layout

```text
app.py              Streamlit workflow, checkpoint selection, and session state
ui/components.py    reusable presentation-only UI components
ui/styles.css       local responsive visual design
src/audio.py        decoding and distinct single-/multi-segment preprocessing
src/model.py        checkpoint validation and strict ECAPA construction
src/inference.py    batched extraction, mean aggregation, and cosine scoring
src/vad.py          WebRTC speech activity detection
src/model_thresholds.py  separate historical and multi-segment threshold mappings
scripts/           diagnostics and CLI verification
tests/             unit tests that avoid requiring huge real checkpoints
```

This project performs speaker verification, not speaker identification.
