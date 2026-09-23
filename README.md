# Vietnamese Speaker Verification Demo

A local Streamlit demonstration of 1:1, text-independent speaker verification
using three thesis ECAPA-TDNN checkpoints. Enroll one utterance, then compare a
verification utterance against it using cosine similarity between normalized
192-dimensional speaker embeddings.

Enrollment and verification utterances do not need to contain the same sentence.
Enrollment exists only in the current Streamlit session; the project does not
create accounts, databases, or persistent speaker profiles.

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

## Inference Pipeline

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

Exact 48,000-sample mono/16 kHz input is used directly. Longer realtime
recordings use WebRTC VAD only to choose one speech-rich contiguous 3-second
window. Separated speech fragments are never concatenated. Too-short or
insufficient-speech recordings are rejected with a clear validation error.

Cosine similarity is not a probability or confidence percentage.

## Thresholds And Decisions

The demo does not invent operational thresholds. The checkpoints include best
validation EER metadata, but a deployable SAME/DIFFERENT threshold must be
calibrated from validation data for the intended operating condition.

Until calibrated thresholds are added in `src/model_thresholds.py`, the app
shows:

```text
Operational Threshold: Not configured
Decision: Not available
```

When real thresholds are configured, the UI will produce `SAME SPEAKER` or
`DIFFERENT SPEAKER` from `similarity >= threshold`.

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
the local audio stack can decode MP3.

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
app.py              Streamlit UI and session state
src/audio.py        decoding, model preprocessing, realtime segment selection
src/model.py        checkpoint validation and strict ECAPA construction
src/inference.py    embedding extraction and cosine scoring
src/vad.py          WebRTC speech activity detection
src/model_thresholds.py  nullable calibrated-threshold mapping
scripts/           diagnostics and CLI verification
tests/             unit tests that avoid requiring huge real checkpoints
```

This project performs speaker verification, not speaker identification.
