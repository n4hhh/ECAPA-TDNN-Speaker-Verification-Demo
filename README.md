# Vietnamese Speaker Verification Demo

A local Streamlit demonstration of 1:1, text-independent speaker verification
using an ECAPA-TDNN speaker embedding model. Enroll one utterance, then compare
multiple verification utterances against it using cosine similarity.

Enrollment and verification utterances **do not need to contain the same
sentence**. Enrollment exists only in the current Streamlit session; the project
does not create accounts, databases, or persistent speaker profiles.

## Demo Workflow

1. Select a compatible model checkpoint.
2. Record or upload an **Enrollment Utterance**.
3. Click **SAVE ENROLLMENT**.
4. Record or upload a **Verification Utterance**.
5. Click **VERIFY SPEAKER**.
6. Read the cosine similarity and **SAME SPEAKER** or **DIFFERENT SPEAKER** result.
7. Repeat verification against the same enrollment if desired.
8. Click **CLEAR ENROLLMENT** to start again.

Supported audio inputs:

- browser microphone recording;
- WAV upload;
- MP3 upload.

## Project Structure

```text
ECAPA-TDNN-Speaker-Verification-Demo/
|-- app.py
|-- checkpoints/
|   `-- ecapa_tdnn_finetuned_3.pt       # supplied separately
|-- src/
|   |-- __init__.py
|   |-- audio.py                        # decoding and audio preparation
|   |-- inference.py                    # embedding extraction and verification
|   |-- model.py                        # strict ECAPA/checkpoint reconstruction
|   `-- vad.py                          # local WebRTC speech activity detection
|-- scripts/
|   |-- __init__.py
|   |-- inspect_checkpoint.py
|   |-- smoke_test.py
|   `-- verify_pair.py
|-- tests/
|   |-- __init__.py
|   |-- test_app.py
|   |-- test_audio_realtime.py
|   |-- test_streamlit_app.py
|   `-- test_verify_pair.py
|-- reference/
|   |-- adp_aug_ecapa_tdnn.ipynb
|   `-- Model_Testing.ipynb
|-- requirements.txt
`-- README.md
```

## Setup

### Requirements

- Windows, Linux, or macOS capable of running the listed Python packages
- Python 3.10 recommended
- A compatible ECAPA-TDNN checkpoint supplied separately
- A browser with microphone permission for direct recording

The project was validated on Windows with Python 3.10.11, PyTorch 2.2.0 CPU,
torchaudio 2.2.0 CPU, SpeechBrain 1.0.3, soundfile 0.12.1, WebRTC VAD wheels
2.0.14, and Streamlit 1.60.0. CUDA is optional; CPU execution is supported.

If you already have a working CUDA-enabled PyTorch installation, avoid
unnecessarily replacing it with a CPU-only build. Confirm the appropriate
PyTorch/torchaudio installation for your CUDA environment before installing the
remaining requirements.

### Clone Repository

```powershell
git clone <REPOSITORY_URL>
cd ECAPA-TDNN-Speaker-Verification-Demo
```

### Create Virtual Environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model Checkpoint

Model checkpoints are intentionally excluded from Git because of their size.
After cloning, place at least one compatible checkpoint in `checkpoints/` before
starting the application.

For the current demo:

```text
checkpoints/
`-- ecapa_tdnn_finetuned_3.pt
```

The application automatically discovers `.pt` and `.pth` files under
`checkpoints/`, validates their basic checkpoint structure, and presents
compatible files in the model selector. Multiple compatible checkpoints can be
available. The selected checkpoint is then loaded with strict state-dictionary
matching; incompatible files produce an error rather than a partial load.

Enrollment embeddings are model-specific. Changing the selected model clears the
current enrollment and requires enrollment again. The project cannot perform
inference without a compatible checkpoint, and this README does not provide a
checkpoint download URL.

## Run the Demo

From the repository root:

```powershell
streamlit run app.py
```

Streamlit opens the application locally in a browser. Grant browser microphone
permission when prompted to use direct recording.

## Using the Demo

### Enrollment

1. Select **Record microphone** or **Upload audio**.
2. Speak naturally for several seconds, or select a WAV/MP3 file.
3. Click **SAVE ENROLLMENT**.
4. Confirm that the UI displays **Enrollment saved** and the preprocessing
   metadata.

The normalized 192-dimensional enrollment embedding is stored only in the
current Streamlit session and is not displayed or written to a permanent speaker
profile.

### Verification

1. Record or upload another utterance. It may contain a different sentence.
2. Click **VERIFY SPEAKER**.
3. Read the cosine similarity, decision threshold, and same/different-speaker
   result.

Additional verification utterances can be evaluated against the same saved
enrollment without recomputing it. Use **CLEAR ENROLLMENT** to remove enrollment
and verification state.

## Audio and Model Pipeline

Microphone recordings, WAV uploads, and MP3 uploads use the same local pipeline:

```text
Input audio
-> mono
-> resample to 16 kHz
-> WebRTC VAD
-> speech-rich contiguous 3-second window selection
-> 48,000 waveform samples
-> SpeechBrain Fbank
-> sentence mean normalization
-> ECAPA-TDNN
-> 192-dimensional embedding
-> L2 normalization
-> cosine similarity
```

The selected model input is always one continuous section of the recording.
Separated speech fragments are never concatenated into an artificial waveform.

## Speech Quality Requirements

Realtime preprocessing requires:

- at least `2.0` seconds of detected speech in the complete recording;
- at least `1.5` seconds of detected speech inside the selected three-second
  window.

These values are input-quality requirements. They determine whether the recording
contains enough useful speech for inference; they are **not** speaker-verification
decision thresholds.

## Speaker Verification Decision

The current provisional operating threshold is `0.1007`:

```text
similarity >= 0.1007  -> SAME SPEAKER
similarity <  0.1007  -> DIFFERENT SPEAKER
```

This value is a provisional, test-derived operating point. Cosine similarity is
not a calibrated probability, confidence score, or percentage probability.

## MP3 Support

MP3 decoding is local and offline. The validated Windows environment uses
torchaudio's SoundFile backend with soundfile 0.12.1 and libsndfile 1.2.0. This
build advertises and successfully decodes MPEG Layer III audio; FFmpeg is not
required in the validated environment.

Check MP3 capability with:

```powershell
python -c "import soundfile as sf; print(sf.available_formats().get('MP3'))"
```

Expected result:

```text
MPEG-1/2 Audio
```

If another system's libsndfile build lacks MP3 support, or an MP3 file cannot be
decoded, the UI asks the user to try another MP3 or upload WAV audio. Browser
microphone recording and WAV upload remain usable when MP3 decoding is
unavailable.

## Command-Line Verification

Verify a local enrollment/verification pair with realtime preprocessing:

```powershell
python -m scripts.verify_pair enrollment.wav verification.wav
```

Override the model decision threshold when required:

```powershell
python -m scripts.verify_pair enrollment.wav verification.wav --threshold 0.12
```

## Tests and Diagnostics

Run all automated tests:

```powershell
python -m unittest discover -s tests -v
```

This covers realtime preprocessing, WAV/MP3 handling, Streamlit state behavior,
model cache reuse, model-change invalidation, CLI behavior, and initial app
rendering.

Run the embedding smoke test:

```powershell
python -m scripts.smoke_test
```

This validates strict model loading, inference mode, embedding shape and finite
values, L2 normalization, and self-similarity. Synthetic audio verifies execution
integrity, not speaker-verification quality.

Inspect checkpoint metadata and strict compatibility:

```powershell
python -m scripts.inspect_checkpoint
```

Real microphone permission, recording quality, room noise, and real
same/different-speaker behavior require manual browser testing.

## Notes

This project performs **speaker verification**, not speaker identification.

Speaker verification asks:

> Given two utterances, were they spoken by the same person?

It does not identify an unknown speaker from a database.
