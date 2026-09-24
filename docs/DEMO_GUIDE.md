# Demo Guide: Vietnamese Speaker Verification

This guide explains the current interactive demo and its operational calibration. For installation and a quick run, start with the [README](../README.md). The code and reviewed reports remain the authority if a later implementation changes.

## 1. Mental Model

The system asks: **Does this verification recording appear to come from the same speaker as the enrolled recording?** It does not answer **Who is this person?** The first question is one-to-one *verification*; the second would be *identification* across a population.

This is **text-independent** verification. The two recordings can contain different words. The result is a similarity score compared with a model-specific operating threshold, not proof of identity.

## 2. End-to-End Flow

```text
Enrollment audio
  -> decode and convert -> VAD and valid windows -> ECAPA embeddings
  -> mean and L2 normalization -> aggregated embedding in Streamlit session

Verification audio
  -> the same preparation and aggregation -> new aggregated embedding
  -> cosine similarity with enrollment -> selected model's multi-segment threshold
  -> SAME SPEAKER / DIFFERENT SPEAKER
```

The same checkpoint must produce both embeddings. Switching models invalidates the enrolled profile, so the user must enroll again. The Streamlit decision is assembled in `app.py`; the lower-level `SpeakerVerifier.verify_multisegment()` in `src/inference.py` deliberately returns a score without an operating threshold. This distinction helps keep the historical and demo APIs separate.

## 3. Detailed Audio Pipeline

The app accepts microphone audio or one WAV/MP3 upload. MP3 requires a local decoder that supports it. `src/audio.py` decodes the file and keeps its source sample rate long enough to measure the original duration. It resamples each channel to **16 kHz** when needed, then averages channels to **mono**. An interactive input over **20.0 seconds** is rejected before model preparation.

WebRTC VAD (`src/vad.py`) runs at mode **2** on **20 ms frames**. It analyzes a clipped 16-bit PCM copy; it does not replace the floating-point waveform sent to the model. A speech mask counts VAD-positive samples globally and inside each candidate window.

Candidates are complete, contiguous **3.0-second** slices beginning every **1.5 seconds**. A window survives when it contains at least **1.5 seconds** of detected speech. The whole recording must contain at least **3.0 seconds** of detected speech and produce at least **two valid windows**. Only the original contiguous waveform slices are passed to ECAPA. Concatenating separate VAD speech fragments would invent timing and waveform boundaries that were never recorded, so this pipeline does not do it.

## 4. Windowing Examples

The table shows *candidate* windows before VAD filtering. Times are in seconds.

| Recording duration | Candidate starts | Complete candidate ranges | Incomplete tail |
| --- | --- | --- | --- |
| 8 s | 0, 1.5, 3, 4.5 | 0-3, 1.5-4.5, 3-6, 4.5-7.5 | 7.5-8 discarded |
| 9 s | 0, 1.5, 3, 4.5, 6 | 0-3, 1.5-4.5, 3-6, 4.5-7.5, 6-9 | None |
| 10 s | 0, 1.5, 3, 4.5, 6 | 0-3, 1.5-4.5, 3-6, 4.5-7.5, 6-9 | 9-10 discarded |

VAD can reject any candidate with too little detected speech. Therefore an 8- or 10-second recording does not guarantee four or five model inputs. At least two must remain.

## 5. ECAPA Inference

The model contract stays at **48,000 samples per window**. Valid windows form one `[N, 48000]` batch for one ECAPA inference call. The frozen frontend applies an **80-bin SpeechBrain FBank**, mean/variance normalization, and the fine-tuned ECAPA-TDNN. Each output is a normalized **192-dimensional** embedding.

The recording representation is the **arithmetic mean** of its retained segment embeddings, followed by another **L2 normalization**. This produces one normalized 192-D vector for enrollment and one for verification. A 10-second waveform is never fed to ECAPA as a single 10-second tensor. See `src/model.py` for the checkpoint and model contract, and `src/inference.py` for aggregation.

## 6. Cosine Similarity

The app scores the two normalized recording vectors with their dot product, which is cosine similarity. Its mathematical range is **-1 to 1**. Higher scores generally mean the embeddings are more similar, but a score alone is not a calibrated probability. For example, a score of `0.80` does **not** mean an 80% chance that the speakers are the same.

The result visualization places the score and operating threshold on the same -1 to 1 scale.

## 7. Threshold Decision

```text
similarity >= selected model's multi-segment threshold -> SAME SPEAKER
similarity <  selected model's multi-segment threshold -> DIFFERENT SPEAKER
```

Equality belongs to **SAME SPEAKER**. If a selected checkpoint has no configured multi-segment threshold, the similarity remains visible and the decision is unavailable. The app does not borrow another checkpoint's threshold or fall back to a historical one.

| Model | Historical single-window | Current multi-segment demo |
| --- | ---: | ---: |
| RAW | `0.16837078332901` | `0.20708201825618744` |
| RANDOM | `0.16918502748012543` | `0.2140672206878662` |
| ADAPTIVE | `0.172615185379982` | `0.21084168553352356` |

Each checkpoint has its own score behavior and operating point. The historical path uses one selected 3-second embedding; the demo averages several embeddings and normalizes again. That changes the score distribution. The historical threshold therefore cannot be transferred by assumption, and the demo thresholds were calibrated separately. A higher numerical threshold does not imply a better model.

## 8. Calibration From First Principles

A **genuine trial** compares two recordings of the same speaker. An **impostor trial** compares recordings from different speakers. At a proposed threshold, scores at or above it are accepted as same speaker.

- **FAR** = impostor trials incorrectly accepted / all impostor trials.
- **FRR** = genuine trials incorrectly rejected / all genuine trials.

The calibration code checks a finite threshold above the highest observed score and each distinct observed score, from high to low. It chooses the empirical point with the smallest `abs(FAR - FRR)` gap; a tie takes the highest threshold. It reports `EER = (FAR + FRR) / 2` at that attainable point. There is **no interpolation**. This is an operating-point selection method for the interactive demo, not an estimate that proves real-world accuracy.

## 9. Exact Calibration Dataset

The reviewed [calibration summary](../reports/multisegment_calibration/calibration_summary.json) and [dataset audit](../reports/multisegment_calibration/dataset_audit.json) record:

| Item | Recorded value |
| --- | ---: |
| Speakers | 9 |
| Independent recordings per speaker | 3 |
| Recordings | 27 |
| Genuine unordered trials | 27 |
| Impostor unordered trials | 324 |
| Total unique unordered trials | 351 |

The audited raw source durations range from about **20.1 to 34.9 seconds**. Each source contributed exactly **one deterministic 10-second center crop**, after mono 16 kHz conversion. Calibration applies the same multi-segment preparation to that crop, then scores every pair with cosine similarity. Raw sources may exceed the interactive 20-second limit because they are cropped *before* the deployment preparation. All three checkpoints used the same frozen [trial list](../reports/multisegment_calibration/trials.csv); no final-test data was used.

The three report EERs happen to be `0.14814814814814814` on this small set. With only 27 genuine trials, this is a coarse operational measurement. It does not show that the three models perform equally, and it is not the thesis comparative or final-test evaluation. The reports are provenance; the running app uses explicit reviewed constants and does not load reports at startup.

The calibration command is `python scripts/calibrate_multisegment_thresholds.py` from the repository root. It writes report files, so a teammate running the demo should use the existing reports instead of rerunning it. `--audit-only` audits inputs but still writes audit provenance.

## 10. What Each Model Means

`RAW`, `RANDOM`, and `ADAPTIVE` are the repository's labels for three thesis **training conditions** using the same ECAPA architecture and different checkpoint weights:

| Label | Current checkpoint | Safe explanation |
| --- | --- | --- |
| RAW | `checkpoints/best_raw.pt` | RAW-labeled thesis condition |
| RANDOM | `checkpoints/best_random.pt` | RANDOM-labeled thesis condition |
| ADAPTIVE | `checkpoints/best_adp.pt` | ADAPTIVE-labeled thesis condition |

The repository does not fully document the training recipe behind each label here. For augmentation details and comparative claims, refer to the thesis experiments rather than inferring them from these names or from the demo calibration EER.

## 11. Session-State Behavior

The enrollment embedding, its model path, metadata, and playback audio live in the **current Streamlit session**. There is no persistent account or voice-profile database. Enrollment playback and retained-segment details are available in the UI.

- **Verify again** clears the latest comparison and verification input while keeping enrollment.
- **Clear enrollment** removes the current profile, playback audio, metadata, and result.
- **Changing the model** invalidates enrollment and its result because embeddings belong to the checkpoint that produced them.

The microphone and uploader are alternative sources for each stage. The app uses fresh widget keys after resets so an old recording is not silently reused. The app writes uploaded/recorded bytes to a private temporary directory only while extracting an embedding, then removes that temporary directory.

## 12. Common Failure Messages

| What the app shows or does | Likely reason | Next step |
| --- | --- | --- |
| “Not enough speech detected. Please speak naturally for at least a few seconds.” | Fewer than 3 seconds of VAD-detected speech overall | Speak longer with less silence. |
| “Not enough usable speech segments were found. Please speak for longer and try again.” | Fewer than two complete windows passed the 1.5-second speech rule | Provide more continuous natural speech. |
| “Recording is longer than 20 seconds. Please record a shorter sample.” | Interactive input exceeded the limit | Trim or rerecord at 20 seconds or less. |
| “No compatible .pt or .pth checkpoints were found in checkpoints/.” | Files are absent or incompatible | Check the three expected paths and checkpoint files. |
| “Unable to decode this MP3 file. Please try another file or upload WAV audio.” | Local MP3 decoding failed | Use WAV or verify the local decoder stack. |
| “Model changed. Create a new voice profile for this model.” | Model switching cleared the old enrollment | Enroll again with the selected checkpoint. |

An unsupported MP3 decoder may also trigger the app's “Local MP3 decoding is unavailable” warning; WAV upload remains available. The app keeps diagnostic detail in terminal logs for other processing failures.

## 13. Manual Demo Script

1. Launch `streamlit run app.py` from the repository root with the environment active and confirm the checkpoint is ready. `ADAPTIVE` is selected by default when available.
2. Record about 10 seconds of natural enrollment speech. Click **CREATE VOICE PROFILE** and point out the retained-segment metadata and optional playback.
3. Have the same speaker say a *different* sentence for about 10 seconds. Click **VERIFY SPEAKER**. Explain the cosine score, the selected model's multi-segment threshold, and the resulting decision.
4. Click **Verify again**. The enrollment remains. Record another speaker and show the new score and decision.
5. If useful, switch models to show that enrollment must be recreated, or use **Clear enrollment** to reset the profile.

Real-room acoustics, microphone changes, and speech variation affect scores. A same-speaker trial is not guaranteed to display SAME, nor an impostor trial DIFFERENT, every time. Explain the observed outcome rather than promising perfect accuracy.

## 14. Defense Cheat Sheet

| Question | Short answer |
| --- | --- |
| Why record 10 seconds if ECAPA input is 3 seconds? | The recording supplies several complete 3-second windows; ECAPA processes those windows in a batch, then their embeddings are aggregated. |
| Why overlapping windows? | Overlap gives multiple contiguous observations and reduces dependence on one window boundary. |
| Why a 1.5-second hop? | It is the fixed half-window stride in this frozen demo protocol; changing it changes the evidence and calibration. |
| Why use VAD? | To count detected speech and reject windows dominated by silence. |
| Why not concatenate detected speech? | That would create artificial timing and waveform boundaries absent from the recording. |
| Why average embeddings? | The fixed, uniform aggregation summarizes retained windows as one recording vector. |
| Why L2-normalize after averaging? | The mean is not generally unit length; normalization restores the unit-vector contract for cosine scoring. |
| Why a new threshold? | Aggregation changes the score distribution, so the single-window operating point cannot be reused. |
| Why three thresholds? | Each checkpoint has its own empirically calibrated operating point under the same demo protocol. |
| Is cosine similarity a probability? | No. It is a geometric similarity score, not a percentage of certainty. |
| Must verification text match enrollment? | No. The system is text-independent. |
| Does it identify unknown speakers? | No. It compares a candidate against one enrolled profile. |
| Does it prevent replay, TTS, or voice-conversion attacks? | No. There is no spoof or liveness detector. |
| Why is anti-spoofing not included? | The implemented scope is speaker-embedding verification; attack detection would require its own methods and evaluation. |
| What if the model changes after enrollment? | The app clears enrollment; enroll again with the new checkpoint. |
| Why not compare models using calibration EER? | The 9-speaker operational set is small and separate from the thesis comparative evaluation. |

## 15. Things Not To Change Casually

The calibration is tied to a specific inference protocol. Treat these as coupled settings:

- 16 kHz mono preparation and the 3.0-second / 48,000-sample model window;
- the 1.5-second / 24,000-sample hop and complete-window tail policy;
- WebRTC VAD settings and the 1.5-second-per-window, 3.0-second-total, and two-window minima;
- uniform arithmetic mean followed by L2 normalization;
- model-to-checkpoint labels and their **separate** multi-segment threshold values.

Changing window selection, VAD behavior, aggregation, or the checkpoint can change the score distribution and invalidate the operational threshold. Review and recalibrate under the changed protocol before using its decisions. Never substitute a historical single-window threshold into the multi-segment app.

## 16. Known Limitations

The operational calibration covers only **9 speakers and 27 recordings** and is intended for the interactive demo. Different microphones, rooms, noise, or speech domains can shift scores. The main target is natural speech; singing is outside that primary domain. The app has no anti-spoofing, replay or synthetic-voice detection, liveness check, or ASR challenge phrase. It cannot guarantee perfect decisions and does not prove legal identity. The cosine score is not a probability.
