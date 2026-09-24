"""Reusable, presentation-only components for the Streamlit interface."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Optional

import streamlit as st

from src.audio import MultiSegmentMetadata, SpeechWindowMetadata

STYLES_PATH = Path(__file__).with_name("styles.css")


def load_styles() -> None:
    """Load the local stylesheet without relying on an external asset host."""

    st.html(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>")


def render_header() -> None:
    st.html(
        """
        <header class="voice-hero">
          <div class="voice-kicker"><span class="voice-mark"></span>VOICE ID</div>
          <h1>Vietnamese Speaker Verification</h1>
          <p>ECAPA-TDNN based speaker verification demonstration</p>
          <div class="voice-context">Final-year thesis demonstration · Text-independent verification</div>
        </header>
        """
    )


def render_step_indicator(current_step: int) -> None:
    labels = ("Enrollment", "Verification", "Result")
    parts: list[str] = []
    for number, label in enumerate(labels, start=1):
        if number < current_step:
            state = "complete"
            marker = "✓"
        elif number == current_step:
            state = "active"
            marker = str(number)
        else:
            state = "upcoming"
            marker = str(number)
        connector = '<span class="step-connector" aria-hidden="true"></span>' if number < 3 else ""
        parts.append(
            f'<div class="flow-step {state}"><span class="step-marker">{marker}</span>'
            f'<span class="step-label">{label}</span></div>{connector}'
        )
    st.html(
        '<nav class="flow-steps" aria-label="Verification progress">'
        + "".join(parts)
        + "</nav>"
    )


def render_section_heading(
    number: int,
    title: str,
    description: str,
    *,
    eyebrow: str,
    muted: bool = False,
) -> None:
    state_class = " is-muted" if muted else ""
    st.html(
        f"""
        <div class="section-heading{state_class}">
          <div class="section-number">{number:02d}</div>
          <div>
            <div class="section-eyebrow">{html.escape(eyebrow)}</div>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
          </div>
        </div>
        """
    )


def metadata_rows(metadata: SpeechWindowMetadata) -> tuple[tuple[str, str], ...]:
    """Return truthful, display-ready timing values from backend metadata."""

    rows: list[tuple[str, str]] = [
        ("Recording duration", f"{metadata.original_duration_seconds:.2f} s"),
    ]
    if metadata.vad_used:
        rows.append(("Detected speech", f"{metadata.total_speech_seconds:.2f} s"))
    rows.append(
        (
            "Selected model segment",
            f"{metadata.selected_start_seconds:.2f}–{metadata.selected_end_seconds:.2f} s",
        )
    )
    if metadata.vad_used:
        rows.append(
            (
                "Speech in selected segment",
                f"{metadata.speech_in_selected_window_seconds:.2f} s",
            )
        )
    return tuple(rows)


def multisegment_metadata_rows(
    metadata: MultiSegmentMetadata,
) -> tuple[tuple[str, str], ...]:
    """Return display values measured by the multi-segment backend."""

    return (
        ("Recording duration", f"{metadata.original_duration_seconds:.2f} s"),
        ("Detected speech", f"{metadata.total_speech_seconds:.2f} s"),
        ("Candidate windows", str(metadata.candidate_window_count)),
        ("Valid segments", str(metadata.valid_window_count)),
        (
            "Window / hop",
            f"{metadata.window_seconds:.1f} s / {metadata.hop_seconds:.1f} s",
        ),
    )


def _metadata_grid(metadata: SpeechWindowMetadata) -> str:
    return "".join(
        '<div class="metadata-item">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        "</div>"
        for label, value in metadata_rows(metadata)
    )


def _multisegment_metadata_grid(metadata: MultiSegmentMetadata) -> str:
    return "".join(
        '<div class="metadata-item">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        "</div>"
        for label, value in multisegment_metadata_rows(metadata)
    )


def render_enrollment_ready(metadata: MultiSegmentMetadata, model_label: str) -> None:
    st.html(
        f"""
        <section class="status-card status-card-success">
          <div class="status-icon" aria-hidden="true">✓</div>
          <div class="status-copy">
            <div class="status-overline">CURRENT SESSION · {html.escape(model_label)}</div>
            <h3>Voice profile ready</h3>
            <p>{metadata.valid_window_count} valid segments were aggregated into one voice profile.</p>
          </div>
          <div class="metadata-grid">{_multisegment_metadata_grid(metadata)}</div>
        </section>
        """
    )


def render_locked_verification() -> None:
    st.html(
        """
        <div class="locked-card">
          <span class="locked-icon" aria-hidden="true">2</span>
          <div><strong>Enrollment required</strong>
          <p>Create a voice profile to unlock verification.</p></div>
        </div>
        """
    )


def render_score_visualization(similarity: float, threshold: Optional[float]) -> None:
    """Render a mathematically valid -1 to 1 cosine comparison scale."""

    if not math.isfinite(similarity):
        return
    score_position = min(100.0, max(0.0, (similarity + 1.0) * 50.0))
    threshold_markup = ""
    threshold_legend = ""
    if threshold is not None and math.isfinite(threshold):
        threshold_position = min(100.0, max(0.0, (threshold + 1.0) * 50.0))
        threshold_markup = (
            f'<span class="threshold-line" style="left:{threshold_position:.3f}%" '
            f'title="Decision threshold {threshold:.4f}"></span>'
        )
        threshold_legend = '<span><i class="legend-threshold"></i>Decision threshold</span>'

    st.html(
        f"""
        <div class="score-visual" role="img" aria-label="Cosine similarity {similarity:.4f}">
          <div class="score-visual-title">Cosine similarity</div>
          <div class="score-axis-labels"><span>−1.00</span><span>0.00</span><span>1.00</span></div>
          <div class="score-track">
            <span class="score-zero"></span>
            {threshold_markup}
            <span class="score-dot" style="left:{score_position:.3f}%" title="Score {similarity:.4f}"></span>
          </div>
          <div class="score-legend"><span><i class="legend-score"></i>Similarity score</span>{threshold_legend}</div>
        </div>
        """
    )


def render_result_card(
    similarity: float,
    threshold: Optional[float],
    same_speaker: Optional[bool],
    model_label: str,
) -> None:
    if same_speaker is True:
        style = "match"
        icon = "✓"
        title = "Same Speaker"
        detail = "The aggregated cosine similarity meets the multi-segment operating threshold."
        decision = "SAME SPEAKER"
    elif same_speaker is False:
        style = "no-match"
        icon = "×"
        title = "Different Speaker"
        detail = "The aggregated cosine similarity is below the multi-segment operating threshold."
        decision = "DIFFERENT SPEAKER"
    else:
        style = "unavailable"
        icon = "—"
        title = "Speaker Similarity"
        detail = "No multi-segment operating threshold is configured for this model."
        decision = "DECISION UNAVAILABLE"

    threshold_text = "Not configured" if threshold is None else f"{threshold:.4f}"
    st.html(
        f"""
        <section class="result-card result-{style}">
          <div class="result-summary">
            <div class="result-icon" aria-hidden="true">{icon}</div>
            <div>
              <div class="result-decision">{decision}</div>
              <h2>{title}</h2>
              <p>{detail}</p>
            </div>
          </div>
          <div class="result-metrics">
            <div><span>Cosine similarity</span><strong>{similarity:.4f}</strong></div>
            <div><span>Operational threshold</span><strong>{threshold_text}</strong></div>
            <div><span>Selected model</span><strong>{html.escape(model_label)}</strong></div>
          </div>
        </section>
        """
    )


def render_sample_details(metadata: MultiSegmentMetadata) -> None:
    segment_rows = "".join(
        "<tr>"
        f"<td>Segment {index}</td>"
        f"<td>{segment.start_seconds:.2f}–{segment.end_seconds:.2f} s</td>"
        f"<td>{segment.speech_seconds:.2f} s speech</td>"
        "</tr>"
        for index, segment in enumerate(metadata.valid_segments, start=1)
    )
    st.html(
        f"""
        <div class="sample-details">
          <div class="metadata-grid">{_multisegment_metadata_grid(metadata)}</div>
          <div class="sample-details-title segment-title">Retained contiguous segments</div>
          <table class="segment-table">
            <thead><tr><th>Segment</th><th>Range</th><th>Detected speech</th></tr></thead>
            <tbody>{segment_rows}</tbody>
          </table>
        </div>
        """
    )


def render_calibration_note() -> None:
    st.html(
        """
        <div class="calibration-note">
          This multi-segment operating threshold was calibrated on a separate
          9-speaker / 27-recording demo set, outside the frozen thesis evaluation.
        </div>
        """
    )


def render_privacy_note() -> None:
    st.html(
        """
        <div class="privacy-note">
          <span aria-hidden="true">●</span>
          Recordings and embeddings are retained only for this active demo session.
        </div>
        """
    )
