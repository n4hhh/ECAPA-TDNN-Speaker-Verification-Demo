"""Reusable, presentation-only components for the Streamlit interface."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Optional

import streamlit as st

from src.audio import SpeechWindowMetadata

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
          <p>ECAPA-TDNN based voice authentication demonstration</p>
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


def _metadata_grid(metadata: SpeechWindowMetadata) -> str:
    return "".join(
        '<div class="metadata-item">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        "</div>"
        for label, value in metadata_rows(metadata)
    )


def render_enrollment_ready(metadata: SpeechWindowMetadata, model_label: str) -> None:
    st.html(
        f"""
        <section class="status-card status-card-success">
          <div class="status-icon" aria-hidden="true">✓</div>
          <div class="status-copy">
            <div class="status-overline">CURRENT SESSION · {html.escape(model_label)}</div>
            <h3>Voice profile ready</h3>
            <p>Your enrollment is ready for a new verification sample.</p>
          </div>
          <div class="metadata-grid">{_metadata_grid(metadata)}</div>
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
        title = "Identity Match"
        detail = "The verification sample matched the enrolled speaker."
        decision = "SAME SPEAKER"
    elif same_speaker is False:
        style = "no-match"
        icon = "×"
        title = "Identity Not Matched"
        detail = "The verification sample did not match the enrolled speaker."
        decision = "DIFFERENT SPEAKER"
    else:
        style = "unavailable"
        icon = "—"
        title = "Decision Unavailable"
        detail = "A validation-calibrated threshold is not configured for this model."
        decision = "NOT AVAILABLE"

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
            <div><span>Similarity score</span><strong>{similarity:.4f}</strong></div>
            <div><span>Decision threshold</span><strong>{threshold_text}</strong></div>
            <div><span>Model</span><strong>{html.escape(model_label)}</strong></div>
          </div>
        </section>
        """
    )


def render_sample_details(metadata: SpeechWindowMetadata) -> None:
    st.html(
        f"""
        <div class="sample-details">
          <div class="sample-details-title">Verification sample processing</div>
          <div class="metadata-grid">{_metadata_grid(metadata)}</div>
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

