"""
Tests for the book profiling pipeline.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.embeddings.summary_fetcher import BookSummary
from backend.embeddings.emotion_extractor import EmotionProfile, AudioTargets
from backend.embeddings.aligner import build_enriched_profile, EnrichedBookProfile


MOCK_SUMMARY = BookSummary(
    title="The Starless Sea",
    author="Erin Morgenstern",
    summary=(
        "A young man discovers a book that leads him to a secret underground world "
        "full of libraries, mystery, and forgotten stories. As he descends deeper, "
        "he finds himself entangled in a fate written long before he was born."
    ),
    source="openlibrary",
)

MOCK_EMOTION = EmotionProfile(
    joy=0.04, sadness=0.61, anger=0.02,
    fear=0.08, surprise=0.03, disgust=0.01, neutral=0.21,
)

MOCK_TARGETS = AudioTargets(
    valence=0.23, energy=0.27, acousticness=0.78,
    instrumentalness=0.72, tempo=74.2,
)


@patch("backend.embeddings.aligner.extract_from_summary")
@patch("backend.embeddings.aligner.fetch_summary")
def test_build_enriched_profile_returns_valid_profile(mock_fetch, mock_extract):
    mock_fetch.return_value = MOCK_SUMMARY
    mock_extract.return_value = (MOCK_EMOTION, MOCK_TARGETS)

    profile = build_enriched_profile("The Starless Sea", "Erin Morgenstern")

    assert isinstance(profile, EnrichedBookProfile)
    assert profile.title == "The Starless Sea"
    assert profile.summary_source == "openlibrary"
    assert 0.0 <= profile.audio_targets.valence <= 1.0
    assert 60 <= profile.audio_targets.tempo <= 180


@patch("backend.embeddings.aligner.extract_from_summary")
@patch("backend.embeddings.aligner.fetch_summary")
def test_dominant_emotions_returns_top_3(mock_fetch, mock_extract):
    mock_fetch.return_value = MOCK_SUMMARY
    mock_extract.return_value = (MOCK_EMOTION, MOCK_TARGETS)

    profile = build_enriched_profile("The Starless Sea", "Erin Morgenstern")
    dominant = profile.dominant_emotions(top_k=3)

    assert len(dominant) == 3
    assert dominant[0][0] == "sadness"   # highest in mock
    assert all(isinstance(score, float) for _, score in dominant)


@patch("backend.embeddings.aligner.fetch_summary")
def test_build_enriched_profile_handles_missing_summary(mock_fetch):
    """Pipeline should return a neutral fallback profile when no summary is found."""
    mock_fetch.return_value = BookSummary(
        title="Unknown Book", author="Unknown Author",
        summary="", source="none",
    )

    profile = build_enriched_profile("Unknown Book", "Unknown Author")

    assert profile.summary_source == "none"
    assert isinstance(profile.audio_targets, AudioTargets)
