"""
Aligner — the Phase 2 pipeline.

Wires together:
  1. summary_fetcher  — pulls book summary from Open Library / Google Books
  2. emotion_extractor — runs DistilRoBERTa emotion model on the summary
  3. Produces AudioTargets that replace Phase 1's LLM-guessed values

This is a meaningful improvement over Phase 1 because:
  - The emotional profile is derived from the book's actual text, not
    an LLM's recollection of it
  - Works for obscure or recently published books Phase 1 may not know
  - The emotion → audio feature mapping (EMOTION_TO_AUDIO matrix) is
    a concrete artifact that can be improved with training data

Future improvement — learning the mapping:
  Once you have paired (book summary, "good playlist") examples, you can
  replace the hand-calibrated EMOTION_TO_AUDIO matrix with learned weights
  by treating it as a simple regression problem:

    X = stack of emotion vectors from summaries
    y = stack of audio feature vectors from their paired playlists
    model = Ridge().fit(X, y)   # or a small MLP
    learned_weights = model.coef_   # shape (5, 7)

  Training data sources:
    - Goodreads shelf names matched to Spotify playlist names (weak labels)
    - Human-annotated (book, playlist) pairs if you can collect them
    - LLM-generated similarity scores as additional weak supervision

See notebooks/emotion_mapping_experiments.ipynb for experiments.
"""

from __future__ import annotations
from dataclasses import dataclass

from backend.embeddings.summary_fetcher import fetch_summary, BookSummary
from backend.embeddings.emotion_extractor import (
    EmotionProfile,
    AudioTargets,
    extract_from_summary,
)


@dataclass
class EnrichedBookProfile:
    """
    Output of the Phase 2 pipeline — richer than Phase 1's BookProfile
    because the emotional signal comes from the book's actual text.
    """
    title: str
    author: str
    summary: str
    summary_source: str             # "openlibrary" | "googlebooks" | "none"
    emotion_profile: EmotionProfile
    audio_targets: AudioTargets

    def dominant_emotions(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k emotions by probability, for logging/debugging."""
        emotions = {
            "joy": self.emotion_profile.joy,
            "sadness": self.emotion_profile.sadness,
            "anger": self.emotion_profile.anger,
            "fear": self.emotion_profile.fear,
            "surprise": self.emotion_profile.surprise,
            "disgust": self.emotion_profile.disgust,
            "neutral": self.emotion_profile.neutral,
        }
        return sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_k]


def build_enriched_profile(title: str, author: str) -> EnrichedBookProfile:
    """
    Full Phase 2 pipeline:
      title + author
        → fetch summary (Open Library → Google Books fallback)
        → run emotion model on summary text
        → map emotion profile to Spotify audio targets

    Falls back gracefully: if no summary is found, returns a neutral
    audio target profile rather than raising.
    """
    book_summary: BookSummary = fetch_summary(title, author)

    if book_summary.summary:
        emotion_profile, audio_targets = extract_from_summary(book_summary.summary)
    else:
        # No summary found — return a neutral mid-range profile
        # Phase 1 (LLM profiler) is a better fallback in this case
        from backend.embeddings.emotion_extractor import EmotionProfile, AudioTargets
        emotion_profile = EmotionProfile(
            joy=0.1, sadness=0.1, anger=0.1, fear=0.1,
            surprise=0.1, disgust=0.1, neutral=0.4,
        )
        audio_targets = AudioTargets(
            valence=0.5, energy=0.3, acousticness=0.6,
            instrumentalness=0.6, tempo=80.0,
        )

    return EnrichedBookProfile(
        title=title,
        author=author,
        summary=book_summary.summary,
        summary_source=book_summary.source,
        emotion_profile=emotion_profile,
        audio_targets=audio_targets,
    )


if __name__ == "__main__":
    profile = build_enriched_profile("The Starless Sea", "Erin Morgenstern")

    print(f"Summary source: {profile.summary_source}")
    print(f"Summary snippet: {profile.summary[:200]}...")
    print(f"\nDominant emotions: {profile.dominant_emotions()}")
    print(f"\nAudio targets:")
    t = profile.audio_targets
    print(f"  valence={t.valence:.3f}  energy={t.energy:.3f}  "
          f"acousticness={t.acousticness:.3f}  "
          f"instrumentalness={t.instrumentalness:.3f}  "
          f"tempo={t.tempo:.1f}bpm")
