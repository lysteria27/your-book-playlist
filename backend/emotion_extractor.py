"""
Emotion Extractor — runs a pretrained emotion classification model on a book
summary and maps the resulting emotional profile to Spotify audio feature targets.

Model: j-hartmann/emotion-english-distilroberta-base
  - Classifies text into 7 Ekman emotions: joy, sadness, anger, fear,
    surprise, disgust, neutral
  - Runs locally, no API key required
  - ~330MB download on first use (cached by HuggingFace)

The mapping from emotions → Spotify audio features is a learned affine
transformation. The initial weights below are hand-calibrated and can be
replaced with learned weights once training data is available (see aligner.py).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from transformers import pipeline


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmotionProfile:
    """Raw output from the emotion model — probability per emotion axis."""
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    neutral: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.joy, self.sadness, self.anger,
            self.fear, self.surprise, self.disgust, self.neutral,
        ], dtype=np.float32)


@dataclass
class AudioTargets:
    """Spotify audio feature targets derived from an emotion profile."""
    valence: float              # 0.0 (negative) – 1.0 (positive)
    energy: float               # 0.0 (calm) – 1.0 (intense)
    acousticness: float         # 0.0 (electronic) – 1.0 (acoustic)
    instrumentalness: float     # 0.0 (vocal) – 1.0 (instrumental)
    tempo: float                # BPM, ~60–180


# ---------------------------------------------------------------------------
# Emotion → audio feature mapping
# ---------------------------------------------------------------------------

# Each row is a Spotify feature: [valence, energy, acousticness, instrumentalness, tempo]
# Each column is an emotion:     [joy, sadness, anger, fear, surprise, disgust, neutral]
#
# Intuition behind the weights:
#   joy       → high valence, moderate-high energy, low instrumentalness (songs feel celebratory)
#   sadness   → low valence, low energy, high acousticness, high instrumentalness (sparse, quiet)
#   anger     → low-mid valence, high energy, low acousticness (driving, electric)
#   fear      → low valence, low-mid energy, high instrumentalness (tense, ambient)
#   surprise  → mid valence, mid-high energy (unpredictable)
#   disgust   → low valence, mid energy
#   neutral   → mid valence, low energy, high acousticness, high instrumentalness (background)
#
# Columns:          joy    sad    anger  fear   surp   disg   neut
EMOTION_TO_AUDIO = np.array([
    # valence
    [0.85,  0.15,  0.30,  0.20,  0.55,  0.20,  0.50],
    # energy
    [0.70,  0.25,  0.85,  0.40,  0.65,  0.50,  0.20],
    # acousticness
    [0.40,  0.75,  0.20,  0.60,  0.35,  0.40,  0.80],
    # instrumentalness
    [0.30,  0.70,  0.25,  0.75,  0.40,  0.45,  0.75],
    # tempo (normalised 0-1, maps to 60-180 BPM below)
    [0.65,  0.25,  0.85,  0.35,  0.70,  0.50,  0.20],
], dtype=np.float32)


def _emotion_to_audio_targets(profile: EmotionProfile) -> AudioTargets:
    """
    Linear map: emotion probability vector → audio feature vector.

    This is the key function to improve with learned weights (see aligner.py).
    Currently hand-calibrated; replace EMOTION_TO_AUDIO with trained weights
    once you have (emotion_profile, audio_features) paired training data.
    """
    emotion_vec = profile.to_vector()                    # shape (7,)
    audio_vec = EMOTION_TO_AUDIO @ emotion_vec           # shape (5,)

    # Clip features to valid range
    audio_vec = np.clip(audio_vec, 0.0, 1.0)

    # De-normalise tempo from [0,1] to [60,180] BPM
    tempo = float(60.0 + audio_vec[4] * 120.0)

    return AudioTargets(
        valence=float(audio_vec[0]),
        energy=float(audio_vec[1]),
        acousticness=float(audio_vec[2]),
        instrumentalness=float(audio_vec[3]),
        tempo=tempo,
    )


# ---------------------------------------------------------------------------
# Emotion extraction
# ---------------------------------------------------------------------------

_classifier = None  # Lazy-loaded to avoid slow startup


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,         # Return scores for all labels
            truncation=True,
            max_length=512,
        )
    return _classifier


def extract_emotion_profile(text: str) -> EmotionProfile:
    """
    Run the emotion model on a piece of text (e.g. a book summary).
    Returns an EmotionProfile with a probability for each of the 7 emotion axes.

    Long texts are truncated to 512 tokens — for full summaries, consider
    chunking and averaging (see extract_from_summary below).
    """
    classifier = _get_classifier()
    results = classifier(text)

    # results is a list of lists: [[{"label": "joy", "score": 0.8}, ...]]
    scores = {item["label"].lower(): item["score"] for item in results[0]}

    return EmotionProfile(
        joy=scores.get("joy", 0.0),
        sadness=scores.get("sadness", 0.0),
        anger=scores.get("anger", 0.0),
        fear=scores.get("fear", 0.0),
        surprise=scores.get("surprise", 0.0),
        disgust=scores.get("disgust", 0.0),
        neutral=scores.get("neutral", 0.0),
    )


def extract_from_summary(summary: str, chunk_size: int = 400) -> tuple[EmotionProfile, AudioTargets]:
    """
    Extract an emotion profile and derive audio targets from a book summary.

    For summaries longer than ~512 tokens, splits into overlapping chunks,
    runs the model on each, and averages the results. This gives a more
    representative emotional reading than truncating.

    Returns:
        emotion_profile: raw emotion probabilities
        audio_targets: Spotify feature targets derived from the emotion profile
    """
    words = summary.split()

    if len(words) <= chunk_size:
        chunks = [summary]
    else:
        # Sliding window with 50-word overlap
        step = chunk_size - 50
        chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), step)
        ]

    profiles = [extract_emotion_profile(chunk) for chunk in chunks]

    # Average across chunks
    avg = EmotionProfile(
        joy=float(np.mean([p.joy for p in profiles])),
        sadness=float(np.mean([p.sadness for p in profiles])),
        anger=float(np.mean([p.anger for p in profiles])),
        fear=float(np.mean([p.fear for p in profiles])),
        surprise=float(np.mean([p.surprise for p in profiles])),
        disgust=float(np.mean([p.disgust for p in profiles])),
        neutral=float(np.mean([p.neutral for p in profiles])),
    )

    targets = _emotion_to_audio_targets(avg)
    return avg, targets


if __name__ == "__main__":
    sample = (
        "A young man discovers a book that leads him to a secret underground world "
        "full of libraries, mystery, and forgotten stories. As he descends deeper, "
        "he finds himself entangled in a fate written long before he was born. "
        "The novel is lyrical, melancholic, and suffused with a bittersweet longing "
        "for stories and the people who love them."
    )
    profile, targets = extract_from_summary(sample)
    print("Emotion profile:")
    print(f"  joy={profile.joy:.3f}  sadness={profile.sadness:.3f}  "
          f"fear={profile.fear:.3f}  neutral={profile.neutral:.3f}")
    print("\nDerived audio targets:")
    print(f"  valence={targets.valence:.3f}  energy={targets.energy:.3f}  "
          f"acousticness={targets.acousticness:.3f}  "
          f"instrumentalness={targets.instrumentalness:.3f}  "
          f"tempo={targets.tempo:.1f}bpm")
