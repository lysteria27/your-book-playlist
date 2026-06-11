"""
FastAPI app — exposes the book-to-playlist pipeline as a REST API.

Two pipeline modes:
  POST /playlist          — Phase 1: Gemini LLM profiling + Spotify
  POST /playlist/v2       — Phase 2: summary fetch + emotion model + Spotify
  POST /profile           — Phase 1 profile only (debug)
  POST /profile/v2        — Phase 2 enriched profile only (debug)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.profiler.book_profiler import profile_book, BookProfile
from backend.spotify.spotify_client import build_playlist, Track
from backend.embeddings.aligner import build_enriched_profile

app = FastAPI(
    title="Book to Playlist API",
    description="Give us a book — we'll give you a playlist.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response schemas ----------

class BookRequest(BaseModel):
    title: str
    author: str
    playlist_size: int = 20


class TrackResponse(BaseModel):
    name: str
    artist: str
    album: str
    spotify_url: str
    preview_url: str | None
    duration_ms: int


class PlaylistResponse(BaseModel):
    book_title: str
    book_author: str
    profile: BookProfile
    tracks: list[TrackResponse]


class EmotionProfileResponse(BaseModel):
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    neutral: float


class AudioTargetsResponse(BaseModel):
    valence: float
    energy: float
    acousticness: float
    instrumentalness: float
    tempo: float


class EnrichedPlaylistResponse(BaseModel):
    book_title: str
    book_author: str
    summary_source: str
    dominant_emotions: list[tuple[str, float]]
    emotion_profile: EmotionProfileResponse
    audio_targets: AudioTargetsResponse
    tracks: list[TrackResponse]


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


# --- Phase 1 ---

@app.post("/playlist", response_model=PlaylistResponse)
def get_playlist(request: BookRequest):
    """
    Phase 1: Gemini LLM profiling → Spotify recommendations.
    Fast, works well for well-known books.
    """
    try:
        profile = profile_book(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Book profiling failed: {e}")

    try:
        tracks = build_playlist(profile, limit=request.playlist_size)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Spotify fetch failed: {e}")

    return PlaylistResponse(
        book_title=request.title,
        book_author=request.author,
        profile=profile,
        tracks=[TrackResponse(**t.__dict__) for t in tracks],
    )


@app.post("/profile", response_model=BookProfile)
def get_profile(request: BookRequest):
    """Phase 1 profile only — useful for debugging."""
    try:
        return profile_book(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Book profiling failed: {e}")


# --- Phase 2 ---

@app.post("/playlist/v2", response_model=EnrichedPlaylistResponse)
def get_playlist_v2(request: BookRequest):
    """
    Phase 2: summary fetch → emotion model → Spotify recommendations.
    More grounded — emotional profile derived from actual book text,
    not LLM memory. Works for obscure books Phase 1 may not know.
    """
    try:
        enriched = build_enriched_profile(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Enriched profiling failed: {e}")

    # Build a minimal BookProfile-compatible object so we can reuse build_playlist
    from backend.profiler.book_profiler import BookProfile as BP
    proxy_profile = BP(
        title=enriched.title,
        author=enriched.author,
        genres=[],
        themes=[],
        emotional_tone=[e for e, _ in enriched.dominant_emotions()],
        pace="slow" if enriched.audio_targets.tempo < 85 else (
            "fast" if enriched.audio_targets.tempo > 115 else "moderate"
        ),
        aesthetic="",
        era_feel="",
        target_valence=enriched.audio_targets.valence,
        target_energy=enriched.audio_targets.energy,
        target_acousticness=enriched.audio_targets.acousticness,
        target_instrumentalness=enriched.audio_targets.instrumentalness,
        target_tempo=enriched.audio_targets.tempo,
        reasoning="Derived from emotion model applied to book summary.",
    )

    try:
        tracks = build_playlist(proxy_profile, limit=request.playlist_size)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Spotify fetch failed: {e}")

    return EnrichedPlaylistResponse(
        book_title=enriched.title,
        book_author=enriched.author,
        summary_source=enriched.summary_source,
        dominant_emotions=enriched.dominant_emotions(),
        emotion_profile=EmotionProfileResponse(**enriched.emotion_profile.__dict__),
        audio_targets=AudioTargetsResponse(**enriched.audio_targets.__dict__),
        tracks=[TrackResponse(**t.__dict__) for t in tracks],
    )


@app.post("/profile/v2", response_model=EnrichedPlaylistResponse)
def get_profile_v2(request: BookRequest):
    """Phase 2 enriched profile only — useful for debugging the emotion pipeline."""
    try:
        enriched = build_enriched_profile(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Enriched profiling failed: {e}")

    return EnrichedPlaylistResponse(
        book_title=enriched.title,
        book_author=enriched.author,
        summary_source=enriched.summary_source,
        dominant_emotions=enriched.dominant_emotions(),
        emotion_profile=EmotionProfileResponse(**enriched.emotion_profile.__dict__),
        audio_targets=AudioTargetsResponse(**enriched.audio_targets.__dict__),
        tracks=[],
    )
