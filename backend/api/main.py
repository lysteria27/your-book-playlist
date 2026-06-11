"""
FastAPI app — book-to-playlist pipeline.

POST /playlist   — fetch summary → emotion model → Spotify recommendations
POST /profile    — same pipeline, returns emotion profile only (no Spotify call)
GET  /health     — health check
"""

from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.embeddings.aligner import build_enriched_profile, EnrichedBookProfile
from backend.spotify.spotify_client import build_playlist

app = FastAPI(
    title="Book to Playlist API",
    description="Give us a book — we'll give you a playlist.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class ProfileResponse(BaseModel):
    book_title: str
    book_author: str
    summary_source: str
    dominant_emotions: list[tuple[str, float]]
    emotion_profile: EmotionProfileResponse
    audio_targets: AudioTargetsResponse


class PlaylistResponse(ProfileResponse):
    tracks: list[TrackResponse]


# ---------- Helpers ----------

def _profile_response(enriched: EnrichedBookProfile) -> dict:
    return dict(
        book_title=enriched.title,
        book_author=enriched.author,
        summary_source=enriched.summary_source,
        dominant_emotions=enriched.dominant_emotions(),
        emotion_profile=EmotionProfileResponse(**enriched.emotion_profile.__dict__),
        audio_targets=AudioTargetsResponse(**enriched.audio_targets.__dict__),
    )


def _spotify_proxy(enriched: EnrichedBookProfile):
    """
    The Spotify client expects an object with named audio target attributes.
    EnrichedBookProfile already has these via enriched.audio_targets — we
    build a lightweight proxy so build_playlist can read them directly.
    """
    @dataclass
    class Proxy:
        target_valence: float
        target_energy: float
        target_acousticness: float
        target_instrumentalness: float
        target_tempo: float
        aesthetic: str = ""

    t = enriched.audio_targets
    return Proxy(
        target_valence=t.valence,
        target_energy=t.energy,
        target_acousticness=t.acousticness,
        target_instrumentalness=t.instrumentalness,
        target_tempo=t.tempo,
    )


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/playlist", response_model=PlaylistResponse)
def get_playlist(request: BookRequest):
    """
    Full pipeline: book title + author → emotion profile → Spotify playlist.
    """
    try:
        enriched = build_enriched_profile(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Profiling failed: {e}")

    try:
        tracks = build_playlist(_spotify_proxy(enriched), limit=request.playlist_size)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Spotify fetch failed: {e}")

    return PlaylistResponse(
        **_profile_response(enriched),
        tracks=[TrackResponse(**t.__dict__) for t in tracks],
    )


@app.post("/profile", response_model=ProfileResponse)
def get_profile(request: BookRequest):
    """
    Returns the emotion profile only — useful for debugging or inspecting
    what the model thinks of a book before fetching music.
    """
    try:
        enriched = build_enriched_profile(request.title, request.author)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Profiling failed: {e}")

    return ProfileResponse(**_profile_response(enriched))
