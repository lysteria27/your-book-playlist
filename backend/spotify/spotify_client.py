"""
Spotify Client — maps a BookProfile's audio feature targets to a curated playlist
using the Spotify Recommendations API.
"""

import os
import base64
from dataclasses import dataclass

import httpx

from backend.profiler.book_profiler import BookProfile


@dataclass
class Track:
    name: str
    artist: str
    album: str
    spotify_url: str
    preview_url: str | None
    duration_ms: int


class SpotifyClient:
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    RECOMMENDATIONS_URL = "https://api.spotify.com/v1/recommendations"
    SEARCH_URL = "https://api.spotify.com/v1/search"

    def __init__(self):
        self.client_id = os.environ["SPOTIFY_CLIENT_ID"]
        self.client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
        self._token: str | None = None

    def _get_token(self) -> str:
        """Fetch a client credentials token (no user login required)."""
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        response = httpx.post(
            self.TOKEN_URL,
            headers={"Authorization": f"Basic {credentials}"},
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()
        return response.json()["access_token"]

    @property
    def token(self) -> str:
        if not self._token:
            self._token = self._get_token()
        return self._token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _genre_seeds(self, profile: BookProfile) -> list[str]:
        """
        Map book aesthetic/themes to Spotify genre seeds.
        Spotify accepts up to 5 seed genres.
        """
        GENRE_MAP = {
            "gothic": ["dark-ambient", "classical", "piano"],
            "whimsical": ["ambient", "folk", "acoustic"],
            "gritty": ["blues", "rock", "industrial"],
            "epic": ["classical", "epic", "cinematic"],
            "romantic": ["romance", "classical", "piano"],
            "dystopian": ["electronic", "ambient", "industrial"],
            "folkloric": ["folk", "acoustic", "world-music"],
            "noir": ["jazz", "blues", "soul"],
            "literary": ["classical", "piano", "ambient"],
        }

        aesthetic = profile.aesthetic.lower()
        for key, genres in GENRE_MAP.items():
            if key in aesthetic:
                return genres[:5]

        # Default fallback — instrumental ambient suits most reading
        return ["ambient", "classical", "piano"]

    def get_recommendations(
        self,
        profile: BookProfile,
        limit: int = 20,
    ) -> list[Track]:
        """
        Fetch track recommendations from Spotify based on BookProfile audio targets.
        """
        seeds = self._genre_seeds(profile)

        params = {
            "limit": limit,
            "seed_genres": ",".join(seeds),
            "target_valence": profile.target_valence,
            "target_energy": profile.target_energy,
            "target_acousticness": profile.target_acousticness,
            "target_instrumentalness": profile.target_instrumentalness,
            "target_tempo": profile.target_tempo,
            # Hard filters — keep it reading-friendly
            "max_energy": min(profile.target_energy + 0.2, 1.0),
            "min_instrumentalness": max(profile.target_instrumentalness - 0.2, 0.0),
        }

        response = httpx.get(
            self.RECOMMENDATIONS_URL,
            headers=self._auth_headers(),
            params=params,
        )
        response.raise_for_status()

        tracks = []
        for item in response.json().get("tracks", []):
            tracks.append(
                Track(
                    name=item["name"],
                    artist=", ".join(a["name"] for a in item["artists"]),
                    album=item["album"]["name"],
                    spotify_url=item["external_urls"]["spotify"],
                    preview_url=item.get("preview_url"),
                    duration_ms=item["duration_ms"],
                )
            )
        return tracks


def build_playlist(profile: BookProfile, limit: int = 20) -> list[Track]:
    """Convenience function — profile → playlist."""
    client = SpotifyClient()
    return client.get_recommendations(profile, limit=limit)
