# 📚🎵 Your Book Playlist

> Give it a book. Get a playlist.

Generates a Spotify playlist to accompany any book by analysing the emotional
content of its summary using a pretrained NLP model, then mapping that emotional
profile to music.

---

## How it works

```
Book title + author
       │
       ▼
Fetch summary
(Open Library → Google Books fallback)
       │
       ▼
Emotion model (DistilRoBERTa)
extracts emotional profile:
  joy, sadness, anger, fear,
  surprise, disgust, neutral
       │
       ▼
Map to Spotify audio features:
  valence, energy, acousticness,
  instrumentalness, tempo
       │
       ▼
Spotify Recommendations API
       │
       ▼
    Playlist 🎵
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/book-to-playlist.git
cd book-to-playlist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Fill in your keys in .env
```

You'll need:
- **Google API key** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free, no credit card required) — used only for summary fallback enrichment
- **Spotify app credentials** — [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) (Client ID + Secret, no user login required)

### 3. Run

```bash
uvicorn backend.api.main:app --reload
```

API docs at `http://localhost:8000/docs`

### 4. Try it

```bash
curl -X POST http://localhost:8000/playlist \
  -H "Content-Type: application/json" \
  -d '{"title": "The Starless Sea", "author": "Erin Morgenstern"}'
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/playlist` | POST | Book → emotion profile + Spotify playlist |
| `/profile` | POST | Book → emotion profile only (no Spotify call) |

### Request body

```json
{
  "title": "The Starless Sea",
  "author": "Erin Morgenstern",
  "playlist_size": 20
}
```

### Response (playlist)

```json
{
  "book_title": "The Starless Sea",
  "book_author": "Erin Morgenstern",
  "summary_source": "openlibrary",
  "dominant_emotions": [["sadness", 0.61], ["neutral", 0.21], ["fear", 0.08]],
  "emotion_profile": { "joy": 0.04, "sadness": 0.61, ... },
  "audio_targets": { "valence": 0.23, "energy": 0.27, "tempo": 74.2, ... },
  "tracks": [
    {
      "name": "...",
      "artist": "...",
      "album": "...",
      "spotify_url": "...",
      "preview_url": "...",
      "duration_ms": 240000
    }
  ]
}
```

---

## Project structure

```
book-to-playlist/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI app
│   ├── embeddings/
│   │   ├── summary_fetcher.py   # Open Library + Google Books
│   │   ├── emotion_extractor.py # DistilRoBERTa emotion model
│   │   └── aligner.py           # Wires the pipeline together
│   └── spotify/
│       └── spotify_client.py    # Spotify Recommendations API
└── tests/
```
## Currently working on:
- **Gathering training data** to make the emotion model to spotify feature mapping more robust.
- **Trainable weight matrix** to better learn the feature mapping. Kaggle Goodreads dataset has mood shelves, from which keywords can be used to search for Spotify playlist names.
```
Goodreads shelf labels → matched Spotify playlists → (emotion vector, audio features) pairs → fit Ridge regression → learned matrix
```
---

## Tech stack

- **Emotion model:** [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) via HuggingFace Transformers
- **Book summaries:** [Open Library API](https://openlibrary.org/developers/api) + [Google Books API](https://developers.google.com/books)
- **Music:** [Spotify Web API](https://developer.spotify.com/documentation/web-api)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com) + [Pydantic](https://docs.pydantic.dev)

---

## License

MIT
