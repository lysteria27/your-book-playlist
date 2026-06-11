"""
Summary Fetcher — retrieves book summaries from Open Library and Google Books.

Both APIs are free with no authentication required for basic use.
Open Library:  https://openlibrary.org/developers/api
Google Books:  https://developers.google.com/books/docs/v1/using
"""

import httpx
from dataclasses import dataclass


@dataclass
class BookSummary:
    title: str
    author: str
    summary: str
    source: str     # "openlibrary" | "googlebooks" | "none"


def _fetch_open_library(title: str, author: str) -> str | None:
    """
    Try Open Library first — community-contributed descriptions,
    works well for literary fiction.
    """
    # Step 1: search for the work
    search_resp = httpx.get(
        "https://openlibrary.org/search.json",
        params={"title": title, "author": author, "limit": 1},
        timeout=10,
    )
    search_resp.raise_for_status()
    docs = search_resp.json().get("docs", [])
    if not docs:
        return None

    work_key = docs[0].get("key")  # e.g. "/works/OL123W"
    if not work_key:
        return None

    # Step 2: fetch the work record for its description
    work_resp = httpx.get(
        f"https://openlibrary.org{work_key}.json",
        timeout=10,
    )
    work_resp.raise_for_status()
    work = work_resp.json()

    description = work.get("description")
    if not description:
        return None

    # Description can be a plain string or a {"type": ..., "value": ...} dict
    if isinstance(description, dict):
        return description.get("value")
    return description


def _fetch_google_books(title: str, author: str) -> str | None:
    """
    Fall back to Google Books — broader coverage, publisher-supplied blurbs.
    """
    resp = httpx.get(
        "https://www.googleapis.com/books/v1/volumes",
        params={
            "q": f'intitle:"{title}" inauthor:"{author}"',
            "maxResults": 1,
            "printType": "books",
        },
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return None

    info = items[0].get("volumeInfo", {})
    return info.get("description")


def fetch_summary(title: str, author: str) -> BookSummary:
    """
    Fetch a book summary, trying Open Library first then Google Books.
    Returns a BookSummary with source="none" and empty summary if both fail.
    """
    summary = _fetch_open_library(title, author)
    if summary:
        return BookSummary(title=title, author=author, summary=summary, source="openlibrary")

    summary = _fetch_google_books(title, author)
    if summary:
        return BookSummary(title=title, author=author, summary=summary, source="googlebooks")

    return BookSummary(title=title, author=author, summary="", source="none")


if __name__ == "__main__":
    result = fetch_summary("The Starless Sea", "Erin Morgenstern")
    print(f"Source: {result.source}")
    print(f"Summary: {result.summary[:300]}...")
