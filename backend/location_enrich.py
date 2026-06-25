"""Resolve a real photo + the best official link for a location card.

No API keys required — every source here is free and hot-linkable:

* Photo: Wikipedia page image (an actual photo of the place) → Openverse
  (Creative-Commons image search) → None (the UI then shows a category tile).
* Official website: Wikidata property P856 ("official website"), used only when
  the Wikipedia match is confident enough to trust.

Every network lookup is cached in SQLite so repeated card loads — and live
demos — are instant and resilient to flaky networks.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .llm_service import _LLMCache  # generic (key -> json) sqlite cache

logger = logging.getLogger(__name__)

_UA = {
    "User-Agent": "ChicagoDoesRecsys/1.0 (UChicago capstone demo; non-commercial)"
}
_TIMEOUT = 6.0

# Tokens that carry no identifying signal when matching a Wikipedia title to a
# venue name (used to decide whether an official-website claim is trustworthy).
_STOP = {
    "the", "a", "an", "of", "and", "at", "in", "on", "co", "inc", "llc", "ltd",
    "chicago", "il", "illinois", "us", "usa",
}


def _get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        logger.info("enrich: GET failed (%s): %s", type(exc).__name__, url[:80])
        return None


def _tokens(text: str) -> set:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in (text or "").lower())
    return {t for t in cleaned.split() if t and t not in _STOP}


def _title_matches(name: str, title: str) -> bool:
    """True when a Wikipedia article title really refers to THIS venue.

    Strict on purpose: the Wikipedia search generator always returns its top
    hit, which can be a loosely related page (e.g. "Emerald Loop" -> "Emerald
    City", "WNDR Museum" -> the founder's bio). A wrong match means a wrong
    photo or a wrong official link, so we only accept a title when (almost) all
    of the venue's identifying words appear in it.
    """
    if not title:
        return False
    nl, tl = name.lower().strip(), title.lower().strip()
    if nl in tl or tl in nl:
        return True
    nt, tt = _tokens(name), _tokens(title)
    if not nt or not tt:
        return False
    overlap = nt & tt
    # Short names (<=2 significant words) must match in full; longer names ~60%.
    need = len(nt) if len(nt) <= 2 else max(2, (len(nt) * 3 + 4) // 5)
    return len(overlap) >= need


def _wikipedia(name: str) -> Dict[str, Any]:
    """One Wikipedia call → best-matching page's thumbnail + wikidata id + title."""
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "prop": "pageimages|pageprops",
        "piprop": "thumbnail",
        "pithumbsize": "640",
        "generator": "search",
        "gsrsearch": f"{name} Chicago",
        "gsrlimit": "1",
        "redirects": "1",
    })
    data = _get_json(f"https://en.wikipedia.org/w/api.php?{params}")
    pages = (data or {}).get("query", {}).get("pages", {})
    if not pages:
        return {}
    page = next(iter(pages.values()))
    return {
        "title": page.get("title"),
        "thumb": (page.get("thumbnail") or {}).get("source"),
        "wikidata": (page.get("pageprops") or {}).get("wikibase_item"),
    }


def _wikidata_official_site(wikidata_id: str) -> Optional[str]:
    if not wikidata_id:
        return None
    data = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json")
    try:
        claims = data["entities"][wikidata_id]["claims"]
        p856 = claims.get("P856") or []
        for claim in p856:
            url = claim["mainsnak"]["datavalue"]["value"]
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url
    except (KeyError, TypeError, IndexError):
        return None
    return None


def _openverse_photo(name: str) -> Optional[str]:
    params = urllib.parse.urlencode({
        "q": f"{name} Chicago",
        "page_size": "1",
        "mature": "false",
    })
    data = _get_json(f"https://api.openverse.org/v1/images/?{params}")
    results = (data or {}).get("results") or []
    if not results:
        return None
    r = results[0]
    return r.get("thumbnail") or r.get("url")


class LocationEnricher:
    """Cached resolver for a card's photo + official website."""

    def __init__(self, cache_path: str = "data/enrich_cache.sqlite", enabled: bool = True) -> None:
        self.cache = _LLMCache(path=cache_path, enabled=enabled)

    def photo_and_site(self, name: str) -> Dict[str, Any]:
        """Return ``{image_url, image_source, image_attribution, official_site}``.

        Cached by venue name. Network failures degrade to an empty result so the
        card still renders (with a category tile and a non-official link).
        """
        name = (name or "").strip()
        empty = {
            "image_url": None,
            "image_source": None,
            "image_attribution": None,
            "official_site": None,
        }
        if not name:
            return empty

        key = self.cache.make_key("card_enrich_v2", name)
        hit = self.cache.get(key)
        if hit is not None:
            try:
                return json.loads(hit)
            except ValueError:
                pass  # corrupt row → recompute

        out = dict(empty)
        wiki = _wikipedia(name)
        title = wiki.get("title")
        # Use the Wikipedia photo/site ONLY when the matched article is really
        # this place — otherwise we'd show a wrong photo or link.
        confident = bool(title) and _title_matches(name, title)
        if confident and wiki.get("thumb"):
            out["image_url"] = wiki["thumb"]
            out["image_source"] = "wikipedia"
            out["image_attribution"] = f"Photo: Wikipedia ({title})"
        if confident and wiki.get("wikidata"):
            out["official_site"] = _wikidata_official_site(wiki["wikidata"])

        if not out["image_url"]:
            ov = _openverse_photo(name)
            if ov:
                out["image_url"] = ov
                out["image_source"] = "openverse"
                out["image_attribution"] = "Photo: Openverse (Creative Commons)"

        # Cache successes and confirmed-empty lookups alike (avoids re-hammering
        # the network for places that simply have no photo).
        self.cache.put(key, json.dumps(out))
        return out
