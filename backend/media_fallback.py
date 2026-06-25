"""Offline-only helpers to find and localize card photos.

Used exclusively by ``scripts/build_location_cards.py`` — never at HTTP request
time. Runtime card browsing must serve files from ``data/location_images/``.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .chicagodoes_media import _firecrawl_scrape, fetch_og_image, pick_media_attribution
from .location_enrich import LocationEnricher, _get_json, _openverse_photo, _title_matches, _wikipedia

logger = logging.getLogger(__name__)

_RE_FC_URL = re.compile(r"^\s*URL:\s*(https?://\S+)", re.M)
_RE_HTTP_IMG = re.compile(
    r"https?://[^\s\"'<>)]+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s\"'<>)]*)?",
    re.I,
)
_RE_MD_IMG = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", re.I)
_RE_YELP_BPHOTO = re.compile(
    r"https://[^)\s\"']+yelpcdn\.com/bphoto/[^)\s\"']+/l\.jpg",
    re.I,
)
_RE_YT_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})",
    re.I,
)
_STOP_TOKENS = {
    "the", "a", "an", "and", "or", "of", "at", "in", "on", "for", "by",
    "hotel", "hotels", "restaurant", "restaurants", "chicago", "illinois",
    "il", "usa", "co", "company", "llc", "inc",
}

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def youtube_video_id(url: str) -> Optional[str]:
    m = _RE_YT_ID.search(str(url or ""))
    return m.group(1) if m else None


def youtube_thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def _name_variants(name: str) -> List[str]:
    """Search variants for chains / multi-location brands."""
    base = (name or "").strip()
    if not base:
        return []
    out: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    add(base)
    if " - " in base:
        add(base.split(" - ", 1)[0].strip())
    if " near " in base.lower():
        add(re.split(r"\s+near\s+", base, flags=re.I)[0].strip())
    if "/" in base:
        add(base.split("/")[0].strip())
    # Raising Cane's / hotel combos
    for sep in (" - ", " | "):
        if sep in base:
            add(base.split(sep)[0].strip())
    return out


def _openverse_queries(name: str) -> List[str]:
    queries: List[str] = []
    seen: set[str] = set()
    for variant in _name_variants(name):
        for q in (
            f"{variant} Chicago",
            f"{variant} Chicago Illinois",
            f'"{variant}" Chicago',
            variant,
        ):
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def _firecrawl_search_text(query: str, cache_dir: Optional[Path]) -> str:
    cache_dir = cache_dir or Path("data/firecrawl_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:100]
    cache_file = cache_dir / f"search_{safe}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    try:
        proc = subprocess.run(
            ["firecrawl", "search", query, "--limit", "5"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if proc.returncode != 0:
            logger.info("firecrawl search failed for %r: %s", query[:60], proc.stderr[:160])
            return ""
        text = proc.stdout or ""
        cache_file.write_text(text, encoding="utf-8")
        return text
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("firecrawl search error for %r: %s", query[:60], exc)
        return ""


def _firecrawl_result_urls(text: str) -> List[str]:
    return [u.rstrip(").,]") for u in _RE_FC_URL.findall(text or "")]


def _urls_from_text(text: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for m in _RE_HTTP_IMG.finditer(text or ""):
        url = m.group(0).rstrip(").,]")
        if url not in seen:
            seen.add(url)
            out.append(url)
    for m in _RE_MD_IMG.finditer(text or ""):
        url = m.group(1).rstrip(").,]")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _scrape_text_and_image_urls(
    page_url: str,
    cache_dir: Optional[Path],
) -> Tuple[str, List[str]]:
    """Return scraped text plus image candidates from one page."""
    data = _firecrawl_scrape(page_url, cache_dir=cache_dir)
    if not data:
        return "", []
    chunks = [str(data.get("markdown") or "")]
    meta = data.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("title", "ogTitle", "og:title", "description", "ogDescription"):
            if meta.get(key):
                chunks.append(str(meta[key]))
        for key in ("ogImage", "og:image", "twitter:image"):
            if meta.get(key):
                chunks.append(str(meta[key]))
    text = "\n".join(chunks)
    urls: List[str] = []
    seen: set[str] = set()
    for url in _RE_YELP_BPHOTO.findall(text) + _urls_from_text(text):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    og = fetch_og_image(page_url)
    if og and og not in seen:
        urls.insert(0, og)
    return text, urls


def _image_urls_from_scrape(page_url: str, cache_dir: Optional[Path]) -> List[str]:
    _, urls = _scrape_text_and_image_urls(page_url, cache_dir)
    return urls


def _tokens_for_match(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return [t for t in cleaned.split() if t and t not in _STOP_TOKENS]


def _raw_name_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return [t for t in cleaned.split() if t]


def _page_matches_place(text: str, url: str, name: str) -> bool:
    """Strict page guard before trusting scraped external images.

    This is intentionally conservative: if the scraped page cannot identify the
    exact venue well enough, we skip it and keep the existing media.
    """
    hay = f"{text}\n{url}".lower()
    variants = _name_variants(name)
    for variant in variants:
        v = variant.strip().lower()
        if len(v) >= 4 and v in hay and ("chicago" in hay or "chicago" in url.lower()):
            return True

    tokens = _tokens_for_match(name)
    if not tokens:
        return False
    # One-token/generic names are too risky unless the exact phrase matched.
    if len(tokens) < 2:
        return False
    matched = sum(1 for t in tokens if t in hay)
    need = len(tokens) if len(tokens) <= 3 else max(3, (len(tokens) * 3 + 3) // 4)
    return matched >= need and ("chicago" in hay or "chicago" in url.lower())


def _strict_page_images(
    page_url: str,
    name: str,
    cache_dir: Optional[Path],
) -> List[str]:
    text, urls = _scrape_text_and_image_urls(page_url, cache_dir)
    if not urls:
        return []
    if not _page_matches_place(text, page_url, name):
        logger.info("strict image page rejected for %r: %s", name, page_url[:90])
        return []
    return urls


def _commons_image(name: str) -> Optional[str]:
    for variant in _name_variants(name):
        params = urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{variant} Chicago",
            "gsrnamespace": "6",
            "gsrlimit": "3",
            "prop": "imageinfo",
            "iiprop": "url|thumburl",
            "iiurlwidth": "800",
        })
        data = _get_json(f"https://commons.wikimedia.org/w/api.php?{params}")
        pages = (data or {}).get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            if url:
                return str(url)
    return None


def _yelp_images(name: str, cache_dir: Optional[Path]) -> List[str]:
    for variant in _name_variants(name):
        text = _firecrawl_search_text(f'{variant} Chicago site:yelp.com/biz', cache_dir)
        for page_url in _firecrawl_result_urls(text):
            if "yelp.com/biz" not in page_url.lower():
                continue
            clean = page_url.split("?")[0]
            imgs = _image_urls_from_scrape(clean, cache_dir)
            if imgs:
                return imgs
    return []


def _tripadvisor_images(name: str, cache_dir: Optional[Path]) -> List[str]:
    for variant in _name_variants(name):
        text = _firecrawl_search_text(
            f'{variant} Chicago site:tripadvisor.com', cache_dir,
        )
        for page_url in _firecrawl_result_urls(text):
            if "tripadvisor.com" not in page_url.lower():
                continue
            imgs = _image_urls_from_scrape(page_url.split("?")[0], cache_dir)
            if imgs:
                return imgs
    return []


def _web_search_images(name: str, cache_dir: Optional[Path]) -> List[str]:
    """General web search → scrape top pages for OG / inline photos."""
    urls: List[str] = []
    seen: set[str] = set()
    for query in (
        f"{name} Chicago restaurant photo",
        f"{name} Chicago hotel photo",
        f"{name} Chicago official website",
        f"{name} Chicago site:google.com/maps",
    ):
        text = _firecrawl_search_text(query, cache_dir)
        for page_url in _firecrawl_result_urls(text)[:4]:
            if any(skip in page_url.lower() for skip in ("facebook.com", "instagram.com", "twitter.com")):
                continue
            for img in _image_urls_from_scrape(page_url, cache_dir)[:3]:
                if img not in seen:
                    seen.add(img)
                    urls.append(img)
        if urls:
            return urls
    return urls


def iter_strict_extra_image_candidates(
    name: str,
    *,
    official_site: Optional[str],
    cache_dir: Optional[Path] = None,
    use_firecrawl: bool = True,
) -> Iterator[Tuple[str, str]]:
    """Yield extra image candidates from exact-match pages only.

    Intended for augmenting cards that already have at least one image. It is
    stricter than ``iter_fallback_image_candidates`` because adding several
    photos increases the risk of a wrong business/branch image.
    """
    name = (name or "").strip()
    if not name or not use_firecrawl:
        return
    if len(_raw_name_tokens(name)) < 2:
        logger.info("strict extra image search skipped for ambiguous short name: %r", name)
        return

    yielded: set[str] = set()

    def offer(url: Optional[str], source: str) -> Iterator[Tuple[str, str]]:
        if not url or url in yielded:
            return
        yielded.add(url)
        yield url, source

    site = (official_site or "").strip()
    if site:
        for img in _strict_page_images(site, name, cache_dir)[:4]:
            yield from offer(img, "official")

    searches = [
        (f'"{name}" Chicago site:yelp.com/biz', "yelp", "yelp.com/biz"),
        (f'"{name}" Chicago site:tripadvisor.com', "tripadvisor", "tripadvisor.com"),
        (f'"{name}" Chicago official website', "official", ""),
    ]
    for query, source, required_host_part in searches:
        text = _firecrawl_search_text(query, cache_dir)
        for page_url in _firecrawl_result_urls(text)[:5]:
            low = page_url.lower()
            if required_host_part and required_host_part not in low:
                continue
            if any(skip in low for skip in ("facebook.com", "instagram.com", "twitter.com", "x.com")):
                continue
            for img in _strict_page_images(page_url.split("?")[0], name, cache_dir)[:4]:
                yield from offer(img, source)


def iter_fallback_image_candidates(
    name: str,
    *,
    enricher: LocationEnricher,
    official_site: Optional[str],
    llm: Any = None,
    cache_dir: Optional[Path] = None,
    use_firecrawl: bool = True,
    bust_cache: bool = False,
) -> Iterator[Tuple[str, str]]:
    """Yield ``(image_url, source)`` pairs in priority order (build-time only)."""
    name = (name or "").strip()
    if not name:
        return

    yielded: set[str] = set()

    def offer(url: Optional[str], source: str) -> Iterator[Tuple[str, str]]:
        if not url or url in yielded:
            return
        yielded.add(url)
        yield url, source

    # Wikipedia (strict title match)
    for variant in _name_variants(name):
        wiki = _wikipedia(variant)
        title = wiki.get("title")
        confident = bool(title) and _title_matches(variant, title)
        if confident and wiki.get("thumb"):
            yield from offer(wiki["thumb"], "wikipedia")

    # Wikimedia Commons (broader)
    commons = _commons_image(name)
    if commons:
        yield from offer(commons, "wikipedia")

    # Openverse
    for q in _openverse_queries(name):
        ov = _openverse_photo(q)
        if ov:
            yield from offer(ov, "openverse")

    site = (official_site or "").strip()
    if not site:
        wiki = _wikipedia(name)
        title = wiki.get("title")
        if title and _title_matches(name, title) and wiki.get("wikidata"):
            from .location_enrich import _wikidata_official_site

            site = _wikidata_official_site(wiki["wikidata"]) or site
    if site:
        og = fetch_og_image(site)
        if og:
            yield from offer(og, "official")

    if use_firecrawl:
        for img in _yelp_images(name, cache_dir)[:5]:
            yield from offer(img, "yelp")
        for img in _tripadvisor_images(name, cache_dir)[:3]:
            yield from offer(img, "tripadvisor")
        for query in (f"{name} Chicago", f"{name} Chicago official website"):
            text = _firecrawl_search_text(query, cache_dir)
            for page_url in _firecrawl_result_urls(text)[:4]:
                if "yelp.com" in page_url:
                    continue
                for img in _image_urls_from_scrape(page_url, cache_dir)[:2]:
                    yield from offer(img, "official")
        for img in _web_search_images(name, cache_dir)[:6]:
            yield from offer(img, "web")

    if llm is not None and getattr(llm, "enabled", False):
        for url, src in llm.find_location_photo_candidates(name):
            yield from offer(url, src or "llm")

    cache_key = enricher.cache.make_key("card_enrich_v5", name)
    enricher.cache.put(cache_key, json.dumps({"image_url": None, "image_source": None}))


def has_local_image(media_items: List[dict], image_file: Optional[str] = None) -> bool:
    if image_file:
        return True
    return any(m.get("type") == "image" and m.get("file") for m in media_items)


def localize_youtube_poster(
    youtube_url: str,
    basename: str,
    download_image: Any,
) -> Optional[dict]:
    vid = youtube_video_id(youtube_url)
    if not vid:
        return None
    fname = download_image(youtube_thumbnail_url(vid), basename)
    if not fname:
        return None
    return {
        "type": "image",
        "source": "youtube",
        "attribution": pick_media_attribution("youtube", ""),
        "file": fname,
        "url": None,
    }
