"""Resolve photos and videos from ChicagoDoes (Mapme) and fallback sources.

Priority for offline card building:
  1. ChicagoDoes / Mapme location page + gallery pages (Firecrawl)
  2. Wikipedia → Openverse (via LocationEnricher) — only when ChicagoDoes has none
  3. Official-site Open Graph image
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAPME_LOCATION_BASE = (
    "https://viewer.mapme.com/chicagodoesinteractivevideomaps/location/"
)
_MEDIA_HOST = "media.mapme.com"
MAX_GALLERY_PAGES = 15
MAPME_VIDEO_HOST = _MEDIA_HOST

_RE_GALLERY = re.compile(
    r"https://media\.mapme\.com/places/([0-9a-f-]{36})/gallery/([0-9a-f-]{36})(?:\.(?:th|hd))?",
    re.I,
)
_RE_VIDEO = re.compile(
    r"https://media\.mapme\.com/places/([0-9a-f-]{36})/videos/([0-9a-f-]{36})(?:\.thumb)?",
    re.I,
)
_RE_YOUTUBE = re.compile(
    r"(?:img\.youtube\.com/vi/|youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})",
    re.I,
)
_RE_OG_IMAGE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
    re.I,
)


@dataclass
class MapmeMedia:
    """All media discovered on ChicagoDoes Mapme pages for one venue."""
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    youtube_urls: List[str] = field(default_factory=list)
    source: str = "none"

    @property
    def best_image(self) -> Optional[str]:
        return self.images[0] if self.images else None

    @property
    def best_video(self) -> Optional[str]:
        return self.videos[0] if self.videos else None


def mapme_location_url(location_id: str) -> str:
    return MAPME_LOCATION_BASE + str(location_id).strip()


def _gallery_hd_url(location_id: str, asset_id: str) -> str:
    return f"https://{_MEDIA_HOST}/places/{location_id}/gallery/{asset_id}.hd"


def _gallery_url(location_id: str, asset_id: str) -> str:
    return f"https://{_MEDIA_HOST}/places/{location_id}/gallery/{asset_id}"


def _video_url(location_id: str, asset_id: str) -> str:
    return f"https://{_MEDIA_HOST}/places/{location_id}/videos/{asset_id}"


def _youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _scrape_text_chunks(data: Dict[str, Any]) -> str:
    chunks: List[str] = [str(data.get("markdown") or "")]
    links = data.get("links") or []
    if isinstance(links, list):
        chunks.extend(str(x) for x in links)
    for key in ("html", "rawHtml", "content"):
        if data.get(key):
            chunks.append(str(data[key]))
    return "\n".join(chunks)


def _parse_mapme_text(text: str, location_id: str) -> MapmeMedia:
    """Extract gallery images, Mapme videos, and YouTube embeds from page text."""
    images_by_id: Dict[str, str] = {}
    videos_by_id: Dict[str, str] = {}
    youtube: List[str] = []
    seen_yt: set[str] = set()

    for m in _RE_GALLERY.finditer(text or ""):
        lid, aid = m.group(1).lower(), m.group(2).lower()
        if lid != location_id.lower():
            continue
        images_by_id[aid] = _gallery_hd_url(lid, aid)

    for m in _RE_VIDEO.finditer(text or ""):
        lid, aid = m.group(1).lower(), m.group(2).lower()
        if lid != location_id.lower():
            continue
        videos_by_id[aid] = _video_url(lid, aid)

    for m in _RE_YOUTUBE.finditer(text or ""):
        vid = m.group(1)
        if vid not in seen_yt:
            seen_yt.add(vid)
            youtube.append(_youtube_watch_url(vid))

    source = "chicagodoes" if (images_by_id or videos_by_id or youtube) else "none"
    return MapmeMedia(
        images=list(images_by_id.values()),
        videos=list(videos_by_id.values()),
        youtube_urls=youtube,
        source=source,
    )


def _merge_mapme_media(base: MapmeMedia, extra: MapmeMedia) -> None:
    seen_img = set(base.images)
    for url in extra.images:
        if url not in seen_img:
            seen_img.add(url)
            base.images.append(url)
    seen_vid = set(base.videos)
    for url in extra.videos:
        if url not in seen_vid:
            seen_vid.add(url)
            base.videos.append(url)
    seen_yt = set(base.youtube_urls)
    for url in extra.youtube_urls:
        if url not in seen_yt:
            seen_yt.add(url)
            base.youtube_urls.append(url)
    if base.images or base.videos or base.youtube_urls:
        base.source = "chicagodoes"


def _firecrawl_scrape(url: str, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Scrape a URL with the Firecrawl CLI; cache JSON responses on disk."""
    cache_dir = cache_dir or Path("data/firecrawl_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url)[-120:]
    cache_file = cache_dir / f"{safe}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            pass

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name

    try:
        proc = subprocess.run(
            ["firecrawl", "scrape", url, "--format", "markdown,links", "-o", out_path],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if proc.returncode != 0:
            logger.info("firecrawl scrape failed for %s: %s", url[:80], proc.stderr[:200])
            return {}
        raw = Path(out_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        cache_file.write_text(raw, encoding="utf-8")
        return data
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
        logger.info("firecrawl scrape error for %s: %s", url[:80], exc)
        return {}
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


def fetch_mapme_media(
    location_id: str,
    *,
    cache_dir: Optional[Path] = None,
    use_firecrawl: bool = True,
) -> MapmeMedia:
    """Scrape the ChicagoDoes Mapme location page and gallery for all media."""
    lid = str(location_id or "").strip().lower()
    if not lid:
        return MapmeMedia()

    url = mapme_location_url(lid)
    merged = MapmeMedia()

    if use_firecrawl:
        main_data = _firecrawl_scrape(url, cache_dir=cache_dir)
        if main_data:
            _merge_mapme_media(merged, _parse_mapme_text(_scrape_text_chunks(main_data), lid))

        if merged.source == "chicagodoes":
            prev_img_count = 0
            for page_num in range(1, MAX_GALLERY_PAGES + 1):
                gallery_url = f"{url}/gallery/{page_num}"
                gdata = _firecrawl_scrape(gallery_url, cache_dir=cache_dir)
                if not gdata:
                    if page_num > 1:
                        break
                    continue
                extra = _parse_mapme_text(_scrape_text_chunks(gdata), lid)
                before = len(merged.images) + len(merged.videos) + len(merged.youtube_urls)
                _merge_mapme_media(merged, extra)
                after = len(merged.images) + len(merged.videos) + len(merged.youtube_urls)
                if page_num > 1 and after == before and after == prev_img_count:
                    break
                prev_img_count = after

    if merged.source == "none":
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ChicagoDoesRecsys/1.0 (capstone demo)"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                _merge_mapme_media(
                    merged, _parse_mapme_text(resp.read().decode("utf-8", "ignore"), lid),
                )
        except (urllib.error.URLError, OSError, ValueError):
            pass

    return merged


def gallery_download_candidates(image_url: str) -> List[str]:
    """Prefer HD gallery assets, then standard size."""
    url = str(image_url or "").strip()
    if not url:
        return []
    if url.endswith(".hd"):
        base = url[:-3]
        return [url, base]
    return [f"{url}.hd", url]


def fetch_og_image(official_site: str) -> Optional[str]:
    """Best-effort Open Graph image from a venue's official website."""
    url = str(official_site or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ChicagoDoesRecsys/1.0 (capstone demo)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read(250_000).decode("utf-8", "ignore")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    m = _RE_OG_IMAGE.search(html)
    if not m:
        return None
    img = m.group(1).strip()
    if img.startswith(("http://", "https://")):
        return img
    return None


def pick_media_attribution(source: str, location_name: str) -> str:
    if source == "chicagodoes":
        return f"Photo/video: ChicagoDoes ({location_name})"
    if source == "wikipedia":
        return f"Photo: Wikipedia ({location_name})"
    if source == "openverse":
        return "Photo: Openverse (Creative Commons)"
    if source == "official":
        return f"Photo: Official website ({location_name})"
    if source == "youtube":
        return "Photo: YouTube thumbnail"
    if source == "yelp":
        return f"Photo: Yelp ({location_name})"
    if source == "tripadvisor":
        return f"Photo: TripAdvisor ({location_name})"
    if source in {"web", "llm"}:
        return f"Photo: Web ({location_name})"
    return ""
