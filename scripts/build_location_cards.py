#!/usr/bin/env python3
"""Pre-build per-location card enrichment so end users never call the API.

For every location in the warehouse this resolves, ONCE and offline:

  * ALL photos + videos from ChicagoDoes (Mapme), via Firecrawl + gallery pages
  * fallback photo only when ChicagoDoes has none: Wikipedia → Openverse → OG
  * a one-line specialty blurb (OpenAI, cached)
  * the venue's official website (Wikidata), when confidently matched

Usage:
    python scripts/build_location_cards.py
    python scripts/build_location_cards.py --media-only --force
    python scripts/build_location_cards.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chicagodoes_media import (  # noqa: E402
    fetch_mapme_media,
    gallery_download_candidates,
    pick_media_attribution,
)
from backend.llm_service import LLMService  # noqa: E402
from backend.location_enrich import LocationEnricher  # noqa: E402
from backend.main import _load_frames  # noqa: E402
from backend.media_fallback import (  # noqa: E402
    has_local_image,
    iter_fallback_image_candidates,
    iter_strict_extra_image_candidates,
    localize_youtube_poster,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
logger = logging.getLogger("build_cards")

OUT_JSON = ROOT / "data" / "location_cards.json"
IMAGES_DIR = ROOT / "data" / "location_images"
VIDEOS_DIR = ROOT / "data" / "location_videos"
FIRECRAWL_CACHE = ROOT / "data" / "firecrawl_cache"

_DL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
_IMG_EXT = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/webp": "webp", "image/gif": "gif",
}
_VID_EXT = {"video/mp4": "mp4", "video/webm": "webm"}
MAX_VIDEO_BYTES = 25_000_000
MIN_IMAGE_AREA = 90_000
MIN_IMAGE_SHORT_SIDE = 220
MIN_IMAGE_BYTES = 12_000
_BAD_IMAGE_URL_PARTS = (
    "static-resources.mapme.com/story/images/basemap",
    "maps.googleapis.com",
    "maps.gstatic.com",
    "google.com/maps",
    "mapme.com/wp-content/uploads/2020/08/webgl_logo",
)


def _sniff_image_ext(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            return None
        marker = data[i]
        i += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            return None
        seg_len = int.from_bytes(data[i:i + 2], "big")
        if seg_len < 2 or i + seg_len > len(data):
            return None
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }:
            h = int.from_bytes(data[i + 3:i + 5], "big")
            w = int.from_bytes(data[i + 5:i + 7], "big")
            return (w, h) if w and h else None
        i += seg_len
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    i = 12
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        chunk_len = int.from_bytes(data[i + 4:i + 8], "little")
        payload = data[i + 8:i + 8 + chunk_len]
        if fourcc == b"VP8X" and len(payload) >= 10:
            w = 1 + int.from_bytes(payload[4:7] + b"\x00", "little")
            h = 1 + int.from_bytes(payload[7:10] + b"\x00", "little")
            return w, h
        if fourcc == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            w = int.from_bytes(payload[6:8], "little") & 0x3FFF
            h = int.from_bytes(payload[8:10], "little") & 0x3FFF
            return (w, h) if w and h else None
        if fourcc == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            b0, b1, b2, b3 = payload[1], payload[2], payload[3], payload[4]
            w = 1 + (((b1 & 0x3F) << 8) | b0)
            h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return w, h
        i += 8 + chunk_len + (chunk_len % 2)
    return None


def _sniff_image_size(data: bytes, ext: str | None = None) -> tuple[int, int] | None:
    ext = ext or _sniff_image_ext(data)
    if ext == "png" and data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if ext in {"jpg", "jpeg"}:
        return _jpeg_size(data)
    if ext == "gif" and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if ext == "webp":
        return _webp_size(data)
    return None


def _image_quality_issue(data: bytes, ext: str | None = None) -> str | None:
    size = _sniff_image_size(data, ext)
    if size:
        w, h = size
        if w * h < MIN_IMAGE_AREA:
            return f"too small ({w}x{h})"
        if min(w, h) < MIN_IMAGE_SHORT_SIDE:
            return f"short side too small ({w}x{h})"
    if len(data) < MIN_IMAGE_BYTES:
        return f"tiny file ({len(data)} bytes)"
    return None


def _bad_image_url(url: str) -> bool:
    low = str(url or "").lower()
    return any(part in low for part in _BAD_IMAGE_URL_PARTS)


def _download_file(
    url: str,
    dest_dir: Path,
    basename: str,
    *,
    allowed_types: dict[str, str],
    max_bytes: int,
) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_DL_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            data = resp.read(max_bytes)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info("  download failed (%s): %s", url[:70], exc)
        return None
    ext = allowed_types.get(ctype)
    if not ext and allowed_types is _IMG_EXT:
        ext = _sniff_image_ext(data)
    if not ext or not data:
        return None
    fname = f"{basename}.{ext}"
    (dest_dir / fname).write_bytes(data)
    return fname


def _download_image(url: str, basename: str) -> str | None:
    if _bad_image_url(url):
        logger.info("  rejected image candidate (%s): map/logo asset", url[:70])
        return None
    headers = dict(_DL_HEADERS)
    if "yelpcdn.com" in url:
        headers["Referer"] = "https://www.yelp.com/"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            data = resp.read(8_000_000)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info("  download failed (%s): %s", url[:70], exc)
        return None
    ext = _IMG_EXT.get(ctype) or _sniff_image_ext(data)
    if not ext or not data:
        return None
    quality_issue = _image_quality_issue(data, ext)
    if quality_issue:
        logger.info("  rejected image candidate (%s): %s", url[:70], quality_issue)
        return None
    fname = f"{basename}.{ext}"
    (IMAGES_DIR / fname).write_bytes(data)
    return fname


def _download_image_best(url: str, basename: str) -> tuple[str | None, str | None]:
    """Try HD then standard gallery URL."""
    for candidate in gallery_download_candidates(url):
        fname = _download_image(candidate, basename)
        if fname:
            return fname, candidate
    return None, url


def _download_video(url: str, basename: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_DL_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            clen = resp.headers.get("Content-Length")
            if clen and int(clen) > MAX_VIDEO_BYTES:
                logger.info("  keeping remote video (too large to localize): %s", url[:70])
                return None
            data = resp.read(MAX_VIDEO_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info("  video download failed (%s): %s", url[:70], exc)
        return None
    if len(data) > MAX_VIDEO_BYTES:
        logger.info("  keeping remote video (over cap): %s", url[:70])
        return None
    ext = _VID_EXT.get(ctype)
    if not ext or not data:
        return None
    fname = f"{basename}.{ext}"
    (VIDEOS_DIR / fname).write_bytes(data)
    return fname


def _empty_media() -> dict[str, Any]:
    return {
        "media_items": [],
        "image_file": None,
        "image_url": None,
        "image_attribution": None,
        "image_source": None,
        "video_file": None,
        "video_url": None,
        "video_source": None,
        "media_source": "none",
    }


def _append_image_item(
    media_items: list[dict[str, Any]],
    *,
    fname: str | None,
    source: str,
    name: str,
    basename: str,
    remote_url: str | None,
) -> bool:
    """Download if needed and append a local image item. Returns True on success."""
    if fname:
        media_items.append({
            "type": "image",
            "source": source,
            "attribution": pick_media_attribution(source, name),
            "file": fname,
            "url": None,
        })
        return True
    if not remote_url:
        return False
    fname = _download_image(remote_url, basename)
    if not fname:
        return False
    media_items.append({
        "type": "image",
        "source": source,
        "attribution": pick_media_attribution(source, name),
        "file": fname,
        "url": None,
    })
    return True


def _fill_fallback_image(
    media_items: list[dict[str, Any]],
    *,
    lid: str,
    name: str,
    enricher: LocationEnricher,
    llm: LLMService,
    official_site: str | None,
    use_firecrawl: bool,
    bust_cache: bool,
) -> None:
    if has_local_image(media_items):
        return
    for idx, (url, source) in enumerate(iter_fallback_image_candidates(
        name,
        enricher=enricher,
        official_site=official_site,
        llm=llm,
        cache_dir=FIRECRAWL_CACHE,
        use_firecrawl=use_firecrawl,
        bust_cache=bust_cache,
    )):
        if _append_image_item(
            media_items,
            fname=None,
            source=source,
            name=name,
            basename=f"{lid}_f{idx}",
            remote_url=url,
        ):
            logger.info("  fallback image: %s via %s", name, source)
            return


def _ensure_local_images(
    media_items: list[dict[str, Any]],
    *,
    lid: str,
) -> None:
    """Re-download any image items that still point at remote URLs."""
    for i, m in enumerate(media_items):
        if m.get("type") != "image" or m.get("file"):
            continue
        remote = m.get("url")
        if not remote:
            continue
        fname = _download_image(str(remote), f"{lid}_i{i}")
        if fname:
            m["file"] = fname
            m["url"] = None


def _local_media_url(item: dict[str, Any]) -> str | None:
    if item.get("file"):
        return str(item["file"])
    if item.get("url"):
        return str(item["url"])
    return None


def _image_count(media_items: list[dict[str, Any]]) -> int:
    return sum(1 for m in media_items if m.get("type") == "image" and _local_media_url(m))


def _augment_thin_media(
    prev: dict[str, Any],
    *,
    lid: str,
    name: str,
    enricher: LocationEnricher,
    target_images: int,
    max_extra_images: int,
    use_firecrawl: bool,
    bust_cache: bool,
) -> dict[str, Any]:
    """Add exact-match images to cards with sparse media, preserving existing media."""
    out = dict(prev or {})
    media_items = [dict(m) for m in (prev.get("media_items") or [])]
    if not media_items:
        media_items = []
        if prev.get("image_file"):
            media_items.append({
                "type": "image",
                "source": prev.get("image_source") or prev.get("media_source"),
                "attribution": prev.get("image_attribution"),
                "file": prev.get("image_file"),
                "url": None,
            })
        if prev.get("video_file") or prev.get("video_url"):
            media_items.append({
                "type": "video",
                "source": prev.get("video_source") or prev.get("media_source"),
                "attribution": prev.get("video_attribution") or prev.get("image_attribution"),
                "file": prev.get("video_file"),
                "url": prev.get("video_url"),
            })

    known = {_local_media_url(m) for m in media_items if _local_media_url(m)}
    added = 0

    def append_image(url: str, source: str, basename: str) -> bool:
        nonlocal added
        if not url or url in known or _bad_image_url(url):
            return False
        fname = _download_image(url, basename)
        if not fname or fname in known:
            return False
        media_items.append({
            "type": "image",
            "source": source,
            "attribution": pick_media_attribution(source, name),
            "file": fname,
            "url": None,
        })
        known.add(url)
        known.add(fname)
        added += 1
        return True

    # Exact ChicagoDoes / Mapme gallery by location_id is the safest source.
    if _image_count(media_items) < target_images:
        mapme = fetch_mapme_media(lid, cache_dir=FIRECRAWL_CACHE, use_firecrawl=use_firecrawl)
        for img_url in mapme.images:
            if _image_count(media_items) >= target_images or added >= max_extra_images:
                break
            for candidate in gallery_download_candidates(img_url):
                if append_image(candidate, "chicagodoes", f"{lid}_aug{added}"):
                    break

    if _image_count(media_items) < target_images and added < max_extra_images:
        pa = enricher.photo_and_site(name)
        official_site = (prev or {}).get("official_site") or pa.get("official_site")
        out["official_site"] = official_site
        for url, source in iter_strict_extra_image_candidates(
            name,
            official_site=official_site,
            cache_dir=FIRECRAWL_CACHE,
            use_firecrawl=use_firecrawl,
        ):
            if _image_count(media_items) >= target_images or added >= max_extra_images:
                break
            append_image(url, source, f"{lid}_aug{added}")

    _ensure_local_images(media_items, lid=lid)
    _finalize_media(media_items, out)
    if added and (not out.get("media_source") or out.get("media_source") == "none"):
        out["media_source"] = next(
            (m.get("source") for m in media_items if m.get("source")), "web",
        )
    out["augmentation_attempted"] = True
    out["augmentation_added"] = int(out.get("augmentation_added") or 0) + added
    return out


def _finalize_media(media_items: list[dict[str, Any]], out: dict[str, Any]) -> None:
    """Drop non-local images; never persist remote image URLs for runtime."""
    cleaned: list[dict[str, Any]] = []
    for m in media_items:
        if m.get("type") == "image" and not m.get("file"):
            continue
        if m.get("type") == "video" and not m.get("file") and not m.get("url"):
            continue
        cleaned.append(m)
    media_items[:] = cleaned

    first_image = next((m for m in media_items if m["type"] == "image"), None)
    first_video = next((m for m in media_items if m["type"] == "video"), None)
    out["media_items"] = media_items
    out["image_file"] = first_image["file"] if first_image else None
    out["image_url"] = None
    out["image_attribution"] = (
        first_image.get("attribution") if first_image else out.get("image_attribution")
    )
    out["video_file"] = (
        first_video["file"] if first_video and first_video.get("file") else None
    )
    out["video_url"] = (
        first_video["url"] if first_video and not first_video.get("file") else None
    )
    if first_image and (not out.get("media_source") or out.get("media_source") == "none"):
        out["media_source"] = first_image.get("source")
        out["image_source"] = first_image.get("source")


def _resolve_media(
    lid: str,
    name: str,
    enricher: LocationEnricher,
    llm: LLMService,
    *,
    use_firecrawl: bool,
    existing: dict | None,
    bust_cache: bool = False,
) -> dict[str, Any]:
    out = _empty_media()
    pa = enricher.photo_and_site(name)
    official_site = (existing or {}).get("official_site") or pa.get("official_site")

    mapme = fetch_mapme_media(lid, cache_dir=FIRECRAWL_CACHE, use_firecrawl=use_firecrawl)
    media_items: list[dict[str, Any]] = []

    if mapme.source == "chicagodoes":
        attr = pick_media_attribution("chicagodoes", name)
        for i, vid_url in enumerate(mapme.videos):
            fname = _download_video(vid_url, f"{lid}_v{i}")
            media_items.append({
                "type": "video",
                "source": "chicagodoes",
                "attribution": attr,
                "file": fname,
                "url": None if fname else vid_url,
            })
        for i, yt_url in enumerate(mapme.youtube_urls):
            media_items.append({
                "type": "video",
                "source": "chicagodoes",
                "attribution": attr,
                "file": None,
                "url": yt_url,
            })
            poster = localize_youtube_poster(yt_url, f"{lid}_yt{i}", _download_image)
            if poster and not has_local_image(media_items):
                media_items.insert(0, poster)
        for i, img_url in enumerate(mapme.images):
            fname, _ = _download_image_best(img_url, f"{lid}_i{i}")
            if fname:
                media_items.append({
                    "type": "image",
                    "source": "chicagodoes",
                    "attribution": attr,
                    "file": fname,
                    "url": None,
                })
        out["media_source"] = "chicagodoes"
        out["image_source"] = "chicagodoes"
        out["image_attribution"] = attr if media_items else None
        out["video_source"] = "chicagodoes" if any(
            m["type"] == "video" for m in media_items
        ) else None

    _fill_fallback_image(
        media_items,
        lid=lid,
        name=name,
        enricher=enricher,
        llm=llm,
        official_site=official_site,
        use_firecrawl=use_firecrawl,
        bust_cache=bust_cache,
    )

    if media_items and (not out.get("media_source") or out.get("media_source") == "none"):
        src = next((m.get("source") for m in media_items if m.get("source")), "web")
        out["media_source"] = src
        out["image_source"] = src

    _ensure_local_images(media_items, lid=lid)
    out["official_site"] = official_site
    _finalize_media(media_items, out)
    return out


def _needs_media_refresh(prev: dict, force: bool) -> bool:
    if force:
        return True
    items = prev.get("media_items") or []
    if not items:
        return True
    if not has_local_image(items, prev.get("image_file")):
        return True
    if any(m.get("type") == "image" and not m.get("file") for m in items):
        return True
    if prev.get("image_url") and not prev.get("image_file"):
        return True
    if prev.get("media_source") != "chicagodoes":
        return True
    cd_items = [m for m in items if m.get("source") == "chicagodoes"]
    if len(cd_items) <= 1 and not any(m.get("type") == "video" for m in cd_items):
        return True
    return False


def _has_local_image_file(prev: dict) -> bool:
    items = prev.get("media_items") or []
    for m in items:
        if m.get("type") == "image" and m.get("file"):
            if (IMAGES_DIR / str(m["file"])).exists():
                return True
    img = prev.get("image_file")
    return bool(img and (IMAGES_DIR / str(img)).exists())


def _local_image_quality_issue(fname: str | None) -> str | None:
    if not fname:
        return "missing image file"
    path = IMAGES_DIR / str(fname)
    if not path.exists():
        return "missing image file"
    try:
        data = path.read_bytes()
    except OSError:
        return "unreadable image file"
    return _image_quality_issue(data, _sniff_image_ext(data))


def _first_image_file(prev: dict) -> str | None:
    for m in prev.get("media_items") or []:
        if m.get("type") == "image" and m.get("file"):
            return str(m["file"])
    img = prev.get("image_file")
    return str(img) if img else None


def _has_bad_local_video(prev: dict) -> bool:
    for m in prev.get("media_items") or []:
        if m.get("type") != "video" or not m.get("file"):
            continue
        path = VIDEOS_DIR / str(m["file"])
        if not path.exists():
            return True
        if path.stat().st_size >= MAX_VIDEO_BYTES:
            return True
    return False


def _card_needs_quality_refresh(prev: dict) -> bool:
    if not prev:
        return True
    first_image = _first_image_file(prev)
    if _local_image_quality_issue(first_image):
        return True
    if _has_bad_local_video(prev):
        return True
    if prev.get("media_source") == "none":
        return True
    return False


def _card_needs_augmentation(prev: dict, target_images: int) -> bool:
    if not prev:
        return False
    if (
        (prev.get("augmentation_attempted") or "augmentation_added" in prev)
        and not int(prev.get("augmentation_added") or 0)
    ):
        return False
    items = prev.get("media_items") or []
    if not has_local_image(items, prev.get("image_file")):
        return False
    return _image_count(items) < target_images and not any(
        m.get("type") == "video" for m in items
    )


def _mapme_has_media(lid: str) -> bool:
    mapme = fetch_mapme_media(lid, cache_dir=FIRECRAWL_CACHE, use_firecrawl=True)
    return bool(mapme.images or mapme.videos or mapme.youtube_urls)


def _card_needs_local_image(prev: dict) -> bool:
    return not _has_local_image_file(prev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild all locations")
    ap.add_argument("--limit", type=int, default=None, help="process at most N locations")
    ap.add_argument("--blurb-model", default="gpt-5.1")
    ap.add_argument("--blurbs-only", action="store_true")
    ap.add_argument("--media-only", action="store_true")
    ap.add_argument("--missing-only", action="store_true", help="only cards without a local image")
    ap.add_argument(
        "--quality-gaps",
        action="store_true",
        help="only cards with a tiny first image, stale source metadata, or capped local videos",
    )
    ap.add_argument(
        "--mapme-gaps",
        action="store_true",
        help="only cards missing a local image but ChicagoDoes Mapme has media",
    )
    ap.add_argument(
        "--augment-thin-media",
        action="store_true",
        help="append exact-match images to cards with fewer than --target-images and no video",
    )
    ap.add_argument("--target-images", type=int, default=3)
    ap.add_argument("--max-extra-images", type=int, default=3)
    ap.add_argument("--no-firecrawl", action="store_true")
    ap.add_argument("--bust-enrich-cache", action="store_true")
    args = ap.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    frames = _load_frames()
    locations = frames.locations
    logger.info("Loaded %d locations", len(locations))

    llm = LLMService(model=args.blurb_model)
    enricher = LocationEnricher(cache_path=str(ROOT / "data" / "enrich_cache.sqlite"))

    store: dict = {}
    if OUT_JSON.exists() and not args.force:
        try:
            store = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            logger.info("Loaded existing store — %d cards", len(store))
        except ValueError:
            store = {}

    processed = 0
    for _, row in locations.iterrows():
        lid = str(row["location_id"])
        name = str(row["location_name"])
        primary = row.get("primary_category")
        cats = list(row.get("categories") or [])
        prev = store.get(lid) or {}

        if args.blurbs_only:
            if lid not in store and not args.force:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            summ = llm.summarize_location(name, primary, cats)
            store.setdefault(lid, {"location_name": name})
            store[lid]["blurb"] = summ["blurb"]
            store[lid]["blurb_source"] = summ["source"]
            processed += 1
            if processed % 10 == 0:
                OUT_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        if args.media_only:
            if args.augment_thin_media:
                if not _card_needs_augmentation(prev, args.target_images):
                    continue
            elif args.quality_gaps:
                if not _card_needs_quality_refresh(prev):
                    continue
            elif args.mapme_gaps:
                if _has_local_image_file(prev):
                    continue
                if not _mapme_has_media(lid):
                    continue
            elif args.missing_only:
                if not _card_needs_local_image(prev):
                    continue
            elif not _needs_media_refresh(prev, args.force):
                continue
            if args.limit is not None and processed >= args.limit:
                break
            if args.augment_thin_media:
                before = _image_count(prev.get("media_items") or [])
                media = _augment_thin_media(
                    prev,
                    lid=lid,
                    name=name,
                    enricher=enricher,
                    target_images=max(1, args.target_images),
                    max_extra_images=max(1, args.max_extra_images),
                    use_firecrawl=not args.no_firecrawl,
                    bust_cache=args.bust_enrich_cache,
                )
                after = _image_count(media.get("media_items") or [])
                if after <= before:
                    logger.info("  no exact extra media accepted for %s", name)
            else:
                media = _resolve_media(
                    lid, name, enricher, llm,
                    use_firecrawl=not args.no_firecrawl,
                    existing=prev,
                    bust_cache=args.bust_enrich_cache,
                )
            store.setdefault(lid, {"location_name": name})
            store[lid].update(media)
            if not store[lid].get("blurb"):
                summ = llm.summarize_location(name, primary, cats)
                store[lid]["blurb"] = summ["blurb"]
                store[lid]["blurb_source"] = summ["source"]
            processed += 1
            logger.info(
                "[%d] %s | media_items=%d src=%s",
                processed, name, len(media["media_items"]), media["media_source"],
            )
            if processed % 3 == 0:
                OUT_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        if not args.force and lid in store and store[lid].get("blurb") and store[lid].get("media_items"):
            continue
        if args.limit is not None and processed >= args.limit:
            break

        media = _resolve_media(
            lid, name, enricher, llm,
            use_firecrawl=not args.no_firecrawl,
            existing=prev,
            bust_cache=args.bust_enrich_cache,
        )
        summ = llm.summarize_location(name, primary, cats)
        store[lid] = {
            "location_name": name,
            **media,
            "blurb": summ["blurb"],
            "blurb_source": summ["source"],
        }
        processed += 1
        logger.info(
            "[%d] %s | media_items=%d src=%s",
            processed, name, len(media["media_items"]), media["media_source"],
        )
        if processed % 3 == 0:
            OUT_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Done. Wrote %d cards (%d processed).", len(store), processed)


if __name__ == "__main__":
    main()
