"""LLM wrapper used by the ChicagoDoes recommender.

Capabilities
------------
1. `describe_location(...)`     - 90-word concierge blurb + tips for one place.
2. `parse_intent(...)`          - free text -> structured RecommendRequest.
3. `generate_itinerary(...)`    - full day-by-day plan from recommendation pool.
4. `refine_request(...)`        - natural-language tweak -> request delta.

All four methods share a single `_chat_json` / `_chat_text` core that:

* Routes through the OpenAI-compatible client (default: local Ollama).
* Falls back to deterministic templates when the SDK or key is absent.
* Caches every successful response in a SQLite file keyed by
  (model, system_prompt, user_prompt, response_format) so live demos
  are fast and reproducible.

The deterministic fallback paths are real, not stubs: they keep the
website fully functional without an API key, which is critical for
classroom demos and for evaluating the recommender in isolation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class _LLMCache:
    """Tiny SQLite cache. Safe to share between threads.

    Schema:
        cache(key TEXT PRIMARY KEY, value TEXT, created_at REAL)
    """

    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        if not self.enabled:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)

    @staticmethod
    def make_key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        with self._conn() as c:
            row = c.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cache(key, value, created_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a Chicago travel concierge writing for the ChicagoDoes interactive "
    "map. Be concrete, friendly, and respect the JSON schema the user requests. "
    "Never invent specific addresses, prices, or hours."
)

# Default: local Ollama (OpenAI-compatible). Override via .env for OpenAI / Groq / etc.
DEFAULT_LLM_API_KEY = "ollama"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "llama3.1:8b"


class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", DEFAULT_LLM_API_KEY)
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL)
        self.base_url = _resolve_llm_base_url(os.getenv("OPENAI_BASE_URL"))
        self.cache = _LLMCache(
            path=cache_path or os.getenv("LLM_CACHE_PATH", "data/llm_cache.sqlite"),
            enabled=_truthy(os.getenv("LLM_CACHE_ENABLED", "1")) if cache_enabled is None else cache_enabled,
        )
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI  # noqa: WPS433
                client_kw: Dict[str, Any] = {
                    "api_key": self.api_key,
                    "timeout": float(os.getenv("LLM_TIMEOUT_SEC", "180")),
                }
                if self.base_url:
                    client_kw["base_url"] = self.base_url.rstrip("/")
                self._client = OpenAI(**client_kw)
                logger.info(
                    "LLMService enabled (model=%s, base_url=%s, cache=%s)",
                    self.model,
                    self.base_url or "https://api.openai.com/v1",
                    self.cache.enabled,
                )
                if self.base_url and not _ping_llm_server(self.base_url):
                    logger.warning(
                        "LLM server not reachable at %s — start it in another terminal: "
                        "ollama serve",
                        self.base_url,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI SDK unavailable, LLM disabled: %s", exc)
                self._client = None
        else:
            logger.info("LLM API key not set, LLM disabled (fallback mode).")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ==================================================================== #
    # 1) Location description
    # ==================================================================== #
    def describe_location(
        self,
        location_name: str,
        primary_category: Optional[str],
        categories: Sequence[str],
        style: str = "friendly",
        warehouse_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fb = self._fallback_describe(location_name, primary_category, categories, warehouse_context)
        if not self.enabled:
            return fb

        ctx = warehouse_context or {}
        prompt = (
            "You are a Chicago travel concierge. Write concrete, visitor-useful copy "
            f"in a {style} tone. Return STRICT JSON only:\n"
            "{\n"
            '  "description": string,     // 2-3 sentences: what it is, vibe, why go\n'
            '  "highlights": [string],    // 3-5 specific bullets (signature items, exhibits, architecture, etc.)\n'
            '  "tips": [string],          // 2-3 practical tips (timing, booking, dress, transit)\n'
            '  "neighborhood": string|null,  // Chicago area name if known, else null\n'
            '  "best_for": string|null,      // e.g. "date night", "families", "first-time visitors"\n'
            '  "website_url": string|null    // official venue URL ONLY if you are confident; else null\n'
            "}\n"
            "Rules:\n"
            "- Be specific to THIS named place — no generic filler.\n"
            "- Do NOT invent URLs; use null for website_url unless you know the official site.\n"
            "- Do not include markdown or extra keys.\n\n"
            f"Location name: {location_name}\n"
            f"Primary category: {primary_category or 'unknown'}\n"
            f"All categories: {', '.join(categories) if categories else 'unknown'}\n"
            f"Warehouse engagement (real data, may inform tone): {json.dumps(ctx)}\n"
        )
        try:
            payload = self._chat_json(prompt, schema_hint="describe")
            website = _clean_url(payload.get("website_url"))
            return {
                "description": str(payload.get("description") or "").strip() or fb["description"],
                "highlights": [
                    str(h).strip() for h in (payload.get("highlights") or []) if str(h).strip()
                ][:6],
                "tips": [str(t).strip() for t in (payload.get("tips") or []) if str(t).strip()][:5],
                "neighborhood": _nullish_str(payload.get("neighborhood")),
                "best_for": _nullish_str(payload.get("best_for")),
                "website_url": website,
                "source": "llm",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("describe_location failed, falling back: %s", exc)
            return fb

    # ==================================================================== #
    # 2) Natural-language intent parser
    # ==================================================================== #
    def parse_intent(
        self,
        free_text: str,
        valid_categories: Sequence[str],
    ) -> Dict[str, Any]:
        """Turn free-form text into a partial RecommendRequest.

        The LLM is constrained to pick `interests` / `avoid_categories`
        from `valid_categories` and to choose `traveler_type` and `vibe`
        from a fixed enum. We post-validate the response so even a
        misbehaving model cannot corrupt the request schema.
        """
        valid_travelers = ["solo", "couple", "family", "group", "business"]
        valid_vibes = ["chill", "adventurous", "foodie", "nightlife", "cultural", "outdoorsy"]

        fallback = self._fallback_parse_intent(free_text, valid_categories, valid_travelers, valid_vibes)
        if not self.enabled or not free_text.strip():
            return {**fallback, "source": "fallback"}

        prompt = (
            "Extract trip preferences from the visitor's message. Return STRICT JSON "
            "with this exact schema:\n"
            "{\n"
            '  "interests": [string],          // subset of allowed_categories\n'
            '  "avoid_categories": [string],   // subset of allowed_categories\n'
            '  "traveler_type": string|null,   // one of: ' + ", ".join(valid_travelers) + "\n"
            '  "vibe": string|null,            // one of: ' + ", ".join(valid_vibes) + "\n"
            '  "trip_days": int,               // 1..7\n'
            '  "summary": string               // 1-sentence echo of what you understood\n'
            "}\n"
            "Rules:\n"
            "- Only use categories from allowed_categories EXACTLY as spelled.\n"
            "- If the visitor expresses dislike for something, put it in avoid_categories.\n"
            "- If a field is not mentioned, use a sensible default (trip_days=2, others null/[]).\n"
            "- Do NOT include any markdown or commentary.\n\n"
            f"allowed_categories: {json.dumps(list(valid_categories))}\n"
            f"visitor_message: {json.dumps(free_text)}\n"
        )

        try:
            payload = self._chat_json(prompt, schema_hint="intent")
            cleaned = _clean_intent_payload(payload, valid_categories, valid_travelers, valid_vibes, fallback)
            cleaned["source"] = "llm"
            return cleaned
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse_intent failed, falling back: %s", exc)
            return {**fallback, "source": "fallback"}

    # ==================================================================== #
    # 3) Itinerary generation (structure + copy in one call)
    # ==================================================================== #
    def generate_itinerary(
        self,
        candidates: Sequence[Dict],
        trip_days: int,
        interests: Sequence[str],
        inferred_interests: Sequence[str],
        avoid_categories: Sequence[str],
        traveler_type: Optional[str] = None,
        vibe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a multi-day plan using only the provided location_ids."""
        empty = {
            "summary": "",
            "days": [],
            "source": "fallback",
        }
        if not candidates:
            empty["summary"] = "No recommendations to plan from."
            return empty

        if not self.enabled:
            return {
                **empty,
                "summary": (
                    "AI itinerary needs an OpenAI API key. "
                    "Start Ollama (ollama serve) and pull the model to enable day planning."
                ),
            }

        days_n = max(1, min(int(trip_days), 7))
        avoid_l = [str(c).strip() for c in avoid_categories if c]
        combined_interests = list(dict.fromkeys([*interests, *inferred_interests]))

        pool = _trim_candidates_for_itinerary(candidates, days_n)
        allowed_ids = [str(c.get("location_id", "")) for c in pool if c.get("location_id")]
        prompts = [
            _itinerary_prompt_full(
                pool, days_n, allowed_ids, combined_interests, avoid_l,
                traveler_type, vibe,
            ),
            _itinerary_prompt_compact(
                pool, days_n, allowed_ids, combined_interests, avoid_l,
            ),
        ]
        last_exc: Optional[Exception] = None
        local_ollama = bool(self.base_url and "11434" in self.base_url)
        for attempt, prompt in enumerate(prompts):
            try:
                logger.info(
                    "generate_itinerary attempt %d/%d (model=%s)",
                    attempt + 1,
                    len(prompts),
                    self.model,
                )
                payload = self._chat_json(
                    prompt,
                    schema_hint=f"itinerary-v{attempt}",
                    cache_ok=_itinerary_payload_ok,
                )
                days = _extract_itinerary_days(payload)
                if not days:
                    raise ValueError("LLM returned no days with stops")
                return {
                    "summary": str(payload.get("summary") or "").strip()
                    or f"A {days_n}-day plan from your top Chicago picks.",
                    "days": days,
                    "source": "llm",
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "generate_itinerary attempt %d failed: %s", attempt + 1, exc,
                )
                # Local 8B models rarely succeed on a second huge JSON call; skip
                # another 1–3 min wait and let the API use deterministic scheduling.
                if local_ollama and attempt == 0:
                    logger.info(
                        "Skipping second LLM itinerary attempt on local Ollama "
                        "(using automatic schedule from Top picks)."
                    )
                    break
        logger.warning("generate_itinerary exhausted retries: %s", last_exc)
        return {
            **empty,
            "summary": _user_facing_llm_error(
                last_exc or ValueError("no valid itinerary"),
                "AI could not format an itinerary. We will use a data-driven layout instead.",
            ),
            "notice": (
                "The local model did not return valid JSON. "
                "Try again, or use the automatic schedule from your Top picks."
            ),
        }

    # ==================================================================== #
    # 4) Refinement: NL tweak -> request delta
    # ==================================================================== #
    def refine_request(
        self,
        instruction: str,
        previous_request: Dict[str, Any],
        valid_categories: Sequence[str],
    ) -> Dict[str, Any]:
        """Return a *delta* dict that should be merged into `previous_request`.

        We deliberately keep the recommender as the source of truth: the
        LLM never produces the final ranking, it only translates the
        user's English tweak into structured preference changes.

        Output shape:
            {
              "add_interests":   [str],
              "remove_interests":[str],
              "add_avoid":       [str],
              "remove_avoid":    [str],
              "set_vibe":        str|null,
              "set_traveler":    str|null,
              "set_trip_days":   int|null,
              "comment":         str
            }
        """
        empty = {
            "add_interests": [], "remove_interests": [],
            "add_avoid": [], "remove_avoid": [],
            "set_vibe": None, "set_traveler": None,
            "set_trip_days": None,
            "comment": "",
            "source": "fallback",
        }
        if not self.enabled or not instruction.strip():
            return empty

        valid_travelers = ["solo", "couple", "family", "group", "business"]
        valid_vibes = ["chill", "adventurous", "foodie", "nightlife", "cultural", "outdoorsy"]

        prompt = (
            "The visitor wants to adjust their itinerary. Translate their instruction "
            "into a structured DELTA over the current preferences. Return STRICT JSON "
            "matching this schema exactly:\n"
            "{\n"
            '  "add_interests":   [string],\n'
            '  "remove_interests":[string],\n'
            '  "add_avoid":       [string],\n'
            '  "remove_avoid":    [string],\n'
            '  "set_vibe":        string|null,\n'
            '  "set_traveler":    string|null,\n'
            '  "set_trip_days":   integer|null,\n'
            '  "comment":         string\n'
            "}\n"
            "Rules:\n"
            "- Categories must come from allowed_categories EXACTLY.\n"
            "- vibe in: " + ", ".join(valid_vibes) + " or null.\n"
            "- traveler in: " + ", ".join(valid_travelers) + " or null.\n"
            "- Use null / [] for anything the instruction does not change.\n\n"
            f"allowed_categories: {json.dumps(list(valid_categories))}\n"
            f"current_preferences: {json.dumps(previous_request)}\n"
            f"instruction: {json.dumps(instruction)}\n"
        )
        try:
            payload = self._chat_json(prompt, schema_hint="refine")
            return _clean_refine_payload(payload, valid_categories, valid_travelers, valid_vibes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("refine_request failed, falling back: %s", exc)
            return empty

    # ==================================================================== #
    # Server-side helper to also produce a personalised rationale
    # ==================================================================== #
    def rationale_for(
        self,
        location_name: str,
        location_categories: Sequence[str],
        user_interests: Sequence[str],
        vibe: Optional[str],
        traveler_type: Optional[str],
        *,
        inferred_interests: Optional[Sequence[str]] = None,
        rank: Optional[int] = None,
        system_reason: Optional[str] = None,
        evidence_summary: Optional[str] = None,
        final_score: Optional[float] = None,
        is_trending: bool = False,
        is_hot_spot: bool = False,
        similarity_score: Optional[float] = None,
        item_collab_score: Optional[float] = None,
        trending_score: Optional[float] = None,
    ) -> str:
        """Personalised 1-2 sentence 'why this is for you' answer."""
        all_interests = list(dict.fromkeys(list(user_interests) + list(inferred_interests or [])))
        if not self.enabled:
            return self._fallback_rationale(
                location_categories, all_interests, rank, system_reason, evidence_summary,
                is_trending=is_trending,
            )

        signals = {
            k: v
            for k, v in {
                "similarity": similarity_score,
                "item_collab": item_collab_score,
                "trending": trending_score,
                "final_score": final_score,
            }.items()
            if v is not None
        }
        prompt = (
            "Write exactly 2 short sentences (max 45 words total) explaining why THIS "
            "Chicago place fits THIS visitor. Be specific:\n"
            "- Name the category overlap with their interests (use inferred_interests too).\n"
            "- If system_reason or evidence_summary is provided, echo ONE concrete fact from it "
            "(e.g. trending, similar users, engagement) — do not contradict them.\n"
            "- If rank <= 5, you may mention it ranks highly for them.\n"
            "- Tie vibe/traveler_type when relevant.\n"
            "No markdown. No greeting. No generic filler.\n\n"
            f"location: {location_name}\n"
            f"location_categories: {list(location_categories)}\n"
            f"user_interests: {list(user_interests)}\n"
            f"inferred_interests: {list(inferred_interests or [])}\n"
            f"vibe: {vibe}\n"
            f"traveler_type: {traveler_type}\n"
            f"rank: {rank}\n"
            f"system_reason: {system_reason}\n"
            f"evidence_summary: {evidence_summary}\n"
            f"is_trending: {is_trending}\n"
            f"is_hot_spot: {is_hot_spot}\n"
            f"score_signals: {signals}\n"
        )
        try:
            text = self._chat_text(prompt).strip()
            return text or self._fallback_rationale(
                location_categories, all_interests, rank, system_reason, evidence_summary,
                is_trending=is_trending,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rationale_for failed, falling back: %s", exc)
            return self._fallback_rationale(
                location_categories, all_interests, rank, system_reason, evidence_summary,
                is_trending=is_trending,
            )

    # ==================================================================== #
    # Core chat helpers (cache + JSON mode)
    # ==================================================================== #
    def _chat_json(
        self,
        prompt: str,
        schema_hint: str,
        *,
        cache_ok: Optional[Callable[[Dict[str, Any]], bool]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        cache_key = self.cache.make_key("json", self.model, SYSTEM_PROMPT, prompt, schema_hint)
        hit = self.cache.get(cache_key)
        if hit is not None:
            try:
                parsed = json.loads(hit)
                if cache_ok is None or cache_ok(parsed):
                    return parsed
            except Exception:  # noqa: BLE001
                pass  # corrupted or stale cache row, regenerate

        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        parsed = _parse_json_lenient(text)
        if cache_ok is None or cache_ok(parsed):
            self.cache.put(cache_key, json.dumps(parsed))
        return parsed

    def _chat_text(self, prompt: str) -> str:
        cache_key = self.cache.make_key("text", self.model, SYSTEM_PROMPT, prompt)
        hit = self.cache.get(cache_key)
        if hit is not None:
            return hit

        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        self.cache.put(cache_key, text)
        return text

    # ==================================================================== #
    # Deterministic fallbacks (work without any API key)
    # ==================================================================== #
    @staticmethod
    def _fallback_describe(
        location_name: str,
        primary_category: Optional[str],
        categories: Sequence[str],
        warehouse_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cat = primary_category or (next(iter(categories), None)) or "Chicago spot"
        ctx = warehouse_context or {}
        users = int(ctx.get("distinct_users") or 0)
        engagement = ""
        if users > 0:
            engagement = (
                f" About {users} visitors on the ChicagoDoes map engaged with it"
                + (f" (avg ~{ctx.get('avg_engagement_sec')}s)" if ctx.get("avg_engagement_sec") else "")
                + "."
            )
        description = (
            f"{location_name} is a well-known {cat.lower()} in Chicago.{engagement} "
            "It fits travelers who want authentic local picks backed by real map activity."
        )
        highlights: List[str] = [
            f"Listed as {cat} on the ChicagoDoes destination map.",
        ]
        if ctx.get("is_hot_spot"):
            highlights.append("Marked as a hot spot — high engagement across many sessions.")
        if users >= 50:
            highlights.append(f"Strong community signal: {users}+ distinct visitors explored it.")
        tips: List[str] = [
            f"Pair with other {cat.lower()} stops nearby on your day plan.",
            "Confirm hours and any ticket requirements before you go.",
        ]
        if cat.lower() == "bars":
            tips.append("Best as an evening stop; consider rideshare if you're bar-hopping.")
        if cat.lower() == "parks":
            tips.append("Morning or golden hour works well for photos and walks.")
        if cat.lower() == "museums":
            tips.append("Budget at least 90 minutes; popular exhibits can have lines.")
        return {
            "description": description,
            "highlights": highlights,
            "tips": tips,
            "neighborhood": None,
            "best_for": f"{cat.lower()} lovers and curious first-time visitors",
            "website_url": None,
            "source": "fallback",
        }

    @staticmethod
    def _fallback_rationale(
        location_categories: Sequence[str],
        all_interests: Sequence[str],
        rank: Optional[int],
        system_reason: Optional[str],
        evidence_summary: Optional[str],
        *,
        is_trending: bool = False,
    ) -> str:
        overlap = [c for c in location_categories if c in all_interests]
        parts: List[str] = []
        if overlap:
            parts.append(f"It matches your {', '.join(overlap[:2])} picks.")
        elif system_reason:
            parts.append(system_reason.strip().rstrip(".") + ".")
        else:
            parts.append("It scored highly against your profile and similar travelers.")
        if rank and rank <= 5:
            parts.append(f"Ranked #{rank} in your list for this trip.")
        if is_trending and not (evidence_summary and "trending" in evidence_summary.lower()):
            parts.append("It's trending on the map right now.")
        elif evidence_summary and len(parts) < 2:
            parts.append(evidence_summary.strip().rstrip(".") + ".")
        return " ".join(parts[:2])

    @staticmethod
    def _fallback_parse_intent(
        free_text: str,
        valid_categories: Sequence[str],
        valid_travelers: Sequence[str],
        valid_vibes: Sequence[str],
    ) -> Dict[str, Any]:
        """Cheap keyword-based parser used when no LLM is available."""
        text = (free_text or "").lower()
        interests: List[str] = []
        for cat in valid_categories:
            if cat.lower() in text:
                interests.append(cat)
        traveler = next((t for t in valid_travelers if t in text), None)
        vibe = next((v for v in valid_vibes if v in text), None)
        return {
            "interests": interests,
            "avoid_categories": [],
            "traveler_type": traveler,
            "vibe": vibe,
            "trip_days": 2,
            "summary": (free_text or "").strip()[:140],
        }


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _trim_candidates_for_itinerary(
    candidates: Sequence[Dict],
    trip_days: int,
) -> List[Dict]:
    """Keep prompts small enough for local 8B models."""
    cap = min(len(candidates), max(12, trip_days * 8))
    slim: List[Dict] = []
    for c in list(candidates)[:cap]:
        lid = str(c.get("location_id", ""))
        if not lid:
            continue
        slim.append({
            "rank": c.get("rank"),
            "location_id": lid,
            "location_name": str(c.get("location_name", "")),
            "primary_category": c.get("primary_category"),
            "score": c.get("score"),
        })
    return slim


def _itinerary_prompt_full(
    pool: Sequence[Dict],
    days_n: int,
    allowed_ids: Sequence[str],
    interests: Sequence[str],
    avoid_l: Sequence[str],
    traveler_type: Optional[str],
    vibe: Optional[str],
) -> str:
    return (
        "Chicago trip planner. Schedule ONLY location_ids from `candidates`.\n"
        "Return JSON with top-level keys `summary` and `days` (array). "
        "Each day: day_number, theme, narrative, stops[]. "
        "Each stop: location_id, slot (breakfast|morning|lunch|afternoon|dinner|drinks), "
        "slot_label, note.\n"
        f"Plan exactly {days_n} day(s). 4-6 stops per day. No duplicate location_ids.\n"
        f"allowed_ids: {json.dumps(list(allowed_ids))}\n"
        f"interests: {json.dumps(list(interests))}\n"
        f"avoid_categories: {json.dumps(list(avoid_l))}\n"
        f"traveler_type: {json.dumps(traveler_type)}\n"
        f"vibe: {json.dumps(vibe)}\n"
        f"candidates: {json.dumps(list(pool))}\n"
    )


def _itinerary_prompt_compact(
    pool: Sequence[Dict],
    days_n: int,
    allowed_ids: Sequence[str],
    interests: Sequence[str],
    avoid_l: Sequence[str],
) -> str:
    return (
        "Return ONLY valid JSON:\n"
        '{"summary":"...", "days":[{"day_number":1,"theme":"...","narrative":"...",'
        '"stops":[{"location_id":"...","slot":"morning","slot_label":"Morning","note":"..."}]}]}\n'
        f"Exactly {days_n} day(s). Use 4-5 stops per day. "
        f"location_id MUST be from this list: {json.dumps(list(allowed_ids))}\n"
        f"interests: {json.dumps(list(interests)[:5])}\n"
        f"places: {json.dumps(list(pool))}\n"
    )


def _parse_json_lenient(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _extract_itinerary_days(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize common LLM JSON shapes into a days array with stops."""
    if not isinstance(payload, dict):
        return []

    days = payload.get("days")
    if isinstance(days, list) and days:
        return _filter_days_with_stops(days)

    nested = payload.get("itinerary")
    if isinstance(nested, dict):
        days = nested.get("days")
        if isinstance(days, list) and days:
            return _filter_days_with_stops(days)

    # e.g. {"day_1": {...}, "day_2": {...}}
    collected: List[Dict[str, Any]] = []
    for key, val in payload.items():
        if not isinstance(key, str) or not key.lower().startswith("day"):
            continue
        if isinstance(val, dict):
            day = dict(val)
            if "day_number" not in day:
                digits = "".join(ch for ch in key if ch.isdigit())
                if digits:
                    day["day_number"] = int(digits)
            collected.append(day)
    if collected:
        return _filter_days_with_stops(collected)

    return []


def _filter_days_with_stops(days: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in days:
        if not isinstance(raw, dict):
            continue
        stops = raw.get("stops")
        if isinstance(stops, list) and stops:
            out.append(raw)
    return out


def _itinerary_payload_ok(payload: Dict[str, Any]) -> bool:
    return bool(_extract_itinerary_days(payload))


def _ping_llm_server(base_url: str, timeout: float = 2.0) -> bool:
    """Best-effort check that a local Ollama (or compatible) server is up."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        req = urllib.request.Request(f"{root}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _user_facing_llm_error(exc: Exception, fallback: str) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "connection" in msg or "connect" in msg or "refused" in msg:
        return (
            "Cannot reach Ollama. Keep `ollama serve` running in a **separate** "
            "terminal, then click Build AI itinerary again."
        )
    if "not found" in msg and "model" in msg:
        return (
            "Ollama model missing. Run: ollama pull llama3.1:8b  then retry."
        )
    return fallback


def _resolve_llm_base_url(raw: Optional[str]) -> Optional[str]:
    """Return OpenAI-compatible base URL.

    Unset → Ollama local. Empty string or ``openai`` → official OpenAI cloud.
    """
    if raw is None:
        return DEFAULT_LLM_BASE_URL.rstrip("/")
    s = raw.strip()
    if not s or s.lower() in {"openai", "none", "default"}:
        return None
    return s.rstrip("/")


def _nullish_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    return s


def _clean_url(value: Any) -> Optional[str]:
    s = _nullish_str(value)
    if not s:
        return None
    if not s.startswith(("http://", "https://")):
        return None
    return s


def _clean_intent_payload(
    payload: Dict[str, Any],
    valid_categories: Sequence[str],
    valid_travelers: Sequence[str],
    valid_vibes: Sequence[str],
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    cat_set = {c.lower(): c for c in valid_categories}

    def _filter_cats(seq: Any) -> List[str]:
        if not isinstance(seq, list):
            return []
        out: List[str] = []
        for v in seq:
            key = str(v).strip().lower()
            if key in cat_set and cat_set[key] not in out:
                out.append(cat_set[key])
        return out

    return {
        "interests": _filter_cats(payload.get("interests")) or fallback["interests"],
        "avoid_categories": _filter_cats(payload.get("avoid_categories")),
        "traveler_type": _pick_enum(payload.get("traveler_type"), valid_travelers),
        "vibe": _pick_enum(payload.get("vibe"), valid_vibes),
        "trip_days": _clamp_int(payload.get("trip_days"), 1, 7, default=2),
        "summary": str(payload.get("summary") or fallback["summary"]).strip(),
    }


def _clean_refine_payload(
    payload: Dict[str, Any],
    valid_categories: Sequence[str],
    valid_travelers: Sequence[str],
    valid_vibes: Sequence[str],
) -> Dict[str, Any]:
    cat_set = {c.lower(): c for c in valid_categories}

    def _filter_cats(seq: Any) -> List[str]:
        if not isinstance(seq, list):
            return []
        out: List[str] = []
        for v in seq:
            key = str(v).strip().lower()
            if key in cat_set and cat_set[key] not in out:
                out.append(cat_set[key])
        return out

    return {
        "add_interests":    _filter_cats(payload.get("add_interests")),
        "remove_interests": _filter_cats(payload.get("remove_interests")),
        "add_avoid":        _filter_cats(payload.get("add_avoid")),
        "remove_avoid":     _filter_cats(payload.get("remove_avoid")),
        "set_vibe":         _pick_enum(payload.get("set_vibe"), valid_vibes),
        "set_traveler":     _pick_enum(payload.get("set_traveler"), valid_travelers),
        "set_trip_days":    _clamp_int(payload.get("set_trip_days"), 1, 7, default=None),
        "comment":          str(payload.get("comment") or "").strip(),
        "source":           "llm",
    }


def _pick_enum(value: Any, valid: Sequence[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    return v if v in {x.lower() for x in valid} else None


def _clamp_int(value: Any, lo: int, hi: int, default: Optional[int]) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}
