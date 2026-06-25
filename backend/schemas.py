"""Pydantic request / response schemas for the ChicagoDoes recommender API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Payload from the website form.

    Either `user_key` (returning user) OR the preference fields (new user)
    can drive recommendation. If both are present, the existing user's
    behavioural profile is blended with the form preferences.
    """

    user_key: Optional[str] = Field(
        default=None,
        description="Existing user_key from user_location_full_features. Optional.",
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Selected category names, e.g. ['Attractions', 'Bars'].",
    )
    traveler_type: Optional[str] = Field(
        default=None,
        description="One of: solo, couple, family, group, business.",
    )
    vibe: Optional[str] = Field(
        default=None,
        description="One of: chill, adventurous, foodie, nightlife, cultural, outdoorsy.",
    )
    avoid_categories: List[str] = Field(default_factory=list)
    trip_days: int = Field(default=2, ge=1, le=7)
    free_text: Optional[str] = Field(
        default=None,
        description="Optional free-form description, e.g. 'I love jazz and deep-dish pizza'.",
    )
    top_k: int = Field(default=40, ge=5, le=80)
    use_ai_itinerary: bool = Field(
        default=False,
        description="If true, /api/itinerary calls the LLM to build a day plan.",
    )
    itinerary_pool_ids: List[str] = Field(
        default_factory=list,
        description=(
            "location_ids from a prior /api/recommend response. When set, the "
            "AI may only schedule stops from this pool (product: data first, AI second)."
        ),
    )
    planner_note: Optional[str] = Field(
        default=None,
        description=(
            "Internal: latest natural-language tweak (from /api/refine) woven into "
            "the AI day plan so the itinerary reflects the user's exact request."
        ),
    )


class LocationEvidence(BaseModel):
    """Real, data-grounded evidence behind a recommendation.

    Sourced directly from the warehouse (`*_all_users` + event table).
    This is the project's data moat vs. a pure LLM recommender —
    these numbers cannot be hallucinated.
    """
    n_users_engaged: int = 0
    n_interactions: int = 0
    n_sessions: int = 0
    avg_engagement_sec: float = 0.0
    is_trending: bool = False
    summary: str = ""


class LocationCard(BaseModel):
    location_id: str
    location_name: str
    primary_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    is_hot_spot: bool = False
    is_trending: bool = False
    popularity_score: float = 0.0
    similarity_score: float = 0.0
    item_collab_score: float = 0.0
    user_collab_score: float = 0.0
    trending_score: float = 0.0
    session_collab_score: float = 0.0   # session co-visit + transition graph
    feedback_adjustment_score: float = 0.0
    final_score: float = 0.0
    reason: Optional[str] = None
    evidence: LocationEvidence = Field(default_factory=LocationEvidence)


class OutboundClickRequest(BaseModel):
    """Client-side click attribution for links leaving this recommender.

    These events are intentionally stored separately from model training
    interactions so recommender-driven traffic can be filtered or corrected.
    """
    location_id: Optional[str] = None
    location_name: Optional[str] = None
    surface: str = "unknown"             # top_picks | ai_itinerary | other
    rank: Optional[int] = None
    link_type: Optional[str] = None       # chicagodoes | official
    href: Optional[str] = None


class Archetype(BaseModel):
    archetype: str
    cluster_id: int
    confidence: float
    cluster_size: int
    top_categories: List[str] = Field(default_factory=list)


class SimilarUser(BaseModel):
    """A real user whose behavioural fingerprint resembles the visitor's."""
    user_key: str
    label: str           # e.g. "Foodie Explorer · 12 visits · Bars, Restaurants"
    archetype: str
    similarity: float    # cosine sim 0..1
    n_interactions: int
    top_categories: List[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    user_key: Optional[str]
    is_returning_user: bool
    inferred_interests: List[str]
    archetype: Optional[Archetype] = None
    similar_users: List[SimilarUser] = Field(default_factory=list)
    recommendations: List[LocationCard]


class ItineraryStop(BaseModel):
    slot: str  # morning | afternoon | evening | breakfast | lunch | dinner | drinks
    slot_label: str = ""            # human label, e.g. "Lunch"
    location_id: str
    location_name: str
    primary_category: Optional[str] = None
    note: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    geo_source: Optional[str] = None
    source: str = "recommended"     # "recommended" (from our model) | "ai" (added by the LLM)
    link_url: Optional[str] = None
    link_type: Optional[str] = None   # "chicagodoes" | "official"
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    media_items: List[CardMediaItem] = Field(default_factory=list)


class ItineraryLeg(BaseModel):
    from_id: str
    to_id: str
    from_name: str
    to_name: str
    km: Optional[float] = None
    mode: str = "unknown"   # "walk" | "transit" | "drive" | "unknown"
    minutes: Optional[int] = None


class ItineraryDay(BaseModel):
    day_number: int
    theme: str
    stops: List[ItineraryStop]
    legs: List[ItineraryLeg] = []
    narrative: Optional[str] = None
    n_stops: int = 0


class ItineraryResponse(BaseModel):
    user_key: Optional[str]
    trip_days: int
    days: List[ItineraryDay]
    summary: str
    notice: Optional[str] = None
    plan_mode: str = "ai_generated"
    feasible: bool = True
    skip_reason: Optional[str] = None
    narrative_source: str = "fallback"
    recommendation_pool_size: int = 0
    stops_from_pool: int = 0


class LocationInfoRequest(BaseModel):
    location_id: str
    style: Optional[str] = Field(
        default="friendly",
        description="LLM tone: friendly | concise | poetic.",
    )


class LocationInfoResponse(BaseModel):
    location_id: str
    location_name: str
    primary_category: Optional[str]
    description: str
    highlights: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)
    neighborhood: Optional[str] = None
    best_for: Optional[str] = None
    website_url: Optional[str] = None
    maps_search_url: Optional[str] = None
    source: str  # "llm" or "fallback"


class LocationCardRequest(BaseModel):
    """Lazy per-card enrichment: photo, one-line blurb, best external link."""
    location_id: str


class CardMediaItem(BaseModel):
    type: str = "image"              # "image" | "video"
    url: str
    source: Optional[str] = None
    attribution: Optional[str] = None


class LocationCardResponse(BaseModel):
    location_id: str
    location_name: str
    image_url: Optional[str] = None
    image_attribution: Optional[str] = None
    video_url: Optional[str] = None
    video_attribution: Optional[str] = None
    media_items: List[CardMediaItem] = Field(default_factory=list)
    blurb: str = ""
    blurb_source: str = "fallback"   # "llm" | "fallback"
    link_url: str
    link_type: str                   # "chicagodoes"
    media_source: Optional[str] = None   # chicagodoes | wikipedia | openverse | official


# --------------------------------------------------------------------------- #
# A1 - Natural-language intent parsing
# --------------------------------------------------------------------------- #
class IntentParseRequest(BaseModel):
    free_text: str = Field(..., description="What the visitor typed in the chat box.")


class IntentParseResponse(BaseModel):
    """Parsed preferences. Front-end uses this to pre-fill the form and
    immediately fire `/api/recommend`."""
    interests: List[str] = Field(default_factory=list)
    avoid_categories: List[str] = Field(default_factory=list)
    traveler_type: Optional[str] = None
    vibe: Optional[str] = None
    trip_days: int = 2
    summary: str = ""
    source: str = "fallback"  # "llm" or "fallback"


# --------------------------------------------------------------------------- #
# A3 - Per-recommendation rationale
# --------------------------------------------------------------------------- #
class ExplainRequest(BaseModel):
    location_id: str
    interests: List[str] = Field(default_factory=list)
    inferred_interests: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    traveler_type: Optional[str] = None
    rank: Optional[int] = Field(default=None, ge=1, le=80)
    system_reason: Optional[str] = None
    evidence_summary: Optional[str] = None
    final_score: Optional[float] = None
    is_trending: bool = False
    is_hot_spot: bool = False
    similarity_score: Optional[float] = None
    item_collab_score: Optional[float] = None
    trending_score: Optional[float] = None


class ExplainResponse(BaseModel):
    location_id: str
    location_name: str
    rationale: str
    source: str = "fallback"


# --------------------------------------------------------------------------- #
# A4 - Refinement: NL tweak -> request delta -> rerun
# --------------------------------------------------------------------------- #
class RefineRequest(BaseModel):
    instruction: str = Field(..., description="e.g. 'make day 2 more chill, no bars'.")
    previous_request: "RecommendRequest"


class RefineResponse(BaseModel):
    delta: dict
    new_request: "RecommendRequest"
    recommendations: "RecommendResponse"
    itinerary: "ItineraryResponse"
    source: str = "fallback"


# Pydantic forward-ref resolution (RecommendRequest / *Response are above)
RefineRequest.model_rebuild()
RefineResponse.model_rebuild()
