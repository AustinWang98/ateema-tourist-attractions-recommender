/* ChicagoDoes recommender — vanilla SPA frontend */

const API = {
  health:      "/api/health",
  categories:  "/api/categories",
  users:       "/api/users",
  recommend:   "/api/recommend",
  itinerary:   "/api/itinerary",
  info:        "/api/location/info",
  explain:     "/api/explain",
  refine:      "/api/refine",
  trending:    "/api/trending",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* --------------------- state --------------------- */
const state = {
  categories: [],
  selectedInterests: new Set(),
  selectedAvoid: new Set(),
  lastRequest: null,
  lastRecommendations: null,
  recoItems: [],
  recoPage: 0,
};

/* --------------------- boot --------------------- */
window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadHealth(), loadCategories(), loadUsers(), loadWelcomeTeasers()]);
  wireForm();
});

async function loadWelcomeTeasers() {
  // Populate the welcome-card stats and a short "trending right now" row.
  // Best-effort: silently no-op if endpoints aren't ready.
  try {
    const h = await fetch(API.health).then((r) => r.json());
    if (h.n_events) $("#welcome-events").textContent = (h.n_events / 1000).toFixed(1) + "k";
    if (h.n_locations) $("#welcome-locations").textContent = String(h.n_locations);
  } catch (_) {}
  try {
    const r = await fetch(API.trending + "?limit=6").then((r) => r.json());
    const row = $("#welcome-trend-row");
    const wrap = $("#welcome-trending");
    if (!r.locations || !r.locations.length) return;
    row.innerHTML = r.locations.map(
      (l) => `<span class="welcome__trend-chip">${escapeHtml(l.location_name)}</span>`
    ).join("");
    wrap.classList.remove("hidden");
  } catch (_) {}
}

async function loadHealth() {
  const el = $("#health");
  try {
    const r = await fetch(API.health).then((r) => r.json());
    const llmTxt = r.llm_enabled ? "LLM on" : "LLM off (fallback)";
    const obs = r.n_observed_locations || 0;
    const trend = r.trending_locations || 0;
    const eventTxt = (r.n_events || 0) > 0 ? ` · ${r.n_events} events · 🔥 ${trend}` : "";
    const dataTxt = r.load_mode === "bigquery"
      ? "BQ live"
      : r.load_mode === "local_cache_fallback"
        ? "cached"
        : "";
    const dataPart = dataTxt ? ` · ${dataTxt}` : "";
    el.textContent = `${r.n_locations} locations (${obs} observed) · ${r.n_users} users${eventTxt} · ${llmTxt}${dataPart}`;
    if (r.load_warning) el.title = r.load_warning;
    if (r.load_mode === "local_cache_fallback") {
      el.classList.add("warn");
    } else {
      el.classList.add(r.ok ? "ok" : "bad");
    }
  } catch (e) {
    el.textContent = "API unavailable";
    el.classList.add("bad");
  }
}

async function loadCategories() {
  const r = await fetch(API.categories).then((r) => r.json());
  state.categories = r.categories || [];
  renderChips("#interests", state.categories, state.selectedInterests);
  renderChips("#avoid", state.categories, state.selectedAvoid);
}

async function loadUsers() {
  const sel = $("#user_key");
  try {
    const r = await fetch(API.users + "?limit=50").then((r) => r.json());
    for (const u of r.users || []) {
      const opt = document.createElement("option");
      if (typeof u === "string") {
        opt.value = u;
        opt.textContent = u.length > 18 ? u.slice(0, 8) + "…" + u.slice(-6) : u;
      } else {
        opt.value = u.user_key;
        opt.textContent = u.label || u.user_key;
        opt.dataset.archetype = u.archetype || "";
      }
      sel.appendChild(opt);
    }
  } catch (e) {
    /* silent */
  }
}

function renderChips(containerSel, items, selectedSet) {
  const root = $(containerSel);
  root.innerHTML = "";
  for (const item of items) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = item;
    if (selectedSet.has(item)) chip.classList.add("active");
    chip.addEventListener("click", () => {
      if (selectedSet.has(item)) {
        selectedSet.delete(item);
        chip.classList.remove("active");
      } else {
        selectedSet.add(item);
        chip.classList.add("active");
      }
    });
    root.appendChild(chip);
  }
  if (items.length === 0) {
    root.innerHTML = `<span class="muted">No categories loaded.</span>`;
  }
}

function wireForm() {
  $("#rec-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await runRecommend();
  });
  $("#plan-btn").addEventListener("click", async () => {
    await runItinerary();
  });
  $("#use_ai_itinerary").addEventListener("change", () => {
    const on = $("#use_ai_itinerary").checked;
    $("#ai-itin-hint").classList.toggle("hidden", !on);
    if (!on) {
      clearItinerary();
    }
    updatePlanButton();
  });
  $("#refine-go").addEventListener("click", async () => {
    await runRefine();
  });
  $("#reco-prev").addEventListener("click", () => changeRecoPage(-1));
  $("#reco-next").addEventListener("click", () => changeRecoPage(1));
  if (!window.__recoResizeBound) {
    window.__recoResizeBound = true;
    window.addEventListener("resize", () => {
      if (state.recoItems.length) renderRecoPage();
    });
  }
}

/** 3×2 on desktop, 2×2 tablet, 1×2 narrow — matches original ~240px-wide cards. */
function recoLayout() {
  const vp = document.querySelector(".reco-carousel__viewport");
  const w = vp?.clientWidth ?? window.innerWidth;
  if (w >= 720) return { cols: 3, rows: 2, perPage: 6 };
  if (w >= 500) return { cols: 2, rows: 2, perPage: 4 };
  return { cols: 1, rows: 2, perPage: 2 };
}

function recoPerPage() {
  return recoLayout().perPage;
}

function applyRecoGridLayout() {
  const grid = $("#reco-grid");
  if (!grid) return;
  const { cols } = recoLayout();
  grid.classList.remove("reco-grid--1", "reco-grid--2", "reco-grid--3");
  grid.classList.add(`reco-grid--${cols}`);
}

function recoPageCount() {
  const n = state.recoItems.length;
  if (!n) return 0;
  return Math.ceil(n / recoPerPage());
}

function changeRecoPage(direction) {
  const pages = recoPageCount();
  if (!pages) return;
  state.recoPage = Math.max(0, Math.min(pages - 1, state.recoPage + direction));
  renderRecoPage();
}

function updateRecoPagerButtons() {
  const prev = $("#reco-prev");
  const next = $("#reco-next");
  if (!prev || !next) return;
  const pages = recoPageCount();
  prev.disabled = state.recoPage <= 0;
  next.disabled = pages <= 1 || state.recoPage >= pages - 1;
}

/* --------------------- requests --------------------- */
function readForm() {
  return {
    user_key:        $("#user_key").value || null,
    traveler_type:   $("#traveler_type").value || null,
    vibe:            $("#vibe").value || null,
    trip_days:       Number($("#trip_days").value || 2),
    free_text:       $("#free_text").value.trim() || null,
    interests:       Array.from(state.selectedInterests),
    avoid_categories:Array.from(state.selectedAvoid),
    top_k:           40,
    use_ai_itinerary: $("#use_ai_itinerary").checked,
  };
}

function clearItinerary() {
  $("#itin-panel").classList.add("hidden");
  $("#itin-days").innerHTML = "";
  $("#itin-summary").textContent = "";
  $("#itin-notice").classList.add("hidden");
  $("#itin-notice").textContent = "";
  const skipBox = $("#itin-skip");
  if (skipBox) {
    skipBox.innerHTML = "";
    skipBox.classList.add("hidden");
  }
}

function updatePlanButton() {
  const aiOn = $("#use_ai_itinerary").checked;
  const hasRecs = !!(state.lastRecommendations?.recommendations?.length);
  $("#plan-btn").disabled = !aiOn || !state.lastRequest || !hasRecs;
}

function recommendationPoolIds() {
  const recs = state.lastRecommendations?.recommendations || [];
  return recs.map((r) => r.location_id).filter(Boolean);
}

async function runRecommend() {
  const payload = readForm();
  state.lastRequest = payload;
  setStatus("Finding the best ChicagoDoes spots", "loading");
  $("#submit-btn").disabled = true;
  $("#plan-btn").disabled = true;

  try {
    const r = await fetch(API.recommend, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.lastRecommendations = data;
    renderRecommendations(data);
    if (!$("#use_ai_itinerary").checked) {
      clearItinerary();
    }
    clearStatus();
    updatePlanButton();
  } catch (err) {
    setStatus(`Recommendation failed: ${err.message}`, "error");
  } finally {
    $("#submit-btn").disabled = false;
  }
}

async function runItinerary() {
  if (!state.lastRequest) return;
  if (!$("#use_ai_itinerary").checked) {
    clearItinerary();
    setStatus("Turn on **Plan my days with AI** to build a schedule.", "error");
    return;
  }
  const poolIds = recommendationPoolIds();
  if (!poolIds.length) {
    setStatus("Get recommendations first — the AI only schedules those Top picks.", "error");
    return;
  }
  setStatus("AI is arranging your Top picks into days…", "loading");
  $("#plan-btn").disabled = true;

  try {
    const r = await fetch(API.itinerary, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...state.lastRequest,
        ...readForm(),
        use_ai_itinerary: true,
        itinerary_pool_ids: poolIds,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderItinerary(data);
    clearStatus();
  } catch (err) {
    setStatus(`Itinerary failed: ${err.message}`, "error");
  } finally {
    updatePlanButton();
  }
}

/* --------------------- A4: refinement --------------------- */
async function runRefine() {
  const instruction = $("#refine-text").value.trim();
  if (!instruction || !state.lastRequest) return;
  $("#refine-status").textContent = "Re-planning…";
  $("#refine-go").disabled = true;
  try {
    const r = await fetch(API.refine, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction,
        previous_request: state.lastRequest,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.lastRequest = data.new_request;
    state.lastRecommendations = data.recommendations;
    renderRecommendations(data.recommendations);
    if (data.new_request?.use_ai_itinerary) {
      renderItinerary(data.itinerary);
    } else {
      clearItinerary();
    }
    syncFormFromRequest(data.new_request);
    $("#refine-status").textContent =
      data.delta?.comment ? `AI: ${data.delta.comment}` : "Updated.";
    $("#refine-text").value = "";
  } catch (err) {
    $("#refine-status").textContent = `Failed: ${err.message}`;
  } finally {
    $("#refine-go").disabled = false;
  }
}

function syncFormFromRequest(req) {
  $("#traveler_type").value = req.traveler_type || "";
  $("#vibe").value = req.vibe || "";
  $("#trip_days").value = String(req.trip_days || 2);
  state.selectedInterests = new Set(req.interests || []);
  state.selectedAvoid = new Set(req.avoid_categories || []);
  renderChips("#interests", state.categories, state.selectedInterests);
  renderChips("#avoid", state.categories, state.selectedAvoid);
}

/* --------------------- A3: per-rec rationale --------------------- */
async function explainTile(loc, btn, target) {
  btn.disabled = true;
  btn.textContent = "Thinking…";
  try {
    const r = await fetch(API.explain, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location_id: loc.location_id,
        interests: state.lastRequest?.interests || [],
        vibe: state.lastRequest?.vibe || null,
        traveler_type: state.lastRequest?.traveler_type || null,
        inferred_interests: state.lastRecommendations?.inferred_interests || [],
        rank: loc._rank,
        system_reason: loc.reason || null,
        evidence_summary: loc.evidence?.summary || null,
        final_score: loc.final_score,
        is_trending: loc.is_trending,
        is_hot_spot: loc.is_hot_spot,
        similarity_score: loc.similarity_score,
        item_collab_score: loc.item_collab_score,
        trending_score: loc.trending_score,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    target.textContent = data.rationale;
    target.classList.remove("hidden");
    btn.remove();
  } catch (err) {
    target.textContent = `Could not explain: ${err.message}`;
    target.classList.remove("hidden");
    btn.disabled = false;
    btn.textContent = "Why this?";
  }
}

async function openLocationInfo(loc) {
  const modal = $("#info-modal");
  $("#modal-title").textContent = loc.location_name;
  $("#modal-body").innerHTML = `<p class="muted">Asking the concierge…</p>`;
  modal.showModal();

  try {
    const r = await fetch(API.info, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: loc.location_id, style: "friendly" }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const info = await r.json();
    const tips = (info.tips || []).map((t) => `<li>${escapeHtml(t)}</li>`).join("");
    const highlights = (info.highlights || [])
      .map((h) => `<li>${escapeHtml(h)}</li>`)
      .join("");
    const links = [];
    if (info.website_url) {
      links.push(
        `<a class="info-link" href="${escapeHtml(info.website_url)}" target="_blank" rel="noopener noreferrer">Official website</a>`
      );
    }
    if (info.maps_search_url) {
      links.push(
        `<a class="info-link" href="${escapeHtml(info.maps_search_url)}" target="_blank" rel="noopener noreferrer">Find on Google Maps</a>`
      );
    }
    const metaBits = [
      info.neighborhood ? `Area: ${escapeHtml(info.neighborhood)}` : "",
      info.best_for ? `Best for: ${escapeHtml(info.best_for)}` : "",
    ].filter(Boolean);
    $("#modal-body").innerHTML = `
      <p>${escapeHtml(info.description)}</p>
      ${metaBits.length ? `<p class="info-meta">${metaBits.join(" · ")}</p>` : ""}
      ${highlights ? `<h4 class="info-heading">What makes it special</h4><ul>${highlights}</ul>` : ""}
      ${tips ? `<h4 class="info-heading">Visitor tips</h4><ul>${tips}</ul>` : ""}
      ${links.length ? `<div class="info-links">${links.join("")}</div>` : ""}
      <div class="source">Generated by: <b>${info.source === "llm" ? "LLM" : "fallback"}</b></div>
    `;
  } catch (err) {
    $("#modal-body").innerHTML =
      `<p class="status error">Could not load info: ${escapeHtml(err.message)}</p>`;
  }
}

/* --------------------- rendering --------------------- */
function createRecoTile(r, rank) {
  const tile = document.createElement("article");
  tile.className = "tile";
  const cats = (r.categories || [])
    .slice(0, 4)
    .map((c) => {
      const cls = c === "HOT SPOTS" ? "tag tag--hot" : "tag";
      return `<span class="${cls}">${escapeHtml(c)}</span>`;
    })
    .join("");
  const trendingBadge = r.is_trending
    ? `<span class="tag tag--trend">🔥 Trending</span>`
    : "";

  const evidence = r.evidence || {};
  const hasEvidence = (evidence.n_users_engaged || 0) > 0;
  const evidenceHtml = `
    <div class="tile__evidence ${hasEvidence ? "" : "empty"}">${escapeHtml(evidence.summary || "No evidence yet.")}</div>
  `;

  tile.innerHTML = `
    <div class="tile__top">
      <div class="tile__name">${escapeHtml(r.location_name)}</div>
      <div class="tile__rank">#${rank}</div>
    </div>
    <div class="tile__cats">${cats || `<span class="muted">No categories</span>`}${trendingBadge}</div>
    <div class="tile__reason">${escapeHtml(r.reason || "")}</div>
    <div class="tile__why hidden"></div>
    ${evidenceHtml}
    ${renderBreakdown(r)}
    <div class="tile__actions">
      <button class="link-btn js-why" data-id="${r.location_id}">Why this?</button>
      <button class="link-btn js-info" data-id="${r.location_id}">More info ✨</button>
    </div>
  `;
  tile.querySelector(".js-info").addEventListener("click", () => openLocationInfo(r));
  tile.querySelector(".js-why").addEventListener("click", (e) =>
    explainTile(r, e.currentTarget, tile.querySelector(".tile__why"))
  );
  return tile;
}

function renderRecoPage() {
  const grid = $("#reco-grid");
  if (!grid) return;

  applyRecoGridLayout();
  const perPage = recoPerPage();
  const total = state.recoItems.length;
  const pages = recoPageCount();
  if (pages) {
    state.recoPage = Math.max(0, Math.min(pages - 1, state.recoPage));
  }

  const start = state.recoPage * perPage;
  const slice = state.recoItems.slice(start, start + perPage);

  grid.innerHTML = "";
  slice.forEach((r) => grid.appendChild(createRecoTile(r, r._rank)));

  const rangeEl = $("#reco-range");
  const pageNum = $("#reco-page-num");
  const pageTotal = $("#reco-page-total");
  const hint = $("#reco-scroll-hint");
  if (rangeEl && total) {
    const end = start + slice.length;
    rangeEl.textContent = `${start + 1}–${end}`;
  }
  if (pageNum) pageNum.textContent = String(pages ? state.recoPage + 1 : 0);
  if (pageTotal) pageTotal.textContent = String(pages || 1);
  if (hint) {
    hint.classList.toggle("hidden", total <= perPage);
  }

  updateRecoPagerButtons();
}

function renderRecommendations(data) {
  const panel = $("#reco-panel");
  const meta = $("#reco-meta");
  panel.classList.remove("hidden");
  const welcome = $("#welcome");
  if (welcome) welcome.classList.add("hidden");

  const nRecs = (data.recommendations || []).length;
  const interestsTxt = (data.inferred_interests || []).slice(0, 5).join(", ");
  const who = data.is_returning_user ? "Returning user" : "New visitor";
  meta.textContent = `${who} · ${nRecs} places ranked · Inferred: ${interestsTxt || "n/a"}.`;

  const countEl = $("#reco-count");
  if (countEl) countEl.textContent = String(nRecs);

  renderArchetype(data.archetype);
  renderSimilarUsers(data.similar_users, data.is_returning_user);

  state.recoItems = (data.recommendations || []).map((r, i) => {
    r._rank = i + 1;
    return r;
  });
  state.recoPage = 0;
  requestAnimationFrame(() => renderRecoPage());
}

function renderArchetype(a) {
  const box = $("#archetype");
  if (!a) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const initials = (a.archetype || "?").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  const conf = Math.round((a.confidence || 0) * 100);
  const tags = (a.top_categories || []).slice(0, 3)
    .map((c) => `<span class="archetype__tag">${escapeHtml(c)}</span>`)
    .join("");
  box.innerHTML = `
    <div class="archetype__icon">${escapeHtml(initials)}</div>
    <div>
      <div class="archetype__name">${escapeHtml(a.archetype)}</div>
      <div class="archetype__meta">
        ${conf}% match · cluster of ${a.cluster_size} similar ChicagoDoes user${a.cluster_size === 1 ? "" : "s"}
      </div>
    </div>
    <div class="archetype__bars">${tags}</div>
  `;
  box.classList.remove("hidden");
}

function renderSimilarUsers(list, isReturning) {
  const box = $("#similar-users");
  if (!list || !list.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const intro = isReturning
    ? `We compared your past activity to other ChicagoDoes visitors. Your <strong>${list.length}</strong> closest neighbours:`
    : `Based on your interests, you behave like these <strong>${list.length}</strong> real ChicagoDoes visitors — their saved locations informed your picks.`;
  const chipsHtml = list.map((u) => `
    <div class="similar__chip" title="${escapeHtml(u.label)}">
      <div class="similar__sim">${Math.round((u.similarity || 0) * 100)}%</div>
      <div>
        <div class="similar__archetype">${escapeHtml(u.archetype)}</div>
        <div class="similar__meta">${u.n_interactions} visits · ${(u.top_categories || []).slice(0,2).map(escapeHtml).join(", ") || "no clicks"}</div>
      </div>
    </div>
  `).join("");
  box.innerHTML = `
    <div class="similar__intro">${intro}</div>
    <div class="similar__row">${chipsHtml}</div>
  `;
  box.classList.remove("hidden");
}

function renderBreakdown(r) {
  const items = [
    { key: "Content",   v: r.similarity_score,      muted: false },
    { key: "Popular",   v: r.popularity_score,      muted: false },
    { key: "Co-visits", v: r.item_collab_score,     muted: false },
    { key: "Sessions",  v: r.session_collab_score,  muted: (r.session_collab_score || 0) === 0 },
    { key: "Neighbors", v: r.user_collab_score,     muted: (r.user_collab_score || 0) === 0 },
    { key: "Trending",  v: r.trending_score,        muted: (r.trending_score   || 0) === 0 },
  ];
  return `<div class="tile__breakdown">` + items.map((it) => {
    const pct = Math.max(0, Math.min(1, it.v || 0)) * 100;
    return `
      <div class="tile__bar-row ${it.muted ? "muted" : ""}">
        <span>${it.key}</span>
        <div class="tile__bar"><div class="tile__bar-fill" style="width:${pct.toFixed(0)}%"></div></div>
        <span>${(it.v || 0).toFixed(2)}</span>
      </div>
    `;
  }).join("") + `</div>`;
}

function renderItinerary(data) {
  const panel = $("#itin-panel");
  const days  = $("#itin-days");
  const sum   = $("#itin-summary");
  const notice = $("#itin-notice");
  const skipBox = $("#itin-skip");
  panel.classList.remove("hidden");

  const skipped = data.feasible === false || !(data.days && data.days.length);

  if (skipped) {
    sum.textContent = data.summary || "Itinerary not available for this selection.";
    if (skipBox) {
      skipBox.innerHTML = formatNoticeHtml(data.notice);
      skipBox.classList.toggle("hidden", !data.notice);
    }
    notice.classList.add("hidden");
    notice.textContent = "";
    days.innerHTML = "";
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  if (skipBox) {
    skipBox.innerHTML = "";
    skipBox.classList.add("hidden");
  }

  const poolMeta =
    data.recommendation_pool_size > 0
      ? ` · ${data.stops_from_pool || 0} stops from ${data.recommendation_pool_size} Top picks`
      : "";
  sum.textContent = (data.summary || "") + poolMeta;

  if (data.notice) {
    notice.innerHTML = formatNoticeHtml(data.notice);
    notice.classList.remove("hidden");
  } else {
    notice.innerHTML = "";
    notice.classList.add("hidden");
  }

  days.innerHTML = "";
  (data.days || []).forEach((d, idx) => {
    const box = document.createElement("div");
    box.className = "day";

    const stops = d.stops || [];
    const items = stops.map((s, i) => renderStop(s, i + 1)).join("");

    const narrative = d.narrative
      ? `<div class="day__narrative">${escapeHtml(d.narrative)}</div>`
      : "";
    const mapId = `day-map-${idx}`;
    const statLabel = "stops";
    const stopsWithCoord = stops.filter((s) => s.lat != null && s.lon != null);
    const showMap = stopsWithCoord.length >= 2;

    box.innerHTML = `
      <div class="day__head">
        <h3>${escapeHtml(d.theme)}</h3>
        <div class="day__stats">
          <span class="day__stat"><b>${stops.length}</b> ${statLabel}</span>
          ${showMap ? `<span class="day__stat muted">route order on map</span>` : ""}
        </div>
      </div>
      ${narrative}
      ${showMap ? `<div id="${mapId}" class="day__map"></div>` : ""}
      <div class="timeline">${items}</div>
    `;
    days.appendChild(box);

    if (showMap) {
      // Defer to next tick so the div has been inserted and has a size.
      setTimeout(() => drawDayMap(mapId, stopsWithCoord), 0);
    }
  });

  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

const CATEGORY_ICON = {
  "Attractions":          "🎡",
  "Bars":                 "🍸",
  "Big Bus Tours Stops":  "🚌",
  "HOT SPOTS":            "🔥",
  "Hotels":               "🏨",
  "Movie & TV Locations": "🎬",
  "Murals":               "🎨",
  "Museums":              "🏛️",
  "Neighborhood Organizations": "🏘️",
  "Parks":                "🌳",
  "Restaurants":          "🍽️",
  "Shops":                "🛍️",
  "Sports Venues":        "🏟️",
  "Theaters and Music Venues": "🎭",
  "Tours":                "🗺️",
  "Favorites":            "⭐",
};

function renderStop(s, idx) {
  const geoFallback = s.geo_source === "fallback"
    ? '<span class="stop__geo-tag" title="Could not geocode this location precisely.">📍?</span>'
    : "";
  const slot = s.slot || "afternoon";
  const label = s.slot_label || "";
  const showLabel = !!label;
  const slotPill = showLabel
    ? `<span class="stop__slot stop__slot--${escapeHtml(slot)}">${escapeHtml(label)}</span>`
    : "";
  const icon = CATEGORY_ICON[s.primary_category] || "📍";
  const catTag = s.primary_category
    ? `<span class="stop__cat-tag">${icon} ${escapeHtml(s.primary_category)}</span>`
    : "";
  const note = s.note
    ? `<div class="stop__meta">${escapeHtml(s.note)}</div>`
    : "";
  const simple = "";
  const whenCol = showLabel
    ? `<div class="stop__when">${slotPill}</div>`
    : "";
  return `
    <div class="stop${simple}">
      <div class="stop__badge">
        <span class="stop__num">${idx}</span>
      </div>
      ${whenCol}
      <div class="stop__body">
        <div class="stop__name">
          <span>${escapeHtml(s.location_name)}</span>
          ${catTag}
          ${geoFallback}
        </div>
        ${note}
      </div>
    </div>
  `;
}

const _dayMaps = {};   // mapId -> L.Map instance (for cleanup on re-render)
function drawDayMap(mapId, stops) {
  if (!window.L) { return; }   // Leaflet not loaded
  const target = document.getElementById(mapId);
  if (!target) return;
  // Tear down any existing map (re-render case)
  if (_dayMaps[mapId]) { try { _dayMaps[mapId].remove(); } catch (_) {} delete _dayMaps[mapId]; }

  const lats = stops.map((s) => s.lat);
  const lons = stops.map((s) => s.lon);
  const map = L.map(mapId, { scrollWheelZoom: false, zoomControl: true }).setView(
    [lats.reduce((a,b)=>a+b,0)/lats.length, lons.reduce((a,b)=>a+b,0)/lons.length],
    13
  );
  // CartoDB Dark Matter — sober dark basemap that matches the page chrome.
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    subdomains: "abcd",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> · &copy; <a href="https://carto.com/attributions">CARTO</a>',
  }).addTo(map);

  // Numbered markers only (no route polyline — straight lines mislead)
  const latlngs = [];
  stops.forEach((s, i) => {
    const ll = [s.lat, s.lon];
    latlngs.push(ll);
    const icon = L.divIcon({
      className: "map-pin",
      html: `<div class="map-pin__num">${i + 1}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 28],
    });
    L.marker(ll, { icon }).addTo(map).bindPopup(
      `<strong>${i + 1}. ${escapeHtml(s.location_name)}</strong><br>` +
      `<small>${escapeHtml(s.primary_category || "")}</small>`
    );
  });
  if (latlngs.length >= 1) {
    map.fitBounds(L.latLngBounds(latlngs).pad(0.2));
  }
  _dayMaps[mapId] = map;
}

/* --------------------- utils --------------------- */
function setStatus(msg, kind) {
  const el = $("#status");
  el.className = "status " + (kind || "");
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearStatus() {
  const el = $("#status");
  el.className = "status hidden";
  el.textContent = "";
}
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** Backend notices use **bold** markdown; render safely as <strong>. */
function formatNoticeHtml(s) {
  if (!s) return "";
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}
