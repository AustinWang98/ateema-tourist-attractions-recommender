/* ChicagoDoes recommender — vanilla SPA frontend */

const API = {
  health:      "/api/health",
  categories:  "/api/categories",
  recommend:   "/api/recommend",
  itinerary:   "/api/itinerary",
  card:        "/api/location/card",
  outbound:    "/api/outbound/click",
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
  trendingItems: [],
  trendingPage: 0,
  wizardStep: 0,
  wizardTripDaysTouched: false,
  wizardChoiceTouched: new Set(),
  cardCache: {},   // location_id -> {image_url, blurb, link_url, link_type, ...}
};

const DEFAULT_TOP_K = 40;
const WIZARD_STEPS = 5;

/* --------------------- boot --------------------- */
window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadHealth(), loadCategories(), loadWelcomeTeasers()]);
  enhanceCustomSelects();
  wireOutboundTracking();
  wireOnboarding();
  wireTopCountControls();
  wireProfileWizard();
  wireForm();
  syncTopCount(DEFAULT_TOP_K);
  updateWizard();
});

function selectedOptionLabel(select) {
  return select?.selectedOptions?.[0]?.textContent?.trim() || "Choose";
}

function refreshCustomSelect(select) {
  if (!select) return;
  const wrap = select.closest(".custom-select");
  const button = wrap?.querySelector(".custom-select__button");
  const menu = wrap?.querySelector(".custom-select__menu");
  if (button) button.textContent = selectedOptionLabel(select);
  if (menu) {
    menu.querySelectorAll("[data-value]").forEach((item) => {
      const isSelected = item.dataset.value === select.value;
      item.classList.toggle("is-selected", isSelected);
      item.setAttribute("aria-selected", isSelected ? "true" : "false");
    });
  }
}

function refreshAllCustomSelects() {
  $$("select").forEach(refreshCustomSelect);
}

function closeCustomSelects(except = null) {
  $$(".custom-select.is-open").forEach((wrap) => {
    if (wrap === except) return;
    wrap.classList.remove("is-open");
    wrap.querySelector(".custom-select__button")?.setAttribute("aria-expanded", "false");
  });
}

function enhanceCustomSelects() {
  $$("select").forEach((select) => {
    if (select.dataset.customSelectEnhanced === "true") return;
    select.dataset.customSelectEnhanced = "true";
    select.classList.add("native-select");

    const wrap = document.createElement("div");
    wrap.className = "custom-select";
    select.parentNode.insertBefore(wrap, select);
    wrap.appendChild(select);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "custom-select__button";
    button.setAttribute("aria-haspopup", "listbox");
    button.setAttribute("aria-expanded", "false");
    button.textContent = selectedOptionLabel(select);

    const menu = document.createElement("div");
    menu.className = "custom-select__menu";
    menu.setAttribute("role", "listbox");

    Array.from(select.options).forEach((option) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "custom-select__option";
      item.dataset.value = option.value;
      item.textContent = option.textContent;
      item.setAttribute("role", "option");
      item.addEventListener("click", () => {
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        refreshCustomSelect(select);
        closeCustomSelects();
      });
      menu.appendChild(item);
    });

    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const willOpen = !wrap.classList.contains("is-open");
      closeCustomSelects(wrap);
      wrap.classList.toggle("is-open", willOpen);
      button.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });

    button.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeCustomSelects();
        button.focus();
      }
      if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        closeCustomSelects(wrap);
        wrap.classList.add("is-open");
        button.setAttribute("aria-expanded", "true");
        menu.querySelector(".is-selected, .custom-select__option")?.focus();
      }
    });

    menu.addEventListener("keydown", (event) => {
      const options = Array.from(menu.querySelectorAll(".custom-select__option"));
      const index = options.indexOf(document.activeElement);
      if (event.key === "Escape") {
        closeCustomSelects();
        button.focus();
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        options[Math.min(options.length - 1, index + 1)]?.focus();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        options[Math.max(0, index - 1)]?.focus();
      }
    });

    wrap.append(button, menu);
    refreshCustomSelect(select);
  });

  document.addEventListener("click", () => closeCustomSelects());
}

async function loadWelcomeTeasers() {
  // Populate the welcome-card stats and a short "trending right now" row.
  // Best-effort: silently no-op if endpoints aren't ready.
  try {
    const h = await fetch(API.health).then((r) => r.json());
    const loc = $("#welcome-locations");
    if (h.n_locations && loc) loc.textContent = String(h.n_locations);
  } catch (_) {}
  try {
    const r = await fetch(API.trending + "?limit=18").then((r) => r.json());
    const row = $("#welcome-trend-row");
    const wrap = $("#welcome-trending");
    if (!r.locations || !r.locations.length) return;
    state.trendingItems = r.locations.map((l, i) => ({
      ...l,
      is_trending: true,
      _rank: i + 1,
    }));
    row.innerHTML = state.trendingItems.slice(0, 6).map(
      (l) => `<span class="welcome__trend-chip">${escapeHtml(l.location_name)}</span>`
    ).join("");
    wrap.classList.remove("hidden");
  } catch (_) {}
}

async function loadHealth() {
  const el = $("#health");
  if (!el) return;
  try {
    const r = await fetch(API.health).then((r) => r.json());
    el.textContent = `${r.n_locations || 350} places across Chicago`;
    el.classList.add(r.ok ? "ok" : "bad");
  } catch (e) {
    el.classList.add("hidden");
  }
}

async function loadCategories() {
  const r = await fetch(API.categories).then((r) => r.json());
  state.categories = r.categories || [];
  renderAllChips();
}

function renderAllChips() {
  renderChips("#interests", state.categories, state.selectedInterests, renderAllChips);
  renderChips("#avoid", state.categories, state.selectedAvoid, renderAllChips);
  renderChips("#wizard-interests", state.categories, state.selectedInterests, renderAllChips);
  renderChips("#wizard-avoid", state.categories, state.selectedAvoid, renderAllChips);
  updateWizardSummary();
}

function renderChips(containerSel, items, selectedSet, onChange = null) {
  const root = $(containerSel);
  if (!root) return;
  root.innerHTML = "";
  for (const item of items) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = item;
    chip.setAttribute("aria-pressed", selectedSet.has(item) ? "true" : "false");
    if (selectedSet.has(item)) chip.classList.add("active");
    chip.addEventListener("click", () => {
      if (selectedSet.has(item)) {
        selectedSet.delete(item);
        chip.classList.remove("active");
        chip.setAttribute("aria-pressed", "false");
      } else {
        selectedSet.add(item);
        chip.classList.add("active");
        chip.setAttribute("aria-pressed", "true");
      }
      if (onChange) onChange();
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
  $("#trending-prev")?.addEventListener("click", () => changeTrendingPage(-1));
  $("#trending-next")?.addEventListener("click", () => changeTrendingPage(1));
  if (!window.__recoResizeBound) {
    window.__recoResizeBound = true;
    window.addEventListener("resize", () => {
      if (state.recoItems.length) renderRecoPage();
      if (state.trendingItems.length && !$("#trending-panel")?.classList.contains("hidden")) {
        renderTrendingPage();
      }
    });
  }
}

function wireOnboarding() {
  $("#start-personalize")?.addEventListener("click", () => {
    activateExperience();
    showProfileWizard();
  });

  $("#start-top")?.addEventListener("click", async () => {
    activateExperience();
    syncTopCount($("#quick_top_k")?.value || DEFAULT_TOP_K);
    resetPreferences();
    hideProfilePanel();
    hideProfileWizard();
    await runRecommend({ source: "instant" });
  });

  $("#edit-profile")?.addEventListener("click", () => {
    showProfilePanel();
    const target = $("#traveler_type");
    if (target) setTimeout(() => target.focus(), 180);
  });

  $("#profile-close")?.addEventListener("click", () => {
    hideProfilePanel();
  });

  $("#welcome-trending")?.addEventListener("click", () => {
    activateExperience();
    showTrendingPanel();
  });
}

function activateExperience() {
  document.documentElement.classList.remove("is-landing");
  document.body.classList.remove("is-landing");
}

function wireTopCountControls() {
  $$(".top-count-select").forEach((select) => {
    select.addEventListener("change", async () => {
      syncTopCount(select.value);
      if (
        select.id === "results_top_k" &&
        state.lastRecommendations &&
        !$("#reco-panel")?.classList.contains("hidden")
      ) {
        await runRecommend({ source: "results-top-k" });
      }
    });
  });
}

function topCountValue() {
  const raw = Number($("#top_k")?.value || DEFAULT_TOP_K);
  return Math.max(5, Math.min(80, Number.isFinite(raw) ? raw : DEFAULT_TOP_K));
}

function syncTopCount(value) {
  const n = Math.max(5, Math.min(80, Number(value) || DEFAULT_TOP_K));
  $$(".top-count-select").forEach((select) => {
    select.value = String(n);
    refreshCustomSelect(select);
  });
  updateWizardSummary();
}

function showProfilePanel(options = {}) {
  const panel = $("#profile-panel");
  if (!panel) return;
  panel.classList.remove("is-dormant");
  document.querySelector(".workspace")?.classList.remove("is-results-only");
  if (options.scroll !== false) {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function hideProfilePanel() {
  $("#profile-panel")?.classList.add("is-dormant");
  document.querySelector(".workspace")?.classList.add("is-results-only");
}

function resetPreferences() {
  state.selectedInterests.clear();
  state.selectedAvoid.clear();
  state.wizardChoiceTouched.clear();
  $("#traveler_type").value = "";
  $("#vibe").value = "";
  $("#trip_days").value = "";
  $("#free_text").value = "";
  $("#use_ai_itinerary").checked = false;
  state.wizardTripDaysTouched = false;
  $("#wizard_trip_days").value = "";
  $("#wizard_free_text").value = "";
  $("#wizard_use_ai_itinerary").checked = false;
  syncTopCount($("#quick_top_k")?.value || DEFAULT_TOP_K);
  $("#ai-itin-hint").classList.add("hidden");
  renderAllChips();
  updateWizardChoiceState();
  updateWizardSummary();
  clearItinerary();
  refreshAllCustomSelects();
}

function wireProfileWizard() {
  $$(".wizard-choice").forEach((btn) => {
    btn.addEventListener("click", () => {
      const field = btn.dataset.field;
      const value = btn.dataset.value || "";
      const input = field ? $(`#${field}`) : null;
      if (input) input.value = value;
      refreshCustomSelect(input);
      if (field) state.wizardChoiceTouched.add(field);
      updateWizardChoiceState();
      updateWizardSummary();
      setTimeout(() => wizardGo(1), 140);
    });
  });

  $("#wizard-back")?.addEventListener("click", () => wizardGo(-1));
  $("#wizard-next")?.addEventListener("click", () => wizardGo(1));
  $("#wizard-finish")?.addEventListener("click", async () => {
    syncProfileFromWizard();
    hideProfileWizard();
    showProfilePanel({ scroll: false });
    await runRecommend({ source: "wizard" });
    $("#reco-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  $("#wizard_trip_days")?.addEventListener("change", () => {
    state.wizardTripDaysTouched = !!$("#wizard_trip_days")?.value;
    syncProfileFromWizard();
  });
  $("#wizard_free_text")?.addEventListener("input", syncProfileFromWizard);
  $("#wizard_use_ai_itinerary")?.addEventListener("change", syncProfileFromWizard);
}

function showProfileWizard() {
  $("#start-panel")?.classList.add("hidden");
  $("#profile-wizard")?.classList.remove("hidden");
  hideProfilePanel();
  state.wizardStep = 0;
  syncWizardFromProfile();
  updateWizard();
  $("#profile-wizard")?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideProfileWizard() {
  $("#profile-wizard")?.classList.add("hidden");
  $("#start-panel")?.classList.remove("hidden");
}

function wizardGo(direction) {
  state.wizardStep = Math.max(0, Math.min(WIZARD_STEPS - 1, state.wizardStep + direction));
  updateWizard();
}

function syncWizardFromProfile() {
  $("#wizard_trip_days").value = state.wizardTripDaysTouched ? ($("#trip_days").value || "2") : "";
  $("#wizard_free_text").value = $("#free_text").value || "";
  $("#wizard_use_ai_itinerary").checked = $("#use_ai_itinerary").checked;
  syncTopCount($("#top_k")?.value || DEFAULT_TOP_K);
  refreshAllCustomSelects();
  updateWizardChoiceState();
  updateWizardSummary();
}

function syncProfileFromWizard() {
  const wizardDays = $("#wizard_trip_days")?.value || "";
  if (wizardDays) {
    $("#trip_days").value = wizardDays;
    refreshCustomSelect($("#trip_days"));
  }
  if ($("#wizard_free_text")) $("#free_text").value = $("#wizard_free_text").value.trim();
  if ($("#wizard_use_ai_itinerary")) {
    $("#use_ai_itinerary").checked = $("#wizard_use_ai_itinerary").checked;
    $("#ai-itin-hint").classList.toggle("hidden", !$("#use_ai_itinerary").checked);
  }
  syncTopCount($("#wizard_top_k")?.value || $("#top_k")?.value || DEFAULT_TOP_K);
  refreshAllCustomSelects();
  updatePlanButton();
  updateWizardSummary();
}

function updateWizardChoiceState() {
  $$(".wizard-choice").forEach((btn) => {
    const field = btn.dataset.field;
    const value = btn.dataset.value || "";
    const current = field ? ($(`#${field}`)?.value || "") : "";
    const selected = !!field && state.wizardChoiceTouched.has(field) && current === value;
    btn.classList.toggle("is-selected", selected);
    btn.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}

function updateWizard() {
  const step = state.wizardStep;
  $$(".wizard__stage").forEach((el) => {
    el.classList.toggle("is-active", Number(el.dataset.wizardStep) === step);
  });
  const current = $("#wizard-step-current");
  const total = $("#wizard-step-total");
  const bar = $("#wizard-progress-bar");
  const back = $("#wizard-back");
  const next = $("#wizard-next");
  const finish = $("#wizard-finish");
  if (current) current.textContent = String(step + 1);
  if (total) total.textContent = String(WIZARD_STEPS);
  if (bar) bar.style.width = `${((step + 1) / WIZARD_STEPS) * 100}%`;
  if (back) back.disabled = step === 0;
  if (next) next.classList.toggle("hidden", step === WIZARD_STEPS - 1);
  if (finish) finish.classList.toggle("hidden", step !== WIZARD_STEPS - 1);
  updateWizardChoiceState();
  updateWizardSummary();
}

function updateWizardSummary() {
  const box = $("#wizard-summary");
  if (!box) return;
  const parts = [];
  const traveler = $("#traveler_type")?.selectedOptions?.[0]?.textContent || "";
  const vibe = $("#vibe")?.selectedOptions?.[0]?.textContent || "";
  if ($("#traveler_type")?.value) parts.push(traveler);
  if ($("#vibe")?.value) parts.push(vibe);
  const interests = Array.from(state.selectedInterests).slice(0, 3);
  if (interests.length) parts.push(interests.join(", "));
  const avoid = Array.from(state.selectedAvoid).slice(0, 2);
  if (avoid.length) parts.push(`Avoid ${avoid.join(", ")}`);
  parts.push(`Top ${topCountValue()}`);
  const days = $("#wizard_trip_days")?.value || "";
  if (state.wizardTripDaysTouched && days) {
    parts.push(`${days} day${Number(days) === 1 ? "" : "s"}`);
  }
  box.textContent = parts.length ? parts.join(" · ") : "No preferences selected yet.";
}

function layoutWidth(viewportSel) {
  const viewport = document.querySelector(viewportSel);
  const panel = viewport?.closest(".results-panel, .trending-panel, .panel");
  const candidates = [
    viewport?.clientWidth,
    viewport?.getBoundingClientRect?.().width,
    document.querySelector(".results-panel")?.clientWidth,
    document.querySelector(".results-panel")?.getBoundingClientRect?.().width,
    panel?.clientWidth,
    panel?.getBoundingClientRect?.().width,
    document.querySelector(".workspace")?.clientWidth,
    document.querySelector(".workspace")?.getBoundingClientRect?.().width,
    window.innerWidth,
  ];
  return candidates.find((value) => Number.isFinite(value) && value > 0) || window.innerWidth;
}

/** 3×2 on desktop, 2×2 tablet, 1×2 narrow — keep cards near their original size. */
function recoLayout() {
  const w = layoutWidth(".reco-carousel__viewport");
  if (w >= 900) return { cols: 3, rows: 2, perPage: 6 };
  if (w >= 580) return { cols: 2, rows: 2, perPage: 4 };
  return { cols: 1, rows: 2, perPage: 2 };
}

function recoPerPage() {
  return recoLayout().perPage;
}

function carouselLayout(viewportSel = ".reco-carousel__viewport") {
  const w = layoutWidth(viewportSel);
  if (w >= 900) return { cols: 3, rows: 2, perPage: 6 };
  if (w >= 580) return { cols: 2, rows: 2, perPage: 4 };
  return { cols: 1, rows: 2, perPage: 2 };
}

function trendingLayout() {
  return carouselLayout("#trending-viewport");
}

function trendingPerPage() {
  return trendingLayout().perPage;
}

function applyRecoGridLayout() {
  const grid = $("#reco-grid");
  if (!grid) return;
  const { cols } = recoLayout();
  grid.classList.remove("reco-grid--1", "reco-grid--2", "reco-grid--3");
  grid.classList.add(`reco-grid--${cols}`);
}

function applyTrendingGridLayout() {
  const grid = $("#trending-grid");
  if (!grid) return;
  const { cols } = trendingLayout();
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

function trendingPageCount() {
  const n = state.trendingItems.length;
  if (!n) return 0;
  return Math.ceil(n / trendingPerPage());
}

function changeTrendingPage(direction) {
  const pages = trendingPageCount();
  if (!pages) return;
  state.trendingPage = Math.max(0, Math.min(pages - 1, state.trendingPage + direction));
  renderTrendingPage();
}

function updateRecoPagerButtons() {
  const prev = $("#reco-prev");
  const next = $("#reco-next");
  if (!prev || !next) return;
  const pages = recoPageCount();
  prev.disabled = state.recoPage <= 0;
  next.disabled = pages <= 1 || state.recoPage >= pages - 1;
}

function updateTrendingPagerButtons() {
  const prev = $("#trending-prev");
  const next = $("#trending-next");
  if (!prev || !next) return;
  const pages = trendingPageCount();
  prev.disabled = state.trendingPage <= 0;
  next.disabled = pages <= 1 || state.trendingPage >= pages - 1;
}

/* --------------------- requests --------------------- */
function readForm() {
  return {
    user_key:        null,
    traveler_type:   $("#traveler_type").value || null,
    vibe:            $("#vibe").value || null,
    trip_days:       Number($("#trip_days").value || 2),
    free_text:       $("#free_text").value.trim() || null,
    interests:       Array.from(state.selectedInterests),
    avoid_categories:Array.from(state.selectedAvoid),
    top_k:           topCountValue(),
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

async function runRecommend(options = {}) {
  const payload = readForm();
  state.lastRequest = payload;
  setStatus("Ranking the best Chicago places for you…", "loading");
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
    if (options.source === "instant") {
      $("#reco-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
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
    setStatus("Turn on **Plan my days with AI** to build a day plan.", "error");
    return;
  }
  const poolIds = recommendationPoolIds();
  if (!poolIds.length) {
    setStatus("Get recommendations first — the AI only arranges your Top picks.", "error");
    return;
  }
  setStatus("Building your day-by-day plan with AI…", "loading");
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
  state.wizardChoiceTouched = new Set(
    ["traveler_type", "vibe"].filter((field) => !!req[field])
  );
  $("#trip_days").value = req.trip_days ? String(req.trip_days) : "";
  state.wizardTripDaysTouched = !!req.trip_days;
  syncTopCount(req.top_k || DEFAULT_TOP_K);
  state.selectedInterests = new Set(req.interests || []);
  state.selectedAvoid = new Set(req.avoid_categories || []);
  renderAllChips();
  syncWizardFromProfile();
}

/* --------------------- rendering --------------------- */
function createRecoTile(r, rank, surface = "top_picks") {
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
    ? `<span class="tag tag--trend">Trending</span>`
    : "";

  const icon = CATEGORY_ICON[r.primary_category] || "📍";
  const defaultLink = chicagoDoesUrl(r.location_id);
  const trackAttrs =
    `data-track-outbound="1" ` +
    `data-location-id="${escapeHtml(r.location_id)}" ` +
    `data-location-name="${escapeHtml(r.location_name)}" ` +
    `data-surface="${escapeHtml(surface)}" ` +
    `data-rank="${rank}" ` +
    `data-link-type="chicagodoes"`;
  tile.innerHTML = `
    <a class="tile__media" data-role="link" ${trackAttrs} href="${escapeHtml(defaultLink)}"
       target="_blank" rel="noopener noreferrer" aria-label="${escapeHtml(r.location_name)}">
      <div class="tile__media-ph" data-role="ph"><span>${icon}</span></div>
      <span class="tile__rank">#${rank}</span>
      ${trendingBadge ? `<span class="tile__media-trend">Trending</span>` : ""}
    </a>
    <div class="tile__body">
      <a class="tile__title-link" data-role="link" ${trackAttrs} href="${escapeHtml(defaultLink)}"
         target="_blank" rel="noopener noreferrer">
        <span class="tile__name">${escapeHtml(r.location_name)}</span>
        <span class="tile__link-ico" aria-hidden="true">↗</span>
      </a>
      <div class="tile__cats">${cats || `<span class="muted">No categories</span>`}</div>
      <div class="tile__blurb" data-role="blurb"><span class="tile__blurb-skel"></span></div>
    </div>
  `;
  hydrateCard(r, tile);
  return tile;
}

const MAPME_LOCATION_BASE =
  "https://viewer.mapme.com/chicagodoesinteractivevideomaps/location/";

function chicagoDoesUrl(locationId) {
  const lid = String(locationId || "").trim();
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(lid)) {
    return MAPME_LOCATION_BASE + lid;
  }
  return "https://www.chicagodoes.com/";
}

function wireOutboundTracking() {
  if (window.__outboundTrackingBound) return;
  window.__outboundTrackingBound = true;
  document.addEventListener("click", handleTrackedOutboundClick, true);
}

function trackedOutboundUrl(href, meta) {
  try {
    const url = new URL(href, window.location.href);
    if (!/^https?:$/.test(url.protocol)) return href;
    url.searchParams.set("utm_source", "ateema_recommender");
    url.searchParams.set("utm_medium", "recommendation");
    url.searchParams.set("utm_campaign", "capstone_recsys");
    url.searchParams.set("rec_source", "ateema_recommender");
    url.searchParams.set("rec_surface", meta.surface || "unknown");
    if (meta.rank) url.searchParams.set("rec_rank", String(meta.rank));
    if (meta.locationId) url.searchParams.set("rec_location_id", meta.locationId);
    if (meta.linkType) url.searchParams.set("rec_link_type", meta.linkType);
    return url.toString();
  } catch (_) {
    return href;
  }
}

function handleTrackedOutboundClick(event) {
  const a = event.target?.closest?.('a[data-track-outbound="1"]');
  if (!a) return;
  const meta = {
    locationId: a.dataset.locationId || "",
    locationName: a.dataset.locationName || "",
    surface: a.dataset.surface || "unknown",
    rank: Number(a.dataset.rank || 0) || null,
    linkType: a.dataset.linkType || "",
  };
  const trackedHref = trackedOutboundUrl(a.href, meta);
  a.href = trackedHref;
  recordOutboundClick(meta, trackedHref);
}

function recordOutboundClick(meta, href) {
  const payload = {
    location_id: meta.locationId || null,
    location_name: meta.locationName || null,
    surface: meta.surface || "unknown",
    rank: meta.rank,
    link_type: meta.linkType || null,
    href,
  };
  const body = JSON.stringify(payload);
  if (navigator.sendBeacon) {
    const blob = new Blob([body], { type: "application/json" });
    navigator.sendBeacon(API.outbound, blob);
    return;
  }
  fetch(API.outbound, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {});
}

const LINK_LABEL = {
  chicagodoes: "View on the ChicagoDoes map",
};

/** Build playable media list from a card API payload. */
function youtubeVideoId(url) {
  const match = String(url || "").match(/(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  return match ? match[1] : null;
}

function isYouTubeUrl(url) {
  return !!youtubeVideoId(url);
}

function normalizeMediaItem(item) {
  if (!item || !item.url) return null;
  if (item.type === "video" && isYouTubeUrl(item.url)) {
    const vid = youtubeVideoId(item.url);
    return {
      ...item,
      type: "image",
      url: `https://img.youtube.com/vi/${vid}/hqdefault.jpg`,
      attribution: item.attribution || "Video thumbnail: YouTube",
    };
  }
  return item;
}

function uniqueMediaItems(items) {
  const seen = new Set();
  return (items || []).filter((item) => {
    if (!item || !item.url) return false;
    const key = `${item.type || "image"}::${String(item.url).split("?")[0].toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function cardMediaItems(card) {
  if (!card) return [];
  if (Array.isArray(card.media_items) && card.media_items.length) {
    const items = uniqueMediaItems(card.media_items
      .map(normalizeMediaItem)
      .filter((m) => m && m.url));
    const playableVideos = items.filter((m) => m.type === "video");
    const images = items.filter((m) => m.type === "image");
    return [...playableVideos, ...images];
  }
  const items = [];
  if (card.video_url) {
    items.push({
      type: "video",
      url: card.video_url,
      attribution: card.video_attribution || card.image_attribution,
    });
  }
  if (card.image_url) {
    items.push({
      type: "image",
      url: card.image_url,
      attribution: card.image_attribution,
    });
  }
  return uniqueMediaItems(items.map(normalizeMediaItem).filter((m) => m && m.url));
}

/** Insert a photo, video, or slideshow into a media container. */
function applyCardMedia(container, card, altText) {
  if (!container || !card) return;
  const items = cardMediaItems(card);
  if (!items.length) return;

  const ph = container.querySelector('[data-role="ph"]');
  container.querySelectorAll(".tile__img, .tile__vid, .tile__gallery").forEach((el) => el.remove());

  const show = (el) => {
    if (ph) ph.classList.add("is-hidden");
    container.insertBefore(el, container.firstChild);
    requestAnimationFrame(() => el.classList.add("is-loaded"));
  };

  if (items.length === 1) {
    const item = items[0];
    if (item.type === "video" && !isYouTubeUrl(item.url)) {
      const vid = document.createElement("video");
      vid.className = "tile__vid";
      vid.muted = true;
      vid.loop = true;
      vid.playsInline = true;
      vid.autoplay = true;
      vid.preload = "metadata";
      vid.setAttribute("aria-label", altText);
      const poster = items.find((m) => m.type === "image");
      if (poster) vid.poster = poster.url;
      vid.onloadeddata = () => show(vid);
      vid.onerror = () => {
        const imgItem = items.find((m) => m.type === "image");
        if (imgItem) applyCardMedia(container, { media_items: [imgItem] }, altText);
      };
      vid.src = item.url;
      if (item.attribution) container.title = item.attribution;
      return;
    }
    if (item.type === "video") {
      const img = document.createElement("img");
      img.className = "tile__img";
      img.alt = `${altText} (video on ChicagoDoes)`;
      img.loading = "eager";
      const poster = items.find((m) => m.type === "image");
      img.src = poster ? poster.url : item.url;
      img.onload = () => show(img);
      container.title = item.attribution || "Video on ChicagoDoes";
      return;
    }
    const img = document.createElement("img");
    img.className = "tile__img";
    img.alt = altText;
    img.loading = "eager";
    img.referrerPolicy = "no-referrer";
    const onReady = () => show(img);
    img.onload = onReady;
    img.onerror = () => {};
    img.src = item.url;
    if (img.complete && img.naturalWidth > 0) onReady();
    if (item.attribution) container.title = item.attribution;
    return;
  }

  // Slideshow for multiple ChicagoDoes photos / videos.
  const gallery = document.createElement("div");
  gallery.className = "tile__gallery";
  gallery.setAttribute("role", "group");
  gallery.setAttribute("aria-label", `${altText} photos`);

  const track = document.createElement("div");
  track.className = "tile__gallery-track";
  gallery.appendChild(track);

  const mkNav = (label, cls, delta) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `tile__gallery-nav tile__gallery-nav--${cls}`;
    btn.setAttribute("aria-label", label);
    btn.innerHTML = cls === "prev" ? "&#8249;" : "&#8250;";
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      go(active + delta);
      resetTimer();
    });
    return btn;
  };
  gallery.appendChild(mkNav("Previous photo", "prev", -1));
  gallery.appendChild(mkNav("Next photo", "next", 1));

  const dots = document.createElement("div");
  dots.className = "tile__gallery-dots";
  gallery.appendChild(dots);

  let active = 0;
  let timer = null;

  const renderSlide = (item) => {
    track.innerHTML = "";
    if (item.type === "video" && !isYouTubeUrl(item.url)) {
      const vid = document.createElement("video");
      vid.className = "tile__vid";
      vid.muted = true;
      vid.loop = true;
      vid.playsInline = true;
      vid.autoplay = true;
      vid.preload = "metadata";
      vid.src = item.url;
      track.appendChild(vid);
      return;
    }
    const img = document.createElement("img");
    img.className = "tile__img";
    img.alt = altText;
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";
    img.src = item.url;
    track.appendChild(img);
  };

  const go = (idx) => {
    active = (idx + items.length) % items.length;
    renderSlide(items[active]);
    dots.querySelectorAll("button").forEach((b, i) => {
      b.classList.toggle("is-active", i === active);
      b.setAttribute("aria-selected", i === active ? "true" : "false");
    });
    if (items[active].attribution) container.title = items[active].attribution;
  };

  items.forEach((_, i) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "tile__gallery-dot";
    dot.setAttribute("aria-label", `Show photo ${i + 1} of ${items.length}`);
    dot.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      go(i);
      resetTimer();
    });
    dots.appendChild(dot);
  });

  const resetTimer = () => {
    if (timer) clearInterval(timer);
    timer = setInterval(() => go(active + 1), 4500);
  };

  gallery.addEventListener("mouseenter", () => { if (timer) clearInterval(timer); });
  gallery.addEventListener("mouseleave", resetTimer);

  go(0);
  resetTimer();
  show(gallery);
  container._galleryTimer = timer;
}

/** Lazily fetch photo + specialty blurb + best link for one card and apply it. */
async function hydrateCard(r, tile) {
  const apply = (card) => {
    if (!card) return;
    const blurbEl = tile.querySelector('[data-role="blurb"]');
    if (blurbEl && card.blurb) blurbEl.textContent = card.blurb;

    const label = LINK_LABEL[card.link_type] || "Open link";
    tile.querySelectorAll('[data-role="link"]').forEach((a) => {
      if (card.link_url) a.href = card.link_url;
      a.title = label;
      a.dataset.linkType = card.link_type || "";
    });

    const media = tile.querySelector(".tile__media");
    if (media && cardMediaItems(card).length) {
      applyCardMedia(media, card, r.location_name);
    }
  };

  const cached = state.cardCache[r.location_id];
  if (cached && cardMediaItems(cached).length) {
    apply(cached);
    return;
  }

  try {
    const resp = await fetch(API.card, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: r.location_id }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const card = await resp.json();
    if (cardMediaItems(card).length) {
      state.cardCache[r.location_id] = card;
    }
    apply(card);
  } catch (err) {
    // Network hiccup: show a neutral line, never the internal ranking reason.
    const blurbEl = tile.querySelector('[data-role="blurb"]');
    if (blurbEl) blurbEl.textContent = "A noteworthy Chicago spot worth a visit.";
  }
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

async function ensureTrendingItems() {
  if (state.trendingItems.length) return state.trendingItems;
  const r = await fetch(API.trending + "?limit=18").then((resp) => resp.json());
  state.trendingItems = (r.locations || []).map((l, i) => ({
    ...l,
    is_trending: true,
    _rank: i + 1,
  }));
  return state.trendingItems;
}

async function showTrendingPanel() {
  setStatus("Loading places trending now…", "loading");
  try {
    await ensureTrendingItems();
    if (!state.trendingItems.length) {
      setStatus("No trending places are available right now.", "error");
      return;
    }
    $("#reco-panel")?.classList.add("hidden");
    $("#itin-panel")?.classList.add("hidden");
    const panel = $("#trending-panel");
    panel?.classList.remove("hidden");
    const meta = $("#trending-meta");
    if (meta) meta.textContent = `${state.trendingItems.length} places from recent ChicagoDoes activity`;
    const countEl = $("#trending-count");
    if (countEl) countEl.textContent = String(state.trendingItems.length);
    state.trendingPage = 0;
    renderTrendingPage();
    clearStatus();
    panel?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus(`Trending places failed: ${err.message}`, "error");
  }
}

function renderTrendingPage() {
  const grid = $("#trending-grid");
  if (!grid) return;

  applyTrendingGridLayout();
  const perPage = trendingPerPage();
  const total = state.trendingItems.length;
  const pages = trendingPageCount();
  if (pages) {
    state.trendingPage = Math.max(0, Math.min(pages - 1, state.trendingPage));
  }

  const start = state.trendingPage * perPage;
  const slice = state.trendingItems.slice(start, start + perPage);
  grid.innerHTML = "";
  slice.forEach((r) => grid.appendChild(createRecoTile(r, r._rank, "trending_now")));

  const rangeEl = $("#trending-range");
  const pageNum = $("#trending-page-num");
  const pageTotal = $("#trending-page-total");
  const hint = $("#trending-scroll-hint");
  if (rangeEl && total) {
    const end = start + slice.length;
    rangeEl.textContent = `${start + 1}–${end}`;
  }
  if (pageNum) pageNum.textContent = String(pages ? state.trendingPage + 1 : 0);
  if (pageTotal) pageTotal.textContent = String(pages || 1);
  if (hint) {
    hint.classList.toggle("hidden", total <= perPage);
  }
  updateTrendingPagerButtons();
}

function renderRecommendations(data) {
  const panel = $("#reco-panel");
  const meta = $("#reco-meta");
  panel.classList.remove("hidden");
  $("#trending-panel")?.classList.add("hidden");

  const nRecs = (data.recommendations || []).length;
  const requestedN = state.lastRequest?.top_k || nRecs;
  const chosen = (state.lastRequest?.interests || []);
  const interestsTxt = chosen.slice(0, 4).join(", ");
  if (interestsTxt) {
    meta.textContent = `${nRecs} of Top ${requestedN} places · ${interestsTxt}`;
  } else {
    meta.textContent = `Top ${nRecs} places across Chicago`;
  }

  const countEl = $("#reco-count");
  if (countEl) countEl.textContent = String(nRecs);

  state.recoItems = (data.recommendations || []).map((r, i) => {
    r._rank = i + 1;
    return r;
  });
  state.recoPage = 0;
  requestAnimationFrame(() => renderRecoPage());
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

    box.querySelectorAll(".stop").forEach((el, i) => {
      hydrateStopMedia(el, stops[i]);
    });

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
    ? '<span class="stop__geo-tag" title="Could not geocode this location precisely.">approx.</span>'
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
  const sourceTag = s.source === "ai"
    ? '<span class="stop__src-tag" title="A well-known Chicago place the AI added to complete your day.">✨ AI pick</span>'
    : "";
  const note = s.note
    ? `<div class="stop__meta">${escapeHtml(s.note)}</div>`
    : "";
  const link = s.link_url || (s.source !== "ai" ? chicagoDoesUrl(s.location_id) : null);
  const linkTitle = s.link_type === "official"
    ? "Official website"
    : "View on the ChicagoDoes map";
  const linkType = s.link_type || (s.source !== "ai" ? "chicagodoes" : "official");
  const trackAttrs =
    `data-track-outbound="1" ` +
    `data-location-id="${escapeHtml(s.location_id || "")}" ` +
    `data-location-name="${escapeHtml(s.location_name)}" ` +
    `data-surface="ai_itinerary" ` +
    `data-rank="${idx}" ` +
    `data-link-type="${escapeHtml(linkType)}"`;
  const nameHtml = link
    ? `<a class="stop__link" ${trackAttrs} href="${escapeHtml(link)}"
         target="_blank" rel="noopener noreferrer"
         title="${escapeHtml(linkTitle)}">
        <span class="stop__link-name">${escapeHtml(s.location_name)}</span>
        <span class="stop__link-ico" aria-hidden="true">↗</span>
      </a>`
    : `<span class="stop__link-name">${escapeHtml(s.location_name)}</span>`;
  const hasAttachedMedia = Boolean(
    s.image_url ||
    s.video_url ||
    (Array.isArray(s.media_items) && s.media_items.length)
  );
  const canFetchCatalogMedia = Boolean(s.location_id && s.source !== "ai");
  const mediaHtml = (hasAttachedMedia || canFetchCatalogMedia)
    ? `<div class="stop__media" data-role="stop-media">
         <div class="stop__media-ph" data-role="ph"><span>${icon}</span></div>
       </div>`
    : "";
  const hasMedia = Boolean(mediaHtml);
  const simple = hasMedia ? " stop--media" : "";
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
          ${nameHtml}
          ${catTag}
          ${sourceTag}
          ${geoFallback}
        </div>
        ${note}
      </div>
      ${mediaHtml}
    </div>
  `;
}

function hydrateStopMedia(stopEl, s) {
  if (!s) return;
  const media = stopEl.querySelector('[data-role="stop-media"]');
  if (!media) return;
  const card = {
    media_items: s.media_items,
    image_url: s.image_url,
    video_url: s.video_url,
    image_attribution: s.image_attribution,
    video_attribution: s.video_attribution,
  };
  if (cardMediaItems(card).length) {
    applyCardMedia(media, card, s.location_name);
    return;
  }
  const lid = s.location_id;
  if (!lid) return;
  const cached = state.cardCache[lid];
  if (cached && cardMediaItems(cached).length) {
    applyCardMedia(media, cached, s.location_name);
    return;
  }
  fetch(API.card, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ location_id: lid }),
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((c) => {
      if (!c) return;
      if (cardMediaItems(c).length) state.cardCache[lid] = c;
      applyCardMedia(media, c, s.location_name);
    })
    .catch(() => {});
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
