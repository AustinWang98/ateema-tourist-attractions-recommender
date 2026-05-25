"""Generate publication-quality diagrams for the capstone paper.

Produces:
  - architecture.png:   layered data warehouse + recommender architecture
  - score_blend.png:    new-user vs returning-user weight comparison
  - mmr_diagram.png:    MMR re-ranking schematic
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------------------------------------------------
# Figure: system architecture (5-layer)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

LAYER_COLORS = {
    "raw":   "#dde7f0",
    "wh":    "#bfd1e5",
    "feat":  "#7ea8c9",
    "model": "#4a78a5",
    "app":   "#274972",
}
TEXT_DARK = "#0d1b2a"
TEXT_LIGHT = "#ffffff"


def box(ax, x, y, w, h, text, color, text_color=TEXT_DARK):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8, edgecolor="#1b2a3a", facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=9.5, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=11,
                        color="#1b2a3a", linewidth=0.9)
    ax.add_patch(a)


box(ax, 0.2, 5.0, 2.3, 0.7, "Raw GA4 Events\n(events_*)", LAYER_COLORS["raw"])
box(ax, 2.8, 5.0, 2.3, 0.7, "Mapme location\n& category lists", LAYER_COLORS["raw"])
box(ax, 5.4, 5.0, 2.3, 0.7, "Crawler bridge\n(loc x cat)", LAYER_COLORS["raw"])

box(ax, 0.4, 3.6, 3.0, 0.7,
    "user_location_category_events", LAYER_COLORS["wh"])
box(ax, 3.6, 3.6, 3.0, 0.7, "location_category_dim", LAYER_COLORS["wh"])
box(ax, 6.8, 3.6, 3.0, 0.7, "location_dim", LAYER_COLORS["wh"])

box(ax, 0.4, 2.3, 4.4, 0.7,
    "user_location_full_features  (+ *_all_users priors)",
    LAYER_COLORS["feat"], text_color=TEXT_LIGHT)
box(ax, 5.0, 2.3, 4.8, 0.7,
    "candidate_user_location_table  (user x all locations)",
    LAYER_COLORS["feat"], text_color=TEXT_LIGHT)

box(ax, 0.4, 1.0, 9.4, 0.7,
    "Hybrid Recommender:  content + CF + session/transition + trending + popularity  \u2192  MMR diversification",
    LAYER_COLORS["model"], text_color=TEXT_LIGHT)

box(ax, 0.4, 0.05, 4.5, 0.6, "FastAPI backend  (/api/recommend ...)",
    LAYER_COLORS["app"], text_color=TEXT_LIGHT)
box(ax, 5.1, 0.05, 4.7, 0.6, "ChicagoDoes frontend  +  LLM concierge",
    LAYER_COLORS["app"], text_color=TEXT_LIGHT)

for x in (1.35, 3.95, 6.55):
    arrow(ax, x, 5.0, x, 4.3)
arrow(ax, 1.9, 3.6, 1.9, 3.0)
arrow(ax, 5.1, 3.6, 5.1, 3.0)
arrow(ax, 8.3, 3.6, 8.3, 3.0)
arrow(ax, 2.6, 2.3, 2.6, 1.7)
arrow(ax, 7.4, 2.3, 7.4, 1.7)
arrow(ax, 2.6, 1.0, 2.6, 0.65)
arrow(ax, 7.4, 1.0, 7.4, 0.65)

ax.text(0.05, 5.85, "Sources",       fontsize=9, fontweight="bold", color="#374151")
ax.text(0.05, 4.45, "Warehouse",     fontsize=9, fontweight="bold", color="#374151")
ax.text(0.05, 3.15, "Modeling features", fontsize=9, fontweight="bold", color="#374151")
ax.text(0.05, 1.85, "Ranking engine",fontsize=9, fontweight="bold", color="#374151")
ax.text(0.05, 0.75, "Application",   fontsize=9, fontweight="bold", color="#374151")

plt.tight_layout()
plt.savefig("architecture.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Figure: score blend (new vs returning weight schedule)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.3))
terms = ["Content\nsim.", "Popularity", "Item-item\nCF", "User-user\nkNN",
         "Trending", "Session +\ntransition"]
new_w = [0.40, 0.20, 0.10, 0.00, 0.10, 0.20]
ret_w = [0.22, 0.15, 0.18, 0.20, 0.05, 0.20]

x = np.arange(len(terms))
width = 0.38
b1 = ax.bar(x - width / 2, new_w, width, label="New visitor",
            color="#7ea8c9", edgecolor="#1b2a3a", linewidth=0.6)
b2 = ax.bar(x + width / 2, ret_w, width, label="Returning user",
            color="#274972", edgecolor="#1b2a3a", linewidth=0.6)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0.005:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(terms)
ax.set_ylabel("Weight in final score")
ax.set_ylim(0, 0.46)
ax.set_title("Score-blend weights by user regime", pad=8)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#cbd5e1")
plt.tight_layout()
plt.savefig("score_blend.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Figure: MMR re-ranking schematic
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis("off")

# --- Top-60 row (centered) ------------------------------------------------
top60_n = 8
top60_w, top60_gap = 0.9, 0.1
top60_total = top60_n * top60_w + (top60_n - 1) * top60_gap
top60_x0 = (10 - top60_total) / 2

ax.text(top60_x0, 3.85, "Top-60 by final score",
        fontsize=10.5, fontweight="bold")
for i, color in enumerate(["#7ea8c9", "#7ea8c9", "#7ea8c9", "#a9c5dc",
                           "#7ea8c9", "#bfd1e5", "#7ea8c9", "#a9c5dc"]):
    box(ax, top60_x0 + i * (top60_w + top60_gap), 3.05, top60_w, 0.55,
        "", color)

# --- Top-K diversified row (centered, same horizontal span) ---------------
topk_n = 8
topk_w, topk_gap = 0.9, 0.1
topk_total = topk_n * topk_w + (topk_n - 1) * topk_gap
topk_x0 = (10 - topk_total) / 2

ax.text(topk_x0, 1.7, "Top-K diversified",
        fontsize=10.5, fontweight="bold")
mix = ["#274972", "#7ea8c9", "#bfd1e5", "#4a78a5", "#a9c5dc",
       "#274972", "#bfd1e5", "#4a78a5"]
labels = ["Rest.", "Park", "Museum", "Bar", "Tour",
          "Rest.", "Shop", "Bar"]
for i, (c, t) in enumerate(zip(mix, labels)):
    box(ax, topk_x0 + i * (topk_w + topk_gap), 0.9, topk_w, 0.55, t, c,
        text_color=("#ffffff" if c in ("#274972", "#4a78a5") else "#0d1b2a"))

# --- Centered downward arrow between the two rows -------------------------
mid_x = 5.0
ax.annotate("", xy=(mid_x, 1.5), xytext=(mid_x, 3.0),
            arrowprops=dict(arrowstyle="-|>", color="#1b2a3a", lw=1.2))

# --- MMR formula next to the arrow ----------------------------------------
ax.text(mid_x + 0.25, 2.3,
        "MMR(i) = lambda * score_i  -  (1 - lambda) * max_j  cos(L_i, L_j)",
        fontsize=9.6, family="monospace", va="center")

# --- Caption under the bottom row -----------------------------------------
ax.text(5, 0.3,
        "lambda = 0.7  \u2192  70 percent relevance,  30 percent distance from already-picked items",
        fontsize=9.2, color="#374151", ha="center")
plt.tight_layout()
plt.savefig("mmr_diagram.png", dpi=200, bbox_inches="tight")
plt.close()

print("Generated: architecture.png, score_blend.png, mmr_diagram.png")
