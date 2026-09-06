# Research Paper and Presentation

This directory publishes the final capstone paper, the team's editable Word
manuscript, the complete presentation, and the source material used to produce
them. The compiled files are preserved unchanged from the team edition,
including authorship, results, and historical technical details.

```text
docs/
├── paper/
│   ├── main.tex
│   ├── ChicagoDoes_Capstone_Paper.pdf
│   ├── ChicagoDoes_Capstone_Paper.docx
│   └── figures/
│       ├── architecture.png
│       ├── score_blend.png
│       ├── mmr_diagram.png
│       ├── fig_behavior_depth.png
│       ├── fig_engagement_funnel.png
│       ├── fig_entry_points.png
│       ├── fig_listing_engagement.png
│       ├── fig_temporal_activity.png
│       ├── fig_user_overview.png
│       └── gen_diagrams.py
└── presentation/
    ├── ChicagoDoes_Final_Presentation.pptx
    ├── build_presentation.py
    └── assets/
        └── formulas/
```

Quantitative figures and model-quality tables are aggregate research results;
the repository does not include the underlying visitor-level event export.

## Rebuild the paper

```bash
cd docs/paper
latexmk -pdf main.tex
```

`main.tex` uses the local `figures/` directory. A two-pass `pdflatex` build is
also possible when `latexmk` is unavailable.

Regenerate the custom architecture and algorithm diagrams with:

```bash
cd docs/paper/figures
python gen_diagrams.py
```

## Rebuild the presentation

The complete final presentation is ready to view as checked in. Rebuilding it
requires the program-supplied template used by the original script:

```bash
pip install python-pptx pillow matplotlib
export CHICAGODOES_TEMPLATE=/path/to/approved-template.pptx
python docs/presentation/build_presentation.py
```

The generated deck is written to
`docs/presentation/ChicagoDoes_Final_Presentation.pptx`.

If `CHICAGODOES_TEMPLATE` is unset, the script uses the checked-in final deck
as the visual source so its master, layouts, and theme remain available.

## Presentation structure

The 19-slide, 15-minute deck follows the same arc as the final capstone talk:

| Slides | Section |
| --- | --- |
| 1–3 | Title, agenda, and situational analysis |
| 4–7 | Business problem, current experience, goals, and deliverables |
| 8–9 | Recommender-system and tourism literature |
| 10–12 | Data sources, data quality, and exploratory analysis |
| 13–16 | Architecture, six-signal ranking, cold start, and diversity |
| 17–18 | Evaluation methodology and findings |
| 19 | Q&A |

The PDF, DOCX, and PPTX are historical final artifacts. Rebuilding from source
is optional and can reflect local tool or font differences.
