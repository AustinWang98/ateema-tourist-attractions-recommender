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
│       ├── aggregate evaluation figures
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
