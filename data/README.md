# Public Data Artifacts

The public repository contains three kinds of data:

- `demo_events.csv` is fully synthetic and reproducible with
  `scripts/generate_demo_events.py`.
- Catalog, coordinate, card, and geocoding files describe public places.
- Offline evaluation, randomized weight search, selected weights, and
  cross-validation files are complete aggregate historical artifacts. They
  contain metrics and model parameters, not visitor records.

The private team repository retains the original restricted event export and
exact operational inventory. Running the public evaluation commands against
`demo_events.csv` exercises the same algorithms but may overwrite result files
with synthetic measurements, so use a separate output path when comparing them
with the checked-in historical aggregates.
