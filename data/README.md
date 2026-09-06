# Public Data Artifacts

The public data directory separates runnable synthetic input from public place
reference data and preserved aggregate research outputs.

| Artifact type | Files | Provenance and intended use |
| --- | --- | --- |
| Synthetic behavior | `demo_events.csv` | Fully generated; use for local application, API, and pipeline demonstrations |
| Public place reference | `location_dim.csv`, `locations_geo.csv`, `location_cards.json`, geocoding audit files | Public Chicago place identities, coordinates, links, and descriptive metadata |
| Public taxonomy seeds | `../warehouse/seeds/` | Place/category relationships used by the warehouse and synthetic generator |
| Historical aggregate results | evaluation CSV/JSON files | Preserved model metrics, selected weights, and experiment summaries; no visitor rows |

## Synthetic event contract

`demo_events.csv` contains 1,920 generated events for 80 synthetic users and
240 sessions. Each session has four place stops and two events per stop. The
fixture deliberately exercises:

- catalog and category-based content features;
- repeated user/place interactions for collaborative signals;
- ordered stops for session co-visitation and transition graphs;
- timestamps across one month for trending calculations;
- multiple public taxonomy categories and the `HOT SPOTS` flag;
- engagement weights and dwell time used by the quality policy.

Place IDs, names, and taxonomy assignments come from the public catalog seeds.
Every user, session, timestamp, event sequence, and interaction is generated
from a fixed seed. No row is sampled, transformed, or copied from the original
visitor export.

Synthetic identity and URL markers are intentionally obvious:

```text
user_key      synthetic-user-001
session_key   synthetic-user-001-session-01
page_location https://demo.invalid/places/<public-location-id>
```

Regenerate it from the repository root:

```bash
python scripts/generate_demo_events.py
python -m pytest tests/test_demo_fixture.py
```

The tests check the exact schema, deterministic bytes, public-catalog and
taxonomy integrity, temporal/session structure, synthetic identity patterns,
reserved URLs, and absence of common visitor or secret fields.

## Important interpretation boundary

The fixture is suitable for demonstrating the same code paths, but it is not a
statistical substitute for real visitor behavior. It intentionally has regular
session sizes, stronger preference patterns, and less noise than production
analytics. Do not use synthetic-demo metrics as evidence of business impact or
compare them directly with the historical capstone results.

The checked-in evaluation artifacts are aggregate results from the original
capstone experiments. Running evaluation commands can overwrite some of those
tracked paths, so follow the [technical guide](../docs/TECHNICAL_GUIDE.md#run-the-evaluation-workflow)
and use a disposable clone.

The private team repository retains the original restricted event export,
authorized operational details, and full handoff record.
