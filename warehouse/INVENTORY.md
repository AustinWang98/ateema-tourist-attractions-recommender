# Public Warehouse Inventory

This file documents the warehouse shape and a historical aggregate size
snapshot. It omits row-level data and deployment identifiers.

## Expected source tables

- GA4 daily event tables matching `events_YYYYMMDD`
- The public `location_dim` seed
- The public `category_mapping` seed
- The public `location_category_bridge` seed

## Derived tables

- `location_category_bridge_final`
- `location_category_dim`
- `user_location_category_events`
- `user_location_full_features`
- `candidate_user_location_table` as an optional offline artifact

## Historical aggregate snapshot

At the August 2026 handoff snapshot, the authorized warehouse covered 107 GA4
daily event partitions from April 27 through August 11, with approximately
131,000 raw event rows. After the latest documented refresh, the relevant
derived-table sizes were:

| Table | Aggregate rows | Role |
| --- | ---: | --- |
| `location_dim` | 350 | Versioned public catalog seed |
| `category_mapping` | 15 | Versioned public taxonomy seed |
| `location_category_bridge` | 430 | Versioned public relationship seed |
| `location_category_dim` | 350 | Derived place/category dimension |
| `user_location_category_events` | 73,171 | Restricted behavioral application input |
| `user_location_full_features` | 10,552 | Restricted user-place feature input |
| `candidate_user_location_table` | 1,239,000 | Optional offline candidate grid |

These are aggregate operational counts, not files shipped in the public repo.
They are included to communicate the scale at which the warehouse was tested;
they should not be treated as current production status.

## Query and refresh inventory

The warehouse consisted of the six versioned SQL stages under [`sql/`](sql/).
The source environment had no scheduled derived-table refresh at the snapshot,
so stages 4 and 5 were refreshed manually after new GA4 exports. Scheduling,
monitoring, and data-quality alerting remain deployment responsibilities.

Production rows, exact project/dataset/service names, account and billing
details, service identities, revision history, and unresolved operational
follow-ups remain in the private team repository. The public schemas define
every field required by the SQL and application without containing table data.
