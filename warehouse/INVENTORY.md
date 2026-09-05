# Public Warehouse Inventory

This file documents the warehouse shape without publishing the private
deployment inventory, resource names, row counts, billing state, or refresh
timestamps.

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

Production rows and exact infrastructure inventory remain in the private team
repository. The public schemas define every field required by the SQL and
application without containing table data.
