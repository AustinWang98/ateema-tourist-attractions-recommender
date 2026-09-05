-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.location_dim` AS
SELECT
  location_id,
  location_name
FROM `__PROJECT_ID__.__DATASET_ID__.official_full_name_id`;
