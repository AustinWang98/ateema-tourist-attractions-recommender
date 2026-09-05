-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.location_category_bridge_final` AS
SELECT DISTINCT
  TRIM(location_id) AS location_id,
  TRIM(category_id) AS category_id
FROM `__PROJECT_ID__.__DATASET_ID__.location_category_bridge`
WHERE location_id IS NOT NULL
  AND category_id IS NOT NULL
  AND TRIM(location_id) != ''
  AND TRIM(category_id) != '';
