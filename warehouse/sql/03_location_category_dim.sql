-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.location_category_dim` AS
SELECT
  ld.location_id,
  ANY_VALUE(ld.location_name) AS master_location_name,
  STRING_AGG(DISTINCT b.category_id, '; ' ORDER BY b.category_id) AS location_category_id,
  STRING_AGG(DISTINCT cm.category,    '; ' ORDER BY cm.category)  AS location_category_name,

  COALESCE(
    MAX(IF(LOWER(TRIM(cm.category)) = 'restaurants', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'bars',        cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'hotels',      cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'museums',     cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'parks',       cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'tours',       cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'shops',       cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'theaters and music venues', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'sports venues', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'big bus tours stops', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'murals',       cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'movie & tv locations', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'attractions', cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'hot spots',   cm.category, NULL)),
    MAX(IF(LOWER(TRIM(cm.category)) = 'favorites',   cm.category, NULL)),
    MIN(cm.category)
  ) AS primary_category,

  COUNT(DISTINCT b.category_id) AS num_categories,
  MAX(CASE WHEN LOWER(TRIM(cm.category)) = 'hot spots' THEN 1 ELSE 0 END) AS is_hot_spot_location,
  MAX(CASE WHEN LOWER(TRIM(cm.category)) = 'favorites' THEN 1 ELSE 0 END) AS is_favorite_location
FROM `__PROJECT_ID__.__DATASET_ID__.location_dim` ld
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.location_category_bridge_final` b
  ON ld.location_id = b.location_id
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.category_mapping` cm
  ON b.category_id = cm.category_id
GROUP BY ld.location_id;
