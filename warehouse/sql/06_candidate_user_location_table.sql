-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.candidate_user_location_table` AS

WITH users AS (
  SELECT DISTINCT
    user_key
  FROM `__PROJECT_ID__.__DATASET_ID__.user_location_full_features`
),

locations AS (
  SELECT
    ld.location_id,
    ld.location_name,

    STRING_AGG(DISTINCT b.category_id, '; ' ORDER BY b.category_id) AS location_category_id,
    STRING_AGG(DISTINCT cm.category, '; ' ORDER BY cm.category) AS location_category_name,

    COUNT(DISTINCT b.category_id) AS num_categories,

    MAX(CASE WHEN LOWER(TRIM(cm.category)) = 'hot spots' THEN 1 ELSE 0 END) AS is_hot_spot_location,
    MAX(CASE WHEN LOWER(TRIM(cm.category)) = 'favorites' THEN 1 ELSE 0 END) AS is_favorite_location

  FROM `__PROJECT_ID__.__DATASET_ID__.location_dim` ld
  LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.location_category_bridge_final` b
    ON ld.location_id = b.location_id
  LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.category_mapping` cm
    ON b.category_id = cm.category_id
  GROUP BY
    ld.location_id,
    ld.location_name
),

candidate_pairs AS (
  SELECT
    u.user_key,
    l.location_id,
    l.location_name,
    l.location_category_id,
    l.location_category_name,
    IFNULL(l.num_categories, 0) AS num_categories,
    IFNULL(l.is_hot_spot_location, 0) AS is_hot_spot_location,
    IFNULL(l.is_favorite_location, 0) AS is_favorite_location
  FROM users u
  CROSS JOIN locations l
),

observed_interactions AS (
  SELECT
    user_key,
    location_id,

    total_interactions_with_location,
    marker_click_count,
    location_thumbnail_count,
    detail_cta_count,
    location_page_view_count,
    scroll_count,
    total_interaction_score,
    avg_interaction_score,
    sessions_with_location,
    first_interaction_time,
    last_interaction_time,
    total_location_engagement_msec,
    avg_location_engagement_msec,
    total_location_engagement_msec_capped,
    avg_location_engagement_msec_capped,
    max_percent_scrolled
  FROM `__PROJECT_ID__.__DATASET_ID__.user_location_full_features`
)

SELECT
  c.user_key,
  c.location_id,
  c.location_name,
  c.location_category_id,
  c.location_category_name,
  c.num_categories,
  c.is_hot_spot_location,
  c.is_favorite_location,

  CASE
    WHEN o.location_id IS NOT NULL THEN 1
    ELSE 0
  END AS interacted_label,

  IFNULL(o.total_interactions_with_location, 0) AS total_interactions_with_location,
  IFNULL(o.marker_click_count, 0) AS marker_click_count,
  IFNULL(o.location_thumbnail_count, 0) AS location_thumbnail_count,
  IFNULL(o.detail_cta_count, 0) AS detail_cta_count,
  IFNULL(o.location_page_view_count, 0) AS location_page_view_count,
  IFNULL(o.scroll_count, 0) AS scroll_count,
  IFNULL(o.total_interaction_score, 0) AS total_interaction_score,
  IFNULL(o.avg_interaction_score, 0) AS avg_interaction_score,
  IFNULL(o.sessions_with_location, 0) AS sessions_with_location,

  o.first_interaction_time,
  o.last_interaction_time,

  IFNULL(o.total_location_engagement_msec, 0) AS total_location_engagement_msec,
  IFNULL(o.avg_location_engagement_msec, 0) AS avg_location_engagement_msec,
  IFNULL(o.total_location_engagement_msec_capped, 0) AS total_location_engagement_msec_capped,
  IFNULL(o.avg_location_engagement_msec_capped, 0) AS avg_location_engagement_msec_capped,
  IFNULL(o.max_percent_scrolled, 0) AS max_percent_scrolled

FROM candidate_pairs c
LEFT JOIN observed_interactions o
  ON c.user_key = o.user_key
 AND c.location_id = o.location_id;
