-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.user_location_full_features` AS

WITH event_level AS (
  SELECT *
  FROM `__PROJECT_ID__.__DATASET_ID__.user_location_category_events`
  WHERE location_id IS NOT NULL
),

-- Count distinct categories seen by each user.
user_category_counts AS (
  SELECT
    user_key,
    COUNT(DISTINCT TRIM(single_category)) AS distinct_location_categories_seen
  FROM event_level e
  JOIN UNNEST(SPLIT(IFNULL(e.location_category_name, ''), ';')) AS single_category
  WHERE TRIM(single_category) != ''
  GROUP BY user_key
),

user_location_agg AS (
  SELECT
    user_key,
    location_id,

    STRING_AGG(DISTINCT user_pseudo_id, '; ' ORDER BY user_pseudo_id) AS user_pseudo_ids,
    ARRAY_AGG(user_pseudo_id IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS representative_user_pseudo_id,

    ANY_VALUE(user_id)    AS user_id,
    ANY_VALUE(cid)        AS cid,
    ANY_VALUE(account_id) AS account_id,
    ANY_VALUE(story_id)   AS story_id,

    ARRAY_AGG(location_name IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS observed_location_name,

    COUNT(*)                                              AS total_interactions_with_location,
    COUNTIF(action_id = 'map-marker-click')               AS marker_click_count,
    COUNTIF(action_id = 'location-thumbnail')             AS location_thumbnail_count,
    COUNTIF(action_id = 'section-details-call-to-action') AS detail_cta_count,
    COUNTIF(event_name = 'page_view')                     AS location_page_view_count,
    COUNTIF(event_name = 'scroll')                        AS scroll_count,

    SUM(interaction_weight) AS total_interaction_score,
    AVG(interaction_weight) AS avg_interaction_score,

    COUNT(DISTINCT session_key) AS sessions_with_location,
    MIN(event_time)             AS first_interaction_time,
    MAX(event_time)             AS last_interaction_time,

    -- Keep AVG null handling consistent with SUM.
    SUM(IFNULL(engagement_time_msec, 0))         AS total_location_engagement_msec,
    AVG(IFNULL(engagement_time_msec, 0))         AS avg_location_engagement_msec,
    SUM(IFNULL(engagement_time_msec_capped, 0))  AS total_location_engagement_msec_capped,
    AVG(IFNULL(engagement_time_msec_capped, 0))  AS avg_location_engagement_msec_capped,
    MAX(percent_scrolled)                        AS max_percent_scrolled,

    -- Use the latest observed device/location/browser value rather than an arbitrary value.
    ARRAY_AGG(device_category          IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS device_category,
    ARRAY_AGG(mobile_brand             IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS mobile_brand,
    ARRAY_AGG(mobile_model             IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS mobile_model,
    ARRAY_AGG(operating_system         IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS operating_system,
    ARRAY_AGG(operating_system_version IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS operating_system_version,
    ARRAY_AGG(device_language          IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS device_language,
    ARRAY_AGG(browser                  IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS browser,
    ARRAY_AGG(browser_version          IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS browser_version,
    ARRAY_AGG(hostname                 IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS hostname,

    ARRAY_AGG(city          IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS city,
    ARRAY_AGG(country       IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS country,
    ARRAY_AGG(continent     IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS continent,
    ARRAY_AGG(region        IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS region,
    ARRAY_AGG(sub_continent IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS sub_continent,
    ARRAY_AGG(metro         IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS metro,

    ANY_VALUE(analytics_storage)    AS analytics_storage,
    ANY_VALUE(ads_storage)          AS ads_storage,
    ANY_VALUE(uses_transient_token) AS uses_transient_token,

    ANY_VALUE(traffic_source)            AS traffic_source,
    ANY_VALUE(traffic_medium)            AS traffic_medium,
    ANY_VALUE(traffic_campaign)          AS traffic_campaign,
    ANY_VALUE(collected_manual_source)   AS collected_manual_source,
    ANY_VALUE(collected_manual_medium)   AS collected_manual_medium,
    ANY_VALUE(collected_manual_campaign) AS collected_manual_campaign,

    ANY_VALUE(user_first_touch_time) AS user_first_touch_time,
    ANY_VALUE(platform)              AS platform,
    ANY_VALUE(stream_id)             AS stream_id

  FROM event_level
  GROUP BY user_key, location_id
),

-- Aggregate all user events from the source table, including events without a location_id.
user_agg AS (
  SELECT
    user_key,

    COUNT(*)                                                          AS total_user_interactions,
    COUNT(DISTINCT location_id)                                       AS distinct_locations_interacted,
    COUNT(DISTINCT session_key)                                       AS total_sessions,

    SUM(interaction_weight) AS total_user_interaction_score,
    AVG(interaction_weight) AS avg_user_interaction_score,

    COUNTIF(action_id = 'map-marker-click')                           AS total_marker_clicks,
    COUNTIF(action_id = 'section-details-call-to-action')             AS total_detail_cta_clicks,
    COUNTIF(event_name = 'map-search' OR action_id = 'map-search')    AS total_search_events,

    MIN(event_time) AS first_seen_time,
    MAX(event_time) AS last_seen_time,

    SUM(IFNULL(engagement_time_msec, 0))         AS total_user_engagement_msec,
    AVG(IFNULL(engagement_time_msec, 0))         AS avg_user_engagement_msec,
    SUM(IFNULL(engagement_time_msec_capped, 0))  AS total_user_engagement_msec_capped,
    AVG(IFNULL(engagement_time_msec_capped, 0))  AS avg_user_engagement_msec_capped

  FROM `__PROJECT_ID__.__DATASET_ID__.user_location_category_events`
  -- Intentionally no location_id filter: retain search and general page events.
  GROUP BY user_key
),

location_agg AS (
  SELECT
    location_id,

    ARRAY_AGG(location_name IGNORE NULLS ORDER BY event_time DESC LIMIT 1)[SAFE_OFFSET(0)] AS observed_global_location_name,

    COUNT(*)                       AS total_location_interactions_all_users,
    COUNT(DISTINCT user_key)       AS distinct_users_interacted_location,
    COUNT(DISTINCT session_key)    AS distinct_sessions_interacted_location,

    SUM(interaction_weight) AS total_location_score_all_users,
    AVG(interaction_weight) AS avg_location_score_all_users,

    COUNTIF(action_id = 'map-marker-click')                AS total_marker_clicks_all_users,
    COUNTIF(action_id = 'section-details-call-to-action')  AS total_detail_cta_all_users,

    SUM(IFNULL(engagement_time_msec, 0))         AS total_location_engagement_all_users_msec,
    AVG(IFNULL(engagement_time_msec, 0))         AS avg_location_engagement_all_users_msec,
    SUM(IFNULL(engagement_time_msec_capped, 0))  AS total_location_engagement_all_users_msec_capped,
    AVG(IFNULL(engagement_time_msec_capped, 0))  AS avg_location_engagement_all_users_msec_capped

  FROM event_level
  GROUP BY location_id
)

SELECT
  ul.user_key,
  ul.user_pseudo_ids,
  ul.representative_user_pseudo_id,
  ul.user_id,
  ul.cid,
  ul.account_id,
  ul.story_id,

  ul.location_id,

  COALESCE(lca.master_location_name, ul.observed_location_name) AS location_name,
  lca.location_category_id,
  lca.location_category_name,
  lca.primary_category,
  IFNULL(lca.num_categories, 0)         AS num_categories,
  IFNULL(lca.is_hot_spot_location, 0)   AS is_hot_spot_location,
  IFNULL(lca.is_favorite_location, 0)   AS is_favorite_location,

  ul.total_interactions_with_location,
  ul.marker_click_count,
  ul.location_thumbnail_count,
  ul.detail_cta_count,
  ul.location_page_view_count,
  ul.scroll_count,
  ul.total_interaction_score,
  ul.avg_interaction_score,

  ul.sessions_with_location,
  ul.first_interaction_time,
  ul.last_interaction_time,
  ul.total_location_engagement_msec,
  ul.avg_location_engagement_msec,
  ul.total_location_engagement_msec_capped,
  ul.avg_location_engagement_msec_capped,
  ul.max_percent_scrolled,

  ua.total_user_interactions,
  ua.distinct_locations_interacted,
  IFNULL(ucc.distinct_location_categories_seen, 0) AS distinct_location_categories_seen,
  ua.total_sessions,
  ua.total_user_interaction_score,
  ua.avg_user_interaction_score,
  ua.total_marker_clicks,
  ua.total_detail_cta_clicks,
  ua.total_search_events,
  ua.first_seen_time,
  ua.last_seen_time,
  ua.total_user_engagement_msec,
  ua.avg_user_engagement_msec,
  ua.total_user_engagement_msec_capped,
  ua.avg_user_engagement_msec_capped,

  COALESCE(lca.master_location_name, la.observed_global_location_name) AS global_location_name,
  la.total_location_interactions_all_users,
  la.distinct_users_interacted_location,
  la.distinct_sessions_interacted_location,
  la.total_location_score_all_users,
  la.avg_location_score_all_users,
  la.total_marker_clicks_all_users,
  la.total_detail_cta_all_users,
  la.total_location_engagement_all_users_msec,
  la.avg_location_engagement_all_users_msec,
  la.total_location_engagement_all_users_msec_capped,
  la.avg_location_engagement_all_users_msec_capped,

  ul.device_category,
  ul.mobile_brand,
  ul.mobile_model,
  ul.operating_system,
  ul.operating_system_version,
  ul.device_language,
  ul.browser,
  ul.browser_version,
  ul.hostname,

  ul.city,
  ul.country,
  ul.continent,
  ul.region,
  ul.sub_continent,
  ul.metro,

  ul.analytics_storage,
  ul.ads_storage,
  ul.uses_transient_token,

  ul.traffic_source,
  ul.traffic_medium,
  ul.traffic_campaign,
  ul.collected_manual_source,
  ul.collected_manual_medium,
  ul.collected_manual_campaign,

  ul.user_first_touch_time,
  ul.platform,
  ul.stream_id

FROM user_location_agg ul
LEFT JOIN user_agg               ua  ON ul.user_key   = ua.user_key
LEFT JOIN user_category_counts   ucc ON ul.user_key   = ucc.user_key
LEFT JOIN location_agg           la  ON ul.location_id = la.location_id
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.location_category_dim` lca
                                     ON ul.location_id = lca.location_id;
