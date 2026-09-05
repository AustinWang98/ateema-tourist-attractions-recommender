-- Public template; identifiers are substituted by warehouse/run_pipeline.py.

CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.user_location_category_events` AS

WITH base AS (
  SELECT
    event_date,
    event_timestamp,
    TIMESTAMP_MICROS(event_timestamp) AS event_time,

    event_name,
    user_pseudo_id,
    user_id,
    COALESCE(user_id, user_pseudo_id) AS user_key,

    user_first_touch_timestamp,
    TIMESTAMP_MICROS(user_first_touch_timestamp) AS user_first_touch_time,

    device.category AS device_category,
    device.mobile_brand_name AS mobile_brand,
    device.mobile_model_name AS mobile_model,
    device.operating_system AS operating_system,
    device.operating_system_version AS operating_system_version,
    device.language AS device_language,
    device.web_info.browser AS browser,
    device.web_info.browser_version AS browser_version,
    device.web_info.hostname AS hostname,

    geo.city AS city,
    geo.country AS country,
    geo.continent AS continent,
    geo.region AS region,
    geo.sub_continent AS sub_continent,
    geo.metro AS metro,

    privacy_info.analytics_storage AS analytics_storage,
    privacy_info.ads_storage AS ads_storage,
    privacy_info.uses_transient_token AS uses_transient_token,

    traffic_source.source AS traffic_source,
    traffic_source.medium AS traffic_medium,
    traffic_source.name   AS traffic_campaign,

    collected_traffic_source.manual_source        AS collected_manual_source,
    collected_traffic_source.manual_medium        AS collected_manual_medium,
    collected_traffic_source.manual_campaign_name AS collected_manual_campaign,

    stream_id,
    platform,
    is_active_user,
    batch_event_index,
    batch_page_id,
    batch_ordering_id,

    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'id')             AS action_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'payload')        AS payload,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location')  AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title')     AS page_title,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_referrer')  AS page_referrer,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'storyId')        AS story_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'accountId')      AS account_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'cid')            AS cid,

    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id')        AS ga_session_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_number')    AS ga_session_number,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engaged_session_event') AS engaged_session_event,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'percent_scrolled')     AS percent_scrolled,

    (SELECT COALESCE(value.string_value, CAST(value.int_value AS STRING))
     FROM UNNEST(event_params)
     WHERE key = 'session_engaged') AS session_engaged

  FROM `__PROJECT_ID__.__DATASET_ID__.events_*`
  WHERE _TABLE_SUFFIX NOT LIKE 'intraday_%'
),

cleaned AS (
  SELECT
    *,
    CONCAT(user_key, '-', COALESCE(CAST(ga_session_id AS STRING), 'no_session')) AS session_key,

    CASE
      WHEN action_id IN ('map-marker-click', 'section-details-call-to-action')
        AND payload LIKE '%/#/%'
        THEN SPLIT(payload, '/#/')[SAFE_OFFSET(0)]
      ELSE NULL
    END AS payload_location_name,

    CASE
      WHEN action_id IN ('map-marker-click', 'section-details-call-to-action')
        AND payload LIKE '%/#/%'
        THEN SPLIT(payload, '/#/')[SAFE_OFFSET(1)]
      WHEN REGEXP_CONTAINS(page_location, r'/location/')
        THEN REGEXP_EXTRACT(page_location, r'/location/([^?/#]+)')
      ELSE NULL
    END AS location_id,

    REGEXP_EXTRACT(page_location, r'categories=([^&]+)')    AS event_category_id,
    REGEXP_EXTRACT(page_location, r'search=([^&]+)')        AS search_term,
    REGEXP_EXTRACT(page_location, r'last_location=([^&]+)') AS previous_location_id,

    CASE
      WHEN action_id = 'section-details-call-to-action' THEN 5.0
      WHEN action_id = 'location-thumbnail'             THEN 4.0
      WHEN action_id = 'map-marker-click'               THEN 3.0
      WHEN action_id = 'list-section-selection'         THEN 2.0
      WHEN action_id = 'single-cat-filter'              THEN 1.5
      WHEN action_id = 'gallery-media-open'             THEN 2.0
      WHEN action_id = 'media-next'                     THEN 1.5
      WHEN event_name = 'map-search' OR action_id = 'map-search'              THEN 2.0
      WHEN event_name = 'view_search_results'                                  THEN 1.5
      WHEN event_name = 'page_view' AND REGEXP_CONTAINS(page_location, r'/location/') THEN 1.5
      WHEN event_name = 'scroll'                                               THEN 0.5
      WHEN action_id IN (
        'slider-card-size-changed',
        'back-arrow-navigation',
        'section-prev',
        'section-next',
        'map-load',
        'branding-header-hide',
        'map-geolocate',
        'share-open',
        'social-share',
        'media-gallery-close'
      ) THEN 0.0
      ELSE 0.5
    END AS interaction_weight

  FROM base
),

filtered AS (
  SELECT *
  FROM cleaned
  WHERE
    action_id IN (
      'map-marker-click',
      'location-thumbnail',
      'section-details-call-to-action',
      'list-section-selection',
      'single-cat-filter',
      'map-search',
      'gallery-media-open',
      'media-next',
      'share-open',
      'social-share'
    )
    OR event_name IN ('page_view', 'scroll', 'map-search', 'view_search_results')
),

filtered_with_id AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      ORDER BY
        event_timestamp,
        user_key,
        event_name,
        COALESCE(action_id, ''),
        COALESCE(page_location, ''),
        COALESCE(payload, ''),
        COALESCE(batch_event_index, 0),
        COALESCE(batch_ordering_id, 0)
    ) AS event_row_id
  FROM filtered
),

event_category_by_event AS (
  SELECT
    f.event_row_id,
    STRING_AGG(DISTINCT cm.category, '; ' ORDER BY cm.category) AS event_category_name,
    COUNT(DISTINCT cm.category_id) AS matched_event_category_count
  FROM filtered_with_id f
  LEFT JOIN UNNEST(SPLIT(IFNULL(f.event_category_id, ''), ',')) AS single_category_id
  LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.category_mapping` cm
    ON TRIM(single_category_id) = cm.category_id
   -- exclude meta-categories that aren't real interest categories
   AND LOWER(TRIM(cm.category)) NOT IN ('chicago does', 'libraries')
  GROUP BY f.event_row_id
)

SELECT
  f.event_date,
  f.event_timestamp,
  f.event_time,

  f.event_name,
  f.action_id,
  f.payload,
  f.interaction_weight,

  f.user_pseudo_id,
  f.user_id,
  f.user_key,
  f.cid,
  f.account_id,
  f.story_id,

  f.user_first_touch_timestamp,
  f.user_first_touch_time,

  f.session_key,
  f.ga_session_id,
  f.ga_session_number,

  f.location_id,
  -- master_location_name from dim wins; payload-parsed name as fallback (URL-encoded ugliness)
  COALESCE(lca.master_location_name, f.payload_location_name) AS location_name,

  f.event_category_id,
  ec.event_category_name,
  IFNULL(ec.matched_event_category_count, 0) AS matched_event_category_count,

  lca.location_category_id,
  lca.location_category_name,
  lca.primary_category,
  IFNULL(lca.num_categories, 0)        AS num_categories,
  IFNULL(lca.is_hot_spot_location, 0)  AS is_hot_spot_location,
  IFNULL(lca.is_favorite_location, 0)  AS is_favorite_location,

  f.search_term,
  f.previous_location_id,

  f.page_location,
  f.page_title,
  f.page_referrer,

  f.engagement_time_msec,
  LEAST(IFNULL(f.engagement_time_msec, 0), 60000) AS engagement_time_msec_capped,
  f.engaged_session_event,
  f.session_engaged,
  f.percent_scrolled,

  f.device_category,
  f.mobile_brand,
  f.mobile_model,
  f.operating_system,
  f.operating_system_version,
  f.device_language,
  f.browser,
  f.browser_version,
  f.hostname,

  f.city,
  f.country,
  f.continent,
  f.region,
  f.sub_continent,
  f.metro,

  f.analytics_storage,
  f.ads_storage,
  f.uses_transient_token,

  f.traffic_source,
  f.traffic_medium,
  f.traffic_campaign,
  f.collected_manual_source,
  f.collected_manual_medium,
  f.collected_manual_campaign,

  f.stream_id,
  f.platform,
  f.is_active_user,
  f.batch_event_index,
  f.batch_page_id,
  f.batch_ordering_id,

  ROW_NUMBER() OVER (
    PARTITION BY f.user_key, f.session_key
    ORDER BY f.event_timestamp, f.batch_ordering_id
  ) AS event_sequence_in_session

FROM filtered_with_id f
LEFT JOIN event_category_by_event ec
  ON f.event_row_id = ec.event_row_id
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.location_category_dim` lca
  ON f.location_id = lca.location_id;
