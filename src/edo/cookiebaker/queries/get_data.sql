WITH vars AS (
SELECT
  DATE('{start_date}') AS start_date,
  DATE('{end_date}') AS end_date),
main as (
SELECT
    t.*
FROM `{project}.{dataset}.{table_features}` t,
vars v
WHERE DATE(t.BOOKING_DATE) between v.start_date and v.end_date -1
    {class_filter}
)
SELECT
    {variables_list_str}
FROM main
WHERE 1 = 1
    {limit_sample}
