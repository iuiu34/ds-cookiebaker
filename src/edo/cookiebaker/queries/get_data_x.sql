WITH vars AS (
SELECT
    DATE('{start_date}') AS start_date,
    DATE('{end_date}') AS end_date),
main AS (
SELECT
    t.*,
FROM `{project}.{dataset}.{table_features}` t,
    vars v
WHERE DATE(t.BOOKING_DATE) BETWEEN v.start_date AND v.end_date - 1
    )
SELECT
    {variables_list_str}
FROM main