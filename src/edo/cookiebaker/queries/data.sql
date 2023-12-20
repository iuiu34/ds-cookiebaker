with vars as (
select date('{start_date}') start_date,
    date('{end_date}') end_date
    )
select bi.*
FROM `{project}.{dataset}.tablename` bi,
    vars v
where date(bi.date) between v.start_date and v.end_date - 1