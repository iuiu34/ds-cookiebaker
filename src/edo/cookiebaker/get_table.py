"""Get table in big query."""
from importlib.resources import files

import fire
from edo.mkt import bq
import datetime as dt
from edo import cookiecutter_sync
from importlib.resources import files

def get_table(tablename: str,
              query:str=None,
              dataset: str = 'ds_mkt_ds-cookiecutter-sync',
              project: str = 'ds-mkt',
              start_date: str = None,
              end_date: str = None,
              dev: bool = False,
              print_sql: bool = False,
              write_disposition: str = 'WRITE_TRUNCATE',
              time_partitioning_field: str = 'booking_date',
              tablename_out: str = None):
    """Export sql to bq."""
    print('get_table')
    print(vars())
    if query is None:
        query = tablename
    start_ = dt.datetime.now()

    project = bq.get_project(project, dev)

    params = dict(
        project=project,
        dataset=dataset,
        tablename=tablename,
        start_date=start_date,
        end_date=end_date
    )

    table = bq.TableName(project, dataset, tablename)

    if tablename_out is None:
        tablename_out = table.bq

    table_out = bq.TableName(tablename_out)

    package_path = files(cookiecutter_sync)
    filename = package_path.joinpath('queries', f'{query}.sql')
    with open(filename) as f:
        sql = f.read()
    sql = sql.format(**params)

    if print_sql:
        print(sql)

    print(f'table: {table_out.bq}')

    bq.get_query(sql, table_out.bq, write_disposition=write_disposition,
                 time_partitioning_field=time_partitioning_field)
    end_ = dt.datetime.now()
    print(f"execution time: {end_ - start_}")


def main():
    """Execute main program."""
    fire.Fire(get_table)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
