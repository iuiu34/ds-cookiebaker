"""Main module."""

from edo.mkt import bq
from importlib.resources import files
import fire
from edo import cookiecutter_sync

def cookiecutter_sync(filename: str,
    dataset: str = 'ds_mkt_ds-cookiecutter-sync',
    project: str = 'ds-mkt'):
    """cookiecutter sync."""
    print('cookiecutter_sync')
    print(vars())
    params = {}
    package_path = files("edo.cookiecutter_sync")
    filename = package_path.joinpath('queries','data.sql')
    with open(filename) as f:
        sql = f.read()
    sql = sql.format(**params)
    data = bq.get_query(sql)


def main():
    """Execute main program."""
    fire.Fire(cookiecutter_sync)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
