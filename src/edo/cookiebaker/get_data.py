"""Get data."""

import json
import os
from datetime import datetime

import fire
import pandas as pd
import importlib.resources as pkg
from edo.create_wrapper_components import BunchKfp, Artifact, Output
from edo.mkt import bq


def get_data(
        output_path: str,
        start_date: str,
        end_date: str,
        project: str = 'ds-mkt',
        dataset: str = 'ds_ltv',
        bq_location: str = 'EU',
        sample: int = 0,
        output_dir: str = 'train',
        variables_file: str = 'variables_types.csv',
        target: bool = True,
        target_field: list = None,
        project_sql: str = None,
        artifact_url: Output[Artifact] = None,
) -> str:
    """Get data."""
    print('get_data')
    print(vars())
    if project_sql is None:
        project_sql = project
    if artifact_url is None:
        artifact_url = BunchKfp()

    bq.get_client(project)  # vertex-ai needs explicit
    output_path = os.path.join(output_path, 'get_data', output_dir)

    # List of features
    package_path = pkg.files("edo.cookiecutter_sync")
    filename = os.path.join(package_path, 'model_configuration', variables_file)
    variables_types = pd.read_csv(filename)
    if variables_types.isna().sum().sum() > 0:
        raise ValueError("variables_types.csv has nan values:", variables_types[variables_types.isna().any(axis=1)])

    variables = variables_types[variables_types['USE'].isin([1, 2])]['VARIABLE']
    # variables = variables or []
    variables = ',\n'.join(variables)
    if target:
        query_file = 'get_data.sql'
    else:
        query_file = 'get_data_x.sql'

    if sample > 0:
        limit_sample = f' and RAND() < {int(sample)}/(SELECT COUNT(*) FROM main) '  # noqa
    else:
        limit_sample = ''

    params = {'start_date': start_date,
              'end_date': end_date,
              'project': project_sql,
              'dataset': dataset,
              'variables': variables,
              'target_field': target_field,
              'limit_sample': limit_sample}

    package_path = pkg.files("edo.cookiecutter_sync")
    filename = os.path.join(package_path, 'queries', query_file)
    with open(filename) as f:
        sql = f.read()

    sql = sql.format(**params)

    data = bq.get_query(sql, download=False, location=bq_location)
    data = data.bq
    data_path = os.path.join(output_path, 'data.csv')
    bq.extract_table(data, data_path)

    execution_date = datetime.today().strftime("%Y-%m-%d")
    data_rows = bq.client.get_table(data).num_rows
    data_columns = len(bq.client.get_table(data).schema)
    print(f"{data_rows=}")
    print(f"{data_columns=}")
    if int(data_rows) < 10:
        raise ValueError("data has no rows.")

    metadata = dict(
        execution_date=execution_date,
        start_date=start_date,
        end_date=end_date,
        data_rows=data_rows,
        data_columns=data_columns
    )

    # Validation data
    # metadata = str(metadata)
    path = os.path.join(output_path, "get_data_metadata.json")
    with bq.blob_open(path, 'w') as f:
        json.dump(metadata, f)

    artifact_url.uri = data_path
    return data_path


def main():
    """Execute main program."""
    fire.Fire(get_data)
    print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')


if __name__ == "__main__":
    main()
