"""Get table in big query."""
import os

import fire
from edo.dagger.local.get_dag import get_start_index, get_dag_tasks
from edo.mkt import bq

from edo.cookiecutter_sync._utils.data_drift import data_drift
from edo.cookiecutter_sync.get_data import get_data
from edo.cookiecutter_sync.predict import predict


def get_cookiecutter_sync_dag(dags: str,
                project: str = 'ds-mkt',
                dev: bool = False,
                dataset: str = 'ds_cookiecutter_sync',
                start_date: str = None,
                end_date: str = None,
                start_step: str = None):
    """Get cookiecutter_sync dag."""
    print('get cookiecutter_sync dag')
    print(vars())
    project = bq.get_project(project, dev)
    params = dict(PROJECT=project,
                  COOKIECUTTER_SYNC_DATASET=dataset,
                  COOKIECUTTER_SYNC_MDL_BUCKET='gs://ds-mkt-cookiecutter_sync-mdl/dags_dev',
                  COOKIECUTTER_SYNC_TABLE="ds_cookiecutter_sync",
                  COOKIECUTTER_SYNC_DAYS_BACKFILL=-7)

    if type(dags) is str:
        dags = [dags]
    for dagname in dags:
        print(f'DAG {dagname}')
        dagname = f"cookiecutter_sync_{dagname}"
        path = os.getcwd()
        path = f"{path[:-3]}-reporting"
        filename = f"{path}/dags/cookiecutter_sync_dags/cookiecutter_sync_structure.yml"
        dag = get_dag_tasks(filename, dagname, params)
        # if start_step is not None and not start_step.startswith('ltv'):
        #     start_step = f"ltv_{start_step}"
        start_index = get_start_index(dag, start_step)

        for idx, d in enumerate(dag.items()):
            if idx < start_index:
                continue
            k, v = d
            print('\n' + k)
            if v['operator'].endswith('DummyOperator'):
                continue
            if v['operator'] == 'partition_sensor':
                continue
            step_argument = v['step_argument']
            args = v['arguments']

            args += ['--start-date', str(start_date),
                     '--end-date', str(end_date)]

            if step_argument == 'cookiecutter_sync_get_data':
                args += ['--project-sql', str(project),
                         '--project', 'ds-mkt']
                fire.Fire(get_data, args)
            elif step_argument == 'cookiecutter_sync_predict':
                fire.Fire(predict, args)
            elif step_argument == 'cookiecutter_sync_data_drift':
                fire.Fire(data_drift, args)
            else:
                raise ValueError


def main():
    """Execute main program."""
    fire.Fire(get_cookiecutter_sync_dag)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
