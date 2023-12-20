"""Run pipeline local."""
import datetime as dt  # noqa
import os  # noqa
import random  # noqa

import fire  # Executes lines of code in the Terminal
from edo.kfp_local.kfp_local import run_kfp_pipeline_local

from edo.cookiecutter_sync.train_me.pipeline import cookiecutter_sync_pipeline
from edo.cookiecutter_sync.train_me.pipeline import (name, dataset, project, project_sql,
                                                    version, output_path, base_image)


def run_pipeline_local(cache_timestamp=None, start_step=None, dev=True, **kwargs):
    """Run pipeline in local (debugging)."""
    # debug with "run with python console".
    func_imports = [
        "from edo.cookiecutter_sync.get_data import get_data",
        "from edo.cookiecutter_sync.train import train",
        "from edo.cookiecutter_sync.evaluate import evaluate",
        "from edo.cookiecutter_sync._utils import ( get_output_path, get_params)",
        "from edo.cookiecutter_sync.analyze import analyze",
        # "from edo.cookiecutter_sync.train_me.train_hp_vertex import train_hp_vertex",
        "train_hp_vertex = None"
    ]

    locals_ = dict(name=name,
                   dataset=dataset,
                   version=version,
                   output_path=output_path,
                   base_image=base_image,
                   project=project,
                   project_sql=project_sql)

    func_steps_output = dict(
        get_data="os.path.join(output_path, 'get_data',kwargs['output_dir'], 'data.csv')",
        analyze='{}',
        train_hp_vertex="{}",
        train="os.path.join(output_path, 'train','model.joblib')",
        evaluate="os.path.join(output_path, 'evaluate', 'metrics.json')",
        predict="os.path.join(output_path, 'predict',kwargs['output_dir'], 'data.csv')",
    )

    args_dict = dict(pipeline=cookiecutter_sync_pipeline,
                     func_steps_output=func_steps_output,
                     start_step=start_step,
                     func_imports=func_imports,
                     locals_=locals_,
                     hp_tuning=False,
                     dev=dev,
                     cache_timestamp=cache_timestamp,
                     **kwargs)
    run_kfp_pipeline_local(**args_dict)


def main():
    """Execute main program."""
    fire.Fire(run_pipeline_local)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
