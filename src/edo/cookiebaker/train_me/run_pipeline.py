"""Run pipeline."""

import fire
from edo.kfp_local.kfp import run_kfp_pipeline

from edo.cookiecutter_sync.train_me.pipeline import (
    name, dataset, version, output_path, base_image)

def run_pipeline(docker_build=True,
                 docker_build_vm=True,
                 kfp_build=True,
                 dev=True,
                 project='ds-mkt', **kwargs):
    """Run vertex pipeline."""
    args_dict = dict(docker_build=docker_build,
                     docker_build_vm=docker_build_vm,
                     kfp_build=kfp_build,
                     base_image=base_image,
                     display_name=name,
                     project=project,
                     pipeline=churn_pipeline,
                     slack=True,
                     output_path=output_path,
                     dev=dev,
                     **kwargs)
    run_kfp_pipeline(**args_dict)


def main():
    """Execute main program."""
    fire.Fire(run_pipeline)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()