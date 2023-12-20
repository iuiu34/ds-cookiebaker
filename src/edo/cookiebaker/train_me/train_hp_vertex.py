"""Train hp vertex."""
import json
import os

import fire
import importlib.resources as pkg
import yaml
from edo.create_wrapper_components import BunchKfp, Metrics, Output, Artifact
from edo.mkt import bq
from edo.mkt.ml.train_hp_vertex_operator import train_hp_vertex_operator



def train_hp_vertex(train_args: str,
                    hp_args: str = None,
                    output_path: str = None,
                    scoring: str = 'auc',
                    n_iter: int = 30,
                    n_jobs_cv: int = -1,
                    entrypoint: str = "cookiecutter_sync_train",
                    display_name: str = "display_name",
                    base_image: str = "base_image",
                    machine_type: str = 'None',
                    project: str = 'ds-mkt',
                    hp_config_file: str = 'hp_config.yaml',
                    artifact_metrics: Output[Metrics] = None,
                    artifact_url: Output[Artifact] = None
                    ) -> str:
    """Train hp vertex."""
    print('train_hp_vertex')
    print(vars())

    if artifact_metrics is None:
        artifact_metrics = BunchKfp()
    if artifact_url is None:
        artifact_url = BunchKfp()

    if type(train_args) is str:
        train_args = eval(train_args)  # noqa

    if hp_args is None:
        hp_args = {}
    elif type(hp_args) is str:
        hp_args = eval(hp_args)  # noqa

    package_path = pkg.files("edo.cookiecutter_sync")
    filename = os.path.join(os.path.dirname(package_path), 'model_configuration', hp_config_file)
    with open(filename) as f:
        hp_config = yaml.full_load(f)

    hp_out = train_hp_vertex_operator(
        hp_config=hp_config,
        train_args=train_args,
        scoring=scoring,
        n_iter=n_iter,
        n_jobs_cv=n_jobs_cv,
        entrypoint=entrypoint,
        display_name=display_name,
        base_image=base_image,
        machine_type=machine_type,
        project=project,
    )
    hp_best_params, hp_vertex_response, hp_vertex_url = hp_out

    path = os.path.join(output_path, 'train_hp_vertex', "hp_vertex_response.json")
    with bq.blob_open(path, 'w') as f:
        json.dump(hp_vertex_response, f)

    hp_best_params.update(hp_args)

    artifact_metrics.metadata.update(hp_best_params)
    artifact_url.uri = hp_vertex_url
    return str(hp_best_params)


def main():
    """Execute main program."""
    fire.Fire(train_hp_vertex)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
