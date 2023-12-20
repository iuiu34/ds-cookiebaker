"""Components."""
from typing import NamedTuple

from edo.ml_plumber import Constants
from kfp.v2.dsl import component

base_image = Constants().DEFAULT_IMAGE


# get_output_path = create_component_from_func_in_base_image(
#     func=get_output_path,
#     base_image=base_image
# )


@component(base_image=base_image, install_kfp_package=False)
def get_output_path(base_output_path: str, cache_timestamp: str = None
                    ) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync._utils import get_output_path
    return get_output_path(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def get_params(dev: bool, start_date: str, end_date: str, end_valid_date: str, machine_type: str, memory_limit: str,
               n_iter: int, n_jobs_cv: int, n_splits: int
               ) -> NamedTuple('Outputs', [('start_date', str),
                                           ('end_date', str),
                                           ('end_valid_date', str),
                                           ('n_iter', int),
                                           ('n_jobs_cv', int),
                                           ('n_splits', int),
                                           ('machine_type', str),
                                           ('memory_limit', str)]):  # noqa:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync._utils import get_params
    return get_params(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def get_data(output_path: str, start_date: str, end_date: str, project: str = 'ds-mkt', dataset: str = 'ds_ftp',
             bq_location: str = 'EU', sample: int = 0, output_dir: str = 'train', website: str = 'ALL',
             variables_file: str = 'variables_types.csv', target: bool = True, target_field: str = 'LABEL_INT',
             table_features: str = 'ds_ftp_features', project_sql: str = None
             ) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync.get_data import get_data
    return get_data(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def analyze(output_path: str, data_path: str = None, report: bool = True, website: str = 'ALL'
            ) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync.analyze import analyze
    return analyze(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def train(data_path: str, output_path: str, hp_tuning: bool = True, hp_tuning_local: bool = False,
          target_field: str = 'LABEL_INT', n_splits: int = 5, verbose: int = 0, scoring: str = 'roc_auc',
          n_iter: int = 20, n_jobs_cv: int = 5, website: str = 'ALL', variables_file: str = 'variables_types.csv',
          hp_args: str = None, hp_args_path: str = None) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync.train import train
    return train(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def train_hp_vertex(data_path: str, output_path: str, target_field: str = 'LABEL_INT', n_splits: int = 5,
                    verbose: int = 0, scoring: str = 'roc_auc', n_iter: int = 20, n_jobs_cv: int = 5,
                    website: str = 'ALL', variables_file: str = 'variables_types.csv', hp_args: str = '{}',
                    entrypoint: str = 'cookiecutter_sync_train', display_name: str = 'display_name',
                    base_image: str = 'base_image', machine_type: str = 'None', project: str = 'ds-mkt'
                    ) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync.train_me.train_hp_vertex import train_hp_vertex
    return train_hp_vertex(**kwargs)


@component(base_image=base_image, install_kfp_package=False)
def evaluate(output_path: str, data_path: str, model_path: str, output_metrics_path_file: str = 'tmp/metrics.txt',
             output_predictions_path_file: str = 'tmp/predictions.txt', target_field: str = 'LABEL_INT',
             eval_dir: str = 'evaluate', website: str = 'ALL', report: bool = True, scoring: str = 'roc_auc',
             variables_file: str = 'variables_types.csv'
             ) -> str:
    """Kfp wrapper."""
    kwargs = locals()
    kwargs = {k: None if v == "None" else v for k, v in kwargs.items()}
    from edo.cookiecutter_sync.evaluate import evaluate
    return evaluate(**kwargs)

