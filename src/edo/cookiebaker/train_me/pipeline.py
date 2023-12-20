"""KubeFlow pipeline definition."""

import yaml
from edo.ml_plumber import Constants
from kfp.v2.dsl import (
    pipeline,
    Condition
)

from edo.cookiecutter_sync.train_me.components import (
    get_output_path, get_params,
    get_data, analyze, train, train_hp_vertex, evaluate)


with open('vars.yaml', 'r') as f:
    variables = yaml.full_load(f)

name = variables['name']
project_sql = variables['project_sql']
dataset = variables['dataset']
version = variables['version']
# output_path = f'gs://{name}'
# base_image = f"{hostname}/{project}/{name}:v{version}"

project = Constants().GCP_PROJECT
base_image = Constants().DEFAULT_IMAGE
output_path = Constants().OUTPUT_PATH


@pipeline(
    name=name,
    description=name,
    pipeline_root=output_path
)
def cookiecutter_sync_pipeline(
                   start_date: str = '2021-06-01',
                   end_date_days: int = 60,
                   end_valid_date_days: int = 15,
                   cache_timestamp: str = 'None',
                   dev: bool = False,
                   output_path: str = output_path,
                   hp_tuning: bool = True,
                   n_iter: int = 30,
                   n_jobs_cv: int = -1,
                   n_splits: int = 5,
                   project: str = project,
                   dataset: str = dataset,
                   target_field: str = 'LABEL_INT',
                   entrypoint: str = "cookiecutter_sync_train",
                   display_name: str = name,
                   base_image: str = base_image,
                   scoring: str = 'roc_auc_ovr_weighted',
                   eval_metric: str = 'auc',
                   objective: str = 'multi:softprob',
                   do_analyze: str = 'True',
                   machine_type: str = 'None',
                   project_sql: str = project_sql,
                    tree_method: str = 'hist'
                   ):
    """Kfp pipeline."""
    print(vars())
    memory_limit = '32G'
    # issue: https://github.com/kubeflow/pipelines/issues/6681
    # cache problems
    args_output_path = dict(base_output_path=output_path,
                        cache_timestamp=cache_timestamp)
    output_path = get_output_path(**args_output_path).outputs['output']

    args_get_params = dict(dev=dev,
                       output_path=output_path,
                       start_date=start_date,
                       end_date_days=end_date_days,
                       end_valid_date_days=end_valid_date_days,
                       machine_type=machine_type,
                       memory_limit='1G',
                       n_iter=n_iter,
                       n_jobs_cv=n_jobs_cv,
                       n_splits=n_splits
                       )
    params = get_params(**args_get_params)
    start_date = params.outputs['start_date']
    end_date = params.outputs['end_date']
    end_valid_date = params.outputs['end_valid_date']
    n_iter = params.outputs['n_iter']  # hp searches
    n_jobs_cv = params.outputs['n_jobs_cv']  # parallel threats in hp
    n_splits = params.outputs['n_splits']  # cv folders
    # memory_limit = params.outputs['memory_limit']
    machine_type = params.outputs['machine_type']

    args_get_data_train = dict(
        output_dir='train',
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        project=project,
        project_sql=project_sql,
        dataset=dataset,
        target=True,
        target_field=target_field,

    )

    get_data_train_ = get_data(**args_get_data_train)
    get_data_train_.set_display_name('get-data-train')

    train_path = get_data_train_.outputs['output']

    args_get_data_valid = args_get_data_train.copy()
    args_get_data_valid.update(
        output_dir='valid',
        start_date=end_date,
        end_date=end_valid_date,
    )

    get_data_valid_ = get_data(**args_get_data_valid)
    get_data_valid_.set_display_name('get-data-valid')

    valid_path = get_data_valid_.outputs['output']

    with Condition(do_analyze == 'True', 'do-analyze'):
        args_analyze = dict(output_path=output_path,
                            data_path=train_path)
        analyze_ = analyze(**args_analyze)
        analyze_.set_memory_limit(memory_limit)

    args_train = dict(
        data_path=str(train_path),
        output_path=str(output_path),

        scoring=str(scoring),
        target_field=str(target_field),
        n_splits=n_splits)
    hp_args = dict(objective=str(objective),
                   eval_metric=str(eval_metric),
                   tree_method=str(tree_method)
                   )
    args_train['hp_args'] = str(hp_args)
    if hp_tuning:
        # with Condition(hp_tuning == 'True'):
        print('hp_tuning')
        args_train['hp_tuning'] = True
        args_train_hp = dict(train_args=str(args_train),  # vertex component don't admit dict type
                             hp_args=str(hp_args),
                             output_path=output_path,
                             entrypoint=entrypoint,
                             display_name=display_name,
                             base_image=base_image,
                             machine_type=machine_type,
                             project=project,
                             scoring=scoring,
                             n_iter=n_iter,
                             n_jobs_cv=n_jobs_cv,
                             )
        train_hp_ = train_hp_vertex(**args_train_hp)
        args_train['hp_args'] = train_hp_.outputs['output']

    args_train['hp_tuning'] = False
    train_ = train(**args_train)
    train_.set_memory_limit(memory_limit)
    model_path = train_.outputs['output']

    args_evaluate = dict(
        data_path=valid_path,
        output_path=output_path,
        model_path=model_path,
        scoring=scoring,
        target_field=target_field
    )

    evaluate_ = evaluate(**args_evaluate)
    evaluate_.set_memory_limit(memory_limit)

    evaluate_.outputs['output']