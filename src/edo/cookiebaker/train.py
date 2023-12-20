"""Train."""
import datetime as dt
import json
import os
from datetime import datetime

import dateutil as dtu
import fire
import joblib
import pandas as pd
import importlib.resources as pkg
import yaml
import numpy as np
from edo.create_wrapper_components import Output, Artifact

CLASSES_ = ['RENEWED', 'CHURN', 'FAILED']


def hp_args_default(
        objective: str = "auc",
        eval_metric: str = 'mlogloss',
        tree_method: str = 'hist',
        max_depth: int = 5,
        n_estimators: int = 20,
        early_stopping_rounds: int = 5,
        learning_rate: float = 0.01,
        colsample_bytree: float = 0.5,
        reg_lambda: float = 1.,
        gamma: float = 0.3,
        min_child_weight: int = 3,
        subsample: float = 0.9,
        max_cat_to_onehot: int = 3,
        n_jobs: int = -1):
    """Get hp args default."""
    return locals()



def get_model(variable_types, args):
    """Define the XGBoost model."""
    variable_types = variable_types.query("USE == 1").copy()
    variable_types.VARIABLE = variable_types.VARIABLE.apply(
        lambda x: x.upper()
    )
    variable_types = variable_types.query("VARIABLE != 'LABEL_INT'").copy()

    variables_num = variable_types[variable_types.TYPE == 'float'].VARIABLE.to_list()
    variables_str = variable_types[variable_types.TYPE == 'str'].VARIABLE.to_list()
    variables_date = variable_types[variable_types.TYPE == 'date'].VARIABLE.to_list()

    date_parts = ['month', 'day', 'dayofweek', 'hour']
    args_ordinal_encoder = dict(handle_unknown='use_encoded_value',
                                unknown_value=np.nan)

    print('pipeline')
    pipeline = []
    # not to be confused with optimus prime
    transformers = [('num', "passthrough", variables_num)]
    transformers += [('str', OrdinalEncoder(**args_ordinal_encoder), variables_str)]
    transformers += [('date', DatePartEncoder(dateparts=date_parts), variables_date)]

    preprocess = ColumnTransformer(transformers, verbose_feature_names_out=False)
    pipeline += [('preprocess', preprocess)]

    feature_types = ml.get_xgb_feature_types(len(variables_num), len(variables_str),
                                          len(variables_date), len(date_parts))

    learner = ml.XGBClassifierWithEarlyStop(
        enable_categorical=True,
        max_cat_to_onehot=3,
        feature_types=feature_types,
        **args)

    # PMMLPipeline
    model = Pipeline([('learner', learner)])
    pipeline += [('model', model)]
    pipeline = Pipeline(pipeline)

    return pipeline


def get_weights(date):
    """Calculate the weight for each row of the data set.

    Most recent data is 10 times more important than old data.

    """
    weights = date.apply(lambda x: dtu.parser.parse(x))
    weights = weights - weights.min()
    weights = 1 + 9 * weights / weights.max()
    return weights


def get_data_from_gs(data_path, variables_types, target_field=None, target=True):
    """Read x and y data for training, and metadata variables."""
    with bq.blob_open(data_path, 'r') as f:
        data = pd.read_csv(f)
    if len(data) < 10:
        raise ValueError(f"data too small. has {len(data)} rows.")
    variables_types.VARIABLE = variables_types.VARIABLE.apply(
        lambda x: x.upper()
    )
    metadata_vars = variables_types[variables_types['USE'] == 2]['VARIABLE']
    train_vars = variables_types[variables_types['USE'] == 1]['VARIABLE']

    data.columns = [v.upper() for v in data.columns]
    x = data[train_vars]
    metadata = data[metadata_vars]

    if target:
        y = data[target_field]
        return x, y, metadata
    else:
        return x, metadata


def get_cv(x_train, y_train, n_splits):
    """Get cv folds."""
    skf = TimeSeriesSplit(n_splits=n_splits)
    cv = skf.split(x_train, y_train)
    return cv


def train_for_hp_tuning(model, x_train, y_train, weights, output_path,
                        scoring, n_splits):
    """Train for hp tuning."""
    print("train for hp tuning")

    cv = get_cv(x_train, y_train, n_splits)

    score_cv = cross_val_score(model, x_train, y_train, scoring=scoring, cv=cv,
                               n_jobs=-1,
                               error_score='raise',
                               fit_params={'model__learner__sample_weight': weights})

    print('hp tuning score cv')
    print(score_cv)

    score = score_cv.mean()
    print('hp tuning score')
    print(f"{scoring}: {score}")

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=scoring,
        metric_value=score,
        global_step=1)

    return output_path


def train_all_data(model, x_train, y_train, weights, output_path):
    """Train the model using all data."""
    print(f"training model with all data. Size: {x_train.shape}")
    start_ = dt.datetime.now()
    model.fit(x_train, y_train, model__learner__sample_weight=weights)
    print('all data fit is done')
    end_ = dt.datetime.now()
    print(f"execution time: {end_ - start_}")

    path = os.path.join(output_path, 'model.joblib')
    with bq.blob_open(path, 'wb') as f:
        joblib.dump(model, f)
    model_path = path

    metadata = dict()
    metadata['execution_date'] = datetime.today().strftime("%Y-%m-%d")
    metadata['scoring'] = model['model']['learner'].eval_metric
    metadata['score'] = model['model']['learner'].best_score
    metadata['model_path'] = model_path
    metadata['data_rows'] = x_train.shape[0]
    metadata['data_columns'] = x_train.shape[1]

    metadata['model_version'] = cookiecutter_sync.__version__
    metadata['commit'] = git_commit

    print('train score')
    print(f"{metadata['scoring']}: {metadata['score']}")

    path = os.path.join(output_path, "model_metadata.json")
    with bq.blob_open(path, 'w') as f:
        json.dump(metadata, f)

    return model_path


def get_hp_config(hp_config_file='hp_config.yaml'):
    """Get hp config."""
    package_path = os.path.dirname(inspect.getfile(edo.cookiecutter_sync))
    filename = f'{package_path}/model_configuration/{hp_config_file}'
    with open(filename) as f:
        out = yaml.full_load(f)
    return out


def get_hp_args(hp_args, hp_args_path, hp_kwargs):
    """Get hp args."""
    # hp_args - p1
    if hp_args is None:
        hp_args = {}
    elif type(hp_args) is str:
        hp_args = eval(hp_args)  # noqa
    elif type(hp_args) is dict:
        pass
    else:
        raise ValueError

    hp_args = hp_kwargs | hp_args  # hp_kwargs - p2

    if hp_args_path is not None:
        with bq.blob_open(hp_args_path, 'r') as f:
            hp_args_json = json.load(f)

        hp_args = hp_args_json | hp_args  # hp_json - p3

    hp_args = hp_args_default(**hp_args)  # hp_default - p4

    # fix vertex hp issue with int type #866
    hp_args_default_ = hp_args_default()
    for k in hp_args.keys():
        if type(hp_args_default_[k]) is int:
            hp_args[k] = int(hp_args[k])
    return hp_args


def train(data_path: str,
          output_path: str,
          hp_tuning: bool = True,
          target_field: str = 'LABEL_INT',
          n_splits: int = 5,
          scoring: str = 'roc_auc',
          variables_file: str = 'variables_types.csv',
          hp_args: str = None,
          hp_args_path: str = None,
          artifact_url: Output[Artifact] = None,
          **hp_kwargs
          ) -> str:
    """Train."""
    print('train')
    print(vars())
    if artifact_url is None:
        artifact_url = BunchKfp()
    if hp_args_path is not None and hp_tuning:
        raise ValueError

    hp_args = get_hp_args(hp_args, hp_args_path, hp_kwargs)
    print(f"{hp_args=}")

    output_path = os.path.join(output_path, 'train')
    package_path = pkg.files("edo.cookiecutter_sync")
    filename = os.path.join(package_path, 'model_configuration', variables_file)
    variables_types = pd.read_csv(filename)
    x, y, metadata = get_data_from_gs(data_path, variables_types, target_field)

    if 'SUBSCR_DATE' in metadata.keys():
        date = metadata.SUBSCR_DATE
    elif 'SUBSCR_DATE' in x.keys():
        date = x.SUBSCR_DATE
    else:
        raise ValueError
    idx = date.copy()
    idx = idx[idx.apply(lambda x: type(x) is str)]
    idx = idx.apply(lambda x: dtu.parser.parse(x))
    idx = idx.index
    x = x.loc[idx]
    y = y.loc[idx]

    metadata = metadata.loc[idx]
    weights = get_weights(date)

    model = get_model(variables_types, hp_args)

    if hp_tuning:
        model_path = train_for_hp_tuning(
            model, x, y, weights, output_path, scoring, n_splits)
    else:
        model_path = train_all_data(
            model, x, y, weights, output_path)

    artifact_url.uri = model_path
    return model_path


def main():
    """Execute main program."""
    fire.Fire(train)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
