"""Evaluate."""

import json
import os

import fire
import joblib
import numpy as np
import pandas as pd
import importlib.resources as pkg
import sklearn as sk
from edo.create_wrapper_components import BunchKfp, Output, HTML, Metrics
from edo.mkt import ml, bq
from sklearn.preprocessing import OneHotEncoder

from edo.cookiecutter_sync.train import get_data_from_gs, CLASSES_


def get_onehot_class(y):
    """Convert classification probability into class."""
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.arange(len(CLASSES_)).reshape(-1, 1))
    y_onehot = enc.transform(y.values.reshape(-1, 1)).toarray()
    y_onehot = pd.DataFrame(y_onehot)
    y_onehot.columns = y_onehot.columns.astype('str')
    return y_onehot


def get_metrics(x, y, model, scoring):
    """Get metrics."""
    p_proba = model.predict_proba(x)
    scorer = sk.metrics.get_scorer(scoring)
    scoring_ = scorer._score_func
    y_onehot = get_onehot_class(y)
    score = scoring_(y_onehot, p_proba, **scorer._kwargs)

    # store here the accuracy and save
    metrics = dict()
    metrics['scoring'] = scoring
    metrics['score'] = score

    return metrics, pd.DataFrame(p_proba)


def feature_agg(x, preprocess: bool = False):
    """Agg features."""
    for s in ['MAX_', 'MIN_', 'AVG_', 'STD_', 'TOP_', 'STDV_']:
        if x.startswith(s):
            x = x[len(s):]

    for s in ['V_', 'S_']:
        if x.startswith(s):
            x = x[len(s):]
    if preprocess:
        preprocess_suffix = [f"_{i}_{j}" for i in ['MONTH', 'DAY', 'DAYOFWEEK', 'HOUR']
                             for j in ['COS', 'SIN']]
        preprocess_suffix += ['_AS_FLOAT']
        for s in preprocess_suffix:
            if x.endswith(s):
                x = x[:len(x) - len(s)]
    return x


def get_feature_importance(model):
    model_input_features = model['preprocess'].get_feature_names_out()
    feature_importance = model['model']['learner'].feature_importances_

    # %%
    model_input_features = [v for v in model_input_features if v != 'LABEL']
    feature_importance = pd.DataFrame(
        {'feature': model_input_features,
         'gain': feature_importance}).sort_values('gain')
    return feature_importance


def evaluate(
        output_path: str,
        data_path: str,
        model_path: str,
        target_field: str = 'LABEL_INT',
        report: bool = True,
        scoring: str = 'roc_auc',
        variables_file: str = 'variables_types.csv',
        artifact_html: Output[HTML] = None,
        artifact_metrics: Output[Metrics] = None,
) -> str:
    """Evaluate."""
    print('evaluate')
    print(vars())
    if artifact_html is None:
        artifact_html = BunchKfp()
    if artifact_metrics is None:
        artifact_metrics = BunchKfp()
    output_path = os.path.join(output_path, 'evaluate')

    with bq.blob_open(model_path, 'rb') as f:
        model = joblib.load(f)
    package_path = os.path.dirname(inspect.getfile(edo.cookiecutter_sync))

    filename = f"{package_path}/model_configuration/{variables_file}"
    variables_types = pd.read_csv(filename)
    x, y, _ = get_data_from_gs(data_path, variables_types, target_field)
    metrics, p_proba = get_metrics(x, y, model, scoring)

    data = y.copy().to_frame()
    data['LABEL'] = data['LABEL_INT'].apply(lambda x: CLASSES_[int(x)])
    # p_proba.columns = [f"MODEL_PRED_{v}" for v in CLASSES_]

    data['REAL'] = data['LABEL'] == 'RENEWED'
    data['REAL'] = data['REAL'].astype('int')
    data['MODEL_PRED_RENEWED'] = p_proba.iloc[:, CLASSES_.index('RENEWED')].copy()

    metrics['avg_ratio'] = abs((data.MODEL_PRED_RENEWED.mean() /
                                data.REAL.mean()) - 1)

    print('validation score')
    print(f"{metrics['scoring']}: {metrics['score']}")

    metrics_path = f"{output_path}/metrics.json"
    with bq.blob_open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    predictions_path = f"{output_path}/predictions.csv"
    with bq.blob_open(predictions_path, 'w') as f:
        p_proba.to_csv(f, index=False)

    if report:
        args_notebook = dict(model_path=model_path,
                             data_path=data_path,
                             target_field=target_field,
                             predictions_path=predictions_path,
                             variables_file=variables_file,
                             metrics_path=metrics_path)
        repo_path = pkg_resources.resource_filename(__name__, '')

        path = f"{output_path}/evaluate.html"

        with bq.blob_open(path, 'w', encoding='utf-8') as f:
            ml.render_step('evaluate_notebook', f, args_notebook=args_notebook,
                           repo_path=repo_path,
                           args_notebook_file='tmp/args_notebook.yml',
                           exclude_input=True)
        artifact_html.uri = path

    artifact_metrics.metadata.update(metrics)
    artifact_metrics.uri = metrics_path

    return metrics_path


def main():
    """Execute main program."""
    fire.Fire(evaluate)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
