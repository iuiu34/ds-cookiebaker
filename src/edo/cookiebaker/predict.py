"""Predict in batch."""
import json
import os

import dateutil as dtu
import fire
import joblib
import pandas as pd
import importlib.resources as pkg
from edo.mkt import bq


from edo.cookiebaker.train import get_data_from_gs, CLASSES_

from edo.cookiebaker.train_me.llm_model import LlmModel


def get_prediction(st_empty=None, **kwargs):
    model = LlmModel(st_empty=st_empty)
    prompt = model.get_prompt(**kwargs)
    print(prompt)
    # messages = Messages()
    # messages.add_assistant({"booking_id": str(kwargs['booking_id'])}, tool_calls=[])
    # messages.add_tool_response(str(kwargs['booking_id']), 'get_booking_id')
    # p = model.predict_stream(prompt)
    # for chunk in p:
    #     chunk_ = chunk.choices[0].delta.content
    #     if chunk_ is not None:
    #         print(chunk_, end="")
    p = model.predict_sample(prompt)
    # p = p.tools_response()
    print(f"{p=}")
    return p

def get_threshold(x, y, cut_type, cut_value):
    """Apply threshold on probabilistic predictions made by the model."""
    x['churn_probability'] = y[CLASSES_[1]]
    x = x.sort_values(by='churn_probability', ascending=False)

    if cut_type == "THRESHOLD":
        x.loc[x['churn_probability'] >= cut_value, 'churn_class'] = 1
        x.loc[x['churn_probability'] < cut_value, 'churn_class'] = 0
    elif cut_type == "PERCENTILE":
        th_pos = int(x.shape[0] * cut_value)
        x.loc[x.index[:th_pos], 'churn_class'] = 1
        x.loc[x.index[th_pos:], 'churn_class'] = 0
    else:
        print("CUT_TYPE parameter missing. Must be set to 'THRESHOLD' or 'PERCENTILE' along with the CUT_VALUE...")
        return None

    x['churn_class'] = x['churn_class'].astype(int)
    x = x.sort_index()
    return x[['churn_probability', 'churn_class']]


def predict(
        output_path: str,
        model_path: str,
        data_path: str,
        start_date: str,
        end_date: str,
        variables_file: str = 'variables_types.csv',
        cut_type: str = 'THRESHOLD',
        cut_value: float = 0.67,
        output_table: str = 'ds_cookiecutter_sync',
        project: str = 'ds-mkt',
        dataset: str = 'ds_mkt_cookiecutter_sync',
        partition_column: str = 'SUBSCR_DATE',
        store_as_table: bool = False,
        store_as_file: bool = True,
        output_dir: str = 'train') -> str:
    """Make predictions and store it in BQ."""
    print('predict')
    print(vars())

    output_path = os.path.join(output_path, 'predict', output_dir)

    with bq.blob_open(model_path, 'rb') as f:
        model = joblib.load(f)

    model_metadata_path = f"{os.path.dirname(model_path)}/model_metadata.json"
    with bq.blob_open(model_metadata_path, 'rb') as f:
        model_metadata = json.load(f)

    package_path = pkg.files("edo.cookiecutter_sync")
    filename = os.path.join(package_path, 'model_configuration', variables_file)
    variables_types = pd.read_csv(filename)

    data_x, metadata = get_data_from_gs(data_path, variables_types, target=False)
    # feature_names_in = model['preprocess'].feature_names_in_.tolist()
    # data_x = data_x[feature_names_in]
    print(f"data shape: {data_x.shape}.")
    data = model.predict_proba(data_x)
    data = pd.DataFrame(data, columns=CLASSES_)
    p_churn = get_threshold(data_x, data, cut_type, cut_value)

    data.columns = [f"MODEL_PRED_{v}" for v in data.columns]
    data['CUT_TYPE'] = cut_type
    data['CUT_VALUE'] = cut_value
    data['MODEL_CLASS_CHURN'] = p_churn['churn_class']
    data['MODEL_PATH'] = model_metadata['model_path']
    data['MODEL_VERSION'] = model_metadata['model_version']
    data[partition_column] = metadata.SUBSCR_DATE.apply(
        lambda x: dtu.parser.parse(x)
    )
    metadata_vars = ['MEMBER_ACCOUNT_ID', 'SUBSCR_BOOKING_ID']
    data = pd.concat([metadata[metadata_vars], data], axis=1)

    if store_as_table:
        package_path = pkg.files("edo.cookiecutter_sync")
        table_schema = f'{package_path}/queries/schema_{output_table}.json'

        output_table = f"{project}.{dataset}.{output_table}"
        print(f"{output_table=}")
        bq.load_table_partitioned_from_dataframe(
            data, output_table, start_date, end_date,
            partition_column, 'SUBSCR_BOOKING_ID',
            table_schema)
        out = output_table

    if store_as_file:
        output_file = os.path.join(output_path, "data.csv")
        print(f"{output_file=}")
        with bq.blob_open(output_file, 'w') as f:
            data.to_csv(f, index=False)

        out = output_file
    return out


def main():
    """Execute main program."""
    fire.Fire(predict)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
