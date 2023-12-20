"""Release model."""
import json
import os

import fire
from edo.mkt import bq
from google.cloud.storage import Blob


def release_model(model_path: str = None,
                  output_path: str = 'gs://ds-mkt-ds-cookiecutter-sync-mdl',
                  evaluate_path: str = None,
                  project: str = 'ds-mkt',
                  dev: bool = True) -> str:
    """Release model."""
    print('release model')
    EVALUATE_METRIC_THRESHOLD: float = 0.57
    EVALUATE_CALIBRATION_METRIC_THRESHOLD: float = 0.05
    TRAIN_SAMPLE_THRESHOLD: int = int(2e5)
    print(locals())
    if evaluate_path is None:
        evaluate_path = os.path.join(model_path, 'evaluate', 'metrics.json')

    if not model_path.startswith('gs://'):
        raise ValueError

    if dev:
        dags_dir = 'dags_dev'
    else:
        print('release in PROD')
        print('validations')
        path = os.path.join(model_path, 'get_data', 'train', 'get_data_metadata.json')
        blob = Blob.from_string(path, bq.storage_client)
        with bq.blob_open(path, 'rb') as f:
            data_metadata = json.load(f)

        train_sample = data_metadata['data_rows']
        blob = Blob.from_string(evaluate_path, bq.storage_client)
        with bq.blob_open(evaluate_path, 'rb') as f:
            metrics = json.load(f)
        evaluate_metric = metrics['score']
        evaluate_calibration_metric = metrics['avg_ratio']

        validations = []
        validations += [['train_sample', train_sample, TRAIN_SAMPLE_THRESHOLD]]  # we require a >= b
        validations += [['evaluate_metric', evaluate_metric, EVALUATE_METRIC_THRESHOLD]]
        validations += [['evaluate_calibration_metric (-)',
                         -evaluate_calibration_metric, -EVALUATE_CALIBRATION_METRIC_THRESHOLD]]

        for k in validations:
            name, metric, threshold, = k
            if metric < threshold:
                raise ValueError(f'{name}: metric<threshold {metric}<{threshold}')
            else:
                print(f'{name}: metric>=threshold {metric}>={threshold}')

        dags_dir = 'dags'

    output_path = f"{output_path}/{dags_dir}"

    # restore option
    bucket_in = f"{output_path}/model"
    bucket_out = f"{output_path}/_old/model"
    bucket_in_ = Blob.from_string(bucket_in)
    blobs = bq.storage_client.list_blobs(bucket_in_.bucket, prefix=bucket_in_.name)
    print('restore option')
    for blob in blobs:
        filename_in = f"gs://{blob.bucket.name}/{blob.name}"
        filename_out = filename_in.replace(bucket_in, bucket_out)
        print(f"gsutil cp {filename_in} {filename_out}")
        # blob_in = Blob.from_string(filename_in, storage_client)
        blob_out = Blob.from_string(filename_out, bq.storage_client)
        blob.bucket.copy_blob(blob, blob_out.bucket, blob_out.name)

    files = {
        os.path.join(model_path, 'train', 'model.joblib'):
            os.path.join(output_path, 'model', 'model.joblib'),
        os.path.join(model_path, 'train', 'model_metadata.json'):
            os.path.join(output_path, 'model', 'model_metadata.json'),
        os.path.join(model_path, 'evaluate', 'predictions_eval.csv'):
            os.path.join(output_path, 'model', 'data_valid_pred.csv'),
        os.path.join(model_path, 'get_data', 'valid', 'data.csv'):
            os.path.join(output_path, 'model', 'data_valid.csv')
    }
    print('release')
    for filename_in, filename_out in files.items():
        print(f"gsutil cp {filename_in} {filename_out}")
        blob_in = Blob.from_string(filename_in, bq.storage_client)
        blob_out = Blob.from_string(filename_out, bq.storage_client)
        blob_in.bucket.copy_blob(blob_in, blob_out.bucket, blob_out.name)

    return output_path


def main():
    """Execute main program."""
    fire.Fire(release_model)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
