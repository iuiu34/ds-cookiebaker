# cookiecutter sync
![](https://img.shields.io/badge/version-v0.0.1-blue.svg)
![](https://img.shields.io/badge/python-3.9-blue.svg)
[![Docs](https://img.shields.io/badge/docs-confluence-013A97)]()
![](https://img.shields.io/badge/dev-orange.svg)

Description.

Repo mdl: 
edo-ds-mkt-mdl-mkt_cookiecutter_sync

Repo dag: 
edo-ds-mkt-lib-mkt_cookiecutter_sync

Dag master branch:
master-pipeliner

Dag: 
mkt_cookiecutter_sync

For any question refer to 
ds-mkt@edo.com

## Getting Started
Install from [nexus](https://jira.odigeo.com/wiki/display/DS/Python+packages+repositories):


```sh
pip install edo-cookiecutter_sync[local]
```

or from source

```sh
pip install -e .[local]
```

which is equivalent to

```sh
pip install -r requirements.txt
pip install -r requirements_local.txt
pip install -e .
```

## Execution

### Train
### Local
to enter debug mode in your computer, run:
```py
src/edo/*/train_me/run_pipeline_local.py
``` 

No need to dockerize, pipeline will run as a py func
in your venv.

### Custom google cloud project
to run pipeline in your own gcloud project, run: 

```py
src/edo/*/train_me/run_pipeline.py
``` 

Also, the function `pipeline` has the option `DEV`, to
just train with a low sample (1 day of data).

#### TrainME
* Pass [jenkins](http://bcn-jenkins-01.odigeo.org/jenkins/).
* Release model pipeline at [trainMe](https://edo-ds-train-me-lab.appspot.com/pipelines).
* Output in [experiments](https://console.cloud.google.com/storage/browser/edo-kf-experiments-prod/experiments).

### Predict
#### Local
To predict, run:

```py
src/edo/*/predict_me/get_dag.py
``` 

#### Airflow
* Pass [jenkins](http://bcn-jenkins-01.odigeo.org/jenkins/).
* Release dag at [airflow](https://edo-ds-train-me-lab.appspot.com/pipelines).

### model configuration 

  * vars.yml: general paramaters (define) 

/src/edo/*/model_configuration
  * variables_types.csv: features to evaluate the model (define with analyze) 
  * hp_config.yml: hp configuration vertex

### operators

  * get_data: get data & feature engineering.
  * analyze (optional): analyze data.
  * train: parameter tunning (hp) & model training. Algorithm: XGBoost.
  * evaluate: evaluate model. (get metrics auc, recall, etc.)
  * predict: predict with the model.

## GCP permissions

Google cloud permissions required to run kfp pipeline.

  * google storage :: read/write
  * bigquery :: read
    * ds-mkt.ds_ftp
  * container registry :: read (docker)
  * vertex pipelines :: read/write
  * vertex training - hyperparameter tuning jobs :: read/write

## Authors

* *DS team* 

## License

This project is property of *edo*# ds-cookiebaker
