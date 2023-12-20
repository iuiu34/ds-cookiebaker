"""# Evaluate cookiecutter sync"""

# %%

import inspect
import json
import os

import joblib
import jupyter_client  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa
import sklearn as sk
import sklearn.metrics  # noqa
import yaml
from edo.mkt import bq
from sklearn.calibration import calibration_curve

import edo.cookiecutter_sync
from edo.cookiecutter_sync.evaluate import get_onehot_class, feature_agg, get_feature_importance
from edo.cookiecutter_sync.train import get_data_from_gs, CLASSES_

pd.set_option('display.max_columns', 999)

# %%
filename = './tmp/args_notebook.yml'
with open(filename) as f:
    args = yaml.full_load(f)

model_path = args['model_path']
data_path = args['data_path']
target_field = args['target_field']
predictions_path = args['predictions_path']
variables_file = args['variables_file']
metrics_path = args['metrics_path']

# %%
with bq.blob_open(model_path, 'rb') as f:
    model = joblib.load(f)

# notebook doesn't read __name__ properly
package_path = os.path.dirname(inspect.getfile(edo.cookiecutter_sync))
filename = f"{package_path}/model_configuration/{variables_file}"
variables_types = pd.read_csv(filename)

x, y, _ = get_data_from_gs(data_path, variables_types, target_field)

with bq.blob_open(predictions_path, 'rb') as f:
    p_proba = pd.read_csv(f)

y_onehot = get_onehot_class(y)

p = p_proba.idxmax(1).astype('float')
p_onehot = get_onehot_class(p)

with bq.blob_open(metrics_path, 'r') as f:
    metrics = json.load(f)

path = os.path.join(os.path.dirname(data_path), 'model_metadata.json')
with bq.blob_open(path, 'rb') as f:
    model_metadata = json.load(f)

# path = os.path.join(os.path.dirname(data_path), 'get_data_metadata.json')
# with bq.blob_open(path, 'rb') as f:
#     model_metadata = json.load(f)

# %%
"""### Model step"""

# %%
"""### Args"""

# %%
args

# %%
"""### Configuration"""

# %%
"""## Train"""

# %%
"""### Model metadata"""

# %%
model_metadata

# %%
"""### Feature importance"""

# %%
feature_importance = get_feature_importance(model)
g = feature_importance.tail(15).plot.barh(x='feature', y='gain')

# %%
"""### Feature importance agg"""

# %%
variables_types_ = variables_types.VARIABLE
variables_types_ = [v.upper() for v in variables_types_]

feature_importance['feature_agg'] = feature_importance.feature.apply(feature_agg)
feature_importance_agg = feature_importance.groupby('feature_agg').sum('gain')
feature_importance_agg = feature_importance_agg.reset_index(0)
feature_importance_agg = feature_importance_agg.sort_values('gain')
g = feature_importance_agg.tail(15).plot.barh(x='feature_agg', y='gain')

# %%
"""## Test"""

# %%
"""### Metrics"""

# %%
metrics

# %%
"""By class"""

# %%
metrics_class_list = ['roc_auc', 'neg_log_loss']
metrics_class = pd.DataFrame({'class': model.classes_})
metrics_class['label'] = CLASSES_
metrics_class['support'] = y.value_counts()
for class_ in model.classes_:
    for metric in metrics_class_list:
        scoring = sk.metrics.get_scorer(metric)
        scoring = scoring._score_func
        metrics_class.loc[class_, metric] = scoring(
            y_onehot[str(class_)], p_proba[str(class_)])

summary = pd.DataFrame({'class': [None], 'label': 'ALL_',
                        'support': y.count()})
for metric in metrics_class_list:
    summary[metric] = np.average(metrics_class[metric], weights=metrics_class.support)
metrics_class = pd.concat([metrics_class, summary])
metrics_class

# %%
print(sk.metrics.classification_report(y, p, target_names=CLASSES_))

# %%
for class_ in model.classes_:
    class_str_ = CLASSES_[int(class_)]
    plt.figure(figsize=(10, 10))
    y_onehot_ = y_onehot[str(class_)]
    p_proba_ = p_proba[str(class_)]
    plt.hist(p_proba_, range=(0, 1), bins=20, label=f"MODEL_PRED_{class_str_}",
             histtype="step", lw=2)

    plt.hist(y_onehot_, range=(0, 1), bins=20, label=f"REAL_{class_str_}",
             histtype="step", lw=2)

    plt.xlabel("Mean predicted value")
    plt.ylabel("Count")
    plt.legend(ncol=2)

# %%
"""## Calibration"""

# %%
"""## Total average"""

# %%
del x, y, p_proba
with bq.blob_open(predictions_calibration_path, 'rb') as f:
    data = pd.read_csv(f)

# %%
data_mean = data[['REAL', 'MODEL_PRED_RENEWED']].mean()
g = data_mean.plot.barh()

# %%
"""## Calibration curve"""

# %%
calibration_curve(data['REAL'], data['MODEL_PRED_RENEWED'])
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
y_test = data['REAL']

model = 'MODEL_PRED_RENEWED'
prob_pos = data[model]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
         label="%s" % (model,))

ax2.hist(prob_pos, range=(0, 1), bins=20, label=model,
         histtype="step", lw=2)
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

# %%
