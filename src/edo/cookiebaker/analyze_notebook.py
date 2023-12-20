"""# Analyze data."""

# %%

import inspect
import os

import jupyter_client  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
import seaborn as sns  # noqa
import yaml
from edo.mkt import bq

import edo.cookiecutter_sync
from edo.cookiecutter_sync.evaluate import feature_agg
from edo.cookiecutter_sync.train import CLASSES_

pd.set_option('display.max_columns', 999)

# %%
# vars
filename = './tmp/args_notebook.yml'
with open(filename) as f:
    args = yaml.full_load(f)

data_path = args['data_path']
output_path = args['output_path']
target_field = args['target_field']

# %%
# data
with bq.blob_open(data_path, 'r') as f:
    data = pd.read_csv(f)

data.columns = [v.upper() for v in data.columns]
data['LABEL'] = data['LABEL_INT'].apply(lambda x: CLASSES_[int(x)])

# %%
"""## Shape"""

# %%
data.shape

# %%
n = min(int(1e5), len(data) - 1)
print(f"Only using a random sample of {n} rows.")
data = data.sample(n)

# %%
"""Features Selected"""

# %%
package_path = os.path.dirname(inspect.getfile(edo.cookiecutter_sync))
filename = os.path.join(package_path, 'model_configuration', 'variables_types.csv')
variables_types = pd.read_csv(filename)

features_selected = variables_types.query('USE==1').shape[0]
features_selected_ratio = features_selected / variables_types.shape[0]
print(f"{features_selected} ({features_selected_ratio:.0%})")

# %%
"""## Missing"""

# %%
sum_na = data.isna().to_numpy().sum()
print(sum_na)
print(sum_na / data.size)
# g = sns.heatmap(data.isna(), cbar=False)
na_list = data.isna().sum() / len(data)
na_list = na_list.sort_values(ascending=False)

na_list.to_frame().transpose()

# %%
na_list.name = 'share_missing_values'
g = sns.kdeplot(na_list)

# %%
"""## Types"""

# %%
data.dtypes.value_counts()

# %%
"""## Describe"""

# %%
describe = data.describe(include='all')

describe.loc['dtype'] = describe.dtypes
describe.loc['size'] = len(data)
describe.loc['% missing'] = describe.isna().mean().apply(lambda x: f"{x:.2%}")
describe

# %%
"""## Target Distribution"""

# %%
g = sns.countplot(x="LABEL", data=data)

# %%
"""## Densities"""

# %%
"""Densities on 95% of data [2.5%, 97.5%]"""

# %%
"""Only numerical features displayed."""

# %%
y_plots = 2
# warnings.filterwarnings("ignore")
columns = variables_types[variables_types.USE == 1].VARIABLE
columns_ = columns.apply(feature_agg)
columns_ = columns_.sort_values().unique().tolist()
columns = [v for v in columns_ if v in columns.tolist()]

target_column = 'LABEL'
target_groups = data[target_column].unique()
for column in columns:
    if data[column].dtype == 'O':
        continue
    print(column)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, y_plots, 1)
    quantiles_95 = np.quantile(data[column], [0.025, 0.975])
    if any([np.isnan(v) for v in quantiles_95]):
        data_aux = data
    else:
        data_aux = data.query(f"{column} >= {quantiles_95[0]} and "
                              f"{column} <= {quantiles_95[1]}")

    ax = sns.histplot(x=column, hue=target_column,
                      data=data_aux,
                      common_norm=False,
                      fill=False,
                      stat='percent',
                      element="step",
                      palette=sns.color_palette('husl', len(target_groups)))
    ax.set_title('density 95% quantile [2.25%, 97.25%]')
    plt.subplot(1, y_plots, 2)
    ax = sns.boxplot(x=target_column, y=column, data=data,
                     palette=sns.color_palette('husl', len(target_groups)))
    ax.set_title('box plot')
    plt.show()

