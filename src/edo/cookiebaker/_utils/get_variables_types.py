"""Get variable types (csv)."""
import os

import fire
import pandas as pd
import importlib.resources as pkg
from edo.mkt import bq
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder

VARIABLES_METADATA = [
    'EMAIL_SHA1',
    'MEMBER_ACCOUNT_ID',
    'SUBSCR_DATE',
    'SUBSCR_BOOKING_ID',
    'RENEWAL_INFO',
    'LABEL',
    'LABEL_INT',
    'SUBSCR_VISIT_ID',
    'SUBSCR_SEARCH_ID',
    'LABEL_BOOL',
    'RENEWED',
    'CHURN',
    'FAIL',
    'VISIT_DATE',
    'BOOKING_DATE',
    'SEARCH_DATE',
    'VISIT_ID',
    'BOOKING_ID',
    'SEARCH_ID',
    'SUBSCR_FEE_CURRENCY',
    'SUBSCR_FEE_AMOUNT'
    # 'DEP_IATA',
    # 'ARR_IATA'

]

VARIABLES_EXCLUDE = [
    'FTP',
    'MEMBER_ID',
    'TYPE',
]


def map_type(x, y):
    """Map type."""
    x = str(x)
    x = x.lower()
    y = str(y)
    y = y.lower()
    if x == 'object':
        return 'str'
    elif 'unix' in y:
        return 'date'
    elif 'float' in x:
        return 'float'
    elif 'int' in x:
        return 'float'
    elif 'numeric' in x:
        return 'float'
    else:
        return 'str'


def get_variable_types(table='ds_cookiecutter_sync_features',
                       project='ds-mkt',
                       dataset='ds_mkt_cookiecutter_sync',
                       start_date='2022-02-07',
                       end_date='2022-02-08',
                       limit_sample=10000):
    """Get variable types (csv)."""
    table = bq.TableName(f"{project}.{dataset}.{table}")
    params = {}
    params['table_features'] = table.name
    params['project'] = project
    params['dataset'] = dataset
    params['start_date'] = start_date
    params['end_date'] = end_date
    params['website_filter'] = ''
    params['class_filter'] = ''
    params['type_filter'] = ''
    params['variables_list_str'] = '*'
    params['limit_sample'] = f'limit {int(limit_sample)}'
    package_path = pkg.files("edo.cookiecutter_sync")
    package_path = package_path.replace('_utils', '')
    filename = os.path.join(package_path, 'queries', 'get_data.sql')
    with open(filename) as f:
        sql = f.read()
        sql = sql.format(**params)
    # sql = f"select * from `{table.bq}` where date(subscr_date) = '{date}'"  # noqa
    data = bq.get_query(sql)
    data.columns = [v.upper() for v in data.columns]

    if len(data) < 10:
        raise ValueError('len data is zero.')

    schema = bq.client.get_table(table.bq).schema
    schema = [[v.name.upper(), v.field_type.lower()] for v in schema]
    out = pd.DataFrame(schema, columns=['VARIABLE', 'TYPE'])

    out.TYPE = out.apply(lambda x: map_type(x.TYPE, x.VARIABLE), axis=1)

    out['FILL'] = 'NONE'
    out['USE'] = 1

    columns_float = out.query("TYPE == 'float'").VARIABLE.to_list()
    columns_str = out.query("TYPE != 'float'").VARIABLE.to_list()
    column_trans = ColumnTransformer(
        [('str', OrdinalEncoder(), columns_str)],
        remainder='passthrough'
    )
    column_trans.fit(data)

    data_num = pd.DataFrame(column_trans.transform(data),
                            columns=columns_str + columns_float)
    sel = VarianceThreshold(0.1)
    data_var = sel.fit(data_num)
    columns_var = data_var.get_feature_names_out()
    out.loc[[v not in columns_var for v in out.VARIABLE], 'USE'] = 2

    out.loc[[v in VARIABLES_METADATA for v in out.VARIABLE], 'USE'] = 2
    out = out.loc[[v not in VARIABLES_EXCLUDE for v in out.VARIABLE]]
    filename = 'tmp/variable_types.csv'
    print(filename)
    out.to_csv(filename, index=False)


def main():
    """Execute main program."""
    fire.Fire(get_variable_types)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
