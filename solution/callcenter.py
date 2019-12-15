import sys
import pandas as pd
import numpy as np
from sklearn.externals import joblib


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    ## load model and data
    df_test = pd.read_csv(input_csv, index_col='id')
    
    ## prepare data
    time_columns = [
        'Время окончания разговора с оператором',
        'Время переключения на оператора',
        'Время постановки в очередь',
        'Время окончания вызова',
        'Время начала вызова',
    ]

    null_columns = [
        'Время постановки в очередь',
        'Время переключения на оператора',
        'Время окончания разговора с оператором',
        'Длительность разговора с оператором, сек',
    ]
    for col in null_columns:
        df_test[f'{col}_isna'] = df_test[col].isna()
    null_columns = [f'{col}_isna' for col in null_columns]

    start_time = pd.Timestamp('00:00:00')

    for col in time_columns:
        df_test[col] = df_test[col].fillna(start_time)
        df_test[col] = df_test[col].apply(pd.Timestamp) - start_time

    for i, col_1 in enumerate(time_columns[:-1]):
        for col_2 in time_columns[i + 1:]:
            df_test[f'{col_1}-{col_2}'] = abs(df_test[col_1] - df_test[col_2])

    for col in df_test.columns.drop(['Длительность разговора с оператором, сек'] + null_columns):
        df_test[f'{col}_seconds'] = df_test[col].apply(lambda x: x.seconds)
    
    use_columns = [
       'Длительность разговора с оператором, сек',
       'Время постановки в очередь_isna',
       'Время переключения на оператора_isna',
       'Время окончания разговора с оператором_isna',
       'Длительность разговора с оператором, сек_isna',
       'Время начала вызова_seconds', 'Время окончания вызова_seconds',
       'Время постановки в очередь_seconds',
       'Время переключения на оператора_seconds',
       'Время окончания разговора с оператором_seconds',
       'Время окончания разговора с оператором-Время переключения на оператора_seconds',
       'Время окончания разговора с оператором-Время постановки в очередь_seconds',
       'Время окончания разговора с оператором-Время окончания вызова_seconds',
       'Время окончания разговора с оператором-Время начала вызова_seconds',
       'Время переключения на оператора-Время постановки в очередь_seconds',
       'Время переключения на оператора-Время окончания вызова_seconds',
       'Время переключения на оператора-Время начала вызова_seconds',
       'Время постановки в очередь-Время окончания вызова_seconds',
       'Время постановки в очередь-Время начала вызова_seconds',
       'Время окончания вызова-Время начала вызова_seconds'
    ]
    X = df_test[use_columns]
    
    pred = []
    for i in range(10):
        callcenter_model = joblib.load(f'callcenter_model_fold_{i}')
        pred.append(callcenter_model.predict_proba(X)[:,1])

    df_test['Метка'] = np.array(pred).mean(axis=0) > 0.44
    df_test['Метка'] = df_test['Метка'].astype(int)
    df_test.to_csv(output_csv)