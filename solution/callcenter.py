import sys
import pandas as pd
from sklearn.externals import joblib


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    ## load model and data
    callcenter_model = joblib.load('callcenter_model')
    df_test = pd.read_csv(input_csv, index_col='id')
    sub_columns = df_test.columns + ['Метка']
    
    ## prepare data
    time_columns = [
        'Время окончания разговора с оператором',
        'Время переключения на оператора',
        'Время постановки в очередь',
        'Время окончания вызова',
        'Время начала вызова',
    ]

    start_time = pd.Timestamp('00:00:00')

    for col in time_columns:
        df_test[col] = df_test[col].fillna(start_time)
        df_test[col] = df_test[col].apply(pd.Timestamp) - start_time

    for i, col_1 in enumerate(time_columns[:-1]):
        for col_2 in time_columns[i + 1:]:
            df_test[f'{col_1}-{col_2}'] = abs(df_test[col_1] - df_test[col_2])

    for col in df_test.columns.drop('Длительность разговора с оператором, сек'):
        df_test[f'{col}_seconds'] = df_test[col].apply(lambda x: x.seconds)

    df_test = df_test[
        ['Длительность разговора с оператором, сек'] + [x for x in df_test.columns if 'seconds' in x]
    ]    
    
    df_test['Метка'] = callcenter_model.predict_proba(df_test)[:,1] > 0.4445045903799856
    df_test['Метка'] = df_test['Метка'].astype(int)
    df_test.to_csv(output_csv)