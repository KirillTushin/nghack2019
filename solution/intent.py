import sys
import pandas as pd
import numpy as np
from sklearn.externals import joblib


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    intent_label_map = {'FAQ - тарифы и услуги': 0, 'мобильная связь - тарифы': 1, 'Мобильный интернет': 2, 'FAQ - интернет': 3, 'тарифы - подбор': 4, 'Баланс': 5, 'Мобильные услуги': 6, 'Оплата': 7, 'Личный кабинет': 8, 'SIM-карта и номер': 9, 'Роуминг': 10, 'запрос обратной связи': 11, 'Устройства': 12, 'мобильная связь - зона обслуживания': 13}
    intent_label_map = {i:key for key, i in intent_label_map.items()}
    
    df_test = pd.read_csv(input_csv, index_col='id')
    X = df_test['text'].fillna('none').str.lower()
    pred = []
    
    for i in range(10):
        intent_model = joblib.load(f'intent_model_fold_{i}')
        pred.append(intent_model.predict_proba(X))

    df_test['label'] = np.array(pred).mean(axis=0).argmax(axis=1)
    df_test['label'] = df_test['label'].map(intent_label_map)
    df_test.to_csv(output_csv)