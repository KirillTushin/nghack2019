import sys
import pandas as pd
import numpy as np
from sklearn.externals import joblib


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    support_label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    support_label_map = {i:key for key, i in support_label_map.items()}
    
    df_test = pd.read_csv(input_csv, index_col='id')
    X = df_test['text'].fillna('none').str.lower()
    pred = []
    
    for i in range(10):
        support_model = joblib.load(f'support_model_fold_{i}')
        pred.append(support_model.predict_proba(X))

    df_test['label'] = np.array(pred).mean(axis=0).argmax(axis=1)
    df_test['label'] = df_test['label'].map(support_label_map)
    df_test.to_csv(output_csv)