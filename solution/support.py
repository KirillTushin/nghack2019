import sys
import pandas as pd
from sklearn.externals import joblib


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    support_label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    support_label_map = {i:key for key, i in support_label_map.items()}
    
    support_model = joblib.load('support_model')
    df_test = pd.read_csv(input_csv, index_col='id')
    
    X = df_test['text'].fillna('none').str.lower()

    df_test['label'] = support_model.predict(X)
    df_test['label'] = df_test['label'].map(support_label_map)
    df_test.to_csv(output_csv)