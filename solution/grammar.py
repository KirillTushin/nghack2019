import sys
import pandas as pd
from sklearn.externals import joblib


def transform_sequences(original_sequences, predictions, token_one, token_two):
    result = []
    for sequence, pred in zip(original_sequences, predictions):
        if pred == 0:
            one_index = sequence.find(token_one)
            two_index = sequence.find(token_two)
            if one_index != -1 and two_index == -1:
                sequence = sequence[:one_index] + token_two + sequence[one_index + len(token_one):]
            if one_index == -1 and two_index != -1:
                sequence = sequence[:two_index] + token_one + sequence[two_index + len(token_two):]
        result.append(sequence)
    return result


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_test = pd.read_csv(input_csv, index_col='id')
    df_test['correct_sentence'] = df_test['sentence_with_a_mistake']
    for token_1, token_2, tresh in zip(
         ['тоже', 'в течении', 'из за', 'тся'],
         ['то же', 'в течение',  'из-за', 'ться'],
         [0.79, 0.42, 0.79, 0.89],
    ):
    
        model = joblib.load(f'model_spelling_{token_1}_{token_2}')
        pred_label = model.predict_proba(df_test['sentence_with_a_mistake'].str.lower())[:,0] < tresh
        df_test['correct_sentence'] = transform_sequences(df_test['correct_sentence'], pred_label, token_1, token_2)
    df_test.to_csv(output_csv)
