import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data() -> pd.DataFrame:
    return pd.read_csv('data/data-v3.1.csv', low_memory=False, dtype=np.float64)


def calculate_correlation(df):
    df = df[df['weight'] != ' ']
    df['weight'] = df['weight'].astype(np.float64)

    for column in df.columns:
        if column == 'weight':
            continue

        df = df[df[column] != ' ']
        df[column] = df[column].astype(np.float64)
        
        corr = df[column].corr(df['weight'])
        if (corr>0.2) or (corr< -0.2):
            print(
                f'Column name {column}, corr: {corr}')


from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split


def regression_for_len():
    df = load_data()
    df = df[['weight', 'len', 'G5']]
    df = (df-df.min())/(df.max()-df.min())

    test_data = df[df['weight'] == np.NaN]
    print(len(test_data))

    # y = df['weight']
    # X = df[['len', 'G5']]

    # reg = LinearRegression().fit(X, y)

    # x_test = test_data[['len', 'G5']]
    # y_test = test_data['weight']

    # y_pred = reg.predict(x_test)

    # test_data['weight'] = y_pred
    # test_data.info()
    # print(test_data['weight'])


def main():
    regression_for_len()

if __name__ == '__main__':
    main()