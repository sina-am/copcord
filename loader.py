import os
import pandas as pd
import numpy as np


def clean_data():
    df = pd.read_csv('data/data.csv', low_memory=False)
    df.drop(['intdate', 'intstart', 'intend', 'code'], axis=1, inplace=True)

    df.replace(' ', np.NaN, inplace=True)
    cols_to_delete = df.columns[df.isnull().sum()/len(df) > 0.1]
    df.drop(cols_to_delete, axis=1, inplace=True)

    for column in df.columns:
        df[column] = df[column].astype('str')
        df = df[~df[column].str.contains('?', regex=False)]

    df.to_csv('data/data-v0.1.csv')


def load_data() -> pd.DataFrame:
    if not os.path.exists('data/data-v0.1.csv'):
        clean_data()

    return pd.read_csv('data/data-v0.1.csv', low_memory=False)
