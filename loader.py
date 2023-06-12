import pandas as pd
import numpy as np


def is_float(value: str):
    try:
        float(value)
    except ValueError:
        return False
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(' ', np.NaN)
    df['age'] = df['age'].astype(np.float64)
    df['birthdat'] = df['birthdat'].astype(np.float64)
    df['LBP'] = df['LBP'].astype(np.float64)
    df['KneeOA'] = df['KneeOA'].astype(np.float64)
    df['len'] = df['len'].astype(np.float64)
    df['weight'] = df['weight'].astype(np.float64)
    df['Agegroup'] = df['Agegroup'].astype(np.float64)
    df['ank.tenderness'] = df['ank.tenderness'].astype(np.float64)
    df['kn.pain'] = df['kn.pain'].astype(np.float64)
    df['knee'] = df['knee'].astype(np.float64)
    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv('data/data.csv', low_memory=False)
    return clean_data(df)


if __name__ == '__main__':
    load_data('data/data.csv')
