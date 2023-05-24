import random
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/data-v0.2.csv', low_memory=False)

for column in df.columns:
    if column == 'LBP':
        continue

    df = df[df[column] != ' ']
    df[column] = df[column].astype(np.float64)
    df[column] = df[column].fillna(df[column].mode()[0])
    
    corr = df[column].corr(df['LBP'])
    if (corr>0.2) or (corr< -0.2):
        print(
            f'Column name {column}, corr: {corr}')
