from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_score,KFold
import pandas as pd


df = pd.read_csv('data-v1-with-NaN.CSV', low_memory=False,dtype=np.float64)

df.fillna(round(df.mean()), inplace=True)

XT = df.iloc[:, :226]
Y = df['LBP']
X = df[['Agegroup', 'weight', 'len', 'C2F1']]

svm_classifier = svm.SVC(kernel='linear', C=1.0)

k = 10
kf = KFold(n_splits=k)

scores = cross_val_score(svm_classifier, X, Y, cv=kf)

average_accuracy = scores.mean()

print(average_accuracy)