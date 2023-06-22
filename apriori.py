from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np

df = pd.read_csv('data-v1-with-NaN.CSV', low_memory=False,dtype=np.float64)

df.fillna(round(df.mean()), inplace=True)

column_names = list(df.columns.values)

for column in column_names:
    column_values = df[column].unique()
    num_values = len(column_values)

    if num_values == 2:
        bins = 2
    elif num_values == 3:
        bins = 3
    else:
        bins = 4

    # Convert column values to intervals
    df[column] = pd.cut(df[column], bins=bins)

    # Convert intervals to categorical data and encode as integers
    df[column] = pd.Categorical(df[column])
    df[column] = df[column].cat.codes
  

data_bin = df.astype(int)
data_transactions = data_bin.values.tolist()


te = TransactionEncoder()
Transaction_Array = te.fit(data_transactions).transform(data_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(frequent_itemsets)
print(rules)