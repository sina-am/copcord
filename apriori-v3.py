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
    if num_values != 2:

        if num_values > 2:
            label1 = f"{column}_set_1"
            label2 = f"{column}_set_2"
            label3 = f"{column}_set_3"

            df[column] = pd.cut(df[column], bins=3,labels=[label1,label2,label3],right=False)
            # add new columns to df named : column_set_1, column_set_2, column_set_3
            df[f"{column}_set_1"] = 0
            df[f"{column}_set_2"] = 0
            df[f"{column}_set_3"] = 0

            for i in range(len(df[column])):
                if df[column][i] == label1:
                    # set column_set_1 to 1
                    df[f"{column}_set_1"][i] = 1
                elif df[column][i] == label2:
                    # set column_set_2 to 1
                    df[f"{column}_set_2"][i] = 1
                else:
                    # set column_set_3 to 1
                    df[f"{column}_set_3"][i] = 1
                # drop column
            df.drop([column], axis=1, inplace=True)
        # else:
        #     label1 = f"{column}_set_1"
        #     label2 = f"{column}_set_2"
        #     label3 = f"{column}_set_3"
        #     label4 = f"{column}_set_4"
        
        #     df[column] = pd.cut(df[column], bins=4,labels=[label1,label2,label3,label4],right=False)

        #     # add new columns to df named : column_set_1, column_set_2, column_set_3, column_set_4
        #     df[f"{column}_set_1"] = 0
        #     df[f"{column}_set_2"] = 0
        #     df[f"{column}_set_3"] = 0
        #     df[f"{column}_set_4"] = 0

        #     for i in range(len(df[column])):
        #         if df[column][i] == label1:
        #             # set column_set_1 to 1
        #             df[f"{column}_set_1"][i] = 1
        #         elif df[column][i] == label2:
        #             # set column_set_2 to 1
        #             df[f"{column}_set_2"][i] = 1
        #         elif df[column][i] == label3:
        #             # set column_set_3 to 1
        #             df[f"{column}_set_3"][i] = 1
        #         else:
        #             # set column_set_4 to 1
        #             df[f"{column}_set_4"][i] = 1
        #     # drop column
        #     df.drop([column], axis=1, inplace=True)

    # Convert intervals to categorical data and encode as integers
    # df[column] = pd.Categorical(df[column])
    # df[column] = df[column].astype('category')
    # df[column] = df[column].cat.codes  


df = df.astype(bool)

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

print(frequent_itemsets)

print(rules)