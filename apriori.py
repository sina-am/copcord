import pandas as pd
import numpy as np
from loader import load_data
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = load_data()

df = df[
    ['marital', 'a3', 'education',
     'MS.prob', 'shoulder', 'elbow',
     'wrist', 'hand', 'hip',
     'knee', 'ankle', 'foot.finger',
     'neck', 'vertebral', 'tra.history',
     'past.MS.prb', 'shoulder.past',
     'shld.pain.first.time', 'shld.pain.past',
     'elbow.past', 'elb.pain.first.time', 'wrist.past',
     'wr.pain.first.time', 'hand.pst', 'ha.pain.first.time',
     'hip.past', 'hip.pain.first.time', 'knee.past', 'kn.pain.first.time',
     'ankle.past', 'ank.pain.first.time', 'foot.finger.problem.past',
     'ff.pain.past', 'ff.pain.first.time', 'neck.past', 'ne.pain.first.time',
     'vertebral.past', 'ver.pain.first.time',
     'disability', 'clothing', 'standing', 'drinking', 'eating', 'walking',
     'KneeOA', 'HipOA', 'HandOA', 'CMC1OA', 'NeckOA', 'TotalOA', 'Hypermobility',
     'RA', 'Behcet', 'SLE', 'fibromyalgia', 'Golfelbow', 'Tenniselbow', 'dequrvan',
     'triggerfinger', 'spondyloarthropathy', 'Gout', 'LBP', 'lumbarradicolopathy',
     'CTS', 'Chondromalaciapatella', 'FrozenShoulder', 'Rotatorcuff',
     'periarthritis', 'neckradicolopathy'
     ]]

df.fillna(round(df.mean()), inplace=True)

transactions = np.zeros(len(df), dtype=np.ndarray)
for row_index, row in enumerate(df.values.tolist()):
    item_sets = []
    for i, item in enumerate(row):
        if item != 0:
            item_sets.append(f'{df.columns[i]}.{int(item)}')

    transactions[row_index] = np.array(item_sets)

te = TransactionEncoder()
Transaction_Array = te.fit(transactions).transform(transactions)

new_df = pd.DataFrame(Transaction_Array, columns=te.columns_)

frequent_items = apriori(new_df, min_support=0.3, use_colnames=True)
print(frequent_items)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.9)

print(rules)
