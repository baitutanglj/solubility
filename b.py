import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('/mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data/CYP1A2_data_clean/smiles_target_CYP1A2_data_clean.csv')
target_name = 'CYP1A2_Class'

label = np.unique(df[target_name])
if 'High' in label:
    label = label[::-1]
else:
    label = label
le = LabelEncoder()
le.fit(label)
le.classes_ = pd.Series(label).unique()
res = {}
for cl in le.classes_:
    res.update({cl:le.transform([cl])[0]})
print(res)
df['target_label'] = le.transform(df[target_name])
# with open(label_encoder_path, 'wb') as f:
#     pickle.dump(le, f)


label_encoder_path = '/mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data/CYP1A2_data_clean/CYP1A2_data_clean.pkl'
with open(label_encoder_path, 'rb') as f:
    le_departure = pickle.load(f)
le_departure.classes_ = pd.Series(label).unique()
df['target_label'] = le_departure.transform(df[target_name])
