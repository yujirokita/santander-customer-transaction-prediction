import numpy as np
import pandas as pd
import os

X_te = pd.read_csv('../input/test.csv.zip')
X_te = X_te.drop(['ID_code'], axis=1).values

unique_count = np.zeros_like(X_te)
for feature in range(X_te.shape[1]):
    _, index_, count_ = np.unique(X_te[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_index = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
fake_index = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

np.save('../data/real_index', real_index)
np.save('../data/fake_index', fake_index)