import gc, os
from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm import tqdm
import sys
import datetime as dt

import numpy as np
import pandas as pd

# load
X_tr = np.load('../data/X_train_scaled.npy')
X_te = np.load('../data/X_test_scaled.npy')
X = np.concatenate((X_tr, X_te), axis=0)

# fake index
fake_index = np.load('../data/fake_index.npy')
fake_index += len(X_tr)

# count encoding
def count_encode(arr):
    series = pd.Series(arr)
    series_raw = series.drop(fake_index)
    value_counts = series_raw.value_counts().to_dict()
    return series.map(value_counts).values

for i in range(2, 5):
    print(f'count-encoding: {i}')
    X_round = np.round(X, i)
    
    with Pool(cpu_count()) as pool:
        cnts = pool.map(count_encode, [X_round[:, col] for col in range(200)])

    X_cnt = np.array(cnts).T
    X_tr_cnt = X_cnt[:len(X_tr)]
    X_te_cnt = X_cnt[len(X_tr):]
        
    np.save(f'../data/X_train_ce_{i}.npy', X_tr_cnt)
    np.save(f'../data/X_test_ce_{i}.npy', X_te_cnt)