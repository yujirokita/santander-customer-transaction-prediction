import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load
X_tr = pd.read_csv('../input/train.csv.zip')
X_te = pd.read_csv('../input/test.csv.zip')

# prepare
y = X_tr.pop('target').values
ids_tr = X_tr.pop('ID_code').values
ids_te = X_te.pop('ID_code').values
X_tr = X_tr.values
X_te = X_te.values

np.save('../data/X_train.npy', X_tr)
np.save('../data/X_test.npy', X_te)
np.save('../data/y.npy', y)
np.save('../data/ids_train.npy', ids_tr)
np.save('../data/ids_test.npy', ids_te)

# scaling
X = np.concatenate((X_tr, X_te), axis=0)
scaler = StandardScaler()
X =scaler.fit_transform(X)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

np.save('../data/X_train_scaled.npy', X_tr)
np.save('../data/X_test_scaled.npy', X_te)