import gc, os
from multiprocessing import cpu_count
from tqdm import tqdm
import sys
import datetime as dt

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# parameters
SCRIPT_NAME = os.path.basename(__file__).split('.')[0]
NOW = dt.datetime.now().strftime('%Y%m%d%H%M%S')
SUBMIT_FILE_PATH = f'../output/{SCRIPT_NAME}_{NOW}.csv.gz'

NFOLD = 10
NROUND = 1000
SEED = np.random.randint(99999); np.random.seed(SEED); print(f'SEED: {SEED}')

lgb_params = {
    'bagging_fraction': 1.0,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.5,
    'max_depth': -1,
    'metric':'binary_logloss',
    'num_threads': cpu_count(),
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
}

# load
X_tr = pd.read_csv('../input/train.csv.zip')
X_te = pd.read_csv('../input/test.csv.zip')
y = X_tr.pop('target')
ids_tr = X_tr.pop('ID_code')
ids_te = X_te.pop('ID_code')

X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)

# drop fake
fake_indexes = np.load('../data/fake_index.npy')
fake_indexes += len(X_tr)

# count encoding
def count_encode(series):
    series_raw = series.drop(fake_indexes)
    value_counts = series_raw.value_counts().to_dict()
    return series.map(value_counts)

X_cnts = [X.round(i).apply(count_encode, axis=0, result_type='broadcast') for i in range(1, 5)]

# train and predict
O = pd.DataFrame(index=X_tr.index, columns=X_tr.columns)
P = pd.DataFrame(index=X_te.index, columns=X_te.columns)
kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)

for i, c in enumerate(X.columns):
    
    x_tr = X_tr[[c]].copy()
    for i in range(4):
        x_tr['cnts_'+str(i)] = X_cnts[i].iloc[:len(x_tr)][c]
    oof = np.zeros(len(x_tr))
    
    x_te = X_te[[c]].copy()
    for i in range(4):
        x_te['cnts_'+str(i)] = X_cnts[i].iloc[len(x_tr):][c]
    pred = np.zeros(len(x_te))
    
    cv_score = 0
    
    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(x_tr, y)):
        
        trn_dataset = lgb.Dataset(x_tr.reindex(trn_idx), label=y.reindex(trn_idx))
        val_dataset = lgb.Dataset(x_tr.reindex(val_idx), label=y.reindex(val_idx))
        
        clf = lgb.train(lgb_params, trn_dataset, valid_sets=[trn_dataset, val_dataset],
                        num_boost_round=NROUND, early_stopping_rounds=100, verbose_eval=False)
        oof[val_idx] = clf.predict(x_tr.reindex(val_idx), num_iteration=clf.best_iteration)
        pred += clf.predict(x_te, num_iteration=clf.best_iteration) / kfold.n_splits
        cv_score += roc_auc_score(y.reindex(val_idx), oof[val_idx]) / kfold.n_splits
    
    print('{}: CV_SCORE={}'.format(c, cv_score))
    
    O[c] = oof
    P[c] = pred

print('ALL:: CV_SCORE={}'.format(roc_auc_score(y, O.mean(axis=1))))

# output
submission = pd.DataFrame({'ID_code': ids_te, 'target': P.mean(axis=1)})
submission.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')