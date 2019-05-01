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
X_tr = np.load('../data/X_train_scaled.npy')
X_te = np.load('../data/X_test_scaled.npy')
X = np.concatenate((X_tr, X_te), axis=0)
y = np.load('../data/y.npy')

# load encoded data
X_tr_list = [X_tr]+[np.load(f'../data/X_train_ce_{i}.npy') for i in range(2, 5)]
X_te_list = [X_te]+[np.load(f'../data/X_test_ce_{i}.npy') for i in range(2, 5)]

# train and predict
P = np.zeros_like(X_te)
kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

for c in range(200):
    x_tr = np.stack([X_tr_[:, c] for X_tr_ in X_tr_list], axis=1)
    x_te = np.stack([X_te_[:, c] for X_te_ in X_te_list], axis=1)    
    score = 0
    
    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(x_tr, y)):
        trn_dataset = lgb.Dataset(x_tr[trn_idx], label=y[trn_idx])
        val_dataset = lgb.Dataset(x_tr[val_idx], label=y[val_idx])
        
        clf = lgb.train(lgb_params, trn_dataset, valid_sets=[trn_dataset, val_dataset],
                        num_boost_round=NROUND, early_stopping_rounds=100, verbose_eval=False)
                        
        oof = clf.predict(x_tr[val_idx], num_iteration=clf.best_iteration)
        score_ = roc_auc_score(y[val_idx], oof)
        score += score_ / kfold.n_splits

        P[:, c] += clf.predict(x_te, num_iteration=clf.best_iteration) / kfold.n_splits
    
    print(f'Column {c}: CV_SCORE={score}')

# output
ids_test = np.load('../data/ids_test.npy')
submission = pd.DataFrame({'ID_code': ids_test, 'target': P.mean(axis=1)})
submission.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')