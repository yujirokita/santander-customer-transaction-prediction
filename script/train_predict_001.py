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
    'bagging_freq': 5,
    'bagging_fraction': 1.0,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.005,
    'max_depth': -1,
    'metric':'binary_logloss',
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 64,
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
oof = np.zeros(len(X_tr))
pred = np.zeros(len(X_te))
kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
X_tr_all = np.concatenate([X_tr_ for X_tr_ in X_tr_list], axis=1)
X_te_all = np.concatenate([X_te_ for X_te_ in X_te_list], axis=1)    
score = 0

for fold_, (trn_idx, val_idx) in enumerate(kfold.split(X_tr, y)):
    trn_dataset = lgb.Dataset(X_tr_all[trn_idx], label=y[trn_idx])
    val_dataset = lgb.Dataset(X_tr_all[val_idx], label=y[val_idx])
    
    clf = lgb.train(lgb_params, trn_dataset, valid_sets=[trn_dataset, val_dataset],
                    num_boost_round=NROUND, early_stopping_rounds=100, verbose_eval=200)
    oof[val_idx] = clf.predict(X_tr_all[val_idx], num_iteration=clf.best_iteration)

    score_ = roc_auc_score(y[val_idx], oof[val_idx])
    print(f'Fold {fold_}: SCORE={score_}')

    score += score_ / kfold.n_splits
    pred += clf.predict(X_te_all, num_iteration=clf.best_iteration) / kfold.n_splits
    

print(f'CV_SCORE={score}')
print('OOF_SCORE={}'.format(roc_auc_score(y, oof)))

# output
ids_test = np.load('../data/ids_test.npy')
submission = pd.DataFrame({'ID_code': ids_test, 'target': pred})
submission.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')