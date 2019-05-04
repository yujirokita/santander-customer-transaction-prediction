import gc, os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

####################
# preparation
####################

# parameters
SCRIPT_NAME = os.path.basename(__file__).split('.')[0]
SEED = np.random.randint(99999); np.random.seed(SEED); print(f'Random-Seed: {SEED}')
NFOLD = 5
NUM_ROUND = 100
NUM_STOPPING_ROUND = 5
METRIC = 'binary_logloss'
LGB_PARAMS = {
    'bagging_fraction': 0.6,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1.0,
    'learning_rate': 0.03,
    'max_depth': 2,
    'num_leaves': 3,
    'metric': METRIC,
    'num_threads': cpu_count(),
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
}

# functions
def score(y, pred):
    return roc_auc_score(y, pred)


####################
# execute training & prediction
####################

# load data
X_tr = np.load('../data/X_train_scaled.npy')
X_te = np.load('../data/X_test_scaled.npy')
X = np.concatenate((X_tr, X_te), axis=0)
y = np.load('../data/y.npy')

# load encoded data
X_tr_list = [X_tr]+[np.load(f'../data/X_train_ce_{i}.npy') for i in range(2, 6)]
X_te_list = [X_te]+[np.load(f'../data/X_test_ce_{i}.npy') for i in range(2, 6)]

# train and predict
kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
X_tr_all = np.concatenate([X_tr_ for X_tr_ in X_tr_list], axis=1)
X_te_all = np.concatenate([X_te_ for X_te_ in X_te_list], axis=1)
oof = np.zeros(len(X_tr_all))
pred = np.zeros(len(X_te_all))
importances = np.zeros(X_tr_all.shape[1])

for fold_, (trn_idx, val_idx) in enumerate(kfold.split(X_tr_all, y)):
    print(f'----- Fold-{fold_} -----')

    trn_dataset = lgb.Dataset(X_tr_all[trn_idx], label=y[trn_idx])
    val_dataset = lgb.Dataset(X_te_all[val_idx], label=y[val_idx])
    
    clf = lgb.train(LGB_PARAMS, trn_dataset, valid_sets=[trn_dataset, val_dataset],
                    num_boost_round=NUM_ROUND, early_stopping_rounds=NUM_STOPPING_ROUND, verbose_eval=1000)
    oof[val_idx] = clf.predict(X_tr_all[val_idx], num_iteration=clf.best_iteration)
    pred += clf.predict(X_te_all, num_iteration=clf.best_iteration) / kfold.n_splits
    importances += clf.feature_importance(importance_type='gain') / kfold.n_splits

    print(f'Score: {score(y[val_idx], oof[val_idx])}')

print(f'Total Score: {score(y, oof)}')

# save feature importances
np.save(f'../data/feature_importances_{SCRIPT_NAME}_{SEED}.npy', importances)

# output
ids_test = np.load('../data/ids_test.npy', allow_pickle=True)
submission = pd.DataFrame({'ID_code': ids_test, 'target': pred})
submission_path = f'../output/{SCRIPT_NAME}_{SEED}.csv.gz'
submission.to_csv(submission_path, index=False, compression='gzip')