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
NUM_ROUND = 10000
NUM_STOPPING_ROUND = 500
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
O = np.zeros_like(X_tr)
P = np.zeros_like(X_te)
kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

for c in range(200):
    print(f'========== var_{c} ==========')

    x_tr = np.stack([X_tr_[:, c] for X_tr_ in X_tr_list], axis=1)
    x_te = np.stack([X_te_[:, c] for X_te_ in X_te_list], axis=1)

    dataset = lgb.Dataset(x_tr, label=y)
    history = lgb.cv(LGB_PARAMS, dataset, num_boost_round=NUM_ROUND,
                    folds=kfold, metrics=METRIC, early_stopping_rounds=NUM_STOPPING_ROUND,
                    verbose_eval=None, seed=SEED)
    best_round = np.argmin(history[f'{METRIC}-mean'])
    best_metric_score = history[f'{METRIC}-mean'][best_round]
    print(f'Mean {METRIC}: {best_metric_score}')
    
    oof = np.zeros(len(x_tr))
    pred = np.zeros(len(x_te))
    importances = np.zeros(x_tr.shape[1])
    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(x_tr, y)):
        trn_dataset = lgb.Dataset(x_tr[trn_idx], label=y[trn_idx])
        val_dataset = lgb.Dataset(x_tr[val_idx], label=y[val_idx])
        
        clf = lgb.train(LGB_PARAMS, trn_dataset, num_boost_round=best_round)

        oof[val_idx] = clf.predict(x_tr[val_idx])
        pred += clf.predict(x_te) / kfold.n_splits
        importances += clf.feature_importance(importance_type='gain') / kfold.n_splits
    
    print(f'Score: {score(y, oof)}')
    print(f'Feature-Importances: {importances}')

    O[:, c] = oof
    P[:, c] = pred
    
# savepoint
np.save(f'../data/O_{SCRIPT_NAME}_{SEED}.npy', O)
np.save(f'../data/P_{SCRIPT_NAME}_{SEED}.npy', P)

# validate
blend_oof = np.log(O).mean(axis=1) - np.log(1-O).mean(axis=1) #mean of logit
blend_score = score(y, blend_oof)
print(f'Score of Blend: {blend_score}')

# output
ids_test = np.load('../data/ids_test.npy', allow_pickle=True)
blend_pred = np.log(P).mean(axis=1) - np.log(1-P).mean(axis=1) #mean of logit
submission = pd.DataFrame({'ID_code': ids_test, 'target': blend_pred})
submission_path = f'../output/{SCRIPT_NAME}_{SEED}.csv.gz'
submission.to_csv(submission_path, index=False, compression='gzip')