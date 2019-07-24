import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from numba import jit
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(
                        np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def train_model_classification(train,
                               test,
                               target,
                               params,
                               model_type='lgb',
                               test_size=0.16,
                               shuffle=False,
                               n_estimators=50000,
                               n_jobs=-1,
                               early_stopping_rounds=200,
                               verbose=200):
    X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=test_size, shuffle=shuffle)
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    if model_type == 'lgb':
        model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric=metrics_dict['auc']['lgb_metric_name'],
                  verbose=verbose,
                  early_stopping_rounds=early_stopping_rounds)
        y_pred = model.predict_proba(test, num_iteration=model.best_iteration_)[:, 1]
        y_pred_valid = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]

    if model_type == 'xgb':
        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=train.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=train.columns)
        watch_list = [(train_data, 'train'), (valid_data, 'valid')]
        model = xgb.train(dtrain=train_data,
                          num_boost_round=n_estimators,
                          evals=watch_list,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose,
                          params=params)
        y_pred = model.predict(xgb.DMatrix(test, feature_names=train.columns), ntree_limit=model.best_ntree_limit)
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=train.columns),
                                     ntree_limit=model.best_ntree_limit)
    if model_type == 'cat':
        model = CatBoostClassifier(iterations=n_estimators,
                                   eval_metric=metrics_dict['auc']['catboost_metric_name'])
        model.fit(X_train,
                  y_train,
                  eval_set=(X_valid, y_valid),
                  cat_features=[],
                  use_best_model=True,
                  verbose=False)
        y_pred = model.predict(test)
        y_pred_valid = model.predict(X_valid)
    auc = fast_auc(y_valid,y_pred_valid)
    return y_pred, auc
