from utils import reduce_mem_usage, train_model_classification
import pandas as pd
import _pickle as pickle
import os
import datetime

today = datetime.date.today()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

params = {'learning_rate': 0.009,
          'max_depth': 10,
          'boosting': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'seed': 4,
          'num_iterations': 10000,
          'early_stopping_round': 100,
          'verbose_eval': 200,
          'num_leaves': 64,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 5}
emails = pickle.load(open('emails.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))
us_emails = ['gmail', 'net', 'edu']


def load_files():
    train_id = pd.read_csv('../input/train_identity.csv')
    train_tr = pd.read_csv("../input/train_transaction.csv")

    test_id = pd.read_csv("../input/test_identity.csv")
    test_tr = pd.read_csv("../input/test_transaction.csv")

    sample = pd.read_csv("../input/sample_submission.csv")
    print("Loading Files Complete")
    return train_id, train_tr, test_id, test_tr, sample


def preprocess(train, test):
    train['nulls1'] = train.isna().sum(axis=1)
    test['nulls1'] = test.isna().sum(axis=1)
    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    for c1, c2 in train.dtypes.reset_index().values:
        if c2 == 'O':
            train[c1] = train[c1].map(lambda x: labels[str(x).lower()])
            test[c1] = test[c1].map(lambda x: labels[str(x).lower()])
    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)
    print("preprocess complete")
    return train, test


def reduce_mem(train, test):
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    print("reduce memory complete")
    return train, test


def train_stage(train, test, target, params, model_type):
    y_pred = train_model_classification(train=train,
                                        test=test,
                                        target=target,
                                        params=params,
                                        model_type=model_type)
    return y_pred


train_id, train_tr, test_id, test_tr, sample = load_files()
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
print('merge complete')

train,test = reduce_mem(train,test)
train, test = preprocess(train, test)
target = train.isFraud
train = train.drop('isFraud', axis=1)
y_pred, auc = train_stage(train, test, target, params, 'lgb')

sample['isFraud'] = y_pred
sample.to_csv("submit-{}-{}.csv".format(str(today), auc), index=False)
