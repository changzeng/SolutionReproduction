# encoding: utf-8

import pandas as pd
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def train(feature_file_name, prefix=""):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 12,
    }

    print("loading data...")
    total_data = pd.read_csv(feature_file_name)
    # total_data = pd.read_csv("data/test_train_user_feature.csv")
    # total_data.drop(columns=["USRID"])
    print("split feature and label...")
    feature, label = total_data.ix[:, 0:-1], total_data.ix[:, -1]
    print("split train and test dataset...")
    fea_train, fea_test, label_train, label_test = train_test_split(feature, label, test_size=0.2,
                                                                    random_state=0)
    print("construct train and test dataset...")
    lgb_train = lgb.Dataset(fea_train, label_train)
    lgb_eval = lgb.Dataset(fea_test, label_test, reference=lgb_train)
    # lgb_eval = lgb.Dataset(fea_train, label_train, reference=lgb_train)
    print("training...")
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval,
                    early_stopping_rounds=5)
    joblib.dump(gbm, "gbm.pkl")
    predict_res = gbm.predict(fea_test, num_iteration=gbm.best_iteration)
    test_data = pd.merge(fea_test["USRID"], total_data[["USRID", "FLAG"]], on="USRID", how="left")
    test_data["predict"] = predict_res
    test_data.to_csv("data/{0}_{1}".format(prefix, "test_data.csv"))


if __name__ == "__main__":
    # train("data/train_user_feature.csv", prefix="raw_fea")
    train("data/user_feature.csv", prefix="word2vec")
    train("data/no_sep_user_feature.csv", prefix="no_sep_word2vec")
