# encoding: utf-8

import os
import time
import numpy as np
import pandas as pd
from pyspark import SparkContext


def timestr_to_timestamp(timestr):
    return int(time.mktime(time.strptime(timestr, "%Y-%m-%d %H:%M:%S")))


def log_to_doc_agg_func(data_frame):
    """
    用户行为聚合函数
    :param data_frame:
    :return:
    """
    res_list = []
    sep = 60
    data = map(lambda x: x.strip().split("_"), list(data_frame))
    data = list(map(lambda x: (int(x[0]), x[1]), data))
    for ind, item in enumerate(data):
        if ind == 0:
            action_item = "%s_start" % item[1]
        else:
            time_interval = int((item[0] - data[ind-1][0])/sep)
            action_item = "%s_%d" % (item[1], time_interval)
        res_list.append(action_item)
    res = "\t".join(res_list)
    return res


def transform_user_log_into_doc_pd():
    """
    使用pandas DataFrame 将用户行为转化为文档
    :return:
    """
    data = pd.read_csv("data/train_log.csv", sep="\t")
    # data = pd.read_csv("data/test_train_log.csv", sep="\t")
    data["timestamp"] = pd.to_datetime(data.OCC_TIM).map(lambda x: x.value/10**9)
    data["timestamp_action"] = None
    for ind, row in data.iterrows():
        merge_str = "%d_%s" % (int(row["timestamp"]), row["EVT_LBL"])
        data.loc[ind, "timestamp_action"] = merge_str
    data = data.sort_values("timestamp_action").groupby("USRID")["timestamp_action"]\
        .agg(log_to_doc_agg_func).reset_index(name="doc")
    data.to_csv("data/uid_to_doc.csv")
    with open("data/user_action_doc.txt", "w") as fd:
        write_list = []
        for ind, row in data.iterrows():
            write_line = "{0}\t{1}".format(row["USRID"], row["doc"])
            write_list.append(write_line)
        fd.write("\n".join(write_list))


def transform_func(data):
    """
    transform func
    :param data_list:
    :return:
    """
    sep = 60
    uid, data_list = data
    data_list = map(lambda x: list(x), data_list)
    data_list = sorted(data_list, key=lambda x: int(x[1]))
    res_list = []
    for ind, item in enumerate(data_list):
        if ind == 0:
            action_item = "start"
        else:
            time_interval = int((item[1] - data_list[ind-1][1])/sep)
            action_item = "min_%d" % time_interval
        res_list.append(action_item)
        res_list.append(item[0])
    res_list.append("end")
    ret = "{0}\t{1}".format(uid, "\t".join(res_list))
    return ret


def transform_user_log_into_doc_spark(file_name):
    """
    使用spark DataFrame将用户行为转化为文档
    :param file_name:
    :return:
    """
    if os.path.exists(file_name):
        return
    sc = SparkContext()
    # sql_context = SQLContext(sc)
    # data = sql_context.read.format('com.databricks.spark.csv')\
    #     .options(header='true', inferschema='true', delimiter="\t")\
    #     .load('data/train_log.csv')
    data = sc.textFile("data/train_log.csv")\
        .map(lambda x: x.split("\t"))\
        .filter(lambda x: x[0] != "USRID")\
        .map(lambda x: [x[0], x[1], timestr_to_timestamp(x[2]), x[3]])\
        .map(lambda x: (x[0], (x[1:]))) \
        .groupByKey() \
        .map(transform_func) \
        .collect()
    with open("data/user_action_doc.txt", "w") as fd:
        fd.write("\n".join(data))


def train_word2vec(file_name):
    """
    训练word2vec模型
    :param file_name:
    :return:
    """
    import gensim
    from gensim.models import word2vec
    model_name = "model/word2vec.model"
    uid_list, text_list = [], []
    with open(file_name) as fd:
        for line in fd:
            split_res = line.strip().split("\t")
            uid, word_list = split_res[0], split_res[1:]
            uid_list.append(uid)
            text_list.append(word_list)
    user_feature_name = "data/user_feature.csv"
    if not os.path.exists(user_feature_name):
        uid_to_vec = []
        if os.path.exists(model_name):
            model = gensim.models.Word2Vec.load(model_name)
        else:
            model = word2vec.Word2Vec(text_list, size=50)
            model.save(model_name)
        for uid, sentence in zip(uid_list, text_list):
            vector, count = None, 0
            for word in sentence:
                if word not in model:
                    continue
                count += 1
                if vector is None:
                    vector = np.array(model[word])
                else:
                    vector += np.array(model[word])
            vector /= count*1.0
            uid_to_vec.append([int(uid)]+list(vector))
        name_list = ["USRID"] + ["fea_{0}".format(ind) for ind in range(len(vector))]
        data = pd.DataFrame(uid_to_vec, columns=name_list)
        label_data = pd.read_csv("data/train_flg.csv", sep="\t")
        data = pd.merge(data, label_data, on="USRID", how="left")
        data.to_csv(user_feature_name)
    else:
        data = pd.read_csv(user_feature_name)
    return


if __name__ == "__main__":
    target_file_name = "data/user_action_doc.txt"
    transform_user_log_into_doc_spark(target_file_name)
    train_word2vec(target_file_name)
