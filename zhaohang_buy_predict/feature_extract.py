# encoding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as sp


def cross_tab_and_merge(table_1: object, table_2: object, cross_list: list, on_key: str,
                        how_method: str, prefix: str) -> pd.DataFrame:
    """
    交叉表格并合并
    :param table_1:
    :param table_2:
    :param cross_list:
    :param on_key:
    :param how_method:
    :param prefix:
    :return:
    """
    tmp_table = table_2[cross_list].copy()
    tmp_table[cross_list[1]] = tmp_table[cross_list[1]].apply(lambda x: prefix+"_"+str(x))
    tmp_table = pd.crosstab(tmp_table[cross_list[0]], tmp_table[cross_list[1]]).reset_index()
    return pd.merge(table_1, tmp_table, on=on_key, how=how_method)


def basic_stat_feature(user_info, column_data, prefix):
    """
    聚合计算基础统计特征
    平均/最大/最小/中位数/众数/方差/偏度/峰度
    :return:
    """
    name_func_list = [
        ["avg", np.average],
        ["max", np.max],
        ["min", np.min],
        ["median", np.median],
        ["mode", lambda x: sp.stats.mode(x)[0][0]],
        ["std", np.std],
        ["skew", sp.stats.skew],
        ["kurtosis", sp.stats.kurtosis]
    ]
    for name, func in name_func_list:
        user_info = pd.merge(user_info, column_data.agg(func).reset_index(
            name="{0}_{1}".format(prefix, name)), on="USRID", how="left")
    return user_info


def click_time_sep(user_info, user_log):
    """
    用户点击间隔统计数据
    平均/最大/最小/中位数/众数/方差/偏度/峰度
    :param user_info:
    :param user_log:
    :return:
    """
    user_act_time_sep = user_log.sort_values("timestamp", ascending=True)
    user_act_time_sep["time_sep"] = user_act_time_sep.groupby("USRID")["timestamp"].diff(1)
    user_act_time_sep = user_act_time_sep.groupby("USRID")["time_sep"]
    return basic_stat_feature(user_info, user_act_time_sep, "act_time_sep")


def log_act_count(user_info, user_log, prefix):
    """
    对行为日志中的点击次数进行统计
    :param user_info:
    :param user_log:
    :param prefix:
    :return:
    """
    group_log_by_uid = user_log.groupby(["USRID"])
    # 用户点击次数
    user_info = pd.merge(user_info, group_log_by_uid["EVT_LBL"].agg(len).
                         reset_index(name="{0}_act_count".format(prefix)), on="USRID",
                         how="left")
    # 用户每个模块的点击次数
    user_info = cross_tab_and_merge(user_info, user_log, ["USRID", "EVT_LBL"], "USRID", "left",
                                    "{0}_module_act_count".format(prefix))
    # 用户一天中每一个小时的总点击次数
    user_info = cross_tab_and_merge(user_info, user_log, ["USRID", "hour"], "USRID", "left",
                                    "{0}_hour_act_count".format(prefix))
    # 用户对各个子模块的点击次数
    for ind in range(1, 4):
        user_info = cross_tab_and_merge(user_info, user_log, ["USRID", "module_{0}_act"
                                        .format(ind)], "USRID", "left",
                                        "{0}_module_{1}_act_count".format(prefix, ind))
    # 用户每种行为类型触发的次数
    user_info = cross_tab_and_merge(user_info, user_log, ["USRID", "TCH_TYP"], "USRID", "left",
                                    "{0}_TCH_TYP_act_count".format(prefix))
    return user_info


def day_act_count_fea(user_info, user_log):
    """
    用户天级别点击次数的统计特征
    :param user_info:
    :param user_log:
    :return:
    """
    day_act_count_df = user_log.groupby(["USRID", "day"])["EVT_LBL"].agg(len)\
        .reset_index(name="day_act_count").groupby("USRID")["day_act_count"]
    user_info = basic_stat_feature(user_info, day_act_count_df, "day_act_count")

    # 计算最大连续活跃天数
    def max_continue_active_day_num(item):
        item = list(item)
        max_day_num, cur_day_num = 1, 1
        for ind in range(1, len(item)):
            if item[ind] - item[ind-1] == 1:
                cur_day_num += 1
                max_day_num = max(max_day_num, cur_day_num)
            else:
                cur_day_num = 1
        return max_day_num
    user_info = pd.merge(user_info, day_act_count_df.agg(max_continue_active_day_num)
                         .reset_index(name="max_continue_act_day_num"), on="USRID", how="left")
    return user_info


def user_act_ratio(user_info):
    """
    用户不同时间段行为次数占总行为次数的比例
    :param user_info:
    :return:
    """
    for hour in range(24):
        key_name = "total_hour_act_count_{0}".format(hour)
        if key_name not in user_info:
            continue
        user_info["total_hour_act_ratio_{0}".format(hour)] = \
            user_info[key_name] / user_info["total_act_count"]
    for week_day in range(7):
        key_name = "week_day_act_count_{0}".format(week_day)
        if key_name not in user_info:
            continue
        user_info["total_week_day_act_ratio_{0}".format(week_day)] = \
            user_info[key_name] / user_info["total_act_count"]

    return user_info


def add_process_feature(user_info, user_log):
    """
    预处理用户行为数据，添加统计特征
    :param user_info: 用户基础信息数据
    :param user_log: 用户行为日志
    :return:
    """
    user_log.OCC_TIM = pd.to_datetime(user_log.OCC_TIM)
    user_log["timestamp"] = list(map(lambda x: x.value/10**9, user_log.OCC_TIM))
    user_log["hour"] = list(map(lambda x: x.hour, user_log.OCC_TIM))
    user_log["day"] = list(map(lambda x: x.day, user_log.OCC_TIM))
    user_log["week_day"] = list(map(lambda x: x.weekday(), user_log.OCC_TIM))
    # 将用户的行为拆分为三个子模块
    for ind in range(1, 4):
        user_log["module_{0}_act".format(ind)] = list(map(lambda x: x.split("-")[ind-1],
                                                          user_log.EVT_LBL))
    user_info = log_act_count(user_info, user_log, "total")
    user_info = log_act_count(user_info, user_log[user_log.day == 31], "last_day")
    user_info = log_act_count(user_info, user_log[user_log.day >= 30], "last_two_day")
    user_info = log_act_count(user_info, user_log[user_log.day >= 20], "last_ten_day")
    user_info = log_act_count(user_info, user_log[(user_log.day >= 10) & (user_log.day < 20)],
                              "middle_ten_day")
    user_info = log_act_count(user_info, user_log[(user_log.day >= 0) & (user_log.day < 10)],
                              "farthest_ten_day")
    user_info = click_time_sep(user_info, user_log)

    # 用户一星期中每一天的点击总次数
    user_info = cross_tab_and_merge(user_info, user_log, ["USRID", "week_day"], "USRID", "left",
                                    "week_day_act_count")
    # 用户活跃天数
    user_info = pd.merge(user_info, user_log.drop_duplicates(["USRID", "day"])
                         .groupby("USRID")["day"].agg(len).reset_index(name="active_day_num"),
                         on="USRID", how="left")
    # 计算用户每天点击次数的基础统计特征
    user_info = day_act_count_fea(user_info, user_log)

    # 距离最后活跃的天数
    user_info = pd.merge(user_info, user_log.drop_duplicates(["USRID", "day"])
                         .groupby("USRID")["day"].agg(lambda x: 31-max(x))
                         .reset_index(name="how_many_day_not_act"), on="USRID", how="left")

    # 用户在不同时间段行为次数占总行为次数的比例
    user_info = user_act_ratio(user_info)

    return user_info


def feature_preprocess(agg_name, log_name, flg_name=None, dst_name="data/user_feature.csv"):
    """
    特征预处理
    :return:
    """
    user_info = pd.read_csv(agg_name, sep="\t")
    user_log = pd.read_csv(log_name, sep="\t")
    user_info = add_process_feature(user_info, user_log)
    if flg_name is not None:
        label_df = pd.read_csv(flg_name, sep="\t")
        user_info = pd.merge(user_info, label_df, on="USRID", how="left")
    user_info.to_csv(dst_name)


if __name__ == "__main__":
    # feature_preprocess("data/test_train_agg.csv", "data/test_train_log.csv",
    #                    "data/test_train_flg.csv", dst_name="data/train_user_feature")
    feature_preprocess("data/train_agg.csv", "data/train_log.csv", "data/train_flg.csv",
                       dst_name="data/train_user_feature.csv")
    feature_preprocess("data/test_agg.csv", "data/test_log.csv",
                       dst_name="data/test_user_feature.csv")
    # print(most_occur([1, 2, 3, 4, 5, 4, 2, 3, 4, 4]))
