### 招商银行信用卡用户优惠券购买预测

数据预处理：

1. 将行为发生的时间转化为天、小时、分、秒、周几

统计特征：

1. 用户的总点击次数，在每个模块的操作次数
2. 每个小时/周几的点击量，用户有多少天点击过
3. 用户的平均/最大/最小点/中位数/偏度/峰度点击间隔
4. 用户不同方式访问的次数。
5. 用户每天点击次数的方差/均值/标准差/中位数/最大/最小/偏度/峰度
6. 最大连续点击天数
7. 已经不活跃多少天
8. 最后(一/两)天用户的点击总次数/各个模块的点击次数
9. 最后(一/两)天在各个小时的点击量
10. 最后(一/两)天各个类型时间发生的数量
11. 各个事件(总量/最后一天/两天)类型次数占总事件次数的比例
12. 用户一周每天的点击量占总点击量的比例
13. 用户每个小时的点击量占总点击量的比例
14. 用户前十/中间十/后十天时间发生次数和各类型事件发生次数

上述统计特征使用lightGBM未调参的auc值为0.84.

亮点特征

1. 将用户行为看成文档的词，这样，用户的所有行为就构成了一个文档。用所有用户文档训练
word2vec，然后得到用户的向量，使用用户向量预测用户是否购买优惠券，测试集auc能达到
0.74。
2. 另外还尝试在用户行为之间加入时间间隔，比如用户发生了行为A和B，行为A和B的时间间隔为
5分钟，那么新的行为序列就为\[A, 5_min, B\]。经实验验证，加入时间间隔对auc的提升并
不明显。
