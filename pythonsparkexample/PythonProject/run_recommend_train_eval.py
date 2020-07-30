'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-28 16:07:33
@LastEditors: Please set LastEditors
@Description: 推荐系统
'''
from time import time
import os
from operator import add

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def set_logger(sc):
    """
    设置logger
    Args:
        sc ([type]): [description]
    """
    logger = sc._jvm.org.apache.log4j
    sc.setLogLevel("FATAL")
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def set_path(sc):
    """
    设置文件目录
    Args:
        sc ([type]): [description]
    """
    global PATH 
    if sc.master[:5] == 'local':
        PATH = 'file:/mnt/data1/workspace/data_analysis_mining/Python+Spark2.0+Hadoop机器学习与大数据实战/pythonsparkexample/PythonProject'
    else:
        PATH = "hdfs://master:9000/user/hduser"

def create_spark_context():
    """
    创建sc
    Returns:
        [type]: [description]
    """
    conf = SparkConf().setAppName('Decision_Tree')\
        .set("spark.ui.shaowConsoleProgress", 'false')
    sc = SparkContext(conf=conf)
    print('master: {}'.format(sc.master))

    set_logger(sc)
    set_path(sc)
    return sc


def prepare_data(sc):
    """
    准备数据
    Args:
        sc ([type]): [description]

    Returns:
        [type]: [description]
    """
    #----------------------1. 建立用户评价数据-------------
    raw_user_data = sc.textFile(os.path.join(PATH, 'data/u.data'))
    print("数据项: {}".format(raw_user_data.count()))
    # (用户id, 物品id, 评分)
    ratings_rdd = raw_user_data.\
        map(lambda line: line.split('\t')[:3])
    #----------------------2. 以随机方式将数据分为3 个分并且返回-------------     
    train_data, validation_data, test_data = ratings_rdd.randomSplit(
        [0.8, 0.1, 0.1], seed=0x123)
    print("train: {}; validation: {}; test: {}".format(
        train_data.count(), validation_data.count(), test_data.count()
    ))
    
    return train_data, validation_data, test_data


def train_model(train_data, validation_data, rank, iterations, lambda_):
    start_time = time()
    model = ALS.train(train_data, rank=rank, 
                      iterations=iterations, lambda_=lambda_)
    loss = RMSE_loss(model, validation_data)
    duration = time() - start_time
    print("训练评估：使用参数rank = {}, lambda_={}, iterations={},所需时间={}, RMSE={}".format(
                 rank, lambda_, iterations, duration, loss))
    return loss, duration, rank, iterations, lambda_, model
    

def RMSE_loss(model, ratings_rdd):
    n = ratings_rdd.count()
    # Returns a list of predicted ratings for input user and product pairs.
    # user product rating
    predict_rdd = model.predictAll(ratings_rdd.map(lambda x: (x[0], x[1])))
    # 以(user, product) 为key, 评分为value 合并
    predict_true = predict_rdd.map(lambda x: ((x[0], x[1]), x[2])).join(
        ratings_rdd.map(lambda x: ((int(x[0]), int(x[1])), float(x[2])))
    ).values()
    rmse = (predict_true.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / n) ** 0.5
    return rmse


def parameters_tunning(train_data, validation_data):
    rank_list = [5, 10, 15, 20, 50, 100]
    lambda_list = [0.05, 0.1, 1.0, 0.5, 1.0, 5.0, 10.0]
    iterations_list  = [5,10,15,20]
    

    print("==============寻找最佳参数=============")
    best_model = eval_all_parameter(train_data, validation_data, 
                   rank_list, iterations_list, lambda_list)
    return best_model


def eval_parameter(train_data, validation_data, eval_parm, 
                   rank_list, iterations_list, lambda_list):
    metrics = [train_model(train_data, validation_data, rank, iterations, lambda_)
                for rank in rank_list
                for iterations in iterations_list
                for lambda_ in lambda_list]
    if eval_parm == 'rank':
        index = rank_list
    elif eval_parm == 'numIterations':
        index = iterations_list
    elif eval_parm == "lambda":
        index = lambda_list
    
    df = pd.DataFrame(metrics, index=index, 
        columns=['RMSE', 'duration', 'rank', 'iterations', 'lambda_', 'model'])
    showchart(df, eval_parm, 'RMSE', 'duration', 0.8, 5)


def eval_all_parameter(train_datam, validation_data, rank_list, 
                        iterations_list, lambda_list):
    metrics = [train_model(train_data, validation_data, rank, iterations, lambda_)
                for rank in rank_list
                for iterations in iterations_list
                for lambda_ in lambda_list]

    best = sorted(metrics, key=lambda m: m[0])[0]
    # loss, duration, rank, iterations, lambda_, model
    print("最佳参数: rank: {}, iterations: {}, lambda_ : {}, RMSE={}".format(
        best[2], best[3], best[4], best[0]))

    return best[-1]


def showchart(df, param, bar_data, line_data, y_min, y_max):
    ax = df[bar_data].plot(kind='bar', title=param, figsize=(10, 6))
    ax.set_xlabel(param, fontsize=12)
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(bar_data, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[line_data]].values, linestype='-', marker='o',
             linewidth=2.0, color='r')
    plt.show()

def save_model(model, sc, model_path):
    try:
        model.save(sc, os.path.join(PATH, model_path))
        print("模型已保存")
    except Exception :
        print('模型存在, 先删除')


if __name__ == "__main__":
    sc = create_spark_context()
    print("==========数据准备阶段===============")
    train_data, validation_data, test_data = prepare_data(sc)
    train_data.persist()
    validation_data.persist()
    test_data.persist()
    print("==========训练评估阶段===============")
    best_model = parameters_tunning(train_data, validation_data)
    print("==========测试阶段===============")
    test_rmse = RMSE_loss(best_model, test_data)
    print("========== 存储Model========== ==")
    save_model(best_model, sc, 'recommend_model')
    print("Saved")
    train_data.unpersist()
    validation_data.unpersist()
    test_data.unpersist()
    print("Unpresist")