'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-29 09:39:02
@LastEditors: Please set LastEditors
@Description: 
'''
import os
import sys
from time import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics



def create_spark_context():
    conf = SparkConf().setAppName('Decision_Tree')\
        .set("spark.ui.shaowConsoleProgress", 'false')
    sc = SparkContext(conf=conf)
    print('master: {}'.format(sc.master))

    set_logger(sc)
    set_path(sc)
    return sc

def set_logger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def set_path(sc):
    global PATH 
    if sc.master[:5] == 'local':
        PATH = 'file:/mnt/data1/workspace/data_analysis_mining/Python+Spark2.0+Hadoop机器学习与大数据实战/pythonsparkexample/PythonProject'
    else:
        PATH = "hdfs://master:9000/user/hduser"


def get_label(fields):
    # 1~7 -> 0~6
    return float(fields[-1]) - 1

def convert(x):
    return 0 if x == "?" else float(x)

def get_features(fields, feature_end):

    numericalFeatures=[convert(field) for field in fields[0: feature_end]] 
    #返回“分类特征字段”+“数值特征字段”
    return numericalFeatures



def prepare_data(sc):
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    raw_data = sc.textFile(os.path.join(PATH, 'data/covtype.data'))
    
    print("共计: {}项".format(raw_data.count()))
    lines_rdd = raw_data.map(lambda line: line.split(","))
    #----------2.建立训练评估所需数据 RDD[LabeledPoint]-------  
    
    labeledpoint_rdd = lines_rdd.map(
                        lambda r: LabeledPoint(
                            get_label(r),
                            get_features(r, len(r)-1)
                        ))
    #-----------3.以随机方式将数据分为3个部分并且返回-------------
    (trainData, validationData, testData) = labeledpoint_rdd.randomSplit([0.8, 0.1, 0.1])
    print("将数据分trainData: {0}, validationData: {1}, testData: {2}".format(
        trainData.count(), validationData.count(), testData.count()
    ))
    print(labeledpoint_rdd.first())
    return trainData, validationData, testData  # 返回数据

    

def predict_data(sc, model):
    print("开始导入数据...")
    raw_data = sc.textFile(os.path.join(PATH, 'data/covtype.data'))
    
    print("共计: {}项".format(raw_data.count()))
    lines_rdd = raw_data.map(lambda line: line.split(","))

    labeledpoint_rdd = lines_rdd.map(
                        lambda r: LabeledPoint(
                            get_label(r),
                            get_features(r, len(r)-1)
                        ))
    # 评估 
    for lp in labeledpoint_rdd.take(100):
        y_predict = model.predict(lp.features)
        y_true = lp.label
        features = lp.features
        result = "正确" if (y_predict == y_true) else "错误"
        print(f"土地条件: 海拔: {features[0]}, 方位: {features[1]}, 斜率: {features[2]}"\
              f"水源垂直距离: {features[3]}, 水源水平距离: {features[4]},"\
              f"9点时阴影: {features[5]},\n ....==>预测: {y_predict}, 实际: {y_true},"\
              f"结果:{result}")

def evaluate_model(model, valid_data):
    score = model.predict(valid_data.map(lambda p: p.features))
    score_and_labels = score.zip(valid_data.map(lambda p: p.label))
    metrics = MulticlassMetrics(score_and_labels)
    accuracy = metrics.accuracy
    return accuracy

def train_evaluate_model(train_data, valid_data, impurity, max_depth, max_bins):
    start_time = time()
    # 训练
    model = DecisionTree.trainClassifier(
        train_data, numClasses=7, categoricalFeaturesInfo={}, 
        impurity=impurity, maxDepth=max_depth, maxBins=max_bins)
    # 评估
    # y_pred y_true
    acc = evaluate_model(model, valid_data)
    duration = time() - start_time
    print(f"训练评估：使用参数 impurity={impurity}, maxDepth={max_depth}, "\
          f"maxBins={max_bins},==>所需时间={duration} 结果accuracy = {acc}")
    return acc, duration, impurity, max_depth, max_bins, model


def parametersEval(trainData, validationData):

    print("----- 评估impurity参数使用 ---------")
    evalParameter(trainData, validationData,"impurity", 
                              impurityList=["gini", "entropy"],   
                              maxDepthList=[10],  
                              maxBinsList=[10 ])  
 
    print("----- 评估maxDepth参数使用 ---------")
    evalParameter(trainData, validationData,"maxDepth", 
                              impurityList=["gini"],                    
                              maxDepthList=[3, 5, 10, 15, 20, 25],    
                              maxBinsList=[10])   
    print("----- 评估maxBins参数使用 ---------")
    evalParameter(trainData, validationData,"maxBins", 
                              impurityList=["gini"],      
                              maxDepthList =[10],        
                              maxBinsList=[3, 5, 10, 50, 100, 200 ])

def evalParameter(trainData, validationData, evalparm,
                  impurityList, maxDepthList, maxBinsList):
    #训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData,  
                                impurity,maxDepth,  maxBins) 
                       for impurity in impurityList
                       for maxDepth in maxDepthList  
                       for maxBins in maxBinsList ]
    #设置当前评估的参数
    if evalparm=="impurity":
        IndexList=impurityList[:]
    elif evalparm=="maxDepth":
        IndexList=maxDepthList[:]
    elif evalparm=="maxBins":
        IndexList=maxBinsList[:]
    #转换为Pandas DataFrame
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['accuarcy', 'duration','impurity', 'maxDepth', 'maxBins','model'])
    #显示图形
    showchart(df,evalparm,'accuarcy','duration',0.5,0.7 )

def showchart(df, evalparm, bar_data, line_data, y_min, y_max):
    ax = df['accuarcy'].plot(kind='bar', title =evalparm,figsize=(10,6),
                    legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([0.55,0.7])
    ax.set_ylabel("accuarcy",fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df['duration'].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    # df['duration'].plot(linestyle='-', marker='o', linewidth=2.0,color='r',  secondary_y=True)
    plt.show()


def eval_all_parameter(train_data, validation_data, impurity_list, max_depth_list, max_bins_list):
    metrics = [train_evaluate_model(
        train_data, validation_data, impurity, max_depth, max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list]
    # 根据accuarcy 选择最佳
    best = sorted(metrics, key=lambda k: k[0], reverse=True)[0]
    print("调校后最佳参数：impurity:{}, maxDepth: {}, maxBins: {}\n Accuarcy={}".format(
        best[2], best[3], best[4], best[0]))
    
    return best[-1]


def save_model(model, sc, model_path):
    try:
        model.save(sc, os.path.join(PATH, model_path))
        print("模型保存")
    except Exception:
        print("模型已存在, 请先删除")


if __name__ == "__main__":
    print("Start DecisonTreeBinary")
    sc = create_spark_context()
    print("==========数据准备阶段===============")
    trainData, validationData, testData = prepare_data(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("==========训练评估阶段===============")
    acc, duration, impurity, max_depth, max_bins, model = \
        train_evaluate_model(trainData, validationData, 'entropy', 15, 50)
    
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif (len(sys.argv) == 2) and (sys.argv[1]=="-a"):
        print("-----所有参数训练评估找出最好的参数组合---------")
        model = eval_all_parameter(trainData, validationData,
                                   ['gini', 'entropy'],
                                   [3, 5, 10, 15],
                                   [3, 5, 10, 50])
    else:
        pass
    print("==========测试阶段===============")
    acc = evaluate_model(model, testData)
    print("使用test Data测试最佳模型,结果 Accuracy:{}".format(acc))
    print("==========预测数据===============")
    predict_data(sc, model)
    trainData.unpersist()
    validationData.unpersist()
    testData.unpersist()
    save_model(model, sc, 'Multi_decision_tree')
    # print(model.toDebugString())
    