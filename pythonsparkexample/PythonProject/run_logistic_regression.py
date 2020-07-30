'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-29 16:39:56
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
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler



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
    return float(fields[-1])

def get_features(fields, categories_map, feature_end):
    # 新闻类别转one_hot
    cate_idx = categories_map[fields[3]]
    cate_features = np.zeros(len(categories_map))
    cate_features[cate_idx] = 1
    
    #提取数值字段 后面的字段都是数值类型
    numericalFeatures=[0 if field=='?' else float(field) for field in fields[4: feature_end]] 
    #返回“分类特征字段”+“数值特征字段”
    return np.concatenate((cate_features, numericalFeatures))


def prepare_data(sc):
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    raw_data_with_header = sc.textFile(os.path.join(PATH, 'data/train.tsv'))
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x!=header)

    # 去除 "" 按 \t 划分一个网页的不同字段
    lines_rdd = raw_data.\
        map(lambda x: x.replace("\"", "")).\
        map(lambda x: x.split('\t'))
    
    print("共计: {}项".format(lines_rdd.count()))
    #---------------------2.数据标准化----------------------- 
    # {新闻类别: 序号, }
    categories_map = lines_rdd.map(lambda fields: fields[3]).\
                        distinct().zipWithIndex().collectAsMap()
    label_rdd = lines_rdd.map(lambda r: get_label(r))
    features_rdd = lines_rdd.map(lambda r: get_features(r, categories_map, len(r)-1))


    scaler = StandardScaler(withMean=True, withStd=True).fit(features_rdd)
    stand_features = scaler.transform(features_rdd)
    #----------3.建立训练评估所需数据 RDD[LabeledPoint]-------   LabeledPoint                    
    labeledpoint_rdd = label_rdd.zip(stand_features).map(lambda r: LabeledPoint(r[0], r[1]))
    #-----------4.以随机方式将数据分为3个部分并且返回-------------
    (trainData, validationData, testData) = labeledpoint_rdd.randomSplit([0.8, 0.1, 0.1])
    print("将数据分trainData: {0}, validationData: {1}, testData: {2}".format(
        trainData.count(), validationData.count(), testData.count()
    ))

    return (trainData, validationData, testData, categories_map) #返回数据

    

def predict_data(sc, model, categories_map):
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(os.path.join(PATH, "data/test.tsv"))
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    rData=rawData.map(lambda x: x.replace("\"", ""))    
    lines = rData.map(lambda x: x.split("\t"))
    print("共计： {} 项".format(lines.count()))

    dataRDD = lines.map(lambda r: (r[0],
                         get_features(r, categories_map, len(r))))
    DescDict = {
           0: "暂时性网页(ephemeral)",
           1: "长青网页(evergreen)"}
    # 评估 
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print(f"网址:{data[0]}\n ===>预测:{predictResult} 说明是: {DescDict[predictResult]}")

def evaluate_model(model, valid_data):
    # 返回的是int 型 的 0, 1
    score = model.predict(valid_data.map(lambda p: p.features))
    # 
    score_and_labels = score.map(lambda x: float(x)).zip(valid_data.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(score_and_labels)
    AUC = metrics.areaUnderROC
    return AUC

def train_evaluate_model(train_data, valid_data, iterations, regParam):
    start_time = time()
    # 训练
    model = LogisticRegressionWithLBFGS.train(
        train_data, numClasses=2, iterations=iterations, regParam=regParam)
    # 评估
    # y_pred y_true
    AUC = evaluate_model(model, valid_data)
    duration = time() - start_time
    print(f"训练评估：使用参数 iterations={iterations}, regParam={regParam} ==>所需时间={duration} 结果AUC = {AUC}")
    return AUC, duration, iterations, regParam, model


def parametersEval(trainData, validationData):

    print("----- 评估iterations参数使用 ---------")
    evalParameter(trainData, validationData,"iterations", 
                              iterations_list=[50, 100, 150, 200, 250],   
                              regParam_list=[0.0])
 
    print("----- 评估regParam参数使用 ---------")
    evalParameter(trainData, validationData,"maxDepth", 
                              iterations_list=[100],                    
                              regParam_list=[0.001, 0.01, 0.1, 0., 1., 5., 10.])   


def evalParameter(trainData, validationData, evalparm,
                  iterations_list, regParam_list):
    #训练评估参数
    metrics = [train_evaluate_model(trainData, validationData,  
                                iterations,regParam) 
                       for iterations in iterations_list
                       for regParam in regParam_list]
    #设置当前评估的参数
    if evalparm=="iterations":
        IndexList=iterations_list[:]
    elif evalparm=="regParam":
        IndexList=regParam_list[:]

    #转换为Pandas DataFrame
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration','iterations', 'regParam', 'model'])
    #显示图形
    showchart(df,evalparm,'AUC','duration',0.5, 0.7)

def showchart(df, evalparm, bar_data, line_data, y_min, y_max):
    ax = df['AUC'].plot(kind='bar', title =evalparm,figsize=(10,6),
                    legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([0.55,0.7])
    ax.set_ylabel("AUC",fontsize=12)
    # ax2 = ax.twinx()
    # ax2.plot(df['duration'].values, linestyle='-', marker='o', linewidth=2.0,color='r')secondary_y=True
    df['duration'].plot(linestyle='-', marker='o', linewidth=2.0,color='r',  secondary_y=True)
    plt.show()


def eval_all_parameter(train_data, validation_data, iterations_list, regParam_list):
    metrics = [train_evaluate_model(trainData, validationData,  
                                iterations,regParam) 
                       for iterations in iterations_list
                       for regParam in regParam_list]
    # 根据AUC 选择最佳
    best = sorted(metrics, key=lambda k: k[0], reverse=True)[0]
    print("调校后最佳参数：iterations:{}, regParam: {}\n AUC={}".format(
        best[2], best[3], best[0]))
    
    return best[-1]


def save_model(model, sc, model_path):
    try:
        model.save(sc, os.path.join(PATH, model_path))
        print("模型已保存")
    except Exception :
        print('模型存在, 先删除')

if __name__ == "__main__":
    print("Start DecisonTreeBinary")
    sc = create_spark_context()
    print("==========数据准备阶段===============")
    trainData, validationData, testData, categories_map = prepare_data(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("==========训练评估阶段===============")
    # AUC, duration, impurity, max_depth, max_bins, model = \
    #     train_evaluate_model(trainData, validationData,100, 0.0)
    
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif (len(sys.argv) == 2) and (sys.argv[1]=="-a"):
        print("-----所有参数训练评估找出最好的参数组合---------")
        model = eval_all_parameter(trainData, validationData,
                                   [50, 100, 150, 200, 250],
                                   [0.001, 0.01, 0.1, 0., 1., 5., 10.])
    else:
        pass
    print("==========测试阶段===============")
    auc = evaluate_model(model, testData)
    print("使用test Data测试最佳模型,结果 AUC:{}".format(auc))
    print("==========预测数据===============")
    predict_data(sc, model, categories_map)
    trainData.unpersist()
    validationData.unpersist()
    testData.unpersist()
    save_model(model, sc, 'logistic')
    