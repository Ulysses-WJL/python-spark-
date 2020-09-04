'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-18 09:42:43
LastEditors: ulysses
@Description: 
'''
import time
import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as typ
from pyspark.sql.functions import udf
from pyspark.sql.functions import col  
import pyspark.ml.feature as ft
import pyspark.ml.classification as cl
import pyspark.ml.evaluation as ev
import pyspark.ml.tuning as tune
from pyspark.ml import Pipeline



def create_spark_session():
    conf = SparkConf().setAppName("ML LR").set(
        "spark.ui.showConsoleProgress", 'false')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    print("Master: {}".format(sc.master))
    set_logger(sc)
    set_path(sc)
    return spark


def set_logger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    sc.setLogLevel("FATAL")


def set_path(sc):
    global PATH
    if sc.master[:5] == 'local':
        PATH = "/mnt/data1/workspace/data_analysis_mining/Python+Spark2.0+Hadoop机器学习与大数据实战/pythonsparkexample/PythonProject"
    else:
        PATH = "hdfs://master:9000/user/hduser"

def replace_question(x):
    return "0" if x == "?" else x

replace_question = udf(replace_question)


def prepare_data(spark):
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    row_df = spark.read.csv(os.path.join(PATH, "data/train.tsv"), header=True, sep='\t')
    # 先替换? 在转为double 类型,
    df = row_df.select(
        ['url', 'alchemy_category']+
        [replace_question(col(column)).cast(typ.DoubleType()).alias(column) 
            for column in row_df.columns[4:]]
        )
    train_df, test_df = df.randomSplit([0.7, 0.3])
    print("将数据分trainData: {}, testData: {}".format(
        train_df.count(), test_df.count()
    ))
    return train_df, test_df

def train_evaluate(train_data, test_data):
    # 将文字的分类特征转为数字
    stringIndexer = ft.StringIndexer(inputCol='alchemy_category', 
                              outputCol="alchemy_category_Index")

    encoder = ft.OneHotEncoder(dropLast=False,
                            inputCol='alchemy_category_Index',
                            outputCol="alchemy_category_IndexVec")

    assemblerInputs =['alchemy_category_IndexVec']  + train_data.columns[4:-1] 
    assembler = ft.VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    # dt = cl.DecisionTreeClassifier(labelCol="label", 
    #                             featuresCol="features")
    rf = cl.RandomForestClassifier(labelCol="label", 
                               featuresCol="features")
    
    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol="probability",
        labelCol='label',
        metricName='areaUnderROC'
    )
    
    grid_search = tune.ParamGridBuilder()\
        .addGrid(rf.impurity, [ "gini","entropy"])\
        .addGrid(rf.maxDepth, [ 5,10,15])\
        .addGrid(rf.maxBins, [10, 15,20])\
        .addGrid(rf.numTrees, [10, 20,30])\
        .build()

    rf_cv = tune.CrossValidator(
        estimator=rf, 
        estimatorParamMaps=grid_search,
        evaluator=evaluator,
        numFolds=5
    )

    # rf_tvs = tune.TrainValidationSplit(
    #     estimator=rf, 
    #     estimatorParamMaps=grid_search,
    #     evaluator=evaluator,
    #     trainRatio=0.7
    # )
    pipeline = Pipeline(stages=[
        stringIndexer, encoder, assembler, rf_cv])
    cv_pipeline_model = pipeline.fit(train_data)
    
    best_model = cv_pipeline_model.stages[-1]
    best_parm = get_best_param(best_model)

    AUC, AP = evaluate_model(cv_pipeline_model, test_data)

    return AUC, AP, best_parm, cv_pipeline_model


def evaluate_model(pipeline_model, data):
    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol="probability",
        labelCol="label")
    results = pipeline_model.transform(data)
    AUC = evaluator.evaluate(results, 
                            {evaluator.metricName: 'areaUnderROC'})
    AP = evaluator.evaluate(results, 
                            {evaluator.metricName: 'areaUnderPR'})
    return AUC, AP


def get_best_param(model):
    result = [
        (
            [
                {key.name, param_value} for key, param_value in zip(
                    param.keys(), param.values())
            ], metric
        ) for param, metric in zip(
            model.getEstimatorParamMaps(),
            model.avgMetrics)
    ]
    # ([{'maxIter': 50}, {'regParam': 0.01}], 0.7385557487596289)
    best_param = sorted(result, key=lambda e: e[1], reverse=True)[0]
    return best_param[0]


def save_model(model, model_path):
    model.write().overwrite().save(
        os.path.join(PATH, model_path)
    )


if __name__ == "__main__":
    print("Start ML Logisitc Regression")
    spark = create_spark_session()
    
    print("==========数据准备阶段===============")
    train_df, test_df = prepare_data(spark)
    train_df.persist()
    test_df.persist()

    print("==========训练评估阶段===============")
    AUC, AP, best_param, pipeline_model = train_evaluate(
            train_df, test_df)

    print("最佳模型使用的参数{}, 测试集: AUC={}, AP={}".format(
        best_param, AUC, AP))
    
    save_model(pipeline_model, 'ML_classifier')
    train_df.unpersist()
    test_df.unpersist()
