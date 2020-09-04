'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-18 09:51:02
LastEditors: ulysses
@Description: 
'''
import os
import time

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.types as typ
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder


def create_spark_session():
    conf = SparkConf().setAppName("ML MultiClassClassification").set(
        "spark.ui.showConsoleProgress", "false"
    )
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


def prepare_data(spark):
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    # raw_data = sc.textFile(os.path.join(PATH, 'data/covtype.data'))
    # lines = raw_data.map(lambda x: x.split(','))
    # field_num = len(lines.first())
    # fields = [typ.StructField(f"f{num}", typ.StringType(), True)
    #          for num in range(len(covtype_df.columns))]
    covtype_df = spark.read.csv(os.path.join(PATH, 'data/covtype.data'), sep=',')
    columns = covtype_df.columns
    field_num = len(covtype_df.columns)
    fields = [f"f{num}" for num in range(field_num)]

    covtype_df = covtype_df.select([
        col(columns[i]).cast('double').alias(fields[i]) 
        for i in range(field_num)])
    # print(covtype_df.columns)
    covtype_df = covtype_df.withColumn(
        "label", covtype_df[fields[-1]]-1).drop(fields[-1])
    train_df, test_df = covtype_df.randomSplit([0.7, 0.3])

    print("将数据分trainData: {}, testData: {}".format(
        train_df.count(), test_df.count()
    ))
    return train_df, test_df
    
def train(train_data):
    feature_columns = train_data.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_columns, 
                                outputCol='features')
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
    grid_search = ParamGridBuilder()\
                .addGrid(dt.impurity, ['gini', 'entropy'])\
                .addGrid(dt.maxBins, [20, 30, 40, 50])\
                .addGrid(dt.maxDepth, [10, 15, 25])\
                .build()
    evaluator = MulticlassClassificationEvaluator(
        predictionCol='prediction',  # 多元分类 使用y_pred
         labelCol='label',
          metricName='accuracy')  # f1
    tvs = TrainValidationSplit(
        estimator=dt, 
        estimatorParamMaps=grid_search,
        evaluator=evaluator,
        trainRatio=0.8)
    tvs_pipeline = Pipeline(stages=[assembler, tvs])
    tvs_pipeline_model = tvs_pipeline.fit(train_data)
    
    best_model = tvs_pipeline_model.stages[-1]
    best_param = get_best_param(best_model)

    return best_param, tvs_pipeline_model

    
def get_best_param(model):
    result = [
        (
            [
                {key.name, param_value} for key, param_value in zip(
                    param.keys(), param.values())
            ], metric
        ) for param, metric in zip(
            model.getEstimatorParamMaps(),
            model.validationMetrics)  # validationMetrics  avgMetrics
    ]
    # ([{'maxIter': 50}, {'regParam': 0.01}], 0.7385557487596289)
    best_param = sorted(result, key=lambda e: e[1], reverse=True)[0]
    return best_param[0]


def evaluate_model(pipeline_model, data):
    evaluator = MulticlassClassificationEvaluator(
        predictionCol='prediction', labelCol='label', metricName='accuracy')
    
    result = pipeline_model.transform(data)

    acc = evaluator.evaluate(result, {evaluator.metricName:'accuracy'})
    f1 = evaluator.evaluate(result, {evaluator.metricName:'f1'})
    return acc, f1


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
    best_param, pipeline_model = train(train_df)

    print("最佳模型使用的参数{}".format(best_param))

    print("==========测试阶段===============")
    acc, f1 = evaluate_model(pipeline_model, test_df)
    print("测试集 accuracy = {}, f1 = {}".format(acc, f1))
    train_df.unpersist()
    test_df.unpersist()
    save_model(pipeline_model, 'ML_Multiclass')
    