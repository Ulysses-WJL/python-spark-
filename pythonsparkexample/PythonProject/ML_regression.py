'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-31 11:28:40
@LastEditors: Please set LastEditors
@Description: 
'''
import os
import time

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.types as typ
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor,\
                                  GBTRegressor
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator


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
    hour_df = spark.read.csv(os.path.join(PATH, 'data/hour.csv'),
                             sep=',', header=True)
    # 丢弃不需要的字段
    hour_df = hour_df.drop("instant").drop('dteday').drop('yr')\
                     .drop('casual').drop('registered')
    # 转为double类型
    hour_df = hour_df.select([col(column).cast('double').alias(column)
                              for column in hour_df.columns])
    
    train_df, test_df = hour_df.randomSplit([0.7, 0.3])

    print("将数据分trainData: {}, testData: {}".format(
        train_df.count(), test_df.count()
    ))
    return train_df, test_df


def train(train_data):
    feature_columns = train_data.columns[:-1]

    assembler = VectorAssembler(inputCols=feature_columns, 
                                outputCol='afeatures')
    # 将星期 月份 小时 等会被视为分类字段
    indexer = VectorIndexer(inputCol='afeatures', 
                            outputCol='features',
                            maxCategories=24)

    
    dt = DecisionTreeRegressor(featuresCol='features', labelCol='cnt')
    rf = RandomForestRegressor(featuresCol='features', labelCol='cnt')
    gbt = GBTRegressor(featuresCol='features', labelCol='cnt')
    
    # grid_search = ParamGridBuilder()\
    #             .addGrid(dt.maxDepth, [ 5,10,15,25])\
    #             .addGrid(dt.maxBins, [25,35,45,50])\
    #             .build()

    grid_search = ParamGridBuilder()\
                .addGrid(rf.numTrees,[10, 20, 30])\
                .addGrid(rf.maxBins, [25, 35, 50])\
                .addGrid(rf.maxDepth, [5, 10, 15])\
                .build()

    # grid_search = ParamGridBuilder() \
    #             .addGrid(gbt.maxDepth, [ 5,10])\
    #             .addGrid(gbt.maxBins, [25,40])\
    #             .addGrid(gbt.maxIter, [10, 50])\
    #             .build()

            
    evaluator = RegressionEvaluator(predictionCol='prediction',  
                                    labelCol='cnt',
                                    metricName='rmse')
 
    tvs = TrainValidationSplit(
        estimator=rf, 
        estimatorParamMaps=grid_search,
        evaluator=evaluator,
        trainRatio=0.8)
    
    # cv = CrossValidator(
    #     estimator=rf,
    #     estimatorParamMaps=grid_search,
    #     evaluator=evaluator,
    #     numFolds=5)
    
    
    # pipeline = Pipeline(stages=[assembler, indexer, tvs])
    pipeline = Pipeline(stages=[assembler, indexer, tvs])
    pipeline_model = pipeline.fit(train_data)
    
    best_model = pipeline_model.stages[-1]
    best_param = get_best_param(best_model)

    return best_param, pipeline_model



def get_best_param(model):
    result = [
        (
            [
                {key.name: param_value} for key, param_value in zip(
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
    evaluator = RegressionEvaluator(
        predictionCol='prediction', labelCol='cnt', metricName='rmse')
    
    result = pipeline_model.transform(data)

    rmse = evaluator.evaluate(result, {evaluator.metricName:'rmse'})
    r2 = evaluator.evaluate(result, {evaluator.metricName:'r2'})
    return rmse, r2


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
    rmse, r2 = evaluate_model(pipeline_model, test_df)
    print("测试集 rmse = {}, r2 = {}".format(rmse, r2))
    train_df.unpersist()
    test_df.unpersist()
    save_model(pipeline_model, 'ML_regression')