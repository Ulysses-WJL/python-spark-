'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 16:50:46
LastEditors: Please set LastEditors
Description: 
'''
# 导入需要用到的模块
import os
import shutil
from pprint import pprint

from pyspark.sql import SparkSession
from pyspark.sql.functions import window, asc
import pyspark.sql.types as typ


# 定义JSON文件的路径常量
TEST_DATA_DIR_SPARK = 'file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/testdata/'

if __name__ == "__main__":
    schema = typ.StructType(fields=[
        typ.StructField("eventTime", typ.TimestampType(), True),
        typ.StructField("action", typ.StringType(), True),
        typ.StructField("district", typ.StringType(), True),
    ])
    spark = SparkSession\
             .builder\
             .appName("StructuredEMallPurchaseCount")\
             .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # 每次最多读100个文件
    lines = spark\
            .readStream\
            .format('json')\
            .schema(schema)\
            .option("maxFilesPerTrigger", 100)\
            .load(TEST_DATA_DIR_SPARK)
    # 定义窗口
    windowDuration = '1 minutes'

    # 统计purchase 按地域, 时间窗口 分组, 每隔一分钟, 按时间升序
    windowed_count = lines\
        .filter("action='purchase'")\
        .groupBy('district', window('eventTime', windowDuration))\
        .count()\
        .sort(asc('window'))
    
    # 输出结果
    query = windowed_count\
            .writeStream\
            .outputMode('complete')\
            .format('console')\
            .option('truncate', 'false')\
            .trigger(processingTime="10 seconds")\
            .start()
    
    query.awaitTermination()

