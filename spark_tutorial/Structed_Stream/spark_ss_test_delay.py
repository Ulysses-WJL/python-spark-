#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入需要用到的模块
import os
import shutil
from functools import partial

from pyspark.sql import SparkSession
from pyspark.sql.functions import window
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import TimestampType, StringType

# 定义CSV文件的路径常量
TEST_DATA_DIR = '/mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/testdata/'
TEST_DATA_DIR_SPARK = 'file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/testdata/'


# 测试的环境搭建，判断CSV文件夹是否存在，如果存在则删除旧数据，并建立文件夹
def test_setUp():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
    os.mkdir(TEST_DATA_DIR)


# 测试环境的恢复，对CSV文件夹进行清理
def test_tearDown():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


# 写模拟输入的函数，传入CSV文件名和数据。注意写入应当是原子性的
# 如果写入时间较长，应当先写到临时文件在移动到CSV目录内。
# 这里采取直接写入的方式。
def write_to_cvs(filename, data):
    with open(TEST_DATA_DIR + filename, "wt", encoding="utf-8") as f:
        f.write(data)


if __name__ == "__main__":
    test_setUp()

    # 定义模式，为字符串类型的word和时间戳类型的eventTime两个列组成
    schema = StructType([
        StructField("word", StringType(), True),
        StructField("eventTime", TimestampType(), True)])

    spark = SparkSession \
        .builder \
        .appName("StructuredNetworkWordCountWindowedDelay") \
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    lines = spark \
        .readStream \
        .format('csv') \
        .schema(schema) \
        .option("sep", ";") \
        .option("header", "false") \
        .load(TEST_DATA_DIR_SPARK)

    # 定义窗口
    windowDuration = '1 hour'

    windowedCounts = lines \
        .withWatermark("eventTime", "1 hour") \
        .groupBy('word', window('eventTime', windowDuration)) \
        .count()

    query = windowedCounts \
        .writeStream \
        .outputMode("update") \
        .format("console") \
        .option('truncate', 'false') \
        .trigger(processingTime="8 seconds") \
        .start()

    # 写入测试文件file1.cvs
    write_to_cvs('file1.cvs', """
正常;2018-10-01 08:00:00
正常;2018-10-01 08:10:00
正常;2018-10-01 08:20:00
""")

    # 处理当前数据
    query.processAllAvailable()

    # 这时候事件时间更新到上次看到的最大的2018-10-01 08:20:00

    write_to_cvs('file2.cvs', """
正常;2018-10-01 20:00:00
一小时以内延迟到达;2018-10-01 10:00:00
一小时以内延迟到达;2018-10-01 10:50:00
""")

    # 处理当前数据
    query.processAllAvailable()

    # 这时候事件时间更新到上次看到的最大的2018-10-01 20:00:00

    write_to_cvs('file3.cvs', """
正常;2018-10-01 20:00:00
一小时外延迟到达;2018-10-01 10:00:00
一小时外延迟到达;2018-10-01 10:50:00
一小时以内延迟到达;2018-10-01 19:00:00
""")

    # 处理当前数据
    query.processAllAvailable()

    query.stop()

    test_tearDown()
