'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 20:31:18
LastEditors: Please set LastEditors
Description: 
'''
#!/usr/bin/env python3

from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("TestRateStreamSource") \
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    # 每秒 发生的数据 5行
    lines = spark \
        .readStream \
        .format("rate") \
        .option('rowsPerSecond', 5) \
        .load()

    print(lines.schema)

    query = lines \
        .writeStream \
        .outputMode("update") \
        .format("console") \
        .option('truncate', 'false') \
        .start()

    query.awaitTermination()
