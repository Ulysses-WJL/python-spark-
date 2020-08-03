'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 15:44:57
LastEditors: Please set LastEditors
Description: 
'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName('StructedSocketWordCount')\
        .master('local[4]')\
        .getOrCreate()
    
    sc =spark.sparkContext
    sc.setLogLevel('WARN')

    # 从socket源读取stream
    lines = spark\
            .readStream\
            .format('socket')\
            .option('host', 'localhost')\
            .option('port', 9999)\
            .load()
    
    words = lines.select(
        explode(
            split(lines.value, ' ')  # 空格拆开
        ).alias('word')  # 将一行列表 打开 一列数据
    )
    # word , count
    wordcounts = words.groupBy('word').count()
    
    # 输出
    query = wordcounts\
            .writeStream\
            .outputMode('complete')\
            .format('console')\
            .trigger(processingTime="8 seconds")\
            .start()
    
    query.awaitTermination()
