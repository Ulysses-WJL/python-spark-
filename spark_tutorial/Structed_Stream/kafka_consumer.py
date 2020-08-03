'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 20:20:44
LastEditors: Please set LastEditors
Description: 
'''
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName('StrucedKafkaWordCount')\
            .getOrCreate()
    
    spark.sparkContext.setLogLevel('WARN')

    # Subscribe to 1 topic
    df = spark \
        .readStream \
        .format("org.apache.spark.sql.kafka010.KafkaSourceProvider") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", 'wordcount-topic') \
        .load()
        
    # Projects a set of SQL expressions and returns a new DataFrame.
    # key value 都是哦 binary类型
    df2 = df.selectExpr("CAST(value AS STRING)")
    
    word_counts = df2.groupBy("value").count()
    
    # Creating a Kafka Sink for Streaming Queries
    query = word_counts \
        .selectExpr(
            "CAST(value AS STRING) as key",
            "CONCAT(CAST(value AS STRING), ':', CAST(count AS STRING))"
         )\
        .writeStream \
        .outputMode('complete') \
        .format('kafka') \
        .option('kafka.bootstrap.servers', 'localhost:9092') \
        .option("topic", 'wordcount-result-topic') \
        .option('checkpointLocation', 'file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/kafka-sink-cp')\
        .trigger(processingTime="8 seconds") \
        .start()
    
    query.awaitTermination()
