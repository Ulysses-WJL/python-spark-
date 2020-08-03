#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.functions import explode
from pyspark.sql.functions import length


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("StructuredNetworkWordCountFileSink") \
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    lines = spark \
        .readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    words = lines.select(
        explode(
            split(lines.value, " ")
        ).alias("word")
    )

    all_length_5_words = words.filter(length("word") == 5)

    query = all_length_5_words \
        .writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "file:///tmp/filesink") \
        .option("checkpointLocation", "file:///tmp/file-sink-cp") \
        .trigger(processingTime="8 seconds") \
        .start()

    query.awaitTermination()
