'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 19:31:36
LastEditors: Please set LastEditors
Description: 
'''
import string
import random
import time

from kafka import KafkaProducer
# pip install kafka-python

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    while True:
        # 随机2个字符
        s2 = (random.choice(string.ascii_lowercase) for _ in range(2))

        word = ''.join(s2)

        # 转为bytes类型
        value = bytearray(word, 'utf8')
        
        # 产生数据 
        # Block until a single message is sent (or timeout)
        producer.send("wordcount-topic", value).get(timeout=10)
        time.sleep(0.1)