'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 16:50:15
LastEditors: Please set LastEditors
Description: 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入需要用到的模块
import os
import shutil
import random
import time

# 新生成的文件
TEST_DATA_TEMP_DIR = '/mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/'
# 测试文件所在
TEST_DATA_DIR = '/mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/tmp/testdata/'

# 操作类型
ACTION_DEF = ['login', 'logout', 'purchase']
# 地域
DISTRICT_DEF = ['fujian', 'beijing', 'shanghai', 'guangzhou']
# json 格式
JSON_LINE_PATTERN = '{{"eventTime": {}, "action": "{}", "district": "{}"}}\n'


# 测试的环境搭建，判断文件夹是否存在，如果存在则删除旧数据，并建立文件夹
def test_setUp():
    print("==========START===========")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
    os.mkdir(TEST_DATA_DIR)


# 测试环境的恢复，对文件夹进行清理
def test_tearDown():
    print("============END==============")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


# 生成测试文件
def write_and_move(filename, data):
    with open(TEST_DATA_DIR + filename,
              'wt', encoding='utf8') as f:
        f.write(data)
    
    shutil.move(TEST_DATA_DIR + filename,
                TEST_DATA_DIR + filename)


if __name__ == "__main__":
    test_setUp()
    
    for i in range(1000):
        filename = 'e-mall-{}.json'.format(i)

        content = ''
        # 每次生成100个文件
        rndcount = list(range(100))
        random.shuffle(rndcount)
        for _  in rndcount:
            content += JSON_LINE_PATTERN.format(
                str(int(time.time())),
                random.choice(ACTION_DEF),
                random.choice(DISTRICT_DEF)
            )
        write_and_move(filename, content)

        time.sleep(1)
    
    test_tearDown()