'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-28 15:35:43
@LastEditors: Please set LastEditors
@Description: 
'''
import sys
import os

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import MatrixFactorizationModel

def set_logger(sc):
    """
    设置logger
    Args:
        sc ([type]): [description]
    """
    logger = sc._jvm.org.apache.log4j
    sc.setLogLevel("FATAL")
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def set_path(sc):
    """
    设置文件目录
    Args:
        sc ([type]): [description]
    """
    global PATH 
    if sc.master[:5] == 'local':
        PATH = 'file:/mnt/data1/workspace/data_analysis_mining/Python+Spark2.0+Hadoop机器学习与大数据实战/pythonsparkexample/PythonProject'
    else:
        PATH = "hdfs://master:9000/user/hduser"

def create_spark_context():
    """
    创建sc
    Returns:
        [type]: [description]
    """
    conf = SparkConf().setAppName('Decision_Tree')\
        .set("spark.ui.shaowConsoleProgress", 'false')
    sc = SparkContext(conf=conf)
    print('master: {}'.format(sc.master))

    set_logger(sc)
    set_path(sc)
    return sc

def get_data(sc):
    print("开始读取电影ID与名称字典...")
    item_rdd = sc.textFile(os.path.join(PATH, 'data/u.item'))
    move_title = item_rdd.map(lambda line: line.split("|")).\
        map(lambda a: (int(a[0]), a[1])).collectAsMap()

    return move_title

def load_model(sc, model_path):
    try:
        model = MatrixFactorizationModel.load(
            sc, os.path.join(PATH, model_path))
        print("模型已加载")
    except Exception:
        print("找不到模型")
        exit(-1)
    return model

def recommend(model, movie_title):
    if sys.argv[1] == "--U":
        recommend_movies(model, movie_title, int(sys.argv[2]))
    elif sys.argv[1] == "--M":
        recommend_users(model, movie_title, int(sys.argv[2]))
    

def recommend_movies(model, movie_title, userid):
    # user product rating
    movies = model.recommendProducts(userid, 10)
    print("为用户id {} 推荐以下电影: \n".format(userid))
    for movie in movies:
        print("电影: {}, 推荐评分: {}".format(movie_title[movie[1]], movie[2]))


def recommend_users(model, movie_title, movieid):
    # user product rating
    users = model.recommendUsers(movieid, 10)
    print("为电影 {} 推荐以下用户: \n".format(movie_title[movieid]))
    for user in users:
        print("推荐用户: {}, 评分: {}".format(user[0], user[2]))
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("请输入2个参数")
        exit(-1)
    sc = create_spark_context()
    print("==========数据准备===============")
    movie_title = get_data(sc)
    print("==========载入模型===============")
    model = load_model(sc, 'recommend_model')
    print("==========进行推荐===============")
    recommend(model, movie_title)
    