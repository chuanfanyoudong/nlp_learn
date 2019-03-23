#!/usr/bin/env python
# encoding: utf-8

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: data_process.py
@time: 2019/2/1 9:13
"""
import os

"""
使用机器学习方法进行文本分类，包括使用了
》》》》》逻辑回归
》》》》》随机森林
》》》》》KNN
》》》》》SVM
"""

import time
import jieba
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import preprocessing
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
ROOT = cur_dir_path + "/../"
sys.path.append(ROOT)
from conf.config import get_config
__config = get_config()
DATA_PATH = __config["classfication"]["data_path"]
ROOT = __config["path"]["root"]
MODEL_PATH = ROOT + __config["traditional_ml"]["model_path"]
sys.path.append(ROOT)
from jiang_classfication.tfidf.TFIDF import TFIDF
vectorizer = None
encoder = None
knn_model = None
rf_model = None
logistic_model = None
## 数据文件夹路路径，不是文件路径
DATA_PATH = ROOT + DATA_PATH

def get_data(file_name):
    """
    准备语料
    :param file_name:
    :return:
    """
    f = open(DATA_PATH + file_name)
    data_save_path = DATA_PATH + file_name + ".pkl"
    if os.path.exists(data_save_path):
        save_path_file = open(data_save_path, "rb")
        data_save_dict = pickle.load(save_path_file)
        corpus_list, label_list = data_save_dict["corpus"], data_save_dict["label"]
        return corpus_list, label_list
    else:
        num = 0
        corpus_list = []
        label_list = []
        for line in f:
            num += 1
            # if num == 20:
            #     break
            line = line.strip()
            label = line[:2]
            content = line[3:]
            ## 结巴分词
            list = jieba.lcut(content)
            if list != []:
                label_list.append(label)
                ## 每一句话每一个词用空格连接
                corpus_list.append(" ".join(list))
        save_path_file = open(data_save_path, "wb")
        data_save_dict = {"corpus":corpus_list, "label":label_list}
        pickle.dump(data_save_dict, save_path_file)
        return corpus_list, label_list


def train(model_name = "logistic"):
    ## 训练语料
    train_corpus_list, train_label_list = get_data("cnews.train.txt")
    ## 验证语料
    val_corpus_list, val_label_list = get_data("cnews.val.txt")
    ## 测试语料
    test_corpus_list, test_label_list = get_data("cnews.test.txt")
    ## 所有语料加起来
    corpus = train_corpus_list + val_corpus_list + test_corpus_list
    label = train_label_list + val_label_list + test_label_list
    encoder = preprocessing.LabelEncoder()
    ## label数字化
    corpus_encode_label = encoder.fit_transform(label)
    ## 可以选择用自己写的TF-IDF方法
    # vectorizer = TFIDF(corpus)
    # tfidf = vectorizer.get_tf_idf()
    ## 也可以选择掉包，直接调用TF-IDF算法
    vectorizer = TfidfVectorizer(min_df=1e-5)  # drop df < 1e-5,去低频词
    tfidf = vectorizer.fit_transform(corpus)
    ## 拆分训练集，验证集，测试集
    len_train, len_test, len_val = len(train_corpus_list), len(test_corpus_list), len(val_corpus_list)
    train_label = corpus_encode_label[:len_train]
    val_label = corpus_encode_label[len_train:len_train + len_val]
    test_label = corpus_encode_label[len_train + len_val:]
    train_set = tfidf[:len_train]
    val_set = tfidf[len_train:len_train + len_val]
    test_set = tfidf[len_train + len_val:]

    ## 逻辑回归
    start = time.time()
    print("开始训练{}模型".format(model_name))
    if model_name == "logistic":
        ## 选择逻辑回归模型
        model = LogisticRegression()
        model.fit(train_set, train_label)
    if model_name == "svm":
        ## 选择svm模型
        model = svm.SVC(gamma='scale')
        model.fit(train_set, train_label)
    if model_name == "knn":
        ## 选择KNN模型
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(train_set, train_label)
    if model_name == "randomforest" or model_name == "rf":
        model = RandomForestClassifier(n_estimators=10)
        model.fit(train_set, train_label)
    end = time.time()
    print("训练{}模型需要的时间是{}".format(model_name, end - start))
    ## 保存模型
    joblib.dump(model, MODEL_PATH + model_name + "_model.model")
    # joblib.dump(vectorizer, MODEL_PATH + "vectorizer.pkl")
    # joblib.dump(encoder, MODEL_PATH + "encode.pkl")
    ## 在验证集上的效果测试
    print("val mean accuracy: {0}".format(model.score(val_set, val_label)))
    val_pred = model.predict(val_set)
    print(classification_report(val_label, val_pred))
    ## 在测试集上的效果测试
    print("val mean accuracy: {0}".format(model.score(test_set, test_label)))
    test_pred = model.predict(test_set)
    print(classification_report(test_label, test_pred))
    return tfidf, label

def predict(sentence, model_name):
    """
    预测文本的类型
    :param sentence:要预测的句子
    :return:
    """
    ##加载模型
    if model_name == "logistic":
        global logistic_model
        if not logistic_model:
            print("加载{}模型".format(model_name))
            logistic_model = joblib.load(MODEL_PATH + model_name + "_model.model")
        model = logistic_model
    if model_name == "knn":
        global knn_model
        if not knn_model:
            print("加载{}模型".format(model_name))
            knn_model = joblib.load(MODEL_PATH + model_name + "_model.model")
        model = knn_model
    if model_name == "rf":
        global rf_model
        if not rf_model:
            print("加载{}模型".format(model_name))
            rf_model = joblib.load(MODEL_PATH + model_name + "_model.model")
        model = rf_model
    if model_name == "svm":
        global svm_model
        if not svm_model:
            print("加载{}模型".format(model_name))
            svm_model = joblib.load(MODEL_PATH + model_name + "_model.model")
        model = svm_model
    ## 将句子转化成特征向量
    global vectorizer
    if not vectorizer:
        print("加载词向量表")
        vectorizer = joblib.load(MODEL_PATH + "vectorizer.pkl")
    ## 加载label的标签dict
    global encoder
    if encoder == None:
        print("加载tag词典")
        encoder = joblib.load(MODEL_PATH + "encode.pkl")
    corpus = vectorizer.transform([" ".join(jieba.lcut(sentence))])
    start_time = time.time()
    result = model.predict(corpus)
    end_time = time.time()
    return result[0], end_time - start_time
    # print([{idx: label} for idx, label in enumerate(list(encoder.classes_[0:10]))])
    # print(result)

if __name__ == '__main__':

    # data_dict = get_data("cnews.test.txt", data_dict)
    train(model_name = "svm")
    sentence = "中国娱乐新闻真的很好看"
    print(predict(sentence, "svm"))


