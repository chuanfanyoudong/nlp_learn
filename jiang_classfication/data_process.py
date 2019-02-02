#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: data_process.py
@time: 2019/2/1 9:13
"""
import jieba
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing


from conf.config import get_config
__config = get_config()

DATA_PATH = __config["classfication"]["data_path"]
ROOT = __config["path"]["root"]
sys.path.append(ROOT)
from jiang_classfication.model.TFIDF import TFIDF

DATA_PATH = ROOT + DATA_PATH

def get_data(file_name):
    f = open(DATA_PATH + file_name)
    num = 0
    corpus_list = []
    label_list = []
    for line in f:
        num += 1
        if num == 20:
            break
        line = line.strip()
        label = line[:2]
        content = line[3:]
        # if type in data_dict:
        list = jieba.lcut(content)
        if list != []:
            label_list.append(label)
            corpus_list.append(" ".join(list))
    return corpus_list, label_list


def train():
    train_corpus_list, train_label_list = get_data("cnews.train.txt")
    val_corpus_list, val_label_list = get_data("cnews.val.txt")
    test_corpus_list, test_label_list = get_data("cnews.test.txt")
    corpus = train_corpus_list + val_corpus_list + test_corpus_list
    label = train_label_list + val_label_list + test_label_list
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(label)
    # vectorizer = TFIDF(corpus)
    # tfidf = vectorizer.get_tf_idf()
    vectorizer = TfidfVectorizer(min_df=1e-5)  # drop df < 1e-5,去低频词
    tfidf = vectorizer.fit_transform(corpus)
    len_train, len_test, len_val = len(train_corpus_list), len(test_corpus_list), len(val_corpus_list)
    train_label = corpus_encode_label[:len_train]
    val_label = corpus_encode_label[len_train:len_train + len_val]
    test_label = corpus_encode_label[len_train + len_val:]
    train_set = tfidf[:len_train]
    val_set = tfidf[len_train:len_train + len_val]
    test_set = tfidf[len_train + len_val:]
    lr_model = LogisticRegression()
    lr_model.fit(train_set, train_label)
    # joblib.dump(lr_model, "model.lr_model")
    # joblib.dump(vectorizer, "vectorizer.pkl")
    # joblib.dump(encoder, "encode.pkl")
    print("val mean accuracy: {0}".format(lr_model.score(val_set, val_label)))
    val_pred = lr_model.predict(val_set)
    print(classification_report(val_label, val_pred))
    print("val mean accuracy: {0}".format(lr_model.score(test_set, test_label)))
    test_pred = lr_model.predict(test_set)
    print(classification_report(test_label, test_pred))
    return tfidf, label

def predict(sentence):
    lr_model = joblib.load("model.lr_model")
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("encode.pkl")
    corpus = vectorizer.transform([" ".join(jieba.lcut(sentence))])
    result = lr_model.predict(corpus)
    print([{idx: label} for idx, label in enumerate(list(encoder.classes_[0:10]))])
    print(result)



if __name__ == '__main__':

    # data_dict = get_data("cnews.test.txt", data_dict)
    train()
    # sentence = "中国娱乐新闻真的很好看"
    # predict(sentence)
