#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: main.py
@time: 2019/3/23 12:33
"""

"""

文本分类主函数，会返回每种分类方法的结果与时间

"""
import os
import sys
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
ROOT = cur_dir_path + "/../"
sys.path.append(ROOT)
from jiang_classfication.ml_method import predict
from jiang_classfication.fasttext.fasttext_classfication import fasttext_predict
tag_dict = {0: '体育', 1: '娱乐', 2: '家居', 3: '房产', 4: '教育', 5: '时尚', 6: '时政', 7: '游戏', 8: '科技', 9: '财经'}
vectorizer = None
def main(sentence):
    result = {}
    logistic_result, logistic_time = predict(sentence, "logistic")
    knn_result, knn_time = predict(sentence, "knn")
    rf_result, rf_time = predict(sentence, "rf")
    fasttext_result, fasttext_time = fasttext_predict(sentence)
    fasttext_result, fasttext_time = [1,1]
    if logistic_result in tag_dict:
        logistic_result = tag_dict[logistic_result]
    if knn_result in tag_dict:
        knn_result = tag_dict[knn_result]
    if rf_result in tag_dict:
        rf_result = tag_dict[rf_result]
    if int(fasttext_result) in tag_dict:
        fasttext_result = tag_dict[int(fasttext_result)]
    result["logistic_result"] = logistic_result
    result["knn_result"] = knn_result
    result["rf_result"] = rf_result
    result["logistic_cost"] = logistic_time
    result["knn_cost"] = knn_time
    result["rf_cost"] = rf_time
    result["fasttext_restlt"] = fasttext_result
    result["fasttext_cost"] = fasttext_time
    return result

if __name__ == '__main__':
    sentence = "今年房价怎么个形式"
    print(main(sentence))
