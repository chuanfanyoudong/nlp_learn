#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: sentence_classfication.py
@time: 2019/3/23 14:06
"""

from flask import Blueprint, request, render_template
from jiang_classfication.main import main



sentence_classfication = Blueprint('sentence_classfication', __name__)

@sentence_classfication.route('/sentence_classfication_demo', methods=['GET', 'POST'])
def sentence_classfication_method():
    """
    分词接口
    :return: 输入要分词的句子，输出分词的结果
    """
    sentence_classfication_result = {}
    if request.method == 'POST':
        # print("POST")
        sentence = request.form.get("sentence")
        if sentence.strip() == "":
            sentence_classfication_result = ""
        else:
            sentence_classfication_result = main(sentence)
        return render_template("sentence_classfic.html", sentence_classfication_result = sentence_classfication_result,
                               lstm_split_result = "分类结果")
    return render_template("sentence_classfic.html", sentence_classfication_result = "分词结果",
                           lstm_split_result = "分类结果")

