#!/usr/bin/env python3
#coding=utf-8
import sys
import os
g_proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_proj_path)
sys.path.append(g_proj_path)
# sys.path.append(g_proj_path+"modules/")

from jiang_fenci.main import TokenGet
from jiang_fenci.lstm_crf.simple_model_test import ModelMain
tg = TokenGet()
mm = ModelMain()



g_proj_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(g_proj_path)
sys.path.append(g_proj_path+"modules/")
from flask import Flask, request, url_for, redirect
from flask import render_template
from flast_practice.module.get_info import GetInfo
gi = GetInfo()
# from modules.common_lib.const.utils import g_config
import json
# from website.modules_api import QueryManage
# from website.test_modules_api import QueryManage as TestQueryManage
import random
import logging


app = Flask(__name__)

# qm = QueryManage()
# test_qm = TestQueryManage()


# 返回主页面的文件
@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/password", methods=['GET', 'POST'])
def password():
    if request.method == 'POST':
        print("POST")
        password = request.form.get("password")
        if password == "3762541okm":
            return redirect('/rate')
        else:
            return render_template("password.html", info = "请重新输入")
    return render_template("password.html")


@app.route("/split", methods=['GET', 'POST'])
def split_sentence():
    """
    分词接口
    :return: 输入要分词的句子，输出分词的结果
    """

    # print(username)
    if request.method == 'POST':
        print("POST")
        sentence = request.form.get("sentence")
        lstm_sentence = request.form.get("lstm_sentence")
        split_result = tg.main(sentence)
        print(split_result)
        return render_template("split_sentence.html", split_result = split_result,
                               lstm_split_result = "分词结果")
        # print(entity_info)
    # result = test_qm.qa(qs)
    return render_template("split_sentence.html", split_result = "分词结果",
                           lstm_split_result = "分词结果")


@app.route("/rate", methods=['GET', 'POST'])
def rate():
    entity_info = [{"date":"", "thing"  :"", "important":""}]
    print(1)

    date = request.form.get("date")
    thing = request.form.get("thing")
    important = request.form.get("important")
    inquire_date = request.form.get("inquire_date")
    print(inquire_date)
    if date != None and thing != None and important != None and date + thing + important != "":
        gi.insert_data({"date":date, "thing"  :thing, "important":important})
    entity_info = gi.get_thing(inquire_date)
    print(entity_info)
        # return render_template("schedule.html", abs = entity_info)
        # print(entity_info)
    # result = test_qm.qa(qs)
    all_info = gi.get_all_info()
    return render_template("schedule.html", single_day_things=entity_info, all_info = all_info)


class test():

    def __init__(self, param):
        self.a = param
        self.b = param + "test"




if __name__ == "__main__":
    debug = True
    # if g_config["PROJ_INFO"]["env"] == "local":
    #     debug = True
    app.run(host="0.0.0.0", port=5010,debug=debug, use_reloader=False)
