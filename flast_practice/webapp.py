#!/usr/bin/env python3
#coding=utf-8
import sys
import os
g_proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_proj_path)
sys.path.append(g_proj_path)
from flast_practice.config import get_config
__conf = get_config()
from flast_practice.view.split_sentence import split
from flast_practice.view.ner import ner
g_proj_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(g_proj_path)
sys.path.append(g_proj_path+"modules/")
from flask import Flask, request, redirect
from flask import render_template
from flast_practice.module.get_info import GetInfo
gi = GetInfo()


app = Flask(__name__)
app.register_blueprint(split, url_prefix='/split')
app.register_blueprint(ner, url_prefix='/ner')


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
    all_info = gi.get_all_info()
    return render_template("schedule.html", single_day_things=entity_info, all_info = all_info)

if __name__ == "__main__":
    debug = True
    # if g_config["PROJ_INFO"]["env"] == "local":
    #     debug = True
    app.run(host="0.0.0.0", port=10006,debug=debug, use_reloader=False)
