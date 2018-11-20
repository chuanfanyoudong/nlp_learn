#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 11:12
# @Author  : cbdeng
# @Software: PyCharm

from flask import Flask, request,url_for
app = Flask(__name__)
import logging
import os,sys
g_proj_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(g_proj_path)
sys.path.append(g_proj_path+"modules/")

@app.route("/test")
def test():
    from common_lib.utils.query_db import MongoUtils
    ins = MongoUtils()
    # ['山', '山东', '山东省', '山东省的', '山东省的人', '山东省的人口', '东', '东省', '东省的', '东省的人', '东省的人口', '省', '省的', '省的人', '省的人口', '的', '的人', '的人口', '人', '人口', '口']
    # ['山东', '山东省']
    # qs = request.args.get('query_str')
    print(ins.mget_word_info("syno",word_list=["省" ],))
    # print(ins.mget_word_info("word_info",word_list=["语言","fcuk11"]))
    logging.error("尼玛")
    return "test"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5019, debug=True, use_reloader=False)