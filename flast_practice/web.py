#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 15:18
# @Author  : cbdeng
# @Software: PyCharm

import os,sys
g_proj_path = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(g_proj_path)
sys.path.append(g_proj_path+"modules/")

from flask import Flask, request
from view.ent_attr_rel import ent_attr_rel
from view.ent_ent_rel import ent_ent_rel

app = Flask(__name__)
app.register_blueprint(ent_attr_rel, url_prefix='/ent_attr_rel')
app.register_blueprint(ent_ent_rel, url_prefix='/ent_ent_rel')


if __name__ == "__main__":
    # t_debug = False
    t_debug = True
    app.run(host="0.0.0.0", port=25005,debug=t_debug)
