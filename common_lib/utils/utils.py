#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: utils.py
@time: 2019/1/9 14:50
"""
import json


def ret_in_json(data={},err_code=0,msg="操作成功！"):
    if type(data) is not dict:
        raise Exception("[ret_in_json]data must be dict")
    the_res = {
        "code":err_code,
        "msg":msg,
        "data": data
    }
    return json.dumps(the_res, indent=4, ensure_ascii=False)