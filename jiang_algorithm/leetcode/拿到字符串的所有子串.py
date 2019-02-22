#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: 拿到字符串的所有子串.py
@time: 2019/2/21 10:44
"""

def get_all_sub(str_single):
    if len(str_single) == 1:
        return [str_single]
    else:
        return [str_single[0] + i for i in get_all_sub(str_single[1:])] + get_all_sub(str_single[1:])

if __name__ == '__main__':
    str_single = "abcd"
    print(get_all_sub(str_single))