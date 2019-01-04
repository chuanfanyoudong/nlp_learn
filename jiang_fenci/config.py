#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: config.py
@time: 2019/1/4 15:28
"""

"""
分词的配置文件
"""

import os
CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(os.getcwd()), "conf/config.conf"))
from configparser import ConfigParser

__config = None


def get_config(config_file_path= CONFIG_PATH):
    """
    单例配置获取
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config


if __name__ == '__main__':
    get_config()