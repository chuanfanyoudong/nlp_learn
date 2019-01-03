# -*- coding: utf-8 -*-
"""
CONFIG
------
对配置的封装
"""
import os
from configparser import ConfigParser
CONF_PATH = os.path.join(os.path.dirname(os.getcwd()), "conf/config.conf")
# print(DICT_PATH, "姜振康")
# DICT_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "nlp_learn/data/ner_data/"))
# print(DICT_PATH)
__config = None


def get_config(config_file_path= CONF_PATH):
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
