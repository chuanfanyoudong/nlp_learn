#-*-coding:utf-8-*- 

import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
from crf_ner.api import recognize

DICT_PATH = os.path.normpath(os.path.join(os.path.dirname(os.getcwd()), "data/dictionary/"))
print(DICT_PATH)
DICT_NAME = "/hanlp_dict.txt"
sentence = u"中华人民共和国今年90岁了"
print(sentence)
predict = recognize(sentence)
print(predict)


