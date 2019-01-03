# # -*- coding: utf-8 -*-
import sys
from jiang_ner.crf_ner.api import *


def manage():
    arg = 'train'
    if arg == 'train':
        train()
    elif arg == 'process':
        pre_process()
    else:
        print('Args must in ["process", "train"].')
    sys.exit()

if __name__ == '__main__':
    manage()
