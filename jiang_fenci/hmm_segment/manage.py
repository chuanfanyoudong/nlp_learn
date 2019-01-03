# -*- coding: utf-8 -*-
import sys


def manage():
    # arg = sys.argv[1]
    arg = "train"
    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    else:
        print('Args must in ["train", "test"].')
    sys.exit()

if __name__ == '__main__':
    manage()
