

import pandas as pd
from torchtext import data, datasets
import pandas as pd
from torchtext.vocab import Vectors
from tqdm import tqdm
from torch.nn import init
import random
import os
import numpy as np
from torchtext.data import Iterator, BucketIterator
from torchtext.data import TabularDataset

DATA_PATH = "../../data/"

# df = pd.read_csv("../../data/renmin_csv.csv")
# print(df.iloc[1,2])
def load_data():
    tokenize = lambda x: x.strip().split("#")
    WORD = data.Field(init_token="<bos>", eos_token="<eos>", tokenize= tokenize)
    UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>", tokenize= tokenize)
    tv_datafields = [("id", None), # 我们不会需要id，所以我们传入的filed是None
    ("word", WORD), ("tag", UD_TAG)]

    trn, vld = TabularDataset.splits(
    path= DATA_PATH, # 数据存放的根目录
    train='pos_train.csv', validation="pos_val.csv",
    format='csv',
    skip_header=True, # 如果你的csv有表头, 确保这个表头不会作为数据处理
    fields=tv_datafields)
    WORD.build_vocab(trn, vld)
    UD_TAG.build_vocab(trn, vld)


    train_iter = BucketIterator(dataset=trn, batch_size=100, shuffle=True, sort_within_batch=False,
                                     repeat=False, device=1)

    val_iter = Iterator(dataset=vld, batch_size=100, shuffle=True, sort_within_batch=False,
                                     repeat=False, device=1)

    return  train_iter, val_iter, len(WORD.vocab), len(UD_TAG.vocab)

load_data()
