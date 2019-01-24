import os
import pickle

import re
import traceback

from pymongo import MongoClient

from model import *

from conf.config import get_config
__config = get_config()
ROOT_DATA = __config["path"]["root"]
EMBEDDING_ROOT = __config["segment"]["embedding"]
TRAIN_DATA_PATH = __config["segment"]["lstm_train_data"]
TEST_DATA_PATH = __config["segment"]["lstm_test_data"]
VAL_DATA_PATH = __config["segment"]["lstm_val_data"]

def normalize(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        return re.sub(" ", "", x)
    if unit == "word":
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()

def load_word_to_idx(filename):
    print("loading word_to_idx...")
    word_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        word_to_idx[line] = len(word_to_idx)
    fo.close()
    return word_to_idx

def save_word_to_idx(filename, word_to_idx):
    fo = open(filename + ".word_to_idx", "w")
    for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

def load_tag_to_idx(filename):
    print("loading tag_to_idx...")
    tag_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tag_to_idx[line] = len(tag_to_idx)
    fo.close()
    return tag_to_idx

def save_tag_to_idx(filename, tag_to_idx):
    fo = open(filename + ".tag_to_idx", "w")
    for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tag)
    fo.close()

def load_checkpoint(filename, model = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving model...")
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def iob_to_txt(txt, tags, unit):
    y = ""
    txt = tokenize(txt, unit)
    for i, j in enumerate(tags):
        if i and j[0] == "B":
            y += " "
        y += txt[i]
    return y

def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0


def process_line_(line = "", max_length = 1000):
    # line = "“  征  而  未  用  的  耕地  和  有  收益  的  土地  ，  不准  荒芜  。"
    if line.strip() == "":
        return None
    line_list = line.strip("\n").split("  ")
    while "" in line_list:
        line_list.remove("")
    sentence = "".join(line_list)
    tag_list = []
    for word in line_list:
        if line_list == []:
            continue
        if len(word) == 1:
            tag_list.append("S")
        elif len(word) == 2:
            tag_list.append("B")
            tag_list.append("E")
        else:
            tmp_tag_list = ["O" for char in word]
            tmp_tag_list[0] = "B"
            tmp_tag_list[-1] = "E"
            tag_list = tag_list + tmp_tag_list
    list_sentence= list(sentence)
    real_length = len(list_sentence)
    if len(list_sentence) != len(tag_list) or len(list_sentence) > max_length:
        print(list_sentence)
        raise Exception("数据处理出错")
    return (list_sentence, tag_list), real_length

def get_embedding(word2id):
    default_vector = [0. for _ in range(200)]
    tag2id = {"B":0, "E":1, "O":2, "S":3, "<pad>":4}
    path = ROOT_DATA + EMBEDDING_ROOT
    if  os.path.exists(path):
        embedding_file = open(path, 'rb')
        embedding_dict =  pickle.load(embedding_file)
    else:
        embedding_dict = {}
        client = MongoClient('mongodb://192.168.10.27:27017/')
        admin = client.admin
        admin.authenticate("root", "nlp_dbroot1234")
        embedding_data = client.embeddings.tenlent_vector_200
        embedding_file = open(ROOT_DATA + EMBEDDING_ROOT, "wb")
        for vector_info in embedding_data.find({"$where":"this.word.length <2"}):
            word = vector_info["word"]
            vector = [float(i) for i in vector_info["vector"]]
            if len(vector) == 200:
            # print(vector)
                embedding_dict[word] = vector
        pickle.dump(embedding_dict, embedding_file)
    rever_word2id = {value:key for key, value in word2id.items()}
    embedding_list = []
    for i in range(len(rever_word2id)):
        if rever_word2id[i] in embedding_dict:
            embedding_list.append(embedding_dict[rever_word2id[i]])
        else:
            embedding_list.append(default_vector)
    return embedding_list

def load_data(file, PAD_IDX, SOS_IDX, BATCH_SIZE):
    data = []
    tag_to_idx = {"B": 3, "E": 4, "O": 5, "S": 6, "<SOS>": 1, "<PAD>": 0, "<EOS>": 2}
    idx_to_tag = {3 :"B", 4:"E", 5:"O", 6:"S", 1:"<SOS>", 0:"<PAD>", 2:"<EOS>"}
    batch_x = []
    batch_y = []
    batch_len = 0 # maximum sequence length of a mini-batch
    print("loading data...")
    # word_to_idx = load_word_to_idx(sys.argv[2])
    # tag_to_idx = load_tag_to_idx(sys.argv[3])
    fo = open(file, "r")
    line_list_all = []
    tag_list_all = []
    char_set = set()
    for line in fo:
        if line.strip("\n").strip(" ") != "":
            try:
                (list_sentence, tag_list),  _ = process_line_(line)
                line_list_all.append(list_sentence)
                tag_list_all.append(tag_list)
                for i in list_sentence:
                    char_set.add(i)
            except:
                traceback.print_exc()
    char_list = list(char_set)
    max_length = max([len(i) for i in line_list_all])
    word_to_idx = {}
    for idx, char in enumerate(char_list, 4):
        word_to_idx[char] = idx
    word_to_idx["PAD"] = 0
    word_to_idx["SOS"] = 1
    word_to_idx["EOS"] = 2
    word_to_idx["UNK"] = 3
    len_list = []
    for i in range(len(line_list_all)):
        line = line_list_all[i]
        tag = tag_list_all[i]
        seq_len = len(line)
        batch_len = max_length
        pad = [PAD_IDX] * (batch_len - seq_len)
        line_toke = [word_to_idx[i] if i in word_to_idx else word_to_idx["UNK"] for i in line]
        tag_toke = [tag_to_idx[i] if i in tag_to_idx else tag_to_idx["UNK"] for i in tag]
        batch_x.append(line_toke + pad)
        batch_y.append([SOS_IDX] + tag_toke + pad)
        len_list.append(len(line_toke))
        if len(batch_x) == BATCH_SIZE:
            data.append((LongTensor(batch_x), LongTensor(batch_y), LongTensor(len_list))) # append a mini-batch
            batch_x = []
            batch_y = []
            len_list = []
    fo.close()
    embedding_list = get_embedding(word_to_idx)
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx, tag_to_idx, idx_to_tag, embedding_list