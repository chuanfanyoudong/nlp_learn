import re
from model import *

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