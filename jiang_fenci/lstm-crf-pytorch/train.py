import sys
ROOT = "/data/users/zkjiang/projects/nlp_learn/"
sys.path.append(ROOT)
import time
from utils import *
from os.path import isfile

from common_lib.utils.sequence_label import decode
from conf.config import get_config
from jiang_fenci.segment_test import segment_test
import torch
__config = get_config()


UNIT = "char" # unit for tokenization (char, word)
BIDIRECTIONAL = int(__config["segment"]["BIDIRECTIONAL"])
BATCH_SIZE = int(__config["segment"]["lstm_batch_size"])
EMBED_SIZE = int(__config["segment"]["embedding_dim"])
LEARNING_RATE = float(__config["segment"]["lr"])
HIDDEN_SIZE = int(__config["segment"]["hidden_dim"])
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_DIRS = 2 if BIDIRECTIONAL else 1
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 1

PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
# PAD_IDX, SOS_IDX, BATCH_SIZE
data, word_to_idx, tag_to_idx, idx_to_tag, embedding_list = load_data("test.txt", PAD_IDX, SOS_IDX, BATCH_SIZE)

def train():
    best_score = 0
    model = lstm_crf(len(word_to_idx), len(tag_to_idx), HIDDEN_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_DIRS, NUM_LAYERS, embedding_list, padding_idx = PAD_IDX, BIDIRECTIONAL = 1, DROPOUT = 0.5)
    print(model)
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    # epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    # filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(20):
        start = time.time()
        loss_sum = 0
        timer = time.time()
        i = 0
        for x, y, len_list in data:
            i += 1
            # print("第{}次batch".format(i))
            model.zero_grad()
            _, idx_sort = torch.sort(len_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            x_sort = x[idx_sort]
            y_sort = y[idx_sort]
            loss = torch.mean(model(x_sort, y_sort))
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = loss.item()
            loss_sum += loss
        loss_sum /= len(data)

        # sentence = "我是中国人"
        # main(model,  sentence)
        F1 = 0
        if ei%3 == 1:
            F1 = segment_test(test_function = main, model = model)
        end = time.time()
        print("第{}次epoch，损失是{}, F1是{},耗时是{}".format(ei, loss_sum, F1, str(end - start)))
        # print("消耗时间为")
        # print(start - end)
        # predict(model, word_to_idx, tag_to_idx, idx_to_tag, "test.txt")
        # predict(model, word_to_idx, tag_to_idx, idx_to_tag, "sample")
        if F1 > best_score:
            best_score = F1
            checkpoint = {
                'state_dict': model.state_dict()
            }
            torch.save(checkpoint, __config["path"]["root"] + __config["segment"]["lstm_model_path"])
            # torch.save(lstm_split, __config["path"]["root"] + __config["segment"]["lstm_model_path"] + "all" + str(val_f1))
            print('Best tmp model f1score: {}'.format(best_score))
        # if F1 < best_score:
        #     model.load_state_dict(torch.load(save_path)['state_dict'])
        #     lr1 *= args.lr_decay
        #     lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
        #     optimizer = model.get_optimizer(lr1, lr2, 0)
        #     print('* load previous best model: {}'.format(best_score))
        #     print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
        #     if lr1 < args.min_lr:
        #         print('* training over, best f1 score: {}'.format(best_score))
        #         break


def run_model(model, idx_to_tag, data):
    # rever_tag_to_id = zip(idx_to_tag.values(), idx_to_tag.keys())

    z = len(data)
    while len(data) < 1:
        data.append([-1, "", [EOS_IDX]])
    data.sort(key = lambda x: len(x[2]), reverse = True)
    batch_len = len(data[0][2])
    batch = [x + [PAD_IDX] * (batch_len - len(x)) for _, _, x in data]
    result = model.decode(LongTensor(batch))
    for i in range(z):
        data[i].append([idx_to_tag[j] for j in result[i]])
    return [(x[1], x[3]) for x in sorted(data[:z])]

def predict(model, word_to_idx, tag_to_idx, idx_to_tag, file_neme):
    idx = 0
    data = []
    # model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(file_neme, "r")
    origin_tag = []
    pre_tag = []
    for line in fo:
        line = line.strip()
        (x, tag),_ = process_line_(line)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([idx, line, x])
        origin_tag += tag
        if len(data) == 1:
            result = run_model(model, idx_to_tag, data)
            for x in result:
                # print("句子", x)

                pre_tag += x[1]
                # if len(tag) != len(x[1]):
                #     print("句子", x)
                # print("原始结果", x[0])
                # print("预测结果", x[1], "\n")
                # print(iob_to_txt(*x, UNIT))
            data = []
        idx += 1
    fo.close()
    # for i in range(len(origin_tag)):
    if len(data):
        result = run_model(model, idx_to_tag, data)
        for x in result:
            pre_tag += x[1]
            # print(x[1])
            # print(iob_to_txt(*x, UNIT))
    print(sum([origin_tag[i] == pre_tag[i] for i in range(len(origin_tag))]) / len(origin_tag))

def main(model,sentence):
    if sentence == "":
        return []
    if sentence.strip("\n").strip(" ") == "":
        return []
    idx = 0
    data = []
    # model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    # fo = open(file_neme, "r")
    origin_tag = []
    pre_tag = []
    # for line in fo:
    line = sentence.strip()
    (x, tag), _ = process_line_(line)
    x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
    # data.append([idx, line, x])
    batch = [x + [PAD_IDX] * (100 - len(x))]
    result = model.decode(LongTensor(batch))[0]
    decode_result = decode(sentence, result, idx_to_tag)
    return decode_result
    # for i in range(z):



if __name__ == "__main__":
    # if len(sys.argv) != 6:
    #     sys.exit("Usage: %s model word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    # print("cuda: %s" % CUDA)
    train()
