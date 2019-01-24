import sys
import time
from utils import *
from os.path import isfile

UNIT = "char" # unit for tokenization (char, word)
BATCH_SIZE = 64
EMBED_SIZE = 300
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
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


def load_data(file = "test.txt"):
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
        (list_sentence, tag_list),  _ = process_line_(line)
        line_list_all.append(list_sentence)
        tag_list_all.append(tag_list)
        for i in list_sentence:
            char_set.add(i)
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
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx, tag_to_idx, idx_to_tag


def train():
    # num_epochs = int(sys.argv[5])

    data, word_to_idx, tag_to_idx, idx_to_tag = load_data()
    model = lstm_crf(len(word_to_idx), len(tag_to_idx))
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
        end = time.time()
        print("第{}次epoch，损失是{}".format(ei, loss_sum))
        print("消耗时间为")
        print(start - end)
        predict(model, word_to_idx, tag_to_idx, idx_to_tag, "test.txt")
        predict(model, word_to_idx, tag_to_idx, idx_to_tag, "sample")
        # if ei % SAVE_EVERY and ei != epoch + num_epochs:
        #     save_checkpoint("", None, ei, loss_sum, timer)
        # else:
        #     save_checkpoint(filename, model, ei, loss_sum, timer)


def run_model(model, idx_to_tag, data):
    # rever_tag_to_id = zip(idx_to_tag.values(), idx_to_tag.keys())

    z = len(data)
    while len(data) < BATCH_SIZE:
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
        if len(data) == BATCH_SIZE:
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

if __name__ == "__main__":
    # if len(sys.argv) != 6:
    #     sys.exit("Usage: %s model word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    # print("cuda: %s" % CUDA)
    train()
