import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from generate_data import load_data
from lstm import LSTMTagger
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.optim import optimizer

EMBEDDING_DIM = 12
HIDDEN_DIM = 12

train_iter, val_iter, vocab_size, target_size = load_data()
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, target_size)

def main():

    if torch.cuda.is_available():
        cuda = True
        device = 1
        torch.manual_seed(111)
    global model

    if cuda:
        torch.cuda.set_device(1)
        # torch.cuda.manual_seed(seed)  # set random seed for gpu
        model.cuda()


    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, target_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for i in range(5):
        print("epoch",i)
        total_loss = 0.0
        correct = 0
        total = 0
        model.train()
        j = 0
        for idx, batch in enumerate(train_iter):
            j += 1
            if j%100 == 0:
                print("已经读取{}".format(j))
            if len(batch) == 1:
                continue
            word, tag = batch.word, batch.tag
            if cuda:
                word, tag = word.cuda(), tag.cuda()
            if len(word) == 0:
                continue
            model.zero_grad()
            # model.hidden = model.init_hidden()
            sentence_in = word
            targets = tag
            tag_scores = model(sentence_in)
            targets = targets.type(torch.LongTensor)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "./lstm.model")

def val_result():
    # 生成测试提交数据csv
    # 将模型设为验证模式
    cuda = 1
    model.load_state_dict(torch.load("./lstm.model"))
    model.eval()
    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for batch in val_iter:
            word, tag = batch.word, batch.tag
            if cuda:
                word, tag = word.cuda(), tag.cuda()
            outputs = model(word)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            pred = outputs.max(1)[1]
            similary(pred, tag)
            # result = np.hstack((result, pred.cpu().numpy()))

    # 生成概率文件npy
    # prob_cat = np.concatenate(probs_list, axis=0)
    #
    # val = pd.read_csv('/data/users/zkjiang/test/data_grand/util/article/val_set.csv')
    # val_id = val['id'].copy()
    # val_pred = pd.DataFrame({'id': val_id, 'class': result})
    # val_pred['class'] = (val_pred['class'] + 1).astype(int)
    #
    # return prob_cat, val_pred

def similary(pros, tags):
    np_pros = list(pros.numpy().reshape(-1))
    np_tags = list(tags.cpu().numpy().reshape(-1))
    score = sum([1 if np_pros[i] == np_tags[i] else 0for i in range(len(np_tags))])/len(np_tags)
    print(score)







if __name__ == '__main__':
    # main()
    val_result()
    # similary(1,2)