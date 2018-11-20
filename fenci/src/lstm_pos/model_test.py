import re


START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
EMBEDDING_DIM = 5
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from generate_data import load_data
from lstm import LSTMTagger
from torch.optim import optimizer
import numpy as np

EMBEDDING_DIM = 12
HIDDEN_DIM = 12

def main():
    if torch.cuda.is_available():
        cuda = True
        device = 1
        torch.manual_seed(111)
    train_iter, val_iter, vocab_size, target_size = load_data()
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, target_size)

    if cuda:
        torch.cuda.set_device(1)
        # torch.cuda.manual_seed(seed)  # set random seed for gpu
        model.cuda()


    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, target_size)
    model.load_state_dict(torch.load("./lstm.model"))
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for idx, batch in enumerate(train_iter):
            word, tag = batch.word, batch.tag
            if cuda:
                word, tag = word.cuda(), tag.cuda()
            outputs = model(word)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            pred = outputs.max(1)[1]
            result = np.hstack((result, pred.cpu().numpy()))
            print(result)


if __name__ == '__main__':
    main()