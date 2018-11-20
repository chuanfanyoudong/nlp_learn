#!/usr/bin/env python
# coding: utf-8

# ## 序列标注模型的LSTM-CRF实现

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


# **做一些准备函数**

# In[2]:


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# In[3]:


print(list("aaa"))


# **准备做模型**

# In[4]:


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size, device = 1))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device = 1),
                torch.randn(2, 1, self.hidden_dim // 2, device = 1))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000., device = 1)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentence = sentence.type(torch.cuda.LongTensor)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device = 1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        tags = tags.type(torch.cuda.LongTensor)
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device = 1)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[5]:


import re


def double(matched):
    line = matched.group(0)[1:-1]
    line_list = line.strip().split("  ")
    final_lien = "".join([word.strip().split("/")[0] for word in line_list])
    # print(final_lien)
    return final_lien + "/"


def pre_process(line):
    pattern1 = re.compile(r'\[.*\]')
    num = re.sub(r'\[.*?\]', double, line)
    return num


pre_process("河南/ns  和/c  [上海/ns  男排/n]nt  仍/d  居/v  积分榜/n  [河南/ns  男排/n]nt 前/f  两/m  位/q")

# In[6]:


with open("/data/users/zkjiang/fenci/data/renmin.txt", "r", encoding="utf-8") as file:
    training_data_origin = []
    for line in file:
        if line != "":
            #         print(line)
            #         line.replace("  "," ")
            #         line.replace(" ", "  ")
            line_final = pre_process(line)
            #             print(line_final)
            b_i_o_list = []
            line_list = line_final.strip().split(" ")[1:]
            word_list = [word_part.strip().split("/")[0] for word_part in line_list]
            sentence_list = []
            for word in word_list:
                sentence_list += list(word)
                if len(word) == 1:
                    b_i_o_list.append("O")
                else:
                    b_i_o_list.append("B")
                    for i in word[:-1]:
                        b_i_o_list.append("I")
            # part_list = [word_part.strip().split("/")[1] for word_part in line_list]
            training_data_origin.append((sentence_list, b_i_o_list))
        # print(training_data_origin[1:10])

# In[ ]:


# In[ ]:


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 12
HIDDEN_DIM = 12
training_data = training_data_origin[1:-1]
# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    torch.cuda.manual_seed(777)  # set random seed for gpu
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
# def train():
i = 0
for epoch in range(
        20):  # again, normally you would NOT do 300 epochs, it is toy data
    i += 1
    print("epoch",i)
    z = 0
    for sentence, tags in training_data:
        z += 1
        print("已经训练", z)
        if len(sentence) == 0:
            continue
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        if z%10 == 0:
            print("已经训练{}数据，损失是{}".format(z, loss))
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), "./lstm_crf_model")
# Check predictions after training
# print("训练数据是",training_data)
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     #     print(precheck_sent)
#     print(model(precheck_sent))
# We got it!


# In[ ]:


reverse_dict = {}
for key in tag_to_ix:
    value = tag_to_ix[key]
    reverse_dict[value] = key


# In[ ]:


### 比较两个list的相似度


def similary(list1, list2):
    score = sum([1 if list1[i] == list2[i] else 0 for i in range(len(list1))])
    return score / len(list1)


# In[ ]:


# reverse_dict = {}
import numpy as np

score_list = []
print(reverse_dict)
inputs = prepare_sequence(["我", "们", "到", "中", "国"], word_to_ix)
#         print(training_data[2])
tag_scores = model(inputs)
# print(tag_scores)
with torch.no_grad():
    for i in training_data:
        if len(i[0]) == 0:
            continue
        inputs = prepare_sequence(i[0], word_to_ix)
#         print(training_data[2])
        tag_scores = model(inputs)
        print("句子是", i[0])
        print("分词结果是", tag_scores)
        print("正确结果是", i[1])

#         # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#         # for word i. The predicted tag is the maximum scoring tag.
#         # Here, we can see the predicted sequence below is 0 1 2 0 1
#         # since 0 is index of the maximum value of row 1,
#         # 1 is the index of maximum value of row 2, etc.
#         # Which is DET NOUN VERB DET NOUN, the correct sequence!
# #         tag_scores.argmax(dim=1)
#     #     print(tag_scores.argmax(dim=1))
#         final_result = [reverse_dict[i] for i in tag_scores[1]]
#         score = similary(i[1],final_result )
#         print(i)
#         print(final_result)
#         score_list.append(score)
# print(np.mean(score_list))


# In[ ]:


# In[ ]:




