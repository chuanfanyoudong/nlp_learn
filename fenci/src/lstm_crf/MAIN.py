import torch

from generate_data import load_data
from lstm_crf import BiLSTM_CRF
from torch.optim import optimizer

EMBEDDING_DIM = 12
HIDDEN_DIM = 12


def main():

    if torch.cuda.is_available():
        cuda = True
        device = 1
        torch.manual_seed(111)

    train_iter, val_iter, vocab_size = load_data()
    model = BiLSTM_CRF(EMBEDDING_DIM, HIDDEN_DIM, vocab_size)
    for i in range(300):
        total_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for idx, batch in enumerate(train_iter):
            if len(batch) == 1:
                continue
            word, tag = batch.word, batch.tag
            if cuda:
                text, label = word.cuda(), tag.cuda()

            # for sentence, tags in training_data:
            if len(word) == 0:
                continue
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(word, tag)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "./lstm_crf_model")


if __name__ == '__main__':
    main()
#
# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# EMBEDDING_DIM = 5
# HIDDEN_DIM = 4
# # Make up some training data
# # training_data = [(
# #     "the wall street journal reported today that apple corporation made money".split(),
# #     "B I I I O O O B I O O".split()
# # ), (
# #     "georgia tech is a university in georgia".split(),
# #     "B I O O O O B".split()
# # )]
#
# word_to_ix = {}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
#
# tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
#
# model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# # if torch.cuda.is_available():
# #     torch.cuda.set_device(2)
# #     torch.cuda.manual_seed(777)  # set random seed for gpu
# #     model.cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
#
# # Check predictions before training
# # with torch.no_grad():
# #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
# #     print(model(precheck_sent)
#
# # Make sure prepare_sequence from earlier in the LSTM section is loaded
# for epoch in range(
#         300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         if len(sentence) == 0:
#             continue
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is,
#         # turn them into Tensors of word indices.
#         # sentence_in = prepare_sequence(sentence, word_to_ix)
#         # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#
#         # Step 3. Run our forward pass.
#         loss = model.neg_log_likelihood(sentence_in, targets)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         # calling optimizer.step()
#         loss.backward()
#         optimizer.step()
# torch.save(model.state_dict(), "./lstm_crf_model")
# # Check predictions after training
# # print("训练数据是",training_data)
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# #     print(precheck_sent)
#     print(model(precheck_sent))