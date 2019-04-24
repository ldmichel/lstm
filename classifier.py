#!/usr/bin/env python
# coding: utf-8

# In[1]:

# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model3_1 import *

torch.manual_seed(1)


###### process data


thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ'  # 1..73
numbers = '๐๑๒๓๔๕๖๗๘๙0123456789'  # => 74 ' '=> 75, '.'=>76
symbols1 = ':;^+-*/_=#!~%\\/\'"`?'  # 77
symbols2 = '()[]<>{}'  # 78
eng_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 79

chars =  tuple(" "+thai_chars+numbers+ symbols1+symbols2 + eng_chars)


#create input tensor
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)




#create training data
#train_file = './datasets/eng.train'  # path to training data
#val_file = './datasets/eng.testa'  # path to validation data
#test_file = './datasets/eng.testb'  # path to test data


#TODO: split data in to train, dev, test  and write code to evaluate the model
#



x = ['โดยตลอดแต่', 'ความเป็น', 'ศาสตร์ที่', 'สอนในสาขา', 'นั้นๆไม่', 'สามารถที่', 'จะช่วยให้', 'ผู้เรียน', 'หลุดพ้นไป', 'จากทัศนะ', 'ครอบงำที่', 'มองปัญหา', 'ความยากจน', 'มองคนจนแบบ', 'เดิมๆและ', 'แสดงออกต่อ', 'คนจนเหล่า', 'นั้นใน', 'ลักษณะที่', 'เป็นภาระ']

y = ['1001000100', '10001000', '100000100', '100101000', '10001100', '100000100', '101000100', '10010000', '100000010', '10010000', '100000100', '10010000', '100010000', '1001010100', '10001100', '1000100100', '101010000', '100010', '100000100', '10001000']

training_data = tuple(zip(x,y))


'''
training_data = [
    (list("โรงแรมน"), list("1001001")),
    (list("ห้องพัก"), list("1000100"))
]
'''

word_to_ix = { c:i for i, c in enumerate(chars)}
#print(word_to_ix)






tag_to_ix = {"0": 0, "1": 1}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


######### model

class LSTMTagger(nn.Module):

    def __init__(self, weights_matrix, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) # replace with trained embedding

        lstm_vector, last_hidden_state = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_vector.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



####### train model

# load embedding weights
hidden_size = 10  # is also embed dim
n_layers = 2
n_chars = len(chars)

charnet = CharRNN(n_chars, hidden_size, n_chars, "lstm", n_layers)

charnet.load_state_dict(torch.load("reviews_th_1000.pt"))
charnet.eval()

#for param_tensor in charnet.state_dict():
    #print(param_tensor, "\t", charnet.state_dict()[param_tensor].size())

#https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


#print(" charnet emb: ", charnet.embed.weight)

#print("--------load -emb")

#new ,c,v= create_emb_layer(charnet.embed.weight)
#print(new.weight)



model = LSTMTagger(charnet.embed.weight, HIDDEN_DIM, len(word_to_ix), 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)




'''
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    #print(tag_scores)
    L = tag_scores

    L = L.argmax(1)
    print(L)

'''
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data

    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


print("------after training-------")
# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence("ออกไปเลย", word_to_ix)
    #inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
    #print(type(tag_scores))
    #print(tag_scores.shape)
    L = tag_scores

    L = L.argmax(1)
    print(L)







