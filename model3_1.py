# -*- coding: utf-8 -*-

import string
import random
import time
import math
import torch
import os

######## parameters

filename = "data/reviews_th_1000.txt"
#filename = "data/Ploy.txt"
model = "lstm"
epochs = 30
printtime = 20 # print every n  epoch
hidden_size = 10  # is also embed dim
n_layers = 2
learning_rate = 0.01
seq_length = 20
batch_size = 36
cuda = False



thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ'  # 1..73
numbers = '๐๑๒๓๔๕๖๗๘๙0123456789'  # => 74 ' '=> 75, '.'=>76
symbols1 = ':;^+-*/_=#!~%\\/\'"`?'  # 77
symbols2 = '()[]<>{}'  # 78
eng_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 79

chars =  tuple(" "+thai_chars+numbers+ symbols1+symbols2 + eng_chars)

#chars = tuple(set(text))
#TODO add padding symbol and UNK
#int2char = dict(enumerate(chars))
#char2int = {ch: ii for ii, ch in int2char.items()}
#encoded = np.array([char2int[ch] for ch in text])

#chars   = string.printable # define thai vocab
n_chars = len(chars) # len vocab
print(n_chars)

def read_file(filename):
    file = open(filename).read()
    return file, len(file)

# convert string in to int tensor
def char_tensor(string):

    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = chars.index(string[c])
        except:
            continue
    return tensor

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



######### model

class CharRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = torch.nn.Embedding(input_size, hidden_size) # hidden_size is embed dim?
        if self.model == "gru":
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, n_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.embed(input)

        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.h2o(output.view(batch_size, -1))

        return output, hidden



    def init_hidden(self, batch_size):

        if self.model == "lstm":
            return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

############# create training data from file


file, file_len = read_file(filename)

# create batches
def train_set(seq_length, batch_size):

    input = torch.LongTensor(batch_size, seq_length)
    target = torch.LongTensor(batch_size, seq_length)

    for i in range(batch_size):

        start_idx = random.randint(0, file_len - seq_length)
        end_idx = start_idx + seq_length + 1
        seq = file[start_idx:end_idx]
        seq =  seq.strip()
        #print("seq :" ,seq)
        while len(seq) <=  seq_length :
            seq = seq + " "
        input[i] = char_tensor(seq[:-1])
        target[i] = char_tensor(seq[1:])


    input = torch.autograd.Variable(input)
    target = torch.autograd.Variable(target)
    if cuda:
        input = input.cuda()
        target = target.cuda()
    #print("input  : " ,input)
    #print("output : " , target)

    return input, target

def train(inp, target):

    hidden = charnet.init_hidden(batch_size)
    if cuda:
        hidden = hidden.cuda()

    charnet.zero_grad()
    loss = 0

    #lookup_tensor = torch.tensor([20], dtype=torch.long)
    #hello_embed = charnet.embed(lookup_tensor)


    #print("before training embed : " ,hello_embed)
    #print("-----------------")
    for c in range(seq_length):
        output, hidden = charnet(inp[:,c], hidden)

        loss += criterion(output.view(batch_size, -1), target[:,c])


    #print("last hidden: " ,output.shape) :  torch.Size([36, 172])
    #print("last hidden: " ,hidden[0].shape) # tuple of 2 tensor
    #last hidden:  torch.Size([2, 5, 20])  : n_hidden, embed_dim, input_size



    loss.backward()
    optimizer.step()

   # print("loss : ",loss)
    #print(type(loss))

   # return loss.data[0] / seq_length
    return loss / seq_length

def save_model():

    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(charnet.state_dict(), save_filename)
    print('saving file as %s' % save_filename)




def generate(decoder, prime_str='ก', predict_len=100, temperature=0.8, cuda=False):
    print("prime   : "+ prime_str)
    hidden = decoder.init_hidden(1)
    prime_input = torch.autograd.Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = chars[top_i]
        predicted += predicted_char
        inp = torch.autograd.Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

##################### train the model



charnet = CharRNN(n_chars, hidden_size, n_chars, model, n_layers)
optimizer = torch.optim.Adam(charnet.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

if cuda:
    charnet.cuda()

start = time.time()
all_losses = []
loss_avg = 0 # when to use this?

try:

    for epoch in range(1, epochs + 1):
        loss = train(*train_set(seq_length, batch_size))  #####
        loss_avg += loss

        #print("time: ",printtime)
        #print("epoch : ", epoch)

        if epoch % printtime == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / epochs * 100, loss))
            print(generate(charnet, 'เ', 100, cuda=cuda), '\n')

    print("Saving...")
    print("Model's state_dict:")
    '''
    #for param_tensor in charnet.state_dict():
        #print(param_tensor, "\t", charnet.state_dict()[param_tensor].size())

        
                Model's state_dict:
        embed.weight 	 torch.Size([172, 10])
        rnn.weight_ih_l0 	 torch.Size([40, 10])
        rnn.weight_hh_l0 	 torch.Size([40, 10])
        rnn.bias_ih_l0 	 torch.Size([40])
        rnn.bias_hh_l0 	 torch.Size([40])
        rnn.weight_ih_l1 	 torch.Size([40, 10])
        rnn.weight_hh_l1 	 torch.Size([40, 10])
        rnn.bias_ih_l1 	 torch.Size([40])
        rnn.bias_hh_l1 	 torch.Size([40])
        h2o.weight 	 torch.Size([172, 10])
        h2o.bias 	 torch.Size([172])
        '''

    save_model()



except KeyboardInterrupt:
    print("backing up...")
# save_model()
