# lstm language model for training char embeddding
source: https://github.com/pytorch/examples/tree/master/word_language_model


model 3.1 should work for thai corpus

-the model predicts the next character of a given sequence
-its outputs will be used as inputs for a lstm binary classifier

Questions:
- What will be the input for the classifier? (last hidden layer or char embedding?)
- input and output data for classifier? and how to represent them
- eg. we want to predict whether "e" a word ending or not, we represent "e" using its context window of 6
- "nic_e_day" -- > input is then embedding of nic_day

TODO:
-PBC as loss function
-create the classifier
-preprocess label data for classifier
-create validation, test set

