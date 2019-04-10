# lstm language model for training char embeddding
model 3.1 should work for thai corpus

the model predicts the next character of a giving sequence 

its output  will be used as input for a lstm binary classifier

Questions:
- What will be the input for the classifier?  (last hidden layer or char embedding?)
- input and output data for classifier? and how to represent them 
eg. want to predict if "e" a word ending or not, we represent "e" using its context window of 6
-  "nic_e_day" -- > input is then embedding of nic_day  

TODO:
- PBC as loss function
- create the classifier
- preprocess label data for classifier
- create validation, test set 
