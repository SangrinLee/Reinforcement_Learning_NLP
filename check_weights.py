# Reinforcement learning -------------------------------------------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
# from word_lstm_model_check import MyLSTM
import torch.nn.functional as F

# Set up DQN
input_dim = 4      # Three features(unigram, bigram, trigram)
output_dim = 1      # Either train or skip
hidden_size = 3
hidden_dropout_prob = 0.2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        # self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
        self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        x = self.fc2(x)

        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)

# Print DQN model configuration
print ("# DQN model configuration")
print (model)

# Load pretraiend DQN model
print ("# Load pretrained DQN model")
for i in range(7):
	model.load_state_dict(torch.load('dqn_models/DQN_' + str(i) + '.pt'))
	print ("##### DQN Model No.", i, "#####")
	for param in model.parameters():
		print (param)