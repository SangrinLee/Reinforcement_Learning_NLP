import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import pickle

from extract_sentences import train, val, ptb_dict, extract_sentence_list

# extract sentence list
sentence_list = extract_sentence_list(train)
sentence_list = sentence_list[len(sentence_list)//2:]   # First half used for LSTM training

# Construct the batchified list
def select_batch(sentence_list):
    # shuffle the sentence list for removing the ordering of the document
    # this is to make independent sentences as the task is more like the sentence focused, not the document
    np.random.shuffle(sentence_list)
    sentence_list = np.concatenate(sentence_list)

    # create the batch with the size of 60
    nbatch = len(sentence_list) // 60
    sentence_list = sentence_list[:nbatch*60]
    batchified_list = np.split(sentence_list, nbatch)

    return batchified_list
    
# Create the feature by converting the sentence -> data (# of words, proportion of uni, bi, tri unseen before)
def create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list, doUpdate):
	# unigram process
	num_uni = len(data)
	num_uni_unseen = 0
	for uni in data:
		if not uni in uni_seen_list:
			num_uni_unseen += 1
			if doUpdate:
				uni_seen_list.append(uni)
	prop_uni_unseen = num_uni_unseen / num_uni # proportion of unseen unigram words

	# bigram process
	num_bi = len(data) - 1
	num_bi_unseen = 0
	for i in range(num_bi):
		bi = list(data[i:i+2])
		if not bi in bi_seen_list:
			num_bi_unseen += 1
			if doUpdate:
				bi_seen_list.append(bi)
	prop_bi_unseen = num_bi_unseen / num_bi # proportion of unseen bigram words

	# trigram process
	num_tri = len(data) - 2
	num_tri_unseen = 0
	for i in range(num_tri):
		tri = list(data[i:i+3])
		if not tri in tri_seen_list:
			num_tri_unseen += 1
			if doUpdate:
				tri_seen_list.append(tri)
	prop_tri_unseen = num_tri_unseen / num_tri # proportion of unseen trigram words

    # create tensor variable
	input_feature = [prop_uni_unseen, prop_bi_unseen, prop_tri_unseen]

	return input_feature, prop_uni_unseen, prop_bi_unseen, prop_tri_unseen

# Set up DQN
input_dim = 3 # Three features(unigram, bigram, trigram)
output_dim = 1 # Q-Value
hidden_size = 2 # Hidden Units
hidden_dropout_prob = 0.2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim) # input layer -> output layer
        # self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        # self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
        # self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        # x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = self.fc2(x)
        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)

N_options = 5   # Number of options to choose from for sampling

with open('sampled_data/data_sampled_random_dqn_2_0','rb') as b:
    dataset=pickle.load(b)

uni_seen_list = [] # Initialize unigram seen list
bi_seen_list = [] # Initialize bigram seen list
tri_seen_list = [] # Initialize trigram seen list

uni = 0
bi = 0
tri = 0
# No. of iterations : 1467
for i in range(len(dataset)//N_options):
    # Print unigram, bigram, trigram unseen list
    prop, pp_uni, pp_bi, pp_tri = create_feature(dataset[i], uni_seen_list, bi_seen_list, tri_seen_list, True)

    uni += pp_uni
    bi += pp_bi
    tri += pp_tri
print ("------------------")
print (uni, bi, tri, uni/(len(dataset)//N_options), bi/(len(dataset)//N_options), tri/(len(dataset)//N_options))
# Data from DQN
# 0.17548350398179757 0.6075663793602124 0.8165823231728855 -> select easy words
# Random Data
# 0.2034698521046645 0.715335222999942 0.9175002942214894