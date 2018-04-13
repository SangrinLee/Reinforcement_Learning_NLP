import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

from extract_sentences import train, val, ptb_dict, extract_sentence_list

# extract sentence list
sentence_list = extract_sentence_list(train)
sentence_list = sentence_list[len(sentence_list)//2:]

# Select the batchified list
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
def create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list):
    # unigram process
    num_uni = len(data)
    num_uni_unseen = 0
    for uni in data:
        if not uni in uni_seen_list:
            num_uni_unseen += 1
            uni_seen_list.append(uni)
    prop_uni_unseen = num_uni_unseen / num_uni # proportion of unseen unigram words

    # bigram process
    num_bi = len(data) - 1
    num_bi_unseen = 0
    for i in range(num_bi):
        bi = list(data[i:i+2])
        if not bi in bi_seen_list:
            num_bi_unseen += 1
            bi_seen_list.append(bi)
    prop_bi_unseen = num_bi_unseen / num_bi # proportion of unseen bigram words

    # trigram process
    num_tri = len(data) - 2
    num_tri_unseen = 0
    for i in range(num_tri):
        tri = list(data[i:i+3])
        if not tri in tri_seen_list:
            num_tri_unseen += 1
            tri_seen_list.append(tri)
    prop_tri_unseen = num_tri_unseen / num_tri # proportion of unseen trigram words

    # create tensor variable
    input_feature = Variable(torch.Tensor(np.array([prop_uni_unseen, prop_bi_unseen, prop_tri_unseen])))
    input_feature = input_feature.view(-1, 3)

    return input_feature, uni_seen_list, bi_seen_list, tri_seen_list

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
        x = F.sigmoid(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = self.fc2(x)
        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)

N_options = 5   # Number of options to choose from for sampling
N_samples = 5 # Number of samples to extract

def sample(sample_num):    
    model.load_state_dict(torch.load('DQN.pt')) # Load pretraiend DQN model

    data_sampled_DQN = [] # Stores the data sampled from the pretrained DQN model
    data_sampled_random = [] # Stores the data sampled randomly

    dataset = select_batch(sentence_list) # Select the batchified data to be fed into the DQN
    uni_seen_list = [] # Initialize unigram unseen list
    bi_seen_list = [] # Initialize bigram unseen list
    tri_seen_list = [] # Initialize trigram unseen list

    # No. of iterations : 1467
    for i in range(len(dataset)//N_options):
        print ("#", i)
        state_value_list = [] # Initialize state value list
        data_list = [] # Initialize data list

        for j in range(N_options):
            data = dataset[i*N_options+j]
            data_list.append(data)
            if j != N_options-1:
                # Construct the state(how different our input is from the the dataset train, represented as scalar value)
                state, _,_,_ = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            else:
                state, uni_seen_list, bi_seen_list, tri_seen_list = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            
            # Store each state value into state value list
            model_output = model(state).data
            state_value_list.append(model_output[0][0])

        choice = np.argmax(state_value_list) # Choose data with highest state value to train 
        data_sampled_DQN.append(data_list[choice]) # Add selected data into dataset

        choice_random = random.randint(0, N_options-1) # Choose data randomly
        data_sampled_random.append(data_list[choice_random]) # Add selected data into dataset

    with open('sampled_data/data_sampled_dqn_' + str(sample_num), 'wb') as handle:
        pickle.dump(data_sampled_DQN, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sampled_data/data_sampled_random_' + str(sample_num), 'wb') as handle:
        pickle.dump(data_sampled_random, handle, protocol=pickle.HIGHEST_PROTOCOL)


for sample in range(N_samples):
    print ("# Sample", sample)
    sample(sample)

# from reinforcement_learning_sampling_train import *