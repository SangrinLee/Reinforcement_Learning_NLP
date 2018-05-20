import numpy as np
import pickle
import random
from extract_sentences import train, val, ptb_dict, extract_sentence_list
import math

# extract sentence list
sentence_list = extract_sentence_list(train)
sentence_list = sentence_list[:len(sentence_list)//2]   # Latter half used for DQN training

# extract validation data
dataset_val = val[:len(val)//2]

uni_seen_freq_list = np.zeros(len(ptb_dict))
bi_seen_freq_list = np.zeros([len(ptb_dict),len(ptb_dict)])
# tri_seen_freq_list = np.zeros([len(ptb_dict),len(ptb_dict),len(ptb_dict)])

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
def create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list, i_num, doUpdate):
    # unigram process
    num_uni = len(data)
    num_uni_unseen = 0
    freq_uni_seen = 0
    for uni in data:
        freq_uni_seen += uni_seen_freq_list[uni]
        if not uni in uni_seen_list:
            num_uni_unseen += 1
            if doUpdate:
                uni_seen_list.append(uni)
        if doUpdate:
            uni_seen_freq_list[uni] += 1            
    prop_uni_unseen = num_uni_unseen / num_uni # proportion of unseen unigram words

    mean_freq_uni = freq_uni_seen / num_uni
    # print (mean_freq_uni)

    # bigram process
    num_bi = len(data) - 1
    num_bi_unseen = 0
    freq_bi_seen = 0
    for i in range(num_bi):
        bi = list(data[i:i+2])
        freq_bi_seen += bi_seen_freq_list[bi[0], bi[1]]
        # print (freq_bi_seen)
        if not bi in bi_seen_list:
            num_bi_unseen += 1
            if doUpdate:
                bi_seen_list.append(bi)
        if doUpdate:
            bi_seen_freq_list[bi[0]][bi[1]] += 1
    prop_bi_unseen = num_bi_unseen / num_bi # proportion of unseen bigram words

    mean_freq_bi = freq_bi_seen / num_bi

    # trigram process
    num_tri = len(data) - 2
    num_tri_unseen = 0
    for i in range(num_tri):
        tri = list(data[i:i+3])
        if not tri in tri_seen_list:
            num_tri_unseen += 1
            if doUpdate:
                tri_seen_list.append(tri)
                # tri_seen_freq_list[tri[0]][tri[1]][tri[2]] += 1
    prop_tri_unseen = num_tri_unseen / num_tri # proportion of unseen trigram words

    # Frequency
    # print (np.sum(bi_seen_freq_list))


    # create tensor variable
    input_feature = torch.Tensor(np.array([prop_uni_unseen, prop_bi_unseen, prop_tri_unseen, mean_freq_uni, mean_freq_bi, math.log(i_num+1)]))
    input_feature = input_feature.view(-1, 6)

    return input_feature
    
# Reinforcement learning -------------------------------------------------------------------------------------------
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from word_lstm_model import MyLSTM
import torch.nn.functional as F

# Set up LSTM
n_letters = len(ptb_dict)
hidden_size_LSTM = 128
nlayers_LSTM = 2
hidden_dropout_prob_LSTM = 0.25
bidirectional_LSTM = False
batch_size_LSTM = 1
cuda_LSTM = True

# Set up DQN
input_dim = 6 # Three features(unigram, bigram, trigram)
output_dim = 1 # Q-Value (421 -> 4101)
hidden_size = 6 # Hidden Units
hidden_dropout_prob = 0.2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, output_dim) # input layer -> output layer
        self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        # self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
        self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        # x = self.fc1_drop(x)
        x = self.fc2(x)
        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)

# Train LSTM
import word_train_RL as w_t_RL

def Q_learning(replay_memory):
    for memory in replay_memory:
        state = memory[0]
        reward = memory[1]
        #next_state = memory[2]
        
        state_action_values = model(state)

        # Not Q Learning
        expected_state_action_values = torch.FloatTensor([reward])
        
        # Train Q Learning
        '''
        if isinstance(next_state, str):     # If terminal
            expected_state_action_values = Variable(torch.zeros(1))
            expected_state_action_values[0] = reward
        else:
            next_model_output = model(next_state).data            
            # Next state value
            next_state_action_value = Variable(torch.zeros(1))
            next_state_action_value[0] = next_model_output

            # Extract the value from the tensor
            # expected_state_action_values = gamma * next_state_action_value + reward
            expected_state_action_values = Variable(torch.FloatTensor([reward]))
        '''
        loss = F.smooth_l1_loss(state_action_values.view(1), expected_state_action_values) # Compute Huber loss

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Train DQN
budget = 0.25 * len(sentence_list) # Max. number of data that can be selected for language modeling
replay_memory = [] # Stores the transition(State, Action, Reward, Next State) for the Q-Learning
gamma = 0.8     # Discount factor
N_ep = 10       # Number of episodes
N_options = 5   # Number of options to choose from for training

# Initialize optimizer to update the DQN
optimizer = optim.RMSprop(model.parameters())

# Initialize LSTM model, allocate the cuda memory
# model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
# model_LSTM.cuda()

# data = datset[0]
# dataset_train = []

# model_LSTM, _, _ = w_t_RL.train(model_LSTM, dataset_train, 0) # train LSTM based on dataset_labelled
# loss_curr = w_t_RL.evaluate(model_LSTM, dataset_val, 0) # Evaluate previous loss

# exit()

# Loop over episodes
for i_ep in range(N_ep):    
    if i_ep > 0:
        # Load the new state dict of DQN model
        model.load_state_dict(torch.load('dqn_models/DQN_' + str(i_ep-1) + '.pt'))
        # Load the replay memory
        with open('dqn_models/replay_memory_' + str(i_ep-1), 'rb') as handle:
            replay_memory = pickle.load(handle)

    # Initialize LSTM model, allocate the cuda memory
    model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
    model_LSTM.cuda()

    dataset = select_batch(sentence_list) # Construct the batchified data from which training data will be selected
    dataset_train = [] # Stores batchified sentences selected for language modeling (training dat)
    
    uni_seen_list = [] # Initialize unigram seen list
    bi_seen_list = [] # Initialize bigram seen list
    tri_seen_list = [] # Initialize trigram seen list

    for i in range(len(dataset)//N_options):        # Loop through groups of N_options options
        
        state_value_list = [] # Initialize state value list
        data_list = [] # Initialize data list

        for j in range(N_options):  # Loop through N_options options
            data = dataset[i*N_options+j] # Select corresponding batch
            data_list.append(data)
            
            # Construct the state (how different our input is from the dataset_train, represented as scalar values) w/o updating seen lists
            state = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list, i, False)

            '''
            if j != N_options-1:
                # Construct the state (how different our input is from the dataset_train, represented as scalar values)
                state, _,_,_ = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            else:
                state, uni_seen_list, bi_seen_list, tri_seen_list = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            '''
            
            # Store each state value into state value list
            # for param in model.parameters():
                # print (param)
            
            model_output = model(state).data
            # print ("#####")
            # for param in model.parameters():
                # print (param)
            
            # print ("$$$$$")
            state_value_list.append(model_output[0][0])
            
        '''
        # Stores transitions into the replay memory
        if i != 0:
            replay_memory.append([state_prev, reward_prev, state])
        '''

        # choice = np.argmax(state_value_list) # Choose data with highest state value to train 
        choice = random.randint(0, N_options-1) # Choose data randomly

        # dataset_train.append(data_list[choice]) # Add selected data into train dataset
        dataset_train = [data_list[choice]] # Add selected data into train dataset

        # Update seen lists
        state = create_feature(data_list[choice], uni_seen_list, bi_seen_list, tri_seen_list, i, True)

        loss_prev = w_t_RL.evaluate(model_LSTM, dataset_val, i_ep) # Evaluate previous loss
        model_LSTM, loss_train, _ = w_t_RL.train(model_LSTM, dataset_train, i_ep) # train LSTM based on dataset_labelled
        loss_curr = w_t_RL.evaluate(model_LSTM, dataset_val, i_ep) # Evaluate current loss
        reward = loss_prev - loss_curr # Reward(Difference between previous loss and current loss)

        print ("#", i, ", loss_prev, loss_cur, reward :", loss_prev, loss_curr, reward)
        # print (loss_train)
        '''
        # Save replay memory with "terminal" state when dataset is exhausted
        if i == len(dataset)//N_options-1:
            replay_memory.append([state,reward,"terminal"])
            break;

        state_prev = state # Save previous state
        reward_prev = reward # Save previous reward
        '''
        
        # Save replay memory
        replay_memory.append([state, reward, loss_prev, loss_curr, loss_train])

        # Q-learning using replay memory
        if i % 100 == 0 and i != 0:
            Q_learning(replay_memory)
            with open('dqn_models/replay_memory_' + str(i_ep) + '_' + str(i), 'wb') as handle:
                pickle.dump(replay_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Save the state dict of DQN model
    torch.save(model.state_dict(), 'dqn_models/DQN_' + str(i_ep) + '.pt')
    # Save the replay memory
    with open('dqn_models/replay_memory_' + str(i_ep), 'wb') as handle:
    	pickle.dump(replay_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

