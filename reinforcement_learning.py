import numpy as np
import torch
import torch.optim as optim

from extract_sentences import train, val, ptb_dict, words_num, extract_sentence_list

# extract sentence list
sentence_list = extract_sentence_list(train)

# select the batchified list
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
	num_bi = len(data) // 2
	num_bi_unseen = 0
	for i in range(num_bi):
		bi = list(data[i*2:i*2+2])
		if not bi in bi_seen_list:
			num_bi_unseen += 1
			bi_seen_list.append(bi)
	prop_bi_unseen = num_bi_unseen / num_bi # proportion of unseen bigram words

	# trigram process
	num_tri = len(data) // 3
	num_tri_unseen = 0
	for i in range(num_tri):
		tri = list(data[i*3:i*3+3])
		if not tri in tri_seen_list:
			num_tri_unseen += 1
			tri_seen_list.append(tri)
	prop_tri_unseen = num_tri_unseen / num_tri # proportion of unseen trigram words

	input_feature = torch.Tensor(np.array([prop_uni_unseen, prop_bi_unseen, prop_tri_unseen]))
	return input_feature, uni_seen_list, bi_seen_list, tri_seen_list
    
# Reinforcement learning -------------------------------------------------------------------------------------------
import torch.nn as nn
from torch.autograd import Variable
from word_lstm_model import MyLSTM
import torch.nn.functional as F
import copy

# Set up LSTM
n_letters = len(ptb_dict)
hidden_size_LSTM = 128
nlayers_LSTM = 2
hidden_dropout_prob_LSTM = 0.25
bidirectional_LSTM = False
batch_size_LSTM = 1
cuda_LSTM = True

# Set up DQN
input_dim = 3       # Three features(unigram, bigram, trigram)
output_dim = 2      # Either train or skip
hidden_size = 10
hidden_dropout_prob = 0.2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
        self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)
    
# Train LSTM
import word_train_RL as w_t_RL

def train_LSTM(dataset, model):
    w_t_RL.model = model
    w_t_RL.sentence_kept_list = dataset # dataset to be trained
    return w_t_RL.train(model, dataset)

def evaluate_LSTM(model):
	val_loss = w_t_RL.evaluate(model)
	
	return val_loss

# Train DQN
budget = 5000        # Max. number of data that can be selected for language modeling
dataset_train = []   # Stores batchified sentences selected for language modeling
replay_memory = []	 # Stores the transition(State, Action, Reward, Next State) for the Q-Learning

N_ep = 10  # Number of episodes

for i_ep in range(N_ep):
	# select the batchified data to be trained
    dataset = select_batch(sentence_list)

    # Initialize LSTM model, allocate the cuda memory
    model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
    model_LSTM.cuda()
    optimizer = optim.RMSprop(model.parameters())
    torch.save(model_LSTM.state_dict(), 'prev.pt')

    uni_seen_list = [] # Initialize uni unseen list
    bi_seen_list = [] # Initialize bi unseen list
    tri_seen_list = [] # Initialize tri unseen list

    idx = 0
    for data in dataset:
        # Construct the state(how different our input is from the the dataset train, represented as scalar value)
        state, uni_seen_list, bi_seen_list, tri_seen_list = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
        
        # create tensor variable
        state = Variable(state)
        state = state.view(1, 3)

        # Action selection, returns 0(skip) or 1(train)
        a = model(state).data.max(1)[1]

        # Extract the value from the tensor
        a = a[0, 0]
        
        # save the model
        torch.save(model_LSTM.state_dict(), 'prev.pt')

        if a == 1:
            dataset_train.append(data)
        
            # train LSTM based on dataset_labelled            
            model_LSTM = train_LSTM(dataset_train, model_LSTM)
           
        # Find difference in loss after training
        torch.save(model_LSTM.state_dict(), 'curr.pt')

        loss_curr = evaluate_LSTM(model_LSTM)
        
        model_LSTM.load_state_dict(torch.load('prev.pt'))
        loss_prev = evaluate_LSTM(model_LSTM)
        model_LSTM.load_state_dict(torch.load('curr.pt'))
        
        r = loss_prev - loss_curr # Reward
        print ("trainset data", data)
        print ("reward", loss_prev, loss_curr, r)
        if len(dataset_train) == budget:
            replay_memory.append([state,a,r])
            break;

        replay_memory.append([state,a,r])

        # Train Q Learning
        state_action_values = model(state).data[0, a]
        state_action_values = Variable(torch.Tensor([state_action_values]))

        data = dataset[idx+1]

        # Construct the state(how different our input is from the the dataset train, represented as scalar value)
        state, _, _, _ = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
        # create tensor variable
        state = Variable(state)
        state = state.view(1, 3)

        expected_state_action_values = model(state).data[0, a] + r
        expected_state_action_values = expected_state_action_values # needs to be the number
        # Compute Huber loss
        print (state_action_values)
        print (expected_state_action_values)
        print (type(state_action_values))
        print (type(expected_state_action_values))
        # exit()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # increase the index of the dataset
        idx += 1