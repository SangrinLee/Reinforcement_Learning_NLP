import numpy as np
import torch
    
# Set up data ------------------------------------------------------------------------------------------------------
cutoff_percent = 0.5 # Threshold for rare words

from extract_sentences import train, val, ptb_dict, words_num, sentence_list, extract_frequencies

freq_list = extract_frequencies(train)
    
# id for <unk>
unk_idx = ptb_dict['<unk>']
# <unk> is rare by definition
freq_list[unk_idx] = 0

# Sort
idx_list_sorted = np.argsort(freq_list) # Low to high
freq_list_sorted = freq_list[idx_list_sorted]

# Get rare word list
idx_list_rare = idx_list_sorted[0:int(words_num*cutoff_percent)]    # Words corresponding to bottom 50% frequency
freq_list_rare = freq_list_sorted[0:int(words_num*cutoff_percent)]

# Convert sentence -> data (# of words, proportion of rare words)
dataset = []
for sentence in sentence_list:
    num_words = len(sentence)           # Number of words
    num_rare = 0                        # Number of rare words
        for word in sentence:           # Loop through every word
            if word in idx_list_rare:   # If word is rare
                num_rare += 1
                
    prop_rare = num_rare/num_words      # Proportion of rare words
    
    data = torch.Tensor(np.array([num_words,prop_rare]))
    dataset.append(data)
    
dataset_idx = np.arange(len(sentence_list))     # Data identification #
dataset_content = sentence_list                 # Actual sentence
dataset_input = dataset                         # Input structure
    
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
input_dim = 2       # Two features
output_dim = 2      # Either train or skip
hidden_size = 10
hidden_dropout_prob = 0.0

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc1_drop = nn.Dropout(p=hidden_dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2(x)

        return x
        
model = DQN(
    input_dim = input_dim, \
    output_dim = output_dim, \
    hidden_size = hidden_size, \
    hidden_dropout_prob = hidden_dropout_prob)
    
# Train LSTM
import word_train_RL as w_t_RL

def train_LSTM():
    w_t_RL.model = model_LSTM
    w_t_RL.sentence_kept_list = dataset_labelled
    w_t_RL.train()

# Train DQN
budget = np.inf         # Max. number of data that can be selected for annotation
dataset_labelled = []   # Stores sentences selected for annotation
replay_memory = []

N_ep = 10  # Number of episodes

for i_ep in range(N_ep):
    # Shuffle data
    np.random.shuffle(dataset_idx)
    
    # Initialize LSTM model
    model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
    model_LSTM.cuda()

    for i_data in dataset_idx:
        s = dataset_content[i_data]         # Data input for LSTM (sentence)
        x = dataset_input[i_data]           # Data input for DQN (representation)
        
        # Action selection
        a = model(state).data.max(1)
        
        if a==1:
            dataset_labelled.append(dataset_content[i_data])
            
            model_LSTM.save_state_dict('prev.pt')
            # train LSTM based on dataset_labelled
            train_LSTM()
           
        # Find difference in perplexity after training
        model_LSTM.save_state_dict('curr.pt')
        perp_curr = get_perplexity(model_LSTM, val)
        
        model_LSTM.load_state_dict(torch.load('prev.pt'))
        perp_prev = get_perplexity(model_LSTM, val)
        model_LSTM.load_state_dict(torch.load('curr.pt'))
        
        r = perp_prev - perp_curr # Reward
        
        if len(dataset_labelled) == budget:
            replay_memory.append([x,a,r])
            break;

        replay_memory.append([x,a,r])

        # Train DQN
        