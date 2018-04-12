def run_random(i):
    import numpy as np
    import torch
    import torch.optim as optim
    import random

    from extract_sentences import train, val, ptb_dict, words_num, extract_sentence_list

    # extract sentence list
    sentence_list = extract_sentence_list(train)
    sentence_list = sentence_list[len(sentence_list)//2:]

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
        
    # Reinforcement learning -------------------------------------------------------------------------------------------
    import torch.nn as nn
    from torch.autograd import Variable
    from word_lstm_model_check import MyLSTM
    import torch.nn.functional as F
    import copy
    import pickle

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
    output_dim = 1      # Either train or skip
    hidden_size = 2
    hidden_dropout_prob = 0.2

    class DQN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(input_dim, output_dim) # input layer -> hidden layer
            # self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
            # self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
            
        def forward(self, x):
            x = F.sigmoid(self.fc1(x))
            # print ("#1", x)
            # x = self.fc1_drop(x)
            # x = self.fc2(x)
            # print ("#2", x)

            return x
            
    model = DQN(
        input_dim = input_dim, \
        output_dim = output_dim, \
        hidden_size = hidden_size, \
        hidden_dropout_prob = hidden_dropout_prob)
      
    # Train DQN
    budget = 0.25 * len(sentence_list)        # Max. number of data that can be selected for language modeling
    dataset_train = []   # Stores batchified sentences selected for language modeling
    replay_memory = []   # Stores the transition(State, Action, Reward, Next State) for the Q-Learning
    gamma = 0.8
    N_ep = 10       # Number of episodes
    N_options = 5   # Number of options to choose from for training

    # Load pretraiend DQN model
    print ("# Load pretrained DQN model")
    model.load_state_dict(torch.load('DQN.pt'))

    sent_selected = []

    # select the batchified data to be trained
    dataset = select_batch(sentence_list)

    # Initialize LSTM model, allocate the cuda memory
    model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
    model_LSTM.cuda()

    optimizer = optim.RMSprop(model.parameters()) # deleted
    torch.save(model_LSTM.state_dict(), 'prev.pt')

    uni_seen_list = [] # Initialize unigram unseen list
    bi_seen_list = [] # Initialize bigram unseen list
    tri_seen_list = [] # Initialize trigram unseen list

    idx = 0
    # Total length of sentences = 2958
    for j in range(len(dataset)//N_options):
        print (j)

        data_list = []
        state_value_list = []

        for k in range(N_options):
            data = dataset[j*N_options+k]
            data_list.append(data)
            if k!=N_options-1:
                # Construct the state(how different our input is from the the dataset train, represented as scalar value)
                state, _,_,_ = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            else:
                state, uni_seen_list, bi_seen_list, tri_seen_list = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)

            # State value
            model_output = model(state).data
            state_value_list.append(model_output[0][0])

        choice = random.randint(0, 4)

        dataset_train.append(data_list[choice])

    with open('sentence_selected_random_' + str(i), 'wb') as handle:
        pickle.dump(dataset_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sentence_selected_random_' + str(i),'rb') as b:
        sentence_selected=pickle.load(b)    



def run(i):
    import numpy as np
    import torch
    import torch.optim as optim

    from extract_sentences import train, val, ptb_dict, words_num, extract_sentence_list

    # extract sentence list
    sentence_list = extract_sentence_list(train)
    sentence_list = sentence_list[len(sentence_list)//2:]
    
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
        
    # Reinforcement learning -------------------------------------------------------------------------------------------
    import torch.nn as nn
    from torch.autograd import Variable
    from word_lstm_model_check import MyLSTM
    import torch.nn.functional as F
    import copy
    import pickle

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
    output_dim = 1      # Either train or skip
    hidden_size = 2
    hidden_dropout_prob = 0.2

    class DQN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size=400, hidden_dropout_prob=0.2):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(input_dim, output_dim) # input layer -> hidden layer
            # self.fc1_drop = nn.Dropout(p=hidden_dropout_prob) # set the dropout
            # self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
            
        def forward(self, x):
            x = F.sigmoid(self.fc1(x))
            # print ("#1", x)
            # x = self.fc1_drop(x)
            # x = self.fc2(x)
            # print ("#2", x)

            return x
            
    model = DQN(
        input_dim = input_dim, \
        output_dim = output_dim, \
        hidden_size = hidden_size, \
        hidden_dropout_prob = hidden_dropout_prob)

    # Train DQN
    budget = 0.25 * len(sentence_list)        # Max. number of data that can be selected for language modeling
    dataset_train = []   # Stores batchified sentences selected for language modeling
    replay_memory = []	 # Stores the transition(State, Action, Reward, Next State) for the Q-Learning
    gamma = 0.8
    N_ep = 10       # Number of episodes
    N_options = 5   # Number of options to choose from for training

    # Load pretraiend DQN model
    print ("# Load pretrained DQN model")
    pretrained_dict = torch.load('DQN.pt')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    # model.load_state_dict(torch.load('DQN.pt'))

    sent_selected = []

    # select the batchified data to be trained
    dataset = select_batch(sentence_list)

    # Initialize LSTM model, allocate the cuda memory
    model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
    model_LSTM.cuda()

    optimizer = optim.RMSprop(model.parameters()) # deleted
    torch.save(model_LSTM.state_dict(), 'prev.pt')

    uni_seen_list = [] # Initialize unigram unseen list
    bi_seen_list = [] # Initialize bigram unseen list
    tri_seen_list = [] # Initialize trigram unseen list

    idx = 0
    # Total length of sentences = 2958
    for j in range(len(dataset)//N_options):
        print (j)

        data_list = []
        state_value_list = []

        for k in range(N_options):
            data = dataset[j*N_options+k]
            data_list.append(data)
            if k!=N_options-1:
                # Construct the state(how different our input is from the the dataset train, represented as scalar value)
                state, _,_,_ = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)
            else:
                state, uni_seen_list, bi_seen_list, tri_seen_list = create_feature(data, uni_seen_list, bi_seen_list, tri_seen_list)

            # State value
            model_output = model(state).data
            state_value_list.append(model_output[0][0])

        choice = np.argmin(state_value_list)

        dataset_train.append(data_list[choice])

    with open('sentence_selected_' + str(i), 'wb') as handle:
        pickle.dump(dataset_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sentence_selected_' + str(i),'rb') as b:
        sentence_selected=pickle.load(b)    

for i in range(1, 5):
    print ("#####", i)
    run(i)

for i in range(5):
    print ("#####", i)
    run_random(i)

# from reinforcement_learning_sampling_train import *