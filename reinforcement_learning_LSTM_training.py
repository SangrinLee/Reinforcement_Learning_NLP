import numpy as np
import time
import pickle
from word_lstm_model import MyLSTM
import word_train_RL as w_t_RL

from extract_sentences import train, val, ptb_dict, words_num, extract_sentence_list

# Set up LSTM
n_letters = len(ptb_dict)
hidden_size_LSTM = 128
nlayers_LSTM = 2
hidden_dropout_prob_LSTM = 0.25
bidirectional_LSTM = False
batch_size_LSTM = 1
cuda_LSTM = True
n_epoch = 10

dataset_val = val[len(val)//2:] # extract validation data
N_samples = 5 # Number of samples to extract

def run(sample_num, isRandom):
    # Initialize LSTM model, allocate the cuda memory
	model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
	model_LSTM.cuda()

	# Load the data based on selection(sampled data or random data)
	if not isRandom:
		with open('sampled_data/data_sampled_dqn_' + str(sample_num),'rb') as b:
			dataset_train=pickle.load(b)  
	else:
		with open('sampled_data/data_sampled_random_' + str(sample_num),'rb') as b:
			dataset_train=pickle.load(b)  

	# LSTM Training Part
	# At any point, you can hit Ctrl + C to break out of training early.
	try:
	    for epoch in range(n_epoch):
	        epoch_start_time = time.time()
	        
	        w_t_RL.train(model_LSTM, dataset_train, epoch) # Train LSTM based on dataset_labelled
	        val_loss = w_t_RL.evaluate(model_LSTM, dataset_val, epoch) # Evaluate current loss

	        print('-' * 89)
	        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'.format(
	            epoch, (time.time() - epoch_start_time),
	            val_loss))
	        print('-' * 89)

	except KeyboardInterrupt:
	   print('-' * 89)
	   print('Exiting from training early')


for sample in range(N_samples):
	run(sample, False) # Train the LSTM with data sampled from DQN
	run(sample, True) # Train the LSTM with data sampled randomly