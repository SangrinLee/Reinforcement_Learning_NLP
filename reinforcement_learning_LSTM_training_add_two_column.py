import numpy as np
import time
import pickle
import csv
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

def run(sample_num, dqn_model, isRandom):
    # Initialize LSTM model, allocate the cuda memory
	model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
	model_LSTM.cuda()

	# Load the data based on selection(sampled data or random data)
	if not isRandom:
		with open('sampled_data/data_sampled_' + str(dqn_model) + '_' + str(sample_num),'rb') as b:
			dataset_train = pickle.load(b)

		with open('sampled_data/value_sampled_' + str(dqn_model) + '_' + str(sample_num),'rb') as b:
			dataset_value_dqn = pickle.load(b)
		
		write_loss = open('sampled_data/data_sampled_dqn_loss_' + str(dqn_model) + '_' + str(sample_num) + '.csv','w', encoding='UTF-8', newline='')
		write_val_diff = open('sampled_data/val_sampled_dqn_diff_' + str(dqn_model) + '_' + str(sample_num) + '.csv','w', encoding='UTF-8', newline='')
	else:
		with open('sampled_data/data_sampled_random_' + str(dqn_model) + '_' + str(sample_num),'rb') as b:
			dataset_train = pickle.load(b)
		
		with open('sampled_data/value_sampled_random_' + str(dqn_model) + '_' + str(sample_num),'rb') as b:
			dataset_value_dqn = pickle.load(b)  
		
		write_loss = open('sampled_data/data_sampled_random_loss_' + str(dqn_model) + '_' + str(sample_num) + '.csv','w', encoding='UTF-8', newline='')
		write_val_diff = open('sampled_data/val_sampled_random_diff_' + str(dqn_model) + '_' + str(sample_num) + '.csv','w', encoding='UTF-8', newline='')

	writer = csv.DictWriter(write_loss, fieldnames=['Epoch', 'Train_loss', 'Train_ppl', 'Val_loss'])
	writer_value = csv.DictWriter(write_val_diff, fieldnames=['Epoch', 'Iteration', 'Reward', 'Value'])
	# LSTM Training Part
	# At any point, you can hit Ctrl + C to break out of training early.
	try:
		for epoch in range(1, n_epoch+1):
			print ("# Epoch", epoch)

			for i in range(len(dataset_train)):        # Loop through groups of N_options options
				loss_prev = w_t_RL.evaluate(model_LSTM, dataset_val, epoch) # Evaluate previous loss
				model_LSTM, train_loss, train_ppl = w_t_RL.train(model_LSTM, [dataset_train[i]], epoch) # Train LSTM based on dataset_labelled
				loss_curr = w_t_RL.evaluate(model_LSTM, dataset_val, epoch) # Evaluate current loss
				reward = loss_prev - loss_curr # Reward(Difference between previous loss and current loss)
				writer_value.writerow({'Epoch':str(epoch),'Iteration':str(i),'Reward':str(reward),'Value':str(dataset_value_dqn[i])})
				# print (reward, dataset_value_dqn[i])

			writer.writerow({'Epoch':str(epoch),'Train_loss':str(train_loss),'Train_ppl':str(train_ppl),'Val_loss':str(loss_curr)})

	except KeyboardInterrupt:
	   print('-' * 89)
	   print('Exiting from training early')

	# write_loss.close()
	write_loss.close()

N_samples = 1 # Number of samples to extract
for n_sample in range(N_samples):
	run(n_sample, 'DQN_0', False) # Train the LSTM with data sampled from DQN
	run(n_sample, 'DQN_1', False) # Train the LSTM with data sampled from DQN
	run(n_sample, 'DQN_2', False) # Train the LSTM with data sampled from DQN
	# run(n_sample, 'DQN_3', False) # Train the LSTM with data sampled from DQN
	run(n_sample, 'DQN_0', True) # Train the LSTM with data sampled randomly
	run(n_sample, 'DQN_1', True) # Train the LSTM with data sampled randomly
	run(n_sample, 'DQN_2', True) # Train the LSTM with data sampled randomly
	# run(n_sample, 'DQN_3', True) # Train the LSTM with data sampled randomly			
