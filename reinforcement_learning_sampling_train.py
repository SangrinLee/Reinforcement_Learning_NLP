def run(num_input, israndom):
	num = num_input
	import numpy as np
	import torch
	import torch.optim as optim
	import argparse
	import time
	import math

	#################################################
	# Hyper-parameters
	#################################################
	parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Character-Level LSTM Model')
	parser.add_argument('--nhid', type=int, default=128,
	                    help='number of hidden units per layer')
	parser.add_argument('--nlayers', type=int, default=2,
	                    help='number of layers')
	parser.add_argument('--lr', type=float, default=3,
	                    help='initial learning rate')
	parser.add_argument('--clip', type=float, default=0.25,
	                    help='gradient clipping')
	parser.add_argument('--load_epochs', type=int, default=0,
	                    help='load epoch')
	parser.add_argument('--epochs', type=int, default=15,
	                    help='upper epoch limit')
	parser.add_argument('--batch_size', type=int, default=1, metavar='N',
	                    help='batch size')
	parser.add_argument('--bptt', type=int, default=100,
	                    help='sequence length')
	parser.add_argument('--dropout', type=float, default=0.25,
	                    help='dropout applied to layers (0 = no dropout)')
	parser.add_argument('--cuda', action='store_true', default=True,
	                    help='use CUDA')
	parser.add_argument('--bidirectional', action='store_true', default=False,
	                    help='use Bi-LSTM')
	parser.add_argument('--serialize', action='store_true', default=False, #False,
	                    help='continue training a stored model')
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
	                    help='report interval')
	args = parser.parse_args()

	from extract_sentences import train, val, ptb_dict, words_num, extract_sentence_list
	    
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

	# Train LSTM
	import word_train_RL_check as w_t_RL

	def train_LSTM(dataset, w_t_model, epoch, num):
	    return w_t_RL.train(w_t_model, dataset, epoch, num)

	def evaluate_LSTM(w_t_model, num):
		return w_t_RL.evaluate(w_t_model, num)

	model_LSTM = MyLSTM(n_letters, hidden_size_LSTM, nlayers_LSTM, True, True, hidden_dropout_prob_LSTM, bidirectional_LSTM, batch_size_LSTM, cuda_LSTM)
	model_LSTM.cuda()

	# with open('sentence_selected', 'wb') as handle:
	    # pickle.dump(dataset_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

	if israndom:
		with open('sentence_selected_random_' + str(num),'rb') as b:
			dataset_train=pickle.load(b)    
	else:
		with open('sentence_selected_' + str(num),'rb') as b:
			dataset_train=pickle.load(b)

	print (len(dataset_train))
	# exit()

	# Training Part
	# At any point you can hit Ctrl + C to break out of training early.
	arr1 = []
	try:

	    for epoch in range(args.load_epochs+1, args.epochs+args.load_epochs+1):
	        epoch_start_time = time.time()
	        train_LSTM(dataset_train, model_LSTM, epoch, num)
	        
	        val_loss = evaluate_LSTM(model_LSTM, num)
	        print('-' * 89)
	        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'.format(
	            epoch, (time.time() - epoch_start_time),
	            val_loss))
	        print('-' * 89)

	except KeyboardInterrupt:
	   print('-' * 89)
	   print('Exiting from training early')


for i in range(1, 5):
	run(i, False)
	run(i, True)