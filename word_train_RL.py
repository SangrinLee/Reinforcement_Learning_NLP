# -*- coding: utf-8 -*-
import argparse
import torch
import torch.cuda as cuda
from torch.autograd import Variable
from word_lstm_model import *
import time
import math
import numpy as np

#################################################
# Hyper-parameters
#################################################
parser = argparse.ArgumentParser(description='Word-Level LSTM Model')
parser.add_argument('--lr', type=float, default=3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
criterion.cuda()

########################################################
# Pre-process training and validation data
########################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0: nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = data.reshape(bsz, -1)
    return data

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
# you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, source.shape[1] - 1 - i) # -1 so that there's data for the target of the last time step
    data = source[:, i: i + seq_len] 
    source_target = source.astype(np.int64)
    target = source_target[:, i + 1: i + 1 + seq_len]

    # create tensor variable
    data = torch.from_numpy(data)
    data = Variable(data, volatile=evaluation)    # Saves memory in purely inference mode

    target = torch.from_numpy(target)
    target = Variable(target, volatile=evaluation)
    target = target.contiguous().view(-1)

    return data.cuda(), target.cuda()

def train(w_t_model, train_data_array, epoch):
    # Turn on training mode which enables dropout.
    # Built-in function, has effect on dropout and batchnorm
    w_t_model.train()

    total_loss = 0
    cur_loss = 0
    start_time = time.time()
    hidden = w_t_model.init_hidden(args.batch_size)

    for i in range(len(train_data_array)):
        train_data = train_data_array[i].reshape(1, -1)
        data, targets = get_batch(train_data, 1)
        hidden = w_t_model.init_hidden(args.batch_size)

        w_t_model.zero_grad()
        output, hidden = w_t_model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(w_t_model.parameters(), args.clip)
        for p in w_t_model.parameters():
            p.data.add_(-args.lr, p.grad.data)   # (scalar multiplier, other tensor)

        total_loss += loss.data
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            total_loss = 0

    return w_t_model, total_loss[0], math.exp(cur_loss)

# Uses training data to generate predictions, calculate loss based on validation/testing data
# Not using bptt
def evaluate(w_t_model, val_data_array, epoch):
    # Turn on evaluation mode which disables dropout.
    w_t_model.eval()

    val_bsz = 5
    val_data = batchify(val_data_array, val_bsz)
    total_loss = 0
    hidden = w_t_model.init_hidden(val_bsz)
    batch_length = val_data_array.size // val_bsz

    for batch, i in enumerate(range(1, val_data.shape[1] - 1, args.bptt)):
        data, targets = get_batch(val_data, i)
        hidden = w_t_model.init_hidden(val_bsz)
        output, hidden = w_t_model(data, hidden)
        loss = criterion(output, targets)
        total_loss += loss.data
    
    return total_loss[0] / (batch_length / args.bptt)
