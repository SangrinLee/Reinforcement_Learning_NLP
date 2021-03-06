# -*- coding: utf-8 -*-
import argparse
import torch
import torch.cuda as cuda
from torch.autograd import Variable
from word_lstm_model import *
import time
import math
import string
import pickle
import numpy as np
import bisect
import word_corpus_data as data
import re
import operator
import chainer

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
parser.add_argument('--epochs', type=int, default=10,
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

#############################################
# Load data
#############################################
import extract_sentences as es

sentence_kept_list = []

train_data_array = es.train
val_data_array = es.val

print (len(train_data_array), len(val_data_array))

n_letters = len(es.ptb_word_id_dict)
n_categories = len(es.ptb_word_id_dict)
path = 'word_data'

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

# Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
# you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, source.shape[1] - 1 - i) # -1 so that there's data for the target of the last time step
    # print ("# get_batch : ", seq_len)
    source_target = source.astype(np.int64)

    if (seq_len <= 0):
        seq_len = 1
        i = 0
        target = source_target[:, i: i + seq_len]
    else:
        target = source_target[:, i + 1: i + 1 + seq_len]

    data = source[:, i: i + seq_len] 
    # source_target = source.astype(np.int64)
    # target = source_target[:, i + 1: i + 1 + seq_len]

    # initialize train_data_tensor, test_data_tensor
    data_embedding = np.zeros((data.shape[0], data.shape[1], n_letters), dtype = np.float32)

    # convert 2D numpy array to 3D numpy embedding
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            data_embedding[i][j][data[i][j]] = 1

    # create tensor variable
    data_embedding = torch.from_numpy(data_embedding)
    data_embedding = Variable(data_embedding, volatile=evaluation)    # Saves memory in purely inference mode

    target = torch.from_numpy(target)
    target = Variable(target, volatile=evaluation)
    if args.bidirectional:
        # r_target of length seq_len - 1
        r_source_target = np.flip(source_target[:, i - 1: i - 1 + seq_len].cpu().numpy(), 1).copy()
        target = torch.cat((Variable(source_target[:, i + 1: i + 1 + seq_len].contiguous().view(-1)),
                            Variable(torch.from_numpy(r_source_target).cuda().contiguous().view(-1))), 0)
    else:
        target = target.contiguous().view(-1)
    if args.cuda:
        return data_embedding.cuda(), target.cuda()
    else:
        return data_embedding, target

def embed(data_array, bsz):
    # convert 1D array to 2D
    data_array = batchify(data_array, bsz)

    # initialize train_data_tensor, test_data_tensor
    data_embedding = np.zeros((data_array.shape[0], data_array.shape[1], n_letters), dtype = np.float32)

    # convert 2D numpy array to np.int64
    data_array = data_array.astype(np.int64)

    # convert 2D numpy array to 3D numpy embedding
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1]):
            data_embedding[i][j][data_array[i][j]] = 1

    # convert 2D numpy array to 2D target tensor
    return data_embedding, data_array

def find_ge(a, x):
    i = bisect.bisect_left(a, x)
    return i

def find_le(a, x):
    i = bisect.bisect_right(a, x)
    return i - 1

###############################################################################
# Build the model
###############################################################################
if args.bidirectional:
    name = 'Bi-LSTM'
else:
    name = 'LSTM'

if args.serialize:
    with open(path + '/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
             name, args.load_epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'rb') as f:
        model = torch.load(f)
else:
    model = MyLSTM(n_letters, args.nhid, args.nlayers, True, True, args.dropout, args.bidirectional, args.batch_size, args.cuda)
    args.load_epochs = 0

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
NLL = nn.NLLLoss()

if args.cuda:
    criterion.cuda()
    softmax.cuda()
    NLL.cuda()

val_bsz = 5
# train_data = batchify(train_data_array, args.batch_size)
val_data = batchify(val_data_array, val_bsz)

def train(w_t_model, sentence_kept_list):
    # Turn on training mode which enables dropout.
    # Built-in function, has effect on dropout and batchnorm
    w_t_model.train()
    total_loss = 0
    start_time = time.time()
    hidden = w_t_model.init_hidden(args.batch_size)

    for i in range(len(sentence_kept_list)):
        train_data = sentence_kept_list[i].reshape(1, -1)

        data, targets = get_batch(train_data, 1)
        
        if not args.bidirectional:
            hidden = w_t_model.init_hidden(args.batch_size)
        else:
            hidden = repackage_hidden(hidden)
        w_t_model.zero_grad()

        output, hidden = w_t_model(data, hidden)

        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(w_t_model.parameters(), args.clip)
        for p in w_t_model.parameters():
            p.data.add_(-lr, p.grad.data)   # (scalar multiplier, other tensor)

        total_loss += loss.data
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:5.2f} |'.format(
                   epoch, i, train_data.shape[1] // args.bptt, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()

    return w_t_model

# Uses training data to generate predictions, calculate loss based on validation/testing data
# Not using bptt
def evaluate(w_t_model):
    # Turn on evaluation mode which disables dropout.
    w_t_model.eval()
    total_loss = 0
    hidden = w_t_model.init_hidden(val_bsz)
    start_time = time.time()

    batch_length = val_data_array.size // val_bsz
    for batch, i in enumerate(range(1, val_data.shape[1] - 1, args.bptt)):
        # returns Variables
        data, targets = get_batch(val_data, i)

        if not args.bidirectional:
            hidden = w_t_model.init_hidden(val_bsz)
        else:
            hidden = repackage_hidden(hidden)

        output, hidden = w_t_model(data, hidden)
        loss = criterion(output, targets)
        total_loss += loss.data

        if batch % (args.log_interval) == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| validation | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  .format(batch, val_data.shape[1] // args.bptt, lr,
                          elapsed * 1000 / (args.log_interval)))
            start_time = time.time()
        
    return total_loss[0] / (batch_length / args.bptt)

# Loop over epochs.
lr = args.lr
best_val_loss = None
