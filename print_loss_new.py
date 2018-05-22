import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

N_ep = 10 # Number of episodes
prop_test = 0.2 # Proportion of test set
shuffle = False # Shuffle data every episode
episode_rec = int(N_ep/10)

# Load data
#f = open('replay_memory_0','rb')
f = open('dqn_models/replay_memory_661_layer_1000_batch_10_trainiter','rb')
dataset = pickle.load(f)

# Data size
num_data = len(dataset)
# Features
X = torch.cat([x[0] for x in dataset]).float()
# Labels
y = torch.from_numpy(np.array([x[1] for x in dataset])).float()

input_dim = X.size(1)

# Initial shuffle
if shuffle:
    permutation = torch.randperm(num_data)
    X = X[permutation]
    y = y[permutation]

num_train = int(num_data*(1-prop_test))

X_train = X[:num_train]
y_train = y[:num_train]
X_test = X[num_train:]
y_test = y[num_train:]

# Normalize
y_train = y_train/abs(y_train).max()
y_test = y_test/abs(y_test).max()

# Network
class net(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_size=5, hidden_dropout_prob=0):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = net(input_dim=input_dim)


optimizer = optim.Adam(model.parameters())

training_loss = []
test_loss = []
# Training
for i_ep in range(N_ep):
    
    if shuffle:
        permutation = torch.randperm(num_train)
        X_train = X_train[permutation]
        y_train = y_train[permutation]
        
    for i_data in range(num_train):
        features = X_train[i_data]
        y_data = y_train[i_data]
        #features = X
        #y_data = y.view(-1,1)
        
        y_pred = model(features)
        
        #loss = F.smooth_l1_loss(y_pred,y_data)
        loss = torch.mean((y_pred - y_data)**2)
        # training_loss.append(loss)
      
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Compute training & test losses and record
    y_train_pred = model(X_train)
    loss_train = torch.mean((y_train-y_train_pred)**2)
    y_test_pred = model(X_test)
    loss_test = torch.mean((y_test-y_test_pred)**2)
    training_loss.append(loss_train)
    test_loss.append(loss_test)
        
    if i_ep != 0 and (i_ep % episode_rec == 0):
        print(i_ep)

plt.plot(training_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.savefig('train_loss_curve.png')
plt.show()

plt.plot(y)
plt.show()

plt.plot(model(X).detach().numpy())
plt.show()