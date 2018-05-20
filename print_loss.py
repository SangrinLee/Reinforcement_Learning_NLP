import pickle
import csv

# ep = 1300
# losses = []
# with open('dqn_models/losses_1_' + str(ep), 'rb') as handle:
#     losses = pickle.load(handle)


# # print (losses[0].data[0])
# # for i in range(len(losses)):
#     # print (losses[i].data[0].numpy()[0])
# print (len(losses))
# # exit()
write_loss = open('losses.csv','w', encoding='UTF-8', newline='')
writer = csv.DictWriter(write_loss, fieldnames=['Iteration', 'Loss'])

# for i in range(len(losses)):
#     print (losses[i].data[0].numpy())
#     writer.writerow({'Iteration':str(i),'Loss':str(losses[i].data[0].numpy())})
#     # print (losses[i].data[0])
#     # writer.writerow({'Iteration':str(i),'Loss':str(losses[i].data[0])})

# write_loss.close()




import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

N_ep = 1 # Number of episodes

# Load data
f = open('dqn_models/replay_memory_0','rb')
dataset = pickle.load(f)

# Data size
num_data = len(dataset)
# num_data = 500
# Features
X = torch.cat([x[0] for x in dataset]).float()
# Labels
y = torch.from_numpy(np.array([x[1] for x in dataset])).float()
# Val
z = torch.from_numpy(np.array([x[3] for x in dataset])).float()

# Network
class net(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_size=6, hidden_dropout_prob=0):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size) # input layer -> hidden layer
        self.fc2 = nn.Linear(hidden_size, output_dim) # hidden layer -> output layer
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = net()
# model.load_state_dict(torch.load('dqn_models/DQN_0.pt'))

#print(model.fc1.weight)
#print(model.fc2.weight)
sample_num=10

print(model(X[sample_num]))
print(y[sample_num])
print(model(X[sample_num])-y[sample_num])


optimizer = optim.Adam(model.parameters())

training_loss = []
val_loss = []
# Training
for i_ep in range(N_ep):
    for i_data in range(num_data):
        features = X[i_data]
        y_data = y[i_data]
        val_data = z[i_data]
        #features = X
        #y_data = y.view(-1,1)
        
        y_pred = model(features)
        
        #loss = F.smooth_l1_loss(y_pred,y_data)
        loss = torch.mean((y_pred - y_data)**2)
        training_loss.append(loss)
        val_loss.append(val_data)
        writer.writerow({'Iteration':str(i_data),'Loss':str(val_data.data[0].numpy())})
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

write_loss.close()
# plt.plot(training_loss)
# plt.savefig('train_loss_curve.png')
# plt.show()

plt.plot(val_loss)
plt.savefig('val_loss_curve.png')
plt.show()