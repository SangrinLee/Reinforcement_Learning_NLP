import torch
import torch.nn as nn
from torch.autograd import Variable


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, bias, batch_first, dropout, bidirectional, batch_size, cuda):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.layers = layers
        self.iscuda = cuda
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        # print ("input_dim : ", input_dim) # added
        # print ("hidden_dim : ", hidden_dim) # added
        self.encoder = nn.Embedding(input_dim, 200) # added

        self.lstm = nn.LSTM(200, hidden_dim, layers, bias, batch_first, dropout, bidirectional)
        self.hidden2category = nn.Linear(hidden_dim, input_dim)

    def init_hidden(self, length):
        if self.iscuda:
            return (Variable(torch.zeros(self.layers * self.direction, length, self.hidden_dim).cuda()),
                    Variable(torch.zeros(self.layers * self.direction, length, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.layers * self.direction, length, self.hidden_dim)),
                    Variable(torch.zeros(self.layers * self.direction, length, self.hidden_dim)))

    def forward(self, input_tensor, hidden): # input_tensor : 5*35*12043, hidden : 2*5*128
        input_tensor = input_tensor.long() # added
        
        # added
        # print (input_tensor)
        # print (hidden)
        # exit()
        input_tensor = self.encoder(input_tensor)
        # print (input_tensor)
        # exit()

        output_tensor, hidden = self.lstm(input_tensor, hidden)
        output_tensor = output_tensor.contiguous().view(output_tensor.size(0)*output_tensor.size(1), output_tensor.size(2))
        if self.direction == 2:
            output_tensor_forward, output_tensor_reverse = torch.chunk(output_tensor, 2, 1)
            output_tensor_forward = self.hidden2category(output_tensor_forward)
            output_tensor_reverse = self.hidden2category(output_tensor_reverse)
            return torch.cat((output_tensor_forward, output_tensor_reverse), 0), hidden
        else:
            output_tensor = self.hidden2category(output_tensor)
            return output_tensor, hidden
