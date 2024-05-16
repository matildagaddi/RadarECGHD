import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import time


from copy import deepcopy
def binarize(base_matrix):
   return np.where(base_matrix < 0, -1, 1)


def encoding_rp(X_data, base_matrix, quantize=False):
   enc_hv = []
   for i in range(len(X_data)):
       if i % int(len(X_data)/20) == 0:
           if log:
               sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
               sys.stdout.flush()
       hv = np.matmul(base_matrix, X_data[i])
       if quantize:
           hv = binarize(hv)
       enc_hv.append(hv)
   return enc_hv


class MatildaNet(nn.Module):
   def __init__(self):
       super(MatildaNet, self).__init__()


       self.layers = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=0),
           nn.Conv1d(in_channels=3, out_channels=8, kernel_size=64, stride=8, padding=0),
           nn.BatchNorm1d(8),
           nn.Tanh(),
           nn.MaxPool1d(2, stride=2),
           nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=0),
           nn.BatchNorm1d(8),
           nn.Tanh(),
           nn.MaxPool1d(2, stride=2),
           nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=0),
           nn.Tanh(),
           nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=0),
           nn.Tanh(),
       )


       self.lstm = nn.LSTM(input_size=8, hidden_size=1, num_layers=1, bidirectional=True)


       # Adjusted the input size of the linear layer to match the LSTM output size
       self.classifier = nn.Sequential(
           nn.Linear(816, 8),  # Changed input size to match the reshaped LSTM output
           nn.Dropout(p=0.2),
           nn.Linear(8, 4),
           nn.Dropout(p=0.2),
           nn.Linear(4, 1)           
       )
          
   def forward(self, x):
       # Reshape input to add a channel dimension
       x = x.unsqueeze(1)
       x = self.layers(x)
       x = x.permute(0, 2, 1)  # LSTM expects (batch_size, sequence_length, input_size)
      
       x, _ = self.lstm(x)
       x = x.view(x.size(0), -1)  # Flatten LSTM output
      
       x = self.classifier(x)
       return x
# x = torch.randn(10, 1024)
# model = MatildaNet()
# y = model(x)
#print(y)

