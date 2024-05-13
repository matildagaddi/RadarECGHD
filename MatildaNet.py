import torch 
import torch.nn as nn
from modwt import modwt

class MatildaNet(nn.Module):
    def __init__(self):
        super(MatildaNet, self).__init__()

        self.layers = nn.Sequential(
            #modwt should go here instead of first conv layer
            #modwtLayer('Level',5,'IncludeLowpass',false,'SelectedLevels',3:5,"Wavelet","sym2")
            #flattenLayer
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=64, stride=8, padding=0),
            nn.BatchNorm1d(120),
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
            
            nn.LSTM(input_size=8, hidden_size=1, num_layers=1, bidirectional=True)
            )

        self.classifier =nn.Sequential(

            nn.Linear(16, 8, bias=True, device=None, dtype=None),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4, bias=True, device=None, dtype=None),
            nn.Dropout(p=0.2),
            nn.Linear(4, 1, bias=True, device=None, dtype=None)            
            )
            
    def forward(self, x):
        x = self.layers(x)
        x = x[0].flatten()
        x = self.classifier(x)
        return x



x = torch.zeros(1,1024)
model = MatildaNet()
y = model(x)
print(y)