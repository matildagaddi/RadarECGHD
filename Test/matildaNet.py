import torch
import torch.nn as nn
import numpy as np
import pywt
from scipy.ndimage import convolve1d

# def modwt(x, filters, level):
#     wavelet = pywt.Wavelet(filters)
#     h = wavelet.dec_hi
#     g = wavelet.dec_lo
#     h_t = np.array(h) / np.sqrt(2)
#     g_t = np.array(g) / np.sqrt(2)
#     wavecoeff = []
#     v_j_1 = x
#     for j in range(level):
#         w = circular_convolve_d(h_t, v_j_1, j + 1)
#         v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
#         wavecoeff.append(w)
#     wavecoeff.append(v_j_1)
#     return np.vstack(wavecoeff)

# def circular_convolve_d(h_t, v_j_1, j):
#     N = len(np.array([v_j_1]))
#     w_j = np.zeros(N)
#     ker = np.zeros(len(np.array([h_t])) * 2**(j - 1))
#     for i, h in enumerate(h_t):
#         ker[i * 2**(j - 1)] = h
#     w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
#     return w_j


def circular_convolve_d(h_t, v_j_1, j):
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2**(j - 1))
    for i, h in enumerate(h_t):
        ker[i * 2**(j - 1)] = h
    w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
    return w_j

def modwt(x, wavelet_name, level):
    wavelet = pywt.Wavelet(wavelet_name)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)

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

        self.classifier = nn.Sequential(
            nn.Linear(816, 8),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.Dropout(p=0.2),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.modwt_layer(x)  # Apply MODWT before unsqueeze
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def modwt_layer(self, x):
        wavelet = 'db2'
        level = 3
        x_np = x.detach().cpu().numpy()
        modwt_coeffs = np.array([modwt(sample, wavelet, level) for sample in x_np])
        modwt_coeffs = torch.tensor(modwt_coeffs, dtype=torch.float32).to(x.device)
        return modwt_coeffs

# Sample usage
# x = torch.randn(10, 1024)  # (batch_size, sequence_length)
# model = MatildaNet()
# y = model(x)
# print(y)
