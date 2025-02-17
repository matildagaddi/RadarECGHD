
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchhd
from torchhd import embeddings
import torchmetrics
from tqdm import tqdm
import numpy as np
import scipy
from scipy import signal
import neurokit2 as nk
import matplotlib.pyplot as plt

from timeit import default_timer as timer

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff * 2 / fs
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def train(train_loader, model, criterion, optimizer, scheduler, epochs, device): # For CNN only
    model.train()
    for epoch in range(epochs):
        for data, target in tqdm(train_loader, desc="Training {}".format(epoch)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # start = timer() #timer just to measure performance
            output = model(data).flatten()
            # end = timer()
            # print(end - start)
            # input()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            
def test(test_dl, model, TARGET_MIN, TARGET_MAX, HD, device):
    mse = torchmetrics.MeanSquaredError()
    samplesArr = np.array([])
    labelsArr = np.array([])
    predictionsArr = np.array([])
    with torch.no_grad():
        for samples, label in tqdm(test_dl, desc="Testing"):
            samples = samples.to(device)
            label = label.to(device)
            if(HD):
                predictions, _ = model(samples)
            else:
                predictions = model(samples)
            predictions = predictions * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
            label = label * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
            mse.update(predictions.flatten().cpu(), label.cpu())
            samplesArr = np.append(samplesArr, samples.cpu())
            labelsArr = np.append(labelsArr, label.cpu())
            predictionsArr = np.append(predictionsArr, predictions.cpu())
    return mse, labelsArr, predictionsArr


class HDradarECG(nn.Module):
    def __init__(self, feat, dim, lr, device):
        super(HDradarECG, self).__init__()
        self.lr = lr
        self.M = torch.zeros(1, dim).to(device)
        self.project2 = embeddings.Sinusoid(feat, dim)
        self.bias = nn.parameter.Parameter(torch.empty(dim), requires_grad=False).to(device)
        self.bias.data.uniform_(0, 2 * math.pi) # bias

    def encode(self, x):
        enc = self.project2(x)
        enc = torch.cos(enc + self.bias) * torch.sin(enc)
        # return enc
        return torchhd.hard_quantize(enc)

    def model_update(self, x, y):
        for x_sample, y_sample in zip(x,y):
            # start = timer()
            out, enc = self(x_sample, True)
            # end = timer()
            # print(end - start)
            # input()
            update = self.M + self.lr * (y_sample - out) * enc
            self.M = update

    def normalize(self, eps=1e-12):
        norms = self.M.norm(dim=0, keepdim=True)
        norms.clamp_(min=eps)
        self.M.div_(norms)

    def forward(self, x, train=False):
        if(train):
            x = x.flatten()
        else:
            x = x.view(x.size(0), -1)
        enc = self.encode(x)
        res = torch.mul(enc, self.M)
        res = torch.sum(res, dim=1)
        return res, enc



class FlavioNet(nn.Module): #to jointly learn regressor and projection matrix
    def __init__(self, feat, dim, device):
        super(FlavioNet, self).__init__()
        self.reg  = nn.parameter.Parameter(torch.rand(dim), requires_grad=True).to(device)
        self.proj = nn.Linear(feat, dim, bias=False, device=device)

    def forward(self, x, train=False):
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        out = self.proj(x)
        out = torch.sum(torch.mul(out, self.reg), dim=1)
        return out




def get_AAEs_medAEs(labelsArr, predictionsArrFiltered, sampleRate):
    msConversion = 5 # 200pts = 1000ms -> 1pt = 5ms: 5/200
    #maybe move below to utils
    ### actual ECG ###

    # Retrieve ECG data from data folder
    ecg_signal = labelsArr
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampleRate) #rate depends on heart rate, so we might need to fix
    rPeaksTrue = np.array(rpeaks['ECG_R_Peaks']) #indexed dictionary for array of peak indices
    # Plot the events using the events_plot function
    # nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    
    # P,Q,S,T peaks
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampleRate, method="peak")
    pPeaksTrue = np.array(waves_peak['ECG_P_Peaks'])
    qPeaksTrue = np.array(waves_peak['ECG_Q_Peaks'])
    sPeaksTrue = np.array(waves_peak['ECG_S_Peaks'])
    tPeaksTrue = np.array(waves_peak['ECG_T_Peaks'])
   
    nk.events_plot([waves_peak['ECG_P_Peaks'], 
                           waves_peak['ECG_Q_Peaks'],
                           rpeaks['ECG_R_Peaks'],
                           waves_peak['ECG_S_Peaks'],
                           waves_peak['ECG_T_Peaks']], ecg_signal)
    
    # Save the plot to a file
    plt.savefig('ecg_true_peaks.png')
    # Display the plot
    #plt.show()

    ### predicted ECG ###

    # Retrieve ECG data from data folder
    ecg_signal = predictionsArrFiltered
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampleRate)
    rPeaksPred = np.array(rpeaks['ECG_R_Peaks'])
    # Plot the events using the events_plot function
    # nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)

    # P,Q,S,T peaks
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampleRate, method="peak")
    pPeaksPred = np.array(waves_peak['ECG_P_Peaks'])
    qPeaksPred = np.array(waves_peak['ECG_Q_Peaks'])
    sPeaksPred = np.array(waves_peak['ECG_S_Peaks'])
    tPeaksPred = np.array(waves_peak['ECG_T_Peaks'])
    
    nk.events_plot([waves_peak['ECG_P_Peaks'], 
                           waves_peak['ECG_Q_Peaks'],
                           rpeaks['ECG_R_Peaks'],
                           waves_peak['ECG_S_Peaks'],
                           waves_peak['ECG_T_Peaks']], ecg_signal)
    
    # Save the plot to a file
    plt.savefig('ecg_pred_peaks.png')
    # Display the plot
    #plt.show()

    minPeaks = min(np.count_nonzero(~np.isnan(pPeaksTrue)), np.count_nonzero(~np.isnan(pPeaksPred)), # nans at the end causing problems
        np.count_nonzero(~np.isnan(qPeaksTrue)), np.count_nonzero(~np.isnan(qPeaksPred)),
        np.count_nonzero(~np.isnan(rPeaksTrue)), np.count_nonzero(~np.isnan(rPeaksPred)),
        np.count_nonzero(~np.isnan(sPeaksTrue)), np.count_nonzero(~np.isnan(sPeaksPred)),
        np.count_nonzero(~np.isnan(tPeaksTrue)), np.count_nonzero(~np.isnan(tPeaksPred)))
    
    pAbsErr = abs(pPeaksTrue[:minPeaks] - pPeaksPred[:minPeaks]) * msConversion
    print(pAbsErr)
    qAbsErr = abs(qPeaksTrue[:minPeaks] - qPeaksPred[:minPeaks]) * msConversion
    rAbsErr = abs(rPeaksTrue[:minPeaks] - rPeaksPred[:minPeaks]) * msConversion
    sAbsErr = abs(sPeaksTrue[:minPeaks] - sPeaksPred[:minPeaks]) * msConversion
    tAbsErr = abs(tPeaksTrue[:minPeaks] - tPeaksPred[:minPeaks]) * msConversion
    

    #AAE of peaks

    aAEs = np.array([np.mean(pAbsErr), np.mean(qAbsErr), np.mean(rAbsErr), np.mean(sAbsErr), np.mean(tAbsErr)])
    
    #median AE of peaks

    medAEs = np.array([np.median(pAbsErr), np.median(qAbsErr), np.median(rAbsErr), np.median(sAbsErr), np.median(tAbsErr)])
    
    return aAEs, medAEs, pAbsErr, qAbsErr, rAbsErr, sAbsErr, tAbsErr  #for overall median we need all the points for MedAE. if number of points is different per patient, we need weighted average for AAE










class Baseline7(nn.Module):
   def __init__(self):
       super(Baseline7, self).__init__()
       self.layers = nn.Sequential(
           nn.Conv1d(in_channels=2, out_channels=16, kernel_size=4, stride=1, padding=0),
           # nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1, padding=0),
           nn.BatchNorm1d(16),
           nn.ReLU(),
           nn.MaxPool1d(2, stride=2),
           nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.MaxPool1d(2, stride=2),
       )

       self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers=1, bidirectional=True)

       self.classifier = nn.Sequential(
           nn.Linear(24064, 1024),
           nn.Linear(1024, 1),    
       )
          
   def forward(self, x):
       # print(x.shape)
       # x = x.unsqueeze(1)
       # print(x.shape)
       x = self.layers(x)
       # print(x.shape)
       # input()
       x = x.permute(0, 2, 1)  # LSTM expects (batch_size, sequence_length, input_size)
       x, _ = self.lstm(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       return x




class Baseline8(nn.Module):
   def __init__(self):
       super(Baseline8, self).__init__()
       self.layersA = nn.Sequential(
           nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),

           nn.MaxPool1d(2, stride=2),
       )

       self.layersB = nn.Sequential(
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),
           nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
           nn.BatchNorm1d(32),
           nn.ReLU(),

           nn.MaxPool1d(2, stride=2),
       )

       self.lstm = nn.LSTM(input_size=8, hidden_size=256, num_layers=3, bidirectional=True)

       self.classifier = nn.Sequential(
           nn.Linear(19456, 1) 
       )
          
   def forward(self, x):
       # print(x.shape)
       # x = x.unsqueeze(1)
       # print(x.shape)
       x = self.layersA(x)
       x = self.layersB(x)
       # print(x.shape)
       # input()
       x = x.permute(0, 2, 1)  # LSTM expects (batch_size, sequence_length, input_size)
      
       x, _ = self.lstm(x)
       x = x.view(x.size(0), -1) 
      
       x = self.classifier(x)
       return x