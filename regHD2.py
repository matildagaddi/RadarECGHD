import math
import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from dataset2 import MyDataset
import matplotlib.pyplot as plt
import numpy as np
from MatildaNet import *
from utils import *
from scipy.signal import find_peaks
import datetime

#for finding peaks
import neurokit2 as nk
import matplotlib.pyplot as plt

DIMENSIONS = [10000]  
WINDOW_SIZE = [200] # Flavios: 256 (NeuroKit sample rate: 170, so 170:256 ~= 200:133) #but its still working with 170
# 200 pts = 1 sec, 1 pt = 0.005 sec, 1 pt = 5 milliseconds
NUM_FEATURES = WINDOW_SIZE 
BATCH_SIZE = 300
LEARN_RATE = [1e-6]
SAMPLE_RATE = [230] #does it have to do with accuracy?
# [36.25       28.75        9.58333333 36.25       17.08333333] AAE p q r s t, 5 iterations, found peaks need to line up
# [34.16666667 33.33333333  6.66666667 50.41666667  4.58333333]  (sr = 100)
# [54.58333333 32.5         9.16666667 52.5         5.41666667] (sr = 130)
# [37.5        34.16666667  6.66666667 51.66666667  6.66666667] (sr = 200) but 170 detected too many
# [27.5        42.08333333  7.5        48.33333333 13.33333333] (sr = 230) 

FS = [600] #has to be bigger than cutoff*2 (to make between 0 and 1)
CUTOFF = 30
PEAK_THRESH = 0.3
TRAIN_ITERS = 50
train_files_r= ['trainVal/radar/GDN0004_Resting_radar_1.mat',
                'trainVal/radar/GDN0004_Resting_radar_2.mat',
                'trainVal/radar/GDN0004_Resting_radar_3.mat',
                'trainVal/radar/GDN0004_Resting_radar_4.mat',
                'trainVal/radar/GDN0004_Resting_radar_5.mat',
                'trainVal/radar/GDN0004_Resting_radar_6.mat',
                'trainVal/radar/GDN0004_Resting_radar_7.mat',
                'trainVal/radar/GDN0004_Resting_radar_8.mat',
                'trainVal/radar/GDN0004_Resting_radar_9.mat',
                'trainVal/radar/GDN0004_Resting_radar_10.mat',
                'trainVal/radar/GDN0004_Resting_radar_11.mat',
                'trainVal/radar/GDN0004_Resting_radar_12.mat',

               ]
train_files_e= ['trainVal/ecg/GDN0004_Resting_ecg_1.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_2.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_3.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_4.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_5.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_6.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_7.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_8.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_9.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_10.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_11.mat',
                'trainVal/ecg/GDN0004_Resting_ecg_12.mat',
               ]

test_files_r= ['trainVal/radar/GDN0004_Resting_radar_13.mat',
               'trainVal/radar/GDN0004_Resting_radar_14.mat',
               'trainVal/radar/GDN0004_Resting_radar_15.mat',

              ]
test_files_e= ['trainVal/ecg/GDN0004_Resting_ecg_13.mat',
               'trainVal/ecg/GDN0004_Resting_ecg_14.mat',
               'trainVal/ecg/GDN0004_Resting_ecg_15.mat',

              ]
path_to_DS = '/Users/matildagaddi/Documents/SEElab/DATASET/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
#print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)

for f in NUM_FEATURES:
  train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, window_size=f,device=device)
  test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e, window_size=f,device=device)
  DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
  train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
  test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
  for d in DIMENSIONS:
    for lr in LEARN_RATE: 
        for fs in FS:
            for sr in SAMPLE_RATE:
                model = HDradarECG(f*channels, d, lr, device).to(device)
                with torch.no_grad():
                    for _ in range(TRAIN_ITERS):
                        for samples, label in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
                            samples = samples.to(device)
                            label = label.to(device)
                            model.model_update(samples, label)
          
                mse, labelsArr, predictionsArr = test(test_dl, model, TARGET_MIN, TARGET_MAX, True, device)
                
                #preserve peaks
                peaks, _ = find_peaks(predictionsArr, height=PEAK_THRESH)
                predictionsArrFiltered = butter_lowpass_filter(predictionsArr, CUTOFF, fs, 5)
                

                #maybe move below to utils
                ### actual ECG ###

                # Retrieve ECG data from data folder
                ecg_signal = labelsArr
                # Extract R-peaks locations
                print(f'sample_rate: {sr}')
                _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sr) #rate depends on heart rate, so we might need to fix
                rPeaksTrue = np.array(rpeaks['ECG_R_Peaks']) #indexed dictionary for array of peak indices
                # Plot the events using the events_plot function
                # nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
                
                # P,Q,S,T peaks
                _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sr, method="peak")
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
                plt.show()

                ### predicted ECG ###

                # Retrieve ECG data from data folder
                ecg_signal = predictionsArrFiltered
                # Extract R-peaks locations
                _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sr)
                rPeaksPred = np.array(rpeaks['ECG_R_Peaks'])
                # Plot the events using the events_plot function
                # nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)

                # P,Q,S,T peaks
                _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sr, method="peak")
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
                plt.show()

                #AAE of peaks
                minPeaks = min(np.count_nonzero(~np.isnan(pPeaksTrue)), np.count_nonzero(~np.isnan(pPeaksPred)), # nans at the end causing problems
                    np.count_nonzero(~np.isnan(qPeaksTrue)), np.count_nonzero(~np.isnan(qPeaksPred)),
                    np.count_nonzero(~np.isnan(rPeaksTrue)), np.count_nonzero(~np.isnan(rPeaksPred)),
                    np.count_nonzero(~np.isnan(sPeaksTrue)), np.count_nonzero(~np.isnan(sPeaksPred)),
                    np.count_nonzero(~np.isnan(tPeaksTrue)), np.count_nonzero(~np.isnan(tPeaksPred)))
                pAAE = (np.sum(abs(pPeaksTrue[:minPeaks] - pPeaksPred[:minPeaks])))/minPeaks
                #print(np.sum(abs(pPeaksTrue[:minPeaks] - pPeaksPred[:minPeaks])), minPeaks, pPeaksTrue[:minPeaks], pPeaksPred[:minPeaks])
                qAAE = (np.sum(abs(qPeaksTrue[:minPeaks] - qPeaksPred[:minPeaks])))/minPeaks
                rAAE = (np.sum(abs(rPeaksTrue[:minPeaks] - rPeaksPred[:minPeaks])))/minPeaks
                sAAE = (np.sum(abs(sPeaksTrue[:minPeaks] - sPeaksPred[:minPeaks])))/minPeaks
                tAAE = (np.sum(abs(tPeaksTrue[:minPeaks] - tPeaksPred[:minPeaks])))/minPeaks
                # in index units, need to convert to time, keep track of sampling rates
                # Contactless and ECG both report ms

                AAEs = np.array([pAAE, qAAE, rAAE, sAAE, tAAE])*5 #convert to ms
                print(f'AAEs (p,q,r,s,t):{AAEs}') 
                #replace filtered peaks with original peaks
                predictionsArrFiltered[peaks] = predictionsArr[peaks]
                corr = torch.corrcoef(torch.tensor([labelsArr.flatten(),predictionsArrFiltered]))[0][1]
                print(f"Testing correlation of {corr}")
                print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
                plt.figure(figsize=(10, 5))
                plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
                plt.plot(np.arange(len(predictionsArrFiltered)), predictionsArrFiltered, label='Predicted', color='red')
                plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, FS:{fs}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}_date')
                plt.annotate(f'AAEs (p,q,r,s,t):{AAEs} \n sample rate: {sr} \n correlation: {corr}',
                            xy = (1.0, -0.2),
                            xycoords='axes fraction',
                            ha='right',
                            va="center",
                            fontsize=10)
                plt.legend()
                plt.savefig(f'HDC{(mse.compute().item()):.4f}_{corr:.4f}_{d}_{f}_{lr}_{fs}_{PEAK_THRESH}_{datetime.datetime.now()}.png') #need date so it doesn't rewrite previous file
                print('saved')
            plt.clf()