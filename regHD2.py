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

DIMENSIONS = [10000]  
WINDOW_SIZE = [200] # Flavios: 256
# 200 pts = 1 sec, 1 pt = 0.005 sec, 1 pt = 5 milliseconds
NUM_FEATURES = WINDOW_SIZE 
BATCH_SIZE = 300
LEARN_RATE = [1e-6]
SAMPLE_RATE = [230]
FS = [600] #has to be bigger than cutoff*2 (to make between 0 and 1)
CUTOFF = 30
PEAK_THRESH = 0.3
TRAIN_ITERS = 5

### FOR FINAL EXPERIMENTS: ADD AS MANY FILES AS POSSIBLE 
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

pAbsErrs, qAbsErrs, rAbsErrs, sAbsErrs, tAbsErrs = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
allCorr = np.array([])
for subject in ['01', '02', '03', '04', '05']: #subject 06 has different indexing bc it's in the test folder
    train_files_r[:][20:22] = subject
    train_files_e[:][18:20] = subject
    test_files_r[:][20:22] = subject
    test_files_r[:][18:20] = subject

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
                        AAEs, medAEs, pAbsErr, qAbsErr, rAbsErr, sAbsErr, tAbsErr = get_AAEs_medAEs(labelsArr, predictionsArrFiltered, sr)
                        print(f'----- Subject {subject}, num train files: {len(train_files_r)}, num test files: {len(test_files_r)} -----')
                        print(f'AAEs (p,q,r,s,t):{AAEs}') 
                        print(f'MedAEs (p,q,r,s,t):{medAEs}') 
                        #replace filtered peaks with original peaks for higher correlation and better R peak height
                        predictionsArrFiltered[peaks] = predictionsArr[peaks]
                        corr = torch.corrcoef(torch.tensor(np.array([labelsArr.flatten(),predictionsArrFiltered])))[0][1]
                        print(f"Testing correlation of {corr}")
                        print(f"Testing mean squared error of {(mse.compute().item()):.20f}")

                        plt.figure(figsize=(10, 5))
                        plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
                        plt.plot(np.arange(len(predictionsArrFiltered)), predictionsArrFiltered, label='Predicted', color='red')
                        plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, FS:{fs}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}_date')
                        plt.annotate(f'AAEs (p,q,r,s,t):{AAEs} \n MedAEs (p,q,r,s,t):{medAEs} \n sample rate: {sr} \n correlation: {corr}',
                                    xy = (1.0, -0.2),
                                    xycoords='axes fraction',
                                    ha='right',
                                    va="center",
                                    fontsize=10,
                                    annotation_clip=False)
                        plt.tight_layout(h_pad=4)
                        plt.legend()
                        plt.savefig(f'HDC{(mse.compute().item()):.4f}_{corr:.4f}_{d}_{f}_{lr}_{fs}_{PEAK_THRESH}_{datetime.datetime.now()}.png') #need date so it doesn't rewrite previous file

                plt.clf()

    pAbsErrs = np.append(pAbsErrs, pAbsErr)
    qAbsErrs = np.append(qAbsErrs, qAbsErr)
    rAbsErrs = np.append(rAbsErrs, rAbsErr)
    sAbsErrs = np.append(sAbsErrs, sAbsErr)
    tAbsErrs = np.append(tAbsErrs, tAbsErr)
    allCorr = np.append(allCorr, corr)

overallMedAE = np.array([
    np.median(pAbsErrs),
    np.median(qAbsErrs),
    np.median(rAbsErrs),
    np.median(sAbsErrs),
    np.median(tAbsErrs)
])

overallMeanAE = np.array([
    np.mean(pAbsErrs),
    np.mean(qAbsErrs),
    np.mean(rAbsErrs),
    np.mean(sAbsErrs),
    np.mean(tAbsErrs)
])
print('-------------------------')
print(f'overallMedAE: {overallMedAE}')
print(f'median corr: {np.median(allCorr)}')
print(f'overallMeanAE: {overallMeanAE}')
print(f'mean corr: {np.mean(allCorr)}')