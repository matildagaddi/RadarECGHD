#regHD3

import math
import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from dataset3 import MyDataset
import matplotlib.pyplot as plt
import numpy as np
from MatildaNet import *
from utils import *
from scipy.signal import find_peaks
import datetime

DIMENSIONS = [10000]  
WINDOW_SIZE = [256] 
NUM_FEATURES = WINDOW_SIZE 
STEP = 10
BATCH_SIZE = 300
LEARN_RATE = [1e-6]
CUTOFF = [30]
FS = [600] #has to be bigger than cutoff*2 (to make between 0 and 1)
PEAK_THRESH = 0.3
TRAIN_ITERS = 40
train_subjects_sections = [[1, 0, 30000]] #[subject num, start, stop]
test_subjects_sections = [[1, 30000, 40000]] #[subject num, start, stop]
path_to_DS = '/Users/matildagaddi/Documents/SEElab/edanami_dataset/Data_set_of_radar_signal_(.csv)/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
#print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)

for f in NUM_FEATURES:
  train_ds = MyDataset(path=path_to_DS,subjects_sections=train_subjects_sections, window_size=f, step=STEP, device=device)
  test_ds = MyDataset(path=path_to_DS,subjects_sections=test_subjects_sections, window_size=f, step=STEP, device=device)
  DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
  train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
  test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
  for d in DIMENSIONS:
    for lr in LEARN_RATE: 
        for fs in FS:
            for cutoff in CUTOFF:
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
                predictionsArrFiltered = butter_lowpass_filter(predictionsArr, cutoff, fs, 5)
                #replace filtered peaks with original peaks
                predictionsArrFiltered[peaks] = predictionsArr[peaks]
                
                print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
                plt.figure(figsize=(10, 5))
                plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
                plt.plot(np.arange(len(predictionsArrFiltered)), predictionsArrFiltered, label='Predicted', color='red')
                plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, CO:{cutoff}, FS:{fs}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}')
                plt.legend()
                plt.savefig(f'ds3_{(mse.compute().item()):.3f}_{d}_{f}_{lr}_{fs}_{PEAK_THRESH}_{datetime.datetime.now()}.png') #need date so it doesn't rewrite previous file
                print('saved')
            plt.clf()