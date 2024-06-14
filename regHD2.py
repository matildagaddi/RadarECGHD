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

DIMENSIONS = [10000]  
WINDOW_SIZE = [256] 
NUM_FEATURES = WINDOW_SIZE 
BATCH_SIZE = 300
LEARN_RATE = [1e-6]
TRAIN_ITERS = 50
train_files_r= ['trainVal/radar/GDN0004_Apnea_radar_1.mat',
                'trainVal/radar/GDN0004_Apnea_radar_2.mat',
                'trainVal/radar/GDN0004_Apnea_radar_3.mat',
                'trainVal/radar/GDN0004_Apnea_radar_4.mat',
                'trainVal/radar/GDN0004_Apnea_radar_5.mat',
                'trainVal/radar/GDN0004_Apnea_radar_6.mat',
                'trainVal/radar/GDN0004_Apnea_radar_7.mat',
                'trainVal/radar/GDN0004_Apnea_radar_8.mat',
                'trainVal/radar/GDN0004_Apnea_radar_9.mat',
                'trainVal/radar/GDN0004_Apnea_radar_10.mat',
                'trainVal/radar/GDN0004_Apnea_radar_11.mat',
                'trainVal/radar/GDN0004_Apnea_radar_12.mat',

               ]
train_files_e= ['trainVal/ecg/GDN0004_Apnea_ecg_1.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_2.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_3.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_4.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_5.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_6.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_7.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_8.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_9.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_10.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_11.mat',
                'trainVal/ecg/GDN0004_Apnea_ecg_12.mat',
               ]

test_files_r= ['trainVal/radar/GDN0004_Apnea_radar_13.mat',
               'trainVal/radar/GDN0004_Apnea_radar_14.mat',
               'trainVal/radar/GDN0004_Apnea_radar_15.mat',

              ]
test_files_e= ['trainVal/ecg/GDN0004_Apnea_ecg_13.mat',
               'trainVal/ecg/GDN0004_Apnea_ecg_14.mat',
               'trainVal/ecg/GDN0004_Apnea_ecg_15.mat',

              ]
path_to_DS = '/Users/matildagaddi/Documents/SEElab/DATASET/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)

for f in NUM_FEATURES:
  train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, window_size=f,device=device)
  test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e, window_size=f,device=device)
  DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
  train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
  test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
  for d in DIMENSIONS:
    for lr in LEARN_RATE:
      model = HDradarECG(f*channels, d, lr, device).to(device)
      with torch.no_grad():
          for _ in range(TRAIN_ITERS):
              for samples, label in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
                  samples = samples.to(device)
                  label = label.to(device)
                  model.model_update(samples, label)

      mse, labelsArr, predictionsArr = test(test_dl, model, TARGET_MIN, TARGET_MAX, True, device)
      predictionsArr = butter_lowpass_filter(predictionsArr, 30, 200, 5)
      
      print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
      plt.figure(figsize=(10, 5))
      plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
      plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted', color='red')
      plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, LR:{LEARN_RATE}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}')
      plt.legend()
      plt.savefig(f'HDC{(mse.compute().item()):.3f}_{d}_{f}_{lr}.png') 