import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://urldefense.com/v3/__https://torchmetrics.readthedocs.io__;!!Mih3wA!AmHSPDRz07C5JBLPFjfTuelm19_PToee5ShDXTdLZFglPtKHqnQlwOPgpLo0WaXRMJJjzmmQ6OhvZ9jj4Q$ 
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from dataset2 import MyDataset
import matplotlib.pyplot as plt
import numpy as np
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


### hyperparameter tuning ###
# train_iters = [100] #30, 50
# window_sizes = [400, 500, 600]
# lrs = [0.00003, 0.000035] # try [0.000005, 0.00001, 0.00002, 0.00003] 
#     #best to worst so far MSE, but first spikes not tall enough, 2nd spikes better but rest messy, 4th spikes too high
#     #want 1st one with spikes of 2nd one

# MSEs = []

# for ti in train_iters:
#     for ws in window_sizes:
#         print(MSEs)
#         for lr in lrs:

DIMENSIONS = 20000  # number of hypervector dimensions
WINDOW_SIZE = 400 
NUM_FEATURES = WINDOW_SIZE  
#WINDOW_SAMPLES = 1024 # points
BATCH_SIZE = 100
LEARN_RATE = 0.00005
#lower learn rate better, maybe use early stopping or fewer iterations, prevent overfitting?
TRAIN_ITERS = 100
train_files_r= ['trainVal/radar/GDN0001_Resting_radar_1.mat',
                'trainVal/radar/GDN0001_Resting_radar_2.mat',
                'trainVal/radar/GDN0001_Resting_radar_3.mat',
                'trainVal/radar/GDN0001_Resting_radar_4.mat',
                'trainVal/radar/GDN0001_Resting_radar_5.mat',
                'trainVal/radar/GDN0001_Resting_radar_6.mat',
                'trainVal/radar/GDN0001_Resting_radar_7.mat',
                'trainVal/radar/GDN0001_Resting_radar_8.mat',
                'trainVal/radar/GDN0001_Resting_radar_9.mat',
                'trainVal/radar/GDN0001_Resting_radar_10.mat',
                'trainVal/radar/GDN0001_Resting_radar_11.mat',
                'trainVal/radar/GDN0001_Resting_radar_12.mat',

               ]
train_files_e= ['trainVal/ecg/GDN0001_Resting_ecg_1.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_2.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_3.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_4.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_5.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_6.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_7.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_8.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_9.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_10.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_11.mat',
                'trainVal/ecg/GDN0001_Resting_ecg_12.mat',
               ]

test_files_r= ['trainVal/radar/GDN0001_Resting_radar_13.mat',
               'trainVal/radar/GDN0001_Resting_radar_14.mat',
               'trainVal/radar/GDN0001_Resting_radar_15.mat',

              ]
test_files_e= ['trainVal/ecg/GDN0001_Resting_ecg_13.mat',
               'trainVal/ecg/GDN0001_Resting_ecg_14.mat',
               'trainVal/ecg/GDN0001_Resting_ecg_15.mat',

              ]
path_to_DS = '/Users/matildagaddi/Documents/SEElab/DATASET/'

print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)

train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, 
    window_size=WINDOW_SIZE,device=device)
test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e,
    window_size=WINDOW_SIZE,device=device)

DATA_MEAN, DATA_STD, TARGET_MEAN, TARGET_STD = test_ds.get_params()


#maybe DataLoader has a sliding window option, but I'm not using it for now. Just using loop indexing later
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model based on RegHD application for Single model regression
class SingleModel(nn.Module):
    def __init__(self, num_classes, size, device):
        super(SingleModel, self).__init__()

        self.lr = LEARN_RATE
        self.M = torch.zeros(1, DIMENSIONS).to(device)
        self.project2 = embeddings.Sinusoid(size, DIMENSIONS)

    def encode(self, x):
        sample_hv = self.project2(x)
        return sample_hv
        # return torchhd.hard_quantize(sample_hv)

    def model_update(self, x, y):
        for x_sample, y_sample in zip(x,y):
            update = self.M + self.lr * (y_sample - (F.linear(x_sample, self.M))) * x_sample
            self.M = update

    def forward(self, x):
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res


model = SingleModel(1, NUM_FEATURES, device)
model = model.to(device)


# Model training
with torch.no_grad():
    for _ in range(TRAIN_ITERS):
        for samples, label in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):

            samples = samples.to(device)
            label = label.to(device)

            samples_hv = model.encode(samples)
            model.model_update(samples_hv, label)

train_time = 0 #set up later
# Model accuracy
mse = torchmetrics.MeanSquaredError()

samplesArr = np.array([])
labelsArr = np.array([])
predictionsArr = np.array([])

# Model testing 
with torch.no_grad():
    for samples, label in tqdm(test_dl, desc="Testing"):
        samples = samples.to(device)
        label = label.to(device)

        predictions = model(samples)
        predictions = predictions * TARGET_STD + TARGET_MEAN
        label = label * TARGET_STD + TARGET_MEAN
        label = torch.reshape(label, (samples.shape[0],1))
        mse.update(predictions.cpu(), label.cpu())
        samplesArr = np.append(samplesArr, samples.cpu())
        labelsArr = np.append(labelsArr, label.cpu())
        predictionsArr = np.append(predictionsArr, predictions.cpu())


print(f"Testing mean squared error of {(mse.compute().item()):.20f}")


plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted', color='red')
plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, LR:{LEARN_RATE}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}')
plt.legend()
plt.savefig(f'Flavio_MSE{(mse.compute().item()):.8f}_.png') #find how to save into folder
plt.show()
# plt.clf()