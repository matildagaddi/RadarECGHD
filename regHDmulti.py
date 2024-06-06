## regHD2 multifile training

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from dataset2multi import MyDataset
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

DIMENSIONS = 10000  # number of hypervector dimensions
WINDOW_SIZE = 400
NUM_FEATURES = WINDOW_SIZE  
#WINDOW_SAMPLES = 1024 # points
BATCH_SIZE = 20
LEARN_RATE = 0.00003 
#lower learn rate better, maybe use early stopping or fewer iterations, prevent overfitting?
TRAIN_ITERS = 10
train_files_r= ['trainVal/radar/GDN0001_Resting_radar_1.mat','trainVal/radar/GDN0001_Resting_radar_2.mat','trainVal/radar/GDN0001_Resting_radar_3.mat']
train_files_e= ['trainVal/ecg/GDN0001_Resting_ecg_1.mat','trainVal/ecg/GDN0001_Resting_ecg_2.mat','trainVal/ecg/GDN0001_Resting_ecg_3.mat']
test_files_r= ['trainVal/radar/GDN0001_Resting_radar_4.mat']
test_files_e= ['trainVal/ecg/GDN0001_Resting_ecg_4.mat']
path_to_DS = '/Users/matildagaddi/Documents/SEElab/DATASET/'

print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)


train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, 
	window_size=WINDOW_SIZE,device=device)
test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e,
	window_size=WINDOW_SIZE,device=device)
# Get necessary statistics for data and target transform
STD_DEVS = train_ds.data.std(0)
MEANS = train_ds.data.mean(0)
TARGET_STD = train_ds.target.std(0)
TARGET_MEAN = train_ds.target.mean(0)

# def transform(x):
#     x = x - MEANS
#     x = x / STD_DEVS
#     return x


# def target_transform(x):
#     x = x - TARGET_MEAN
#     x = x / TARGET_STD
#     return x


# train_ds.transform = transform
# train_ds.target_transform = target_transform

# test_ds.transform = transform
# test_ds.target_transform = target_transform

#maybe DataLoader has a sliding window option, but I'm not using it for now. Just using loop indexing later
train_dl = data.DataLoader(train_ds, batch_size=1) 
test_dl = data.DataLoader(test_ds, batch_size=1)


# Model based on RegHD application for Single model regression
class SingleModel(nn.Module):
    def __init__(self, num_classes, size):
        super(SingleModel, self).__init__()

        self.lr = LEARN_RATE
        self.M = torch.zeros(1, DIMENSIONS)
        self.project2 = embeddings.Sinusoid(size, DIMENSIONS)

    def encode(self, x):
        sample_hv = self.project2(x)
        return torchhd.hard_quantize(sample_hv)

    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0)
        self.M = update

    def forward(self, x):
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res


model = SingleModel(1, NUM_FEATURES)
model = model.to(device)

trainSamplesArr = np.array([])
# Model training
with torch.no_grad():
    for _ in range(TRAIN_ITERS):
        for file in tqdm(range(len(train_ds.data)), desc="Iteration {}".format(_ + 1)):
            for i in range(len(train_ds.data[file])-WINDOW_SIZE): ##check why its slower now
                #sliding window
                samples = train_ds.data[file][i:i+WINDOW_SIZE]
                label = train_ds.target[file][i+WINDOW_SIZE]
                #print(label) #tensor(-0.0559)
    
                samples = samples.to(device)
                label = label.to(device)
    
                samples_hv = model.encode(samples)
                model.model_update(samples_hv, label)
    
                trainSamplesArr = np.append(trainSamplesArr, samples)

train_time = 0 #set up later

# Model accuracy
mse = torchmetrics.MeanSquaredError()

samplesArr = np.array([])
labelsArr = np.array([])
predictionsArr = np.array([])

# Model testing 
with torch.no_grad():
	for file in tqdm(range(len(test_ds.data)), desc="Testing"):
	    for i in range(len(test_ds.data[file])-WINDOW_SIZE):
	        samples = test_ds.data[file][i:i+WINDOW_SIZE] #####test
	        label = test_ds.target[file][i+WINDOW_SIZE] #####test
	        #print(label) #tensor(0.0491)

	        samples = samples.to(device)

	        predictions = model(samples)
	        #predictions = predictions * TARGET_STD + TARGET_MEAN
	        #label = label * TARGET_STD + TARGET_MEAN # this it turns into 1024 dimensions
	        label = torch.reshape(label, (1,)) 

	        mse.update(predictions.cpu(), label)

	        samplesArr = np.append(samplesArr, samples)
	        labelsArr = np.append(labelsArr, label)
	        predictionsArr = np.append(predictionsArr, predictions)


print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
### MSEs.append([f'{(mse.compute().item()):.10f}', lr, ws, ti])
#print(len(samplesArr), len(labelsArr), len(predictionsArr))
# plt.plot(np.arange(len(samplesArr.flatten())), samplesArr.flatten(), label='test radar', alpha=0.5)
# plt.plot(np.arange(len(trainSamplesArr.flatten())), trainSamplesArr.flatten(), label='train radar', alpha=0.5)
# plt.title('radar data')
# plt.legend()
# plt.show()

## ALSO PLOT TRAINING ECGS TO COMPARE FOR OVERFITTING / HOW IT REALLY LEARNS
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted', color='red')
plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, LR:{LEARN_RATE}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}, {test_files_r}')
plt.legend()
plt.savefig(f'regMulti_MSE{(mse.compute().item()):.8f}_{datetime.datetime.now()}.png') #find how to save into folder
plt.clf()
