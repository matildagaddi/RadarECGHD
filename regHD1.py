
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
from dataset1_1 import MyDataset
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

SEED = 0
DIMENSIONS = 10000  # number of hypervector dimensions
WINDOW_SIZE = 400
NUM_FEATURES = WINDOW_SIZE  
STEP_SIZE = 10
BATCH_SIZE = 20
LEARN_RATE = 0.00003 
#lower learn rate better, maybe use early stopping or fewer iterations, prevent overfitting?
TRAIN_ITERS = 10
train_file_r= 'trainVal/radar/GDN0001_Resting_radar_1.mat'
train_file_e= 'trainVal/ecg/GDN0001_Resting_ecg_1.mat'
test_file_r= 'trainVal/radar/GDN0001_Resting_radar_2.mat'
test_file_e= 'trainVal/ecg/GDN0001_Resting_ecg_2.mat'
path_to_DS = '/Users/matildagaddi/Documents/SEElab'

print('hyperparameters: ti: ', TRAIN_ITERS, ' ws: ', WINDOW_SIZE, ' lr: ', LEARN_RATE)

train_ds = MyDataset(path=f"{path_to_DS}/apnea_dataset/", train=True, window_size=WINDOW_SIZE, 
    step_size=STEP_SIZE, device=device, seed=SEED)
#train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE)

test_ds = MyDataset(path=f"{path_to_DS}/apnea_dataset/", train=False, window_size=WINDOW_SIZE, 
    step_size=STEP_SIZE, device=device, seed=SEED)
#test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE)


# Get necessary statistics for data and target transform
STD_DEVS = train_ds.data_i.std(0)
MEANS = train_ds.data_i.mean(0)
TARGET_STD = train_ds.target.std(0)
TARGET_MEAN = train_ds.target.mean(0)


### we might not want to standardize if we want to simulate real time ##
def transform(x):
    x = x - MEANS
    x = x / STD_DEVS
    return x


def target_transform(x):
    x = x - TARGET_MEAN
    x = x / TARGET_STD
    return x

train_ds.transform = transform
train_ds.target_transform = target_transform

test_ds.transform = transform
test_ds.target_transform = target_transform

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
        for i in tqdm(range(len(train_ds)-WINDOW_SIZE), desc="Iteration {}".format(_ + 1)):
            #sliding window
            samples = train_ds.data_i[i:i+WINDOW_SIZE]
            label = train_ds.target[i+WINDOW_SIZE]

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
    for i in tqdm(range(len(test_ds)-WINDOW_SIZE), desc="Testing"):
        samples = test_ds.data_i[i:i+WINDOW_SIZE]
        label = test_ds.target[i+WINDOW_SIZE]

        samples = samples.to(device)

        predictions = model(samples)
        predictions = predictions * TARGET_STD + TARGET_MEAN
        label = label * TARGET_STD + TARGET_MEAN
        label = torch.reshape(label, (1,))

        mse.update(predictions.cpu(), label)

        samplesArr = np.append(samplesArr, samples)
        labelsArr = np.append(labelsArr, label)
        predictionsArr = np.append(predictionsArr, predictions)


print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
### MSEs.append([f'{(mse.compute().item()):.10f}', lr, ws, ti]) ### HP Tuning
#print(len(samplesArr), len(labelsArr), len(predictionsArr))
# plt.plot(np.arange(len(samplesArr.flatten())), samplesArr.flatten(), label='test radar', alpha=0.5)
# plt.plot(np.arange(len(trainSamplesArr.flatten())), trainSamplesArr.flatten(), label='train radar', alpha=0.5)
# plt.title('radar data')
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted', color='red')
plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, LR:{LEARN_RATE}, window:{WINDOW_SIZE}- MSE:{(mse.compute().item()):.10f}, {test_file_r}')
plt.legend()
plt.savefig(f'Pred_MSE{(mse.compute().item()):.8f}_{datetime.datetime.now()}.png') #find how to save into folder
plt.clf()
