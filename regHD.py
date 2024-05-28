
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
from dataset2 import MyDataset
import matplotlib.pyplot as plt
import numpy as np
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
WINDOW_SIZE = 400 # out of 1024, 1 second is  # HOW TO MAKE WINDOW SLIDE
NUM_FEATURES = WINDOW_SIZE  #1 # number of features in dataset # SHOULD THIS BE WINDOW SIZE? ############
#WINDOW_SAMPLES = 1024 # points
BATCH_SIZE = 20
LEARN_RATE = 0.0001 #could implement a decreasing schedule #0.0003 explodes 
#lower learn rate better, maybe use early stopping or fewer iterations, prevent overfitting?
TRAIN_ITERS = 100

print('hyperparameters: lr: ', LEARN_RATE, ' train iters: ', TRAIN_ITERS)


train_ds = MyDataset(radar_path="/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/radar/GDN0001_Resting_radar_1.mat", 
    ecg_path="/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/ecg/GDN0001_Resting_ecg_1.mat", window_size=WINDOW_SIZE,
    device=device)
test_ds = MyDataset(radar_path="/Users/matildagaddi/Documents/SEElab/DATASET/test/radar/GDN0006_Apnea_radar_1.mat", 
    ecg_path="/Users/matildagaddi/Documents/SEElab/DATASET/test/ecg/GDN0006_Apnea_ecg_1.mat", window_size=WINDOW_SIZE,
    device=device)

# Get necessary statistics for data and target transform
STD_DEVS = train_ds.data.std(0)
MEANS = train_ds.data.mean(0)
TARGET_STD = train_ds.target.std(0)
TARGET_MEAN = train_ds.target.mean(0)

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

#maybe DataLoader has a sliding window option, but I'm not using it for now. Just using loop indexing
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

# Model training
with torch.no_grad():
    for _ in range(TRAIN_ITERS):
        for i in tqdm(range(len(train_ds)-WINDOW_SIZE), desc="Iteration {}".format(_ + 1)):
            #sliding window
            samples = train_ds.data[i:i+WINDOW_SIZE]
            label = train_ds.target[i+WINDOW_SIZE]

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

with torch.no_grad():
    for i in tqdm(range(len(test_ds)-WINDOW_SIZE), desc="Testing"):
        samples = train_ds.data[i:i+WINDOW_SIZE]
        label = train_ds.target[i+WINDOW_SIZE]

        samples = samples.to(device)

        predictions = model(samples)
        predictions = predictions * TARGET_STD + TARGET_MEAN
        label = label * TARGET_STD + TARGET_MEAN
        #label = torch.FloatTensor(label) #doesn't change 
        label = torch.reshape(label, (1,))

        mse.update(predictions.cpu(), label)

        samplesArr = np.append(samplesArr, samples)
        labelsArr = np.append(labelsArr, label)
        predictionsArr = np.append(predictionsArr, predictions)


print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
#print(len(samplesArr), len(labelsArr), len(predictionsArr))
# plt.plot(np.arange(len(samplesArr.flatten())), samplesArr.flatten(), label='Actual X')
# plt.title('radar data')
# plt.show()
plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='green')
plt.title('ecg target')
plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted', color='purple')
plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, LR:{LEARN_RATE}, MSE:{(mse.compute().item()):.10f}, time:{train_time}, window:{WINDOW_SIZE}')
plt.legend()
plt.savefig(f'Pred_i{TRAIN_ITERS}_{datetime.datetime.now()}.png') #find how to save into folder
plt.show()
