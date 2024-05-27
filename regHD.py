
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_FEATURES = 1  # number of features in dataset # SHOULD THIS BE WINDOW SIZE? ############
WINDOW_SIZE = 400 # out of 1024, 1 second is  # HOW TO MAKE WINDOW SLIDE
WINDOW_SAMPLES = 1024 # points
BATCH_SIZE = 20

train_ds = MyDataset(radar_path="/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/radar/GDN0001_Resting_radar_1.mat", 
    ecg_path="/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/ecg/GDN0001_Resting_ecg_1.mat", window_size=WINDOW_SIZE,
    device=device)
test_ds = MyDataset(radar_path="/Users/matildagaddi/Documents/SEElab/DATASET/test/radar/GDN0006_Apnea_radar_1.mat", 
    ecg_path="/Users/matildagaddi/Documents/SEElab/DATASET/test/ecg/GDN0006_Apnea_ecg_1.mat", window_size=WINDOW_SIZE,
    device=device)

# Get necessary statistics for data and target transform
STD_DEVS = ds.data.std(0)
MEANS = ds.data.mean(0)
TARGET_STD = ds.target.std(0)
TARGET_MEAN = ds.target.mean(0)

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

train_dl = data.DataLoader(train_ds, batch_size=1)
test_dl = data.DataLoader(test_ds, batch_size=1)


# Model based on RegHD application for Single model regression
class SingleModel(nn.Module):
    def __init__(self, num_classes, size):
        super(SingleModel, self).__init__()

        self.lr = 0.00001
        self.M = torch.zeros(1, DIMENSIONS) # why 1?
        self.project = embeddings.Sinusoid(size, DIMENSIONS)

    def encode(self, x):
        sample_hv = self.project(x)
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
    for _ in range(2):
        for samples, extra_q, labels in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.model_update(samples_hv, labels)

# Model accuracy
mse = torchmetrics.MeanSquaredError()

samplesArr = np.array([])
labelsArr = np.array([])
predictionsArr = np.array([])

with torch.no_grad():
    for samples, extra_q, labels in tqdm(test_dl, desc="Testing"):
        samples = samples.to(device)

        predictions = model(samples)
        predictions = predictions * TARGET_STD + TARGET_MEAN
        labels = labels * TARGET_STD + TARGET_MEAN
        mse.update(predictions.cpu(), labels)

        samplesArr = np.append(samplesArr, samples)
        labelsArr = np.append(labelsArr, labels)
        predictionsArr = np.append(predictionsArr, predictions)


print(f"Testing mean squared error of {(mse.compute().item()):.20f}")
print(len(samplesArr), len(labelsArr), len(predictionsArr))
plt.plot(np.arange(len(samplesArr.flatten())), samplesArr.flatten(), label='Actual X')
plt.title('radar data')
plt.show()
plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual y')
plt.title('ecg target')
plt.show()
plt.plot(np.arange(len(predictionsArr)), predictionsArr, label='Predicted ps')
plt.title('predicted data')
plt.show()
