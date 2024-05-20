import torch
import torch.nn as nn
from torch.utils import data
from dataset import MyDataset
from matildaNet import MatildaNet
import matplotlib.pyplot as plt
import numpy as np

WINDOW_SIZE = 10000
WINDOW_SAMPLES = 1024
SEED = 0
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_ds = MyDataset(root="./", name="Radar", train=True, window_size=WINDOW_SIZE, 
                       window_samples=WINDOW_SAMPLES, device=device, seed=SEED)
train_ld = data.DataLoader(example_ds, batch_size=BATCH_SIZE)

def train_model(model, train_ld, optimizer, criterion, chunk_size=1024, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for sample_i, sample_q, label in train_ld:
            optimizer.zero_grad()
            inputs = sample_i.to(device)
            outputs = model(inputs)
            target = label.to(device).view(-1, 1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")

model = MatildaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_model(model, example_ds, optimizer, criterion)

# Example plot code (ensure data is passed correctly)
X = example_ds[5000:7048][0].reshape((1, 2048))
y = example_ds[5000:7048][2].reshape((1, 2048))
ps_list = train_model(model, X, y, optimizer, np.sqrt(criterion))
