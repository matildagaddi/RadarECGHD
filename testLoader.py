import torch
from torch.utils import data
from dataset import MyDataset
from FatemehNet import MatildaNet
import matplotlib.pyplot as plt
import numpy as np

WINDOW_SIZE = 10000 # 5 seconds
WINDOW_SAMPLES = 1024 # points
SEED = 0
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_ds = MyDataset(root="./", name="Radar", train=True, window_size=WINDOW_SIZE, 
	window_samples=WINDOW_SAMPLES, device=device, seed=SEED)
train_ld = data.DataLoader(example_ds, batch_size=BATCH_SIZE)

for samples_i, samples_q, samples_ecg in train_ld:
    # samples_i = samples_i.to(device)
    # samples_q = samples_q.to(device)
    # samples_ecg = samples_ecg.to(device)
    print("NEW", samples_i.shape, samples_q.shape, samples_ecg.shape) #size [20] bc of batch size


out = torch.Tensor()
for i in range(10000): #might want to parallelize #len(example_ds)-WINDOW_SAMPLES
	x = example_ds.data_i[i:i+WINDOW_SAMPLES].reshape((1,1024))
	model = MatildaNet()
	y = model(x)

	out = torch.cat((out, y))

out = out.reshape((len(out),))
np_y = out.detach().numpy()
print(np_y)
plt.plot(np_y[1000:2000])
plt.show()

######
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    for batch_idx, data_i, data_q, target in enumerate(train_loader):
        data, target = data_i.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

criterion = torch.nn.MSELoss()
criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=.1)

for epoch in range(1, args.epochs+1):
    train(train_ld, MatildaNet, criterion, optimizer, device)
    scheduler.step()
#####