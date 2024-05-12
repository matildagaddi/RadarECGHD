import torch
from torch.utils import data
from dataset import MyDataset

WINDOW_SIZE = 10000 # 5 seconds
WINDOW_SAMPLES = 1024 # points
SEED = 0
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_ds = MyDataset(root="./", name="Radar", train=True, window_size=WINDOW_SIZE, 
	window_samples=WINDOW_SAMPLES, device=device, seed=SEED)
train_ld = data.DataLoader(example_ds, batch_size=BATCH_SIZE)