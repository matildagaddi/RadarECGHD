import torch
import torch.nn as nn
from torch.utils import data
from dataset2 import MyDataset
from MatildaNet import MatildaNet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

WINDOW_SIZE = 10000 # 5 seconds
WINDOW_SAMPLES = 1024 # points
SEED = 0
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_file_r= 'trainVal/radar/GDN0001_Resting_radar_1.mat'
train_file_e= 'trainVal/ecg/GDN0001_Resting_ecg_1.mat'
test_file_r= 'trainVal/radar/GDN0001_Resting_radar_2.mat'
test_file_e= 'trainVal/ecg/GDN0001_Resting_ecg_2.mat'
path_to_DS = '/Users/matildagaddi/Documents/SEElab'

train_ds = MyDataset(radar_path=f"{path_to_DS}/DATASET/{train_file_r}",
    ecg_path=f"{path_to_DS}/DATASET/{train_file_e}", window_size=WINDOW_SIZE,
    device=device)
test_ds = MyDataset(radar_path=f"{path_to_DS}/DATASET/{test_file_r}", 
    ecg_path=f"{path_to_DS}/DATASET/{test_file_e}", window_size=WINDOW_SIZE,
    device=device)

train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE)
test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE)

def train_model(model, X, y, optimizer, criterion, chunk_size=1024, epochs=50): 
#how to add stopping threshold (if MSE < ___) or take weights of lowest MSE
#also needs to be much faster
    model.train()
    ps_list = np.array([])
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(X[0]) - chunk_size + 1):  # Slide the window over the entire sequence
            optimizer.zero_grad()

            # Get the chunk of input data
            inputs = torch.tensor(X[:, i:i + chunk_size], dtype=torch.float32).to(device).clone().detach()

            # Forward pass
            outputs = model(inputs)

            # Get the corresponding target value (only the last point in the window)
            target = torch.tensor(y[:, i + chunk_size - 1], dtype=torch.float32).to(device).view(-1, 1).clone().detach()

            # Compute the loss
            loss = criterion(outputs, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert outputs to numpy array and then store in ps_list
            outputs_np = outputs.squeeze().detach().cpu().numpy()
            ps_list = np.append(ps_list, outputs_np)

        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")

    return ps_list


#see what train_ld and train_ds return
#print(train_ds[0]) #sample 0's (radar_tensor, ecg_tensor)
X=None
y=None
ps_list = np.array([])
for radar, ecg in tqdm(train_ld):

    # Generate sample data
    X = radar
    y = ecg

    # Instantiate model, optimizer, and criterion
    model = MatildaNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model and get predicted values
    ps_list = np.append(ps_list, train_model(model, X, y, optimizer, criterion)) #matlab example uses RMSE down to .17 w/o wodwt, .10 w/ modwt

    # Save actual array X and predicted values ps to text files
    np.savetxt('actual_array_X.txt', X.flatten())
    np.savetxt('predicted_values_ps.txt', ps_list)

# Plot the actual input and the sequence of predicted values
print(X.shape, y.shape, ps_list.shape) #torch.Size([10, 1024]) torch.Size([10, 1024]) (500,)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(X.flatten())), X.flatten(), label='Actual X')
plt.title('radar data')
plt.show()
plt.plot(np.arange(len(y.flatten())), y.flatten(), label='Actual y')
plt.title('ecg target')
plt.show()
plt.plot(np.arange(len(ps_list)), ps_list, label='Predicted ps')
plt.title('predicted data')
plt.show()
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Actual Array X and Predicted Values ps')
# plt.legend()
# plt.savefig('/results/predicted_values_plot.png')  # Save the plot
# plt.show()
