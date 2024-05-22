import torch
import torch.nn as nn
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


def train_model(model, X, y, optimizer, criterion, chunk_size=1024, epochs=50): 
#how to add stopping threshold (if MSE < ___) or take weights of lowest MSE
#also needs to be much faster
    model.train()
    ps_list = np.zeros((X.shape[1] - chunk_size + 1,))
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
            ps_list[i] = outputs_np

        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")

    return ps_list

# Generate sample data
X = example_ds[5000:7048][0].reshape((1, 2048))
y = example_ds[5000:7048][2].reshape((1, 2048))

# Instantiate model, optimizer, and criterion
model = MatildaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model and get predicted values
ps_list = train_model(model, X, y, optimizer, criterion) #matlab example uses RMSE down to .17 w/o wodwt, .10 w/ modwt
#ps_list[-1] = np.sqrt(ps_list[-1]) # I dont think this works (for RMSE)

# Save actual array X and predicted values ps to text files
np.savetxt('actual_array_X.txt', X.flatten())
np.savetxt('predicted_values_ps.txt', ps_list)

# Plot the actual input and the sequence of predicted values
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(X.flatten())), X.flatten(), label='Actual X')
plt.plot(np.arange(len(y.flatten())), y.flatten(), label='Actual y')
plt.plot(np.arange(len(ps_list))+1024, ps_list, label='Predicted ps')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual Array X and Predicted Values ps')
plt.legend()
plt.savefig('predicted_values_plot.png')  # Save the plot
plt.show()
