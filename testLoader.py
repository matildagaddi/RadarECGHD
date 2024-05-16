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


# for samples_i, samples_q, samples_ecg in train_ld:
#     # samples_i = samples_i.to(device)
#     # samples_q = samples_q.to(device)
#     # samples_ecg = samples_ecg.to(device)
#     print("NEW", samples_i.shape, samples_q.shape, samples_ecg.shape) #size [20] bc of batch size


# out = torch.Tensor()
# for i in range(10000): #might want to parallelize #len(example_ds)-WINDOW_SAMPLES
# 	x = example_ds.data_i[i:i+WINDOW_SAMPLES].reshape((1,1024))
# 	model = MatildaNet()
# 	y = model(x)

# 	out = torch.cat((out, y))

# out = out.reshape((len(out),))
# np_y = out.detach().numpy()
# print(np_y)
# plt.plot(np_y[1000:2000])
# plt.show()

def train_model(model, X, y, optimizer, criterion, chunk_size=1024, epochs=10):
    model.train()
    ps_list = np.zeros((X.shape[1],))
    print(X.shape[1]) ###################
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(X.shape[1] - chunk_size + 1):  # Slide the window over the entire sequence
            optimizer.zero_grad()

            # Get the chunk of input data
            inputs = torch.tensor(X[:, i:i + chunk_size], dtype=torch.float32).to(device).clone().detach()

            # Forward pass
            output = model(inputs)

            # Get the corresponding target value (the entire sequence in the window)
            target = torch.tensor(y[i], dtype=torch.float32).to(device).clone().detach()

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Store outputs in ps_list
            outputs_np = outputs.squeeze().detach().cpu().numpy()
            ps_list[i:i + chunk_size] = outputs_np

        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")

    return ps_list

# Generate sample data
X = example_ds[5000:7048][0].reshape((1, 7048))  
y = example_ds[5000:7048][2].reshape((1, 7048))  

# Instantiate model, optimizer, and criterion
model = MatildaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model and get predicted values
ps_list = train_model(model, X, y, optimizer, criterion)

# Save actual array X and predicted values ps to text files
np.savetxt('actual_array_X.txt', X.flatten())
np.savetxt('predicted_values_ps.txt', ps_list)

# Plot the actual input and the sequence of predicted values
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(X.flatten())), X.flatten(), label='Actual X')
plt.plot(np.arange(len(y.flatten())), y.flatten(), label='Actual y')
plt.plot(np.arange(len(ps_list)), ps_list, label='Predicted ps')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual Array X and Predicted Values ps')
plt.legend()
plt.savefig('predicted_values_plot.png')  # Save the plot
plt.show()