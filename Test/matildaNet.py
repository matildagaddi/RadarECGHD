import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataset import MyDataset  # Assuming you have a custom dataset class
from torch.utils import data

WINDOW_SIZE = 10000  # 5 seconds
WINDOW_SAMPLES = 1024  # points
SEED = 0
BATCH_SIZE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the dataset and data loader
example_ds = MyDataset(root="./", name="Radar", train=True, window_size=WINDOW_SIZE, 
    window_samples=WINDOW_SAMPLES, device=device, seed=SEED)
train_ld = data.DataLoader(example_ds, batch_size=BATCH_SIZE)

class MatildaNet(nn.Module):
    def __init__(self):
        super(MatildaNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=64, stride=8, padding=0),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=0),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.MaxPool1d(2, stride=2),
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=0),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=0),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, bidirectional=True)

        # Calculate the input size for the first linear layer
        conv_output_size = self._get_conv_output_size(WINDOW_SAMPLES)
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size * 8 * 2, 8),  # Adjusted input size
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.Dropout(p=0.2),
            nn.Linear(4, 1)
        )

    def _get_conv_output_size(self, input_length):
        x = torch.randn(1, 1, input_length)
        x = self.layers(x)
        return x.size(2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, X, y, optimizer, criterion, chunk_size=1024, epochs=10):
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
# X = example_ds[5000:7048][0].reshape((1, 2048))  # Example input data representing one sample
# y = example_ds[5000:7048][2].reshape((1, 2048))  # Ensure y is reshaped correctly
X = example_ds[5000:7048][0].reshape((1, 2048))  # Example input data representing one sample
y = example_ds[5000:7048][2].reshape((1, 2048))

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
