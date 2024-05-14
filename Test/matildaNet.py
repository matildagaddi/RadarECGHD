import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

        self.lstm = nn.LSTM(input_size=8, hidden_size=1, num_layers=1, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(816, 8),  
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.Dropout(p=0.2),
            nn.Linear(4, 1)            
        )
            
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)  
        
        x, _ = self.lstm(x)
        x = x.view(x.size(0), -1)  
        
        x = self.classifier(x)
        return x

def train_model(model, X, y, optimizer, criterion, chunk_size=1024, epochs=10):
    ps_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(X[0]), chunk_size):
            optimizer.zero_grad()

            # Get the chunk of input data
            inputs = torch.tensor(X[:, i:i+chunk_size], dtype=torch.float32)

            # Forward pass
            outputs = model(inputs)

            # Get the corresponding target values
            targets = torch.tensor(y[i:i+chunk_size], dtype=torch.float32).view(-1, 1)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert outputs to numpy array and then append to ps_list
            outputs_np = np.atleast_1d(outputs.squeeze().detach().numpy())
            ps_list.extend(outputs_np)

        print(f"Epoch {epoch+1}, Loss: {running_loss}")

    return ps_list


# Generate sample data
X = np.random.randn(1, 1024)  # Example input data representing one sample
y = np.random.randn(1024)  # Example target data for one sample

# Instantiate model, optimizer, and criterion
model = MatildaNet()
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
plt.plot(np.arange(len(ps_list)), ps_list, label='Predicted ps')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual Array X and Predicted Values ps')
plt.legend()
plt.savefig('predicted_values_plot.png')  # Save the plot
plt.show()
