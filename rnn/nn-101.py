# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        HIDDEN_SIZE = 1000
        super(NN, self).__init__() # Call nn.Module.__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE) # Setup our input layer
        self.fch1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, num_classes) # Setup our output layer

    def forward(self, x):
        #print("Shape of x: ", x.shape)
        x = F.relu(self.fc1(x)) # Apply activation function to input
        x = F.relu(self.fch1(x))
        x = self.fc2(x) # Apply activation function to output
        return x # Return output

def load_dataset():
    # Load dataset
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# Train network
def train():
    for epoch in range(num_epochs):
        # The call to enumerate add's the batch index, batch_idx
        # to the enumerate object
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Get data to correct shape
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward() # Weights update

            # gradient descent or adam step
            optimizer.step()

# Check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() # We don't need to keep track of the gradients

    with torch.no_grad(): 
        # Context manager that allows us to disable the gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Initialize network
model = NN(input_size, num_classes).to(device)

# Load the data
train_loader, test_loader = load_dataset()

# Setup the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
train()

# Check accuracy
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

