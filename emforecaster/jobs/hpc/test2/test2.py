import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Define a simple feed-forward neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("exp", type=str, default="test", help="Dummy experiment arg")
    args = parser.parse_args()

    # Check the number of CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_cuda_devices}")

    # Print the name of each CUDA device
    for i in range(num_cuda_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")

    # Select the first CUDA device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the network and move it to the GPU if available
    net = Net().to(device)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create a random dataset and dataloader
    epochs = 10
    x = torch.randn(128, 32)
    y = torch.randint(0, 10, (128,))
    dataset = TensorDataset(x, y)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Training for {epochs} epochs...")
    # For each epoch
    for epoch in range(20):  # loop over the dataset multiple times

        epoch_loss = 0.0
        # For each batch of data (assume trainloader is an iterable over batches)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_loss / len(trainloader)}")

    print("Finished Training")
