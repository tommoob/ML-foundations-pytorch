import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, kernel_size):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=1
        )  # (32, 28, 28)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )  # (64, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 14 * 14, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
