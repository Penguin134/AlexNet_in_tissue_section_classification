import torch
import torch.nn as nn
import numpy as np


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768*3, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        # print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=2),
            nn.MaxPool2d(6),
            nn.Conv2d(8, 16, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = self.layers(x)
        # print(x.size())
        x = self.linear(x)
        return x


