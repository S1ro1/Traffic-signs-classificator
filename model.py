import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, num_classes, in_channels=3):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=3)
    self.p1 = nn.MaxPool2d(2, 2)

    self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    self.p2 = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout(0.20)

    self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    self.p3 = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(128*4*4, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.p1(x)

    x = F.relu(self.conv2(x))
    x = self.p2(x)
    x = self.dropout(x)

    x = F.relu(self.conv3(x))
    x = self.p3(x)

    x = x.reshape(x.shape[0], -1)
    x = F.relu(self.fc1(x))

    return self.fc2(x)