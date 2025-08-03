import torch
import torch.nn as nn
import torch.nn.functional as F



# Hyperparams
conv1_out = 16
conv2_out = 32
conv3_out = 64
dropout_prob = 0.3
fc_hidden = 128
num_classes = 4

# fixed image size
input_height = 96
input_width = 64


# Model
class SpeakerCountCNN(nn.Module):
    def __init__(self, conv1_out, conv2_out, conv3_out, fc_hidden, dropout_prob, input_height=input_height, input_width=input_width):
        super(SpeakerCountCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_out)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_prob)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
