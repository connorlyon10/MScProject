import torch
import torch.nn as nn
import torch.nn.functional as F



# Model config
config = {
    'lr': 0.001,
    'dropout_prob': 0.3,
    'fc_hidden': 128,
    'conv1_out': 32,
    'conv2_out': 16,
    'conv3_out': 64,
    'conv4_out': 64,
    'input_height': 96,
    'input_width': 64,
    'num_classes': 6
}



# 0-5 Speaker Counting CNN
class ConvCount(nn.Module):

    # init defines the layer types that will be called in forward().
    def __init__(self, 
                 conv1_out,
                 conv2_out,
                 conv3_out,
                 conv4_out,
                 fc_hidden,
                 dropout_prob,
                 input_height=96,               # refers to spectrogram image height
                 input_width=64,                # refers to spectrogram image width
                 num_classes=6,                 # 0-5 = 6 classes
                 **kwargs):                     # kwargs isn't accessed but catches config errors
        super(ConvCount, self).__init__()

        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_out)

        self.conv4 = nn.Conv2d(conv3_out, conv4_out, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv4_out)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_prob)

        # Passes a dummy tensor through the layers to find the correct output shape for the fc layers
        # torch.no_grad() disables gradient calculation --> avoids wasting memory/compute 
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)



    # forward() shows the chronological flow of the model:
    # Apply 4 rounds of conv → BN → ReLU → pool, then flatten and dropout
    # Finally, two FC layers perform feature learning and output prediction
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# If file is run as script, creates the model using defaults in `config`
if __name__ == '__main__':
    model = ConvCount(**config)
    print("Model initialised.")