import torch
import torch.nn as nn
import torch.nn.functional as F


class simple_cnn(nn.Module):    # nn.Module is the base class for all PyTorch neural networks.

    def __init__(self, num_classes=10):     # architecture of the CNN (num_classes is 0-9)

        super(simple_cnn, self).__init__()  # super() is used to initialize everything that nn.Module needs.

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # first convolutional layer: input channels = 3 (RGB), output channels = 16, kernel size = 3x3, padding = 1 (padding helps to maintain the output size same as input size)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # second convolutional layer: input channels = 16, output channels = 32, kernel size = 3x3, padding = 1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # third convolutional layer: input channels = 32, output channels = 64, kernel size = 3x3, padding = 1

        self.pool = nn.MaxPool2d(2, 2)          # halves width and height of the input, makes the model more efficient and more robust.
        self.fc1 = nn.Linear(64 * 4 * 4, 128)   # fully connected layer: converts extracted features into 128 learned values.
        self.fc2 = nn.Linear(128, num_classes)  # output layer: 10 neurons for 10 classes

    def forward(self, x):  # defines how the input image moves through the network

        x = self.pool(F.relu(self.conv1(x)))  # applies the first convolutional layer, followed by ReLU activation function and max pooling.
        x = self.pool(F.relu(self.conv2(x)))  # applies the second convolutional layer, followed by ReLU activation function and max pooling.
        x = self.pool(F.relu(self.conv3(x)))  # applies the third convolutional layer, followed by ReLU activation function and max pooling.

        x = x.view(-1, 64 * 4 * 4)       # flatten the 3D feature maps into a single vector -> enters the fully connected layers. The -1 means that the batch size will be inferred automatically.

        x = F.relu(self.fc1(x))          # applies the first fully connected layer -> ReLU activation function.
        x = self.fc2(x)                  # applies the final linear layer that produces the class prediction scores.

        return x                         # returns the output of the network
