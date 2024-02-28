# ERA-V2-ASSIGNMENT-6




from __future__ import print_function              |
import torch                                       |
import torch.nn as nn                               Here we have imported all the libraries required for Our CNN    TORCH  LIBRARY USED FOR OUR MODEL
import torch.nn.functional as F                    |
import torch.optim as optim                        |
from torchvision import datasets, transforms       |





                            //Here we have used class name CNN for creating Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(                 //First Convolution layer used for our dataset  
            in_channels=1, // This used for channel  As our image is black white so it there would be only 1 channel
            out_channels=16,  // This tells us 
            kernel_size=5,
            stride=3,
            padding=2
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=3,
            padding=2
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.out = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

# Instantiate the model
model = CNN()







