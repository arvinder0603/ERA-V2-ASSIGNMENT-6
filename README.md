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
            out_channels=16,  // This tells us number of  kernels used in the first layer 
            kernel_size=5, // this is the kernel size used 5* 5 this is used for convolve in images
            stride=3,  //This tells us the stride used like going to every x pixels // like if stride is one  kernel will convolve ignoring x pixels 
            padding=2  // This padding like outer sides of images so, outer layer feature are not fadded away 
        )
        self.relu1 = nn.ReLU()   //This is activation function to get non linearity 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) //This used to get matrix used to convolve and get maximum out of pixels, this cause reduce in features

        //This is the 2nd layer used. this takes input from the output of last layer. and give output for next layer.
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=3,
            padding=2
        )
        self.relu2 = nn.ReLU() 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.out = nn.Linear(32 * 5 * 5, 10) //This fully connected layer which gives outcome. 

    def forward(self, x):
        x = self.conv1(x) //use of first layer
        x = self.relu1(x) //RELU Apllied on outcome of first layer
        x = self.maxpool1(x)  //MAXPOOL Apllied on outcome of first layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1) //This layer is used to give data to neural network in flatten.
        output = self.out(x)
        return output, x  # return x for visualization

# Instantiate the model
model = CNN()





torch.manual_seed(1)
batch_size = 128 //size of batch given to neural network to train model

use_cuda = True //for gpu

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

//These are data loader provided by torch to load annotate your data and distribute data in train ,test and validation dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), //This is tensor used in pytorch, this actually helps in computation alot 
                        transforms.Normalize((0.1307,), (0.3081,))  //Normalise data to certain value
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)


     //This is the test loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
    
//this is the validate data loader 
valid_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
    ]))
)




//Here in down we have written functions for training of data 


from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, valid_loader):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')

    # Validate the model after each epoch
    validate(model, device, valid_loader)

def validate(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    # Print validation score
    print('Validation Accuracy: {:.2f}%'.format(100. * correct / len(valid_loader.dataset)))




