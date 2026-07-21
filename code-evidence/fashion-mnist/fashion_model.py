import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700).
    """
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            #block 1:
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            #block 2: 
            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                                                                      
            #block 3
            nn.Conv2d(64,96,kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        

        self.gap = nn.AdaptiveAvgPool2d((1,1,))

        self.classifier = nn.Sequential(
            nn.Linear(96,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10),
                     
        )




                
            
        # nn.init module contains initialization methods:
        # https://docs.pytorch.org/docs/main/nn.init.html
        # This particular one is called Kaiming initialization (also known as
        # He initialization) described in He, K. et al. (2015)


    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        
        return x
