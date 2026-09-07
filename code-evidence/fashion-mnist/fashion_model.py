import torch
import torch.nn as nn


class Net(nn.Module):
    """Compact CNN used for Fashion-MNIST image classification."""
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                                                                      
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




                
            
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        
        return x
