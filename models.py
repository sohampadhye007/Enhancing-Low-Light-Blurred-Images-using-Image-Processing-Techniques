import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(True))
        
        self.adaptive_kernel = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.ReLU(True))
        
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),  
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.adaptive_kernel(x)
        x = self.decoder(x)
        return x
