import torch.nn as nn
		
class DefaultHead(nn.Module):

    def __init__(self, in_channels, out_channels):
    
        super(DefaultHead, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
    
        return self.conv(x)
