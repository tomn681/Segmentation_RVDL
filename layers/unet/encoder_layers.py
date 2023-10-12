import sys
import torch.nn as nn

sys.path.append('../')

class DefaultLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bias=False):
	
		super(DefaultLayer, self).__init__()

		self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True),
			    		    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True))
        	
	def forward(self, x):
		return self.layer(x)
