import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

import layers.unet.upsample_layers as Upsample

class NoUpsamplingLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=False, bias=False):
	
		super(NoUpsamplingLayer, self).__init__()
		    
		self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True),
			    		    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True))

	def forward(self, x, skip):

		# Concatenate
		x = torch.cat([skip, x], dim=1)

		# Layer
		return self.layer(x)

class DefaultLayer(NoUpsamplingLayer):

	def __init__(self, in_channels, out_channels, bilinear=False, bias=False):
	
		super(DefaultLayer, self).__init__(in_channels, out_channels, bilinear=False, bias=False)

		self.upsample = Upsample.DefaultLayer(in_channels, out_channels, bilinear=bilinear)
			    		    
	def forward(self, x, skip):
	
		# Upsample
		x = self.upsample(x, skip)

		# Concatenate
		x = torch.cat([skip, x], dim=1)

		# Layer
		return self.layer(x)	
