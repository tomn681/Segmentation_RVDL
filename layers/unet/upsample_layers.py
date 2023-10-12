import torch
import torch.nn as nn
import torch.nn.functional as F

class DefaultLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=False, scale=None):
	
		super(DefaultLayer, self).__init__()
		
		scale_factor = in_channels//out_channels if scale is None else scale

		self.upsample = nn.ConvTranspose2d(in_channels, 
							out_channels, 
							kernel_size=scale_factor, 
							stride=scale_factor)

		if bilinear:
		    self.upsample = nn.Upsample(scale_factor=scale_factor, 
		    					mode='bilinear', 
		    					align_corners=True)
			    		    
	def forward(self, x, skip):
	
		# Upsample
		x = self.upsample(x)

		# Pad
		diffY = skip.size()[2] - x.size()[2]
		diffX = skip.size()[3] - x.size()[3]

		x = F.pad(x, [diffX // 2, diffX - diffX // 2,
				diffY // 2, diffY - diffY // 2])

		return x
		
class SimpleLayer(DefaultLayer):

	def __init__(self, in_channels, out_channels, bilinear=False, scale=None):
	
		super(SimpleLayer, self).__init__(in_channels, out_channels, bilinear=bilinear, scale=scale)
			    		    
	def forward(self, x):
	
		# Upsample
		x = self.upsample(x)

		return x
