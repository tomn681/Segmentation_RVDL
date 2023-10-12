import sys
import torch
import torch.nn as nn

sys.path.append('../')

import layers.unet.upsample_layers as Upsample
import layers.unet.encoder_layers as Encoder

from torchvision.models.swin_transformer import SwinTransformerBlock#, SwinTransformerBlockV2

class DefaultLayer(nn.Module):

	def __init__(self, in1_channels, in2_channels, bilinear=False):
		super(DefaultLayer, self).__init__()
		
	def forward(self, x):
		return x
		
class UpChannellingLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=False, bias=False):
		super(UpChannellingLayer, self).__init__()
		
		self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
		
	def forward(self, x):
		return self.layer(x)
		
class ThreeHeadAttentionLayer(nn.Module):
	'''
		
		- in_channels	Encoder/Decoder output channels.
		- ag_channels	Attention gate channels (third input).
		- r		Squeeze and Excitation Networks reduction ratio for FC layers.
		- bias		Use bias. Default: False
	'''
	def __init__(self, in_channels, ag_channels, r=16, bias=False, bilinear=False):
		super(ThreeHeadAttentionLayer, self).__init__()
		
		self.bilinear = bilinear
		
		self.sigmoid = nn.Sigmoid()
		
		# Encoder Spatial Attention
		self.Ws1 = nn.Sequential(Encoder.DefaultLayer(in_channels, in_channels, bias=bias),
					    nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=bias))
			    	
		# AG Spatial Attention	    
		self.Ws2 = nn.Sequential(Encoder.DefaultLayer(in_channels, in_channels, bias=bias),
					    nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=bias))
			    		    
		# Upsample
		self.upsample = Upsample.DefaultLayer(ag_channels, in_channels, bilinear=bilinear)
			    		    
		# Combined Spatial Attention
		self.Ws = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=bias))
		
		# Squeeze and Excitation Block
		self.Wc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
					    nn.Flatten(),
					    nn.Linear(in_channels, in_channels//r, bias=bias),
					    nn.ReLU(inplace=True),
					    nn.Linear(in_channels//r, in_channels, bias=bias))
					    
		# Correccion Bilineal
		#if bilinear:
		#	self.down_conv = nn.Conv2d(ag_channels, in_channels, kernel_size=3, padding=1, bias=bias)
					    
	def forward(self, x, y, ag=False):
		#use_ag = ag
	
		# Upsampling
		ag = self.upsample(ag, x)# if use_ag is not False else y
		
		#if self.bilinear and use_ag is not False:
		#	ag = self.down_conv(ag)
		
		# Si es bilineal, reducir canales de ag a la mitad
		
		# Channel Attention
		xy = x + y
		Wc = self.sigmoid(self.Wc(xy))
		Wc = Wc.unsqueeze(dim=-1).unsqueeze(dim=-1)
		
		xy = xy * Wc
		
		#Spatial Attention
		Ws1 = self.sigmoid(self.Ws1(x))
		Ws2 = self.sigmoid(self.Ws2(ag))
		
		Ws = torch.cat((Ws1, Ws2), 1)
		Ws = self.sigmoid(self.Ws(Ws))
		
		return xy * Ws
		
class SpatialAttentionLayer(nn.Module):
	def __init__(self, in_channels, ag_channels, r=16, bias=False, bilinear=False, scale=None):
		super(SpatialAttentionLayer, self).__init__()
		
		self.sigmoid = nn.Sigmoid()
		
		# Key Attention
		self.K1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=bias) 	    
		self.K2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=bias)
		
		self.K = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias)
		
		# Query Attention
		self.Q1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias) 	    
		self.Q2 = nn.Conv2d( in_channels, in_channels, kernel_size=3, padding=1, bias=bias)
		
		# Value Attention
		self.V1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias) 	    
		self.V2 = nn.Conv2d( in_channels, in_channels, kernel_size=3, padding=1, bias=bias)
			    		    
		# Upsample
		self.upsample = Upsample.DefaultLayer(ag_channels, in_channels, bilinear=bilinear, scale=scale)
		
		# Squeeze and Excitation Block
		self.C = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
					    nn.Flatten(),
					    nn.Linear(in_channels, in_channels//r, bias=bias),
					    nn.ReLU(inplace=True),
					    nn.Linear(in_channels//r, in_channels, bias=bias),
					    nn.ReLU(inplace=True))
					    
		self.softmax = nn.Softmax2d()
					    
	def forward(self, x, y, ag):
	
		# Upsampling
		ag = self.upsample(ag, x)
		
		# Channel Attention
		C = self.sigmoid(self.C(x))
		C = C.unsqueeze(dim=-1).unsqueeze(dim=-1)
		
		# Key Attention
		K1 = self.sigmoid(self.K1(x))
		K2 = self.sigmoid(self.K2(ag))
		
		K = torch.cat((K1, K2), 1)
		K = self.sigmoid(self.K(K))
		
		# Query Attention
		Q = self.sigmoid(self.Q1(y))
		Q = self.sigmoid(self.Q2(Q))
		
		Q = Q.permute(0, 1, 3, 2)
		
		# Value Attention
		V = self.sigmoid(self.V1(x))
		V = self.sigmoid(self.V2(V))

		return self.softmax(K @ Q) * V * C
		
class MonoSpatialAttentionLayer(nn.Module):
	def __init__(self, in_channels, ag_channels, r=16, bias=False, bilinear=False, scale=None):
		super(MonoSpatialAttentionLayer, self).__init__()
		
		self.sigmoid = nn.Sigmoid()
		
		# Key Attention
		self.K1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=bias) 	    
		self.K2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=bias)
		
		self.K = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=bias)
		
		# Query Attention
		self.Q1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias) 	    
		self.Q = nn.Conv2d( in_channels, 1, kernel_size=3, padding=1, bias=bias)
		
		# Value Attention
		self.V1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias) 	    
		self.V2 = nn.Conv2d( in_channels, in_channels, kernel_size=3, padding=1, bias=bias)
			    		    
		# Upsample
		self.upsample = Upsample.DefaultLayer(ag_channels, in_channels, bilinear=bilinear, scale=scale)
		
		# Squeeze and Excitation Block
		self.C = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
					    nn.Flatten(),
					    nn.Linear(in_channels, in_channels//r, bias=bias),
					    nn.ReLU(inplace=True),
					    nn.Linear(in_channels//r, in_channels, bias=bias),
					    nn.ReLU(inplace=True))
					    
		self.softmax = nn.Softmax2d()
					    
	def forward(self, x, y, ag):
	
		# Upsampling
		ag = self.upsample(ag, x)
		
		# Channel Attention
		C = self.sigmoid(self.C(x))
		C = C.unsqueeze(dim=-1).unsqueeze(dim=-1)
		
		# Key Attention
		K1 = self.sigmoid(self.K1(x))
		K2 = self.sigmoid(self.K2(ag))
		
		K = torch.cat((K1, K2), 1)
		K = self.sigmoid(self.K(K))
		
		# Query Attention
		Q = self.sigmoid(self.Q1(y))
		Q = self.sigmoid(self.Q(Q))
		
		Q = Q.permute(0, 1, 3, 2)
		
		# Value Attention
		V = self.sigmoid(self.V1(x))
		V = self.sigmoid(self.V2(V))

		return self.softmax(K @ Q) * V * C
		
class FullSpatialAttentionLayer(SpatialAttentionLayer):
	def __init__(self, in_channels, ag_channels, r=16, bias=False, bilinear=False):
	
		super(FullSpatialAttentionLayer, self).__init__(in_channels, 
									ag_channels, 
									r, 
									bias, 
									bilinear)
		
		# K1, K2 Conv Layer
		self.K = nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1, bias=bias)
		self.KK = nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1, bias=bias)
		
	def forward(self, x, y, ag):
		# Upsampling
		ag = self.upsample(ag, x)
		
		# Channel Attention
		C = self.sigmoid(self.C(x))
		C = C.unsqueeze(dim=-1).unsqueeze(dim=-1)
		
		# Key Attention
		K1 = self.sigmoid(self.K1(x))
		K2 = self.sigmoid(self.K2(ag))
		
		K1 = self.sigmoid(self.K(K1))
		K2 = self.sigmoid(self.KK(K2))
		
		# Query Attention
		Q = self.sigmoid(self.Q1(y))
		Q = self.sigmoid(self.Q2(Q))
		
		Q = Q.permute(0, 1, 3, 2)
		
		# Value Attention
		V = self.sigmoid(self.V1(x))
		V = self.sigmoid(self.V2(V))

		return self.softmax(K1 @ Q @ K2) * V * C
		
class DoubleSpatialAttentionLayer(nn.Module):
	def __init__(self, in_channels, ag_channels, r=16, bias=False, bilinear=False):
	
		super(DoubleSpatialAttentionLayer, self).__init__()
									
		self.SA1 = SpatialAttentionLayer(in_channels, in_channels, r=16, bias=False, bilinear=False)
		self.SA2 = SpatialAttentionLayer(in_channels, ag_channels, r=16, bias=False, bilinear=False)
		
	def forward(self, x, y, ag):
	
		x = self.SA1(x, x, x)
		
		return self.SA2(x, y, ag)
		
class AttentionLayer(nn.Module):
	def __init__(self, in_channels, ag_channels, d=512, img_size=512, min_freq=1e-4, bias=False, bilinear=False, scale=None, cpu=False):#, r=16):
		super(AttentionLayer, self).__init__()
		
		# Key and Value
		self.KV_1x1 = nn.Conv2d(2*in_channels, d, kernel_size=1, bias=bias)
		
		# Query
		self.Q_1x1 = nn.Conv2d(in_channels, d, kernel_size=1, bias=bias)
		
		# Query Attention
		self.Q = nn.Linear(d, d, bias=bias)
		
		# Key Attention
		self.K = nn.Linear(d, d, bias=bias)
		
		# Value Attention
		self.V = nn.Linear(d, d, bias=bias)
			    		    
		# Upsample
		self.upsample = Upsample.DefaultLayer(ag_channels, in_channels, bilinear=bilinear, scale=scale)
					    
		self.softmax = nn.Softmax2d()
		
		self.d = d
		
		d = torch.tensor(d, dtype=torch.float32)
		self.sqrtd = torch.sqrt(d)
		
		# Positional Encoding
		pos = torch.arange(0, img_size*img_size, dtype=torch.float32)
		msk = torch.arange(0, d)
		sin_msk = (msk%2).float()
		cos_msk = 1-sin_msk
		exp = torch.div(msk, 2, rounding_mode='floor')
		exp = 2*exp
		exp = exp.float()/d
		freqs = min_freq**exp
		angles = torch.einsum('i,j->ij', pos, freqs)
		positional_encoding = torch.cos(angles)*cos_msk + torch.sin(angles)*sin_msk
		
		device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
		
		self.positional_encoding = positional_encoding.to(device=device)
					    
	def forward(self, x, y, ag):
		
		# Upsampling
		ag = self.upsample(ag, x)

		# Key and Value
		KV = torch.cat((x, ag), 1)
		KV = self.KV_1x1(KV)
		KV = KV.reshape(KV.shape[2]*KV.shape[3], self.d)
		KV += self.positional_encoding
		
		# Query
		Q = self.Q_1x1(y)
		Q = Q.reshape(Q.shape[2]*Q.shape[3], self.d)
		Q += self.positional_encoding
		
		# Linear Layers
		Q = self.Q(Q)
		K = self.K(KV)
		V = self.V(KV)
		
		K = K.permute(1, 0)
		
		QK = Q @ K / self.sqrtd
		QK = QK.unsqueeze(0)
		
		out = self.softmax(QK) @ V

		return out.unsqueeze(0).reshape(y.shape)
		
class SwinAttentionLayer(nn.Module):
	def __init__(self, in_channels, output_chanels, bilinear=False):
		super(SwinAttentionLayer, self).__init__()
		
		# Attention
		self.swin_block = SwinTransformerBlock(dim=in_channels ,num_heads=1, window_size=[7, 7], shift_size=[3, 3])
					    
	def forward(self, x):
	
		x = x.permute(0, 2, 3, 1)

		return self.swin_block(x).permute(0, 3, 1, 2)
