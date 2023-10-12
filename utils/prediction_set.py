import os
import cv2
import logging
import numpy as np
import pandas as pd

import torch
from PIL import Image

from torch.utils.data import Dataset

'''
Class Default Dataset:

Constructs a standard-image one-mask segmentation dataset.
'''
class DefaultDataset(Dataset):
	
	'''
	Constructor Method
	
	Inputs:
		- basepath: (String) Annotations file directory path. 
	    
		        Directory must contain files:
		        
		          File Name:                File Column Data:
		            - 'test.txt':              (String, String) Image Path
		            - 'test_mask.txt':         (String, String, ...) Mask 1 Path, Mask 2 Path, ...
		            - 'test_class.txt':        (Int, Int, ...) Mask 1 Class, Mask 2 Class, ...

		            - 'train.txt':             (String, String) Image Path
		            - 'train_mask.txt':        (String, String, ...) Mask 1 Path, Mask 2 Path, ...
		            - 'train_class.txt':       (Int, Int, ...) Mask 1 Class, Mask 2 Class, ...
		            
		           Notes: 
		            - All files must have tab-separated columns
		            - Train files includes validation data
		            - Class 0 reserved
		            
		- img_size: (Int) Image preprocessing resize, default=512
		            
		- transforms: (object containing torchvision.tranforms) Data Augmentation Transforms.
		
		- train: (Boolean) If True, opens train and validation files, default=True.
			If 'test' opens test files and transforms = None.
			
	Outputs:
		- dataset: (DefaultDataset Object) Dataset Object containing the given data:
			
			Attributes:
			
			 - self.img_size:	Given image preprocessing resize, default: 512
			 - self.transforms:	Given set of data augmentation transforms
			 - self.data:		(Dict) Data: image paths, classes path and masks path
			 			       Keys: <image>, <mask>, <class>
			 			       Value Type: String, List(String), List(Int)
			 - self.size:		Amount of images containing in the dataset
			 - self.classes:	Number of classes in dataset
			 - self.objects:	Number of maximum masks for each image
			 
			Methods:
			 
			 - len:		Default len method, returns amount of images contained
			 - load:		Image file open, structured over multiple file extensions
			 - preprocess:		Image preprocessing and transforms applicator
			 - __getitem__: 	Default __getitem__ method for data retrieval
			 
			Note: For further details, methods are explained in it's corresponding class
	'''
	def __init__(self, file_path: str, img_size: int=512, transforms=None, clahe=False):
	
		super(DefaultDataset, self).__init__()
		
		# Ensure image_size is correctly formatted
		assert img_size > 0, 'Size must be greater than 0'
			
		# Store image_size and transforms
		self.img_size = (img_size, img_size)
		self.transforms = transforms
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None
		
		# Read file

		imgs = pd.read_csv(os.path.join(file_path, 'pred.txt'), sep='\t', names=('image',), index_col=False)
			
		# Set values
		self.data = imgs.to_dict('list')
		self.size = len(imgs)
		self.path = file_path
		
		# Get number of channels
		ch_img = self.load(os.path.join(self.path, self.data['image'][0]))
		ch_img = self.preprocess(ch_img, is_mask=False)
		self.channels = ch_img.shape[0]
		
		# Ensure images in dataset
		assert 0 < self.size, 'Empty Dataset' 'No classes other than 0'
			
		# Log the dataset creation
		logging.info(f'Creating {"Train" if train else "Test"} dataset with {self.size} examples with {self.objects} objects from {self.classes} different classes')
		
	'''
	len Method
	
	Default len method. Allows to get the amount of images in dataset.
	
	Inputs: 
		- None
		
	Outputs:
		- len: (Int) Number of images in dataset
	'''
	def __len__(self):
		return self.size
		
	'''
	
	'''
	def preprocess(self, pil_img, is_mask=False):
		# Resize
		pil_img = pil_img.resize(self.img_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
		
		# Ensure mask channels to 1
		if is_mask:
			pil_img = pil_img.convert('L')
		
		# To ndarray
		img_ndarray = np.asarray(pil_img, dtype=np.uint32)
		
		img_ndarray = (img_ndarray * 255) // np.max(img_ndarray)
		
		img_ndarray = img_ndarray.astype(np.uint8)
		
		# Apply CLAHE
		if self.clahe is not None:
			img_ndarray = self.clahe.apply(img_ndarray)

		# Ensure image channels on index 0
		if not is_mask:
			if img_ndarray.ndim == 2:
				img_ndarray = img_ndarray[np.newaxis, ...]
			else:
				img_ndarray = img_ndarray.transpose((2, 0, 1))
			
			# Image to float notation
			img_ndarray = img_ndarray / 255
			
		# Ensure binary mask
		elif np.max(img_ndarray) > 1:
			threshold = (np.max(img_ndarray) - np.min(img_ndarray))//2
			img_ndarray = (img_ndarray > threshold) * 1

		return img_ndarray
	
	'''
	
	'''
	@staticmethod
	def load(filename):
		ext = filename[-4:]
		if ext in ['.npz', '.npy']:
			return Image.fromarray(np.load(filename))
		elif ext in ['.pt', '.pth']:
			return Image.fromarray(torch.load(filename).numpy())
		else:
			return Image.open(filename)
	
	'''
	getitem Method
	
	Default __getitem__ method. Allows to iterate on the dataset.
	
	Inputs: 
		- idx: (Int) Item number to retrieve
		
	Outputs:
		- target: (dict)
			Keys:
				- image: (torch.Tensor) image
				- masks: (list(torch.Tensor)) image mask list
				- labels: (list(torch.Tensor)) image mask label list
				- image_id: (torch.Tensor) images ids
				- mask_area: (torch.Tensor) images box areas
				- img_path: (String) images relative paths
	'''	    
	def __getitem__(self, idx):
		# Image ID
		image_id = torch.tensor(idx)
	
		# Load
		img = self.load(os.path.join(self.path, self.data['image'][idx]))
		
		# Preprocess
		img = self.preprocess(img, is_mask=False)
		
		# Labels To Tensor
		img = torch.as_tensor(img.copy()).float().contiguous()
		
		# Data Augmentation
		if self.transforms is not None:
		    img, target = self.transforms(img, target)
		
		# Target Dictionary
		target = {}
		target['image'] = img
		target['image_id'] = image_id
		target['mask_area'] = torch.sum(mask) #[torch.sum(msk) for msk in mask]
		target['img_path'] = self.data['image'][idx]
		target['img_size'] = self.img_size
		
		return target
		
	'''
	getinfo Method
	
	Retrieves the basic dataset information.
	
	Inputs:
		- None
	
	Outputs:
		- target: (dict)
			Keys:
				- n_classes: (int) number of detected classes
				- n_objects: (int) maximum number of detected objects
				- n_channels: (int) number of detected channels
				- img_size: (int) image size
	'''
	def getinfo(self):
		info = {}
		info['n_channels'] = self.channels
		info['n_images'] = self.size
		
		return info
		
if __name__ == '__main__':

	import matplotlib.pyplot as plt

	dataset = DefaultDataset('../data/Montgomery/')#'./test_imgs/')
	
	print(dataset.getinfo())
	
	tgt = dataset[0]
	img = tgt['image']

	fig = plt.figure(figsize=(8, 8))

	plt.title(f'Original Image of id: {int(tgt["image_id"])}')
	plt.imshow(img.permute(1, 2, 0))
	plt.show()
	
	tgt['image'] = f'Image of shape {tgt["image"].shape}'
	print(tgt)
	
