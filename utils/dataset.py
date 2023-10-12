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
	def __init__(self, file_path: str, img_size: int=512, train=True, transforms=None, clahe=False, histeq=False):
	
		super(DefaultDataset, self).__init__()
		
		# Ensure image_size is correctly formatted
		assert img_size > 0 if img_size is not None else True, 'Size must be greater than 0 or None for full size'
			
		# Store image_size and transforms
		self.img_size = (img_size, img_size) if img_size is not None else None
		self.transforms = transforms
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if clahe else None
		self.histeq = histeq
		
		# Read train/test files
		if train:

			imgs = pd.read_csv(os.path.join(file_path, 'train.txt'), sep='\t', names=('image',), index_col=False)
			msks = pd.read_csv(os.path.join(file_path, 'train_mask.txt'), sep='\t', header=None, index_col=False)
			msks = msks.fillna('N/A')
			msks['mask'] = msks.values.tolist()
			imgs = imgs.join(msks['mask'], lsuffix='_left', rsuffix='_right')
			cls = pd.read_csv(os.path.join(file_path, 'train_class.txt'), sep='\t', header=None, index_col=False)
			cls = cls.fillna(0)
			cls['class'] = cls.values.tolist()
			imgs = imgs.join(cls['class'], lsuffix='_left', rsuffix='_right')
			
		else:
			imgs = pd.read_csv(os.path.join(file_path, 'test.txt'), sep='\t', names=('image',), index_col=False)
			msks = pd.read_csv(os.path.join(file_path, 'test_mask.txt'), sep='\t', header=None, index_col=False)
			msks = msks.fillna('N/A')
			msks['mask'] = msks.values.tolist()
			imgs = imgs.join(msks['mask'], lsuffix='_left', rsuffix='_right')
			cls = pd.read_csv(os.path.join(file_path, 'test_class.txt'), sep='\t', header=None, index_col=False)
			cls = cls.fillna(0)
			cls['class'] = cls.values.tolist()
			imgs = imgs.join(cls['class'], lsuffix='_left', rsuffix='_right')
			
		# Set values
		self.data = imgs.to_dict('list')
		self.size = len(imgs)
		self.objects = len(self.data['mask'][0])
		self.classes = len(set([item for sublist in cls.loc[:, cls.columns != 'class'].values.tolist() for item in sublist]))
		self.path = file_path
		
		# Get number of channels
		ch_img = self.load(os.path.join(self.path, self.data['image'][0]))
		ch_img = self.preprocess(ch_img, is_mask=False)
		self.channels = ch_img.shape[0]
		
		# Ensure basic image-mask-class element-count relations
		assert len(self.data['image']) == len(self.data['mask']) == len(self.data['class']), 'Number of images not equal to number of masks'
		assert len(self.data['mask'][0]) == len(self.data['class'][0]), 'Number of masks and classes should be the same for every image'
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
		if self.img_size:
			pil_img = pil_img.resize(self.img_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
		
		# Ensure mask channels to 1
		if is_mask:
			pil_img = pil_img.convert('L')
		
		# To ndarray
		img_ndarray = np.asarray(pil_img, dtype=np.uint32)
		
		img_ndarray = (img_ndarray * 255) // np.max(img_ndarray)
		
		img_ndarray = img_ndarray.astype(np.uint8)
		
		# Apply CLAHE
		if self.clahe is not None and not is_mask:
			img_ndarray = self.clahe.apply(img_ndarray)
			
		if self.histeq and not is_mask:
			img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2YCrCb)
			img_ndarray[:, :, 0] = cv2.equalizeHist(img_ndarray[:, :, 0]) 
			img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_YCrCb2BGR)
			#img_ndarray = np.asarray(img_ndarray, dtype=np.uint32)

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
		mask = [self.load(os.path.join(self.path, mask)) for mask in self.data['mask'][idx] if mask != 'N/A']
		
		# Preprocess
		img = self.preprocess(img, is_mask=False)
		mask = np.array([self.preprocess(msk, is_mask=True) for msk in mask])
		
		# Labels To Tensor
		img = torch.as_tensor(img.copy()).float().contiguous()
		mask = torch.as_tensor(mask.copy()).long().contiguous() #[torch.as_tensor(msk.copy()).long().contiguous() for msk in mask]
		mask = torch.squeeze(mask)
		labels = torch.as_tensor(self.data['class'][idx])#[torch.as_tensor(label) for label in self.data['class'][idx] if label != 0] #########3
		
		# Data Augmentation
		if self.transforms is not None:
		    img, target = self.transforms(img, target)
		
		# Target Dictionary
		target = {}
		target['image'] = img
		target['masks'] = mask
		target['labels'] = labels
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
		info['n_classes'] = self.classes
		info['n_objects'] = self.objects
		info['n_channels'] = self.channels
		info['n_images'] = self.size
		
		return info
		
class Grayscale2RGBDataset(DefaultDataset):
	
	@staticmethod
	def load(filename):
		ext = filename[-4:]
		if ext in ['.npz', '.npy']:
			return Image.fromarray(np.load(filename)).convert('RGB')
		elif ext in ['.pt', '.pth']:
			return Image.fromarray(torch.load(filename).numpy()).convert('RGB')
		else:
			return Image.open(filename).convert('RGB')
		
class GrayscaleDataset(DefaultDataset):

	def preprocess(self, pil_img, is_mask=False):
		pil_img = pil_img.convert('L')
		return super().preprocess(pil_img, is_mask)
		


if __name__ == '__main__':

	import matplotlib.pyplot as plt
	
	dataset_dict = {'DefaultDataset': DefaultDataset, 
			'Grayscale2RGBDataset': Grayscale2RGBDataset,
			'GrayscaleDataset': GrayscaleDataset}
	
	for dataset_name in dataset_dict.keys():
	
		print('-'*30)
		print(f'\nTesting {dataset_name}:\n')

		dataset = dataset_dict[dataset_name]('../data/Montgomery/')#'./test_imgs/')
		#dataset = dataset_dict[dataset_name]('./test_imgs/')
		
		print(dataset.getinfo())
		
		tgt = dataset[0]
		img = tgt['image']
		
		w = 10
		h = 10
		fig = plt.figure(figsize=(8, 8))
		columns = 3
		rows = 1
		fig.add_subplot(rows, columns, 1)
		plt.title(f'Original Image of id: {int(tgt["image_id"])}')
		plt.imshow(img.permute(1, 2, 0))
		fig.add_subplot(rows, columns, 2)
		plt.title(f'Mask of Class: {tgt["labels"]}')
		plt.imshow(tgt['masks'], cmap='Greys_r')
		fig.add_subplot(rows, columns, 3)
		plt.title('Masked Image')
		plt.imshow((img*tgt['masks']).permute(1, 2, 0))
		plt.show()
		
		tgt['masks'] = f'Found {len(tgt["masks"])} masks of size {tgt["masks"].shape}'
		tgt['image'] = f'Image of shape {tgt["image"].shape}'
		print(tgt)
		
		print(f'\nFinished testing {dataset_name}:\n')
	
