import argparse
import os
import os.path as osp

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as T
import torch.nn.functional as F

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            images.append(path)
    return images

class FolderDataset(Dataset):
	def __init__(self, root, transform=None, preprocess=None):
		self.paths = sorted(make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = pil_loader(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im,from_path


class SDFlowDataset(Dataset):
    def __init__(self,
                 index_file,
                 image_root,
                 latents_file,
                 preds_file,
                 train=True,
                 transform=None):
        
        self.index_file=index_file
        self.image_root = image_root
        self.transform = transform
        self.latents = torch.load(latents_file,map_location='cpu')
        self.preds = torch.load(preds_file,map_location='cpu')
        
        # process the attributes
        df = pd.read_csv(self.index_file, index_col=None)
        df = df[df['split'].values.astype(bool) == (not train)]
        
        self.image_list = df['path'].values

    def _lookup_precomputed(self, store, file):
        if isinstance(store, dict) and 'paths' in store and 'values' in store:
            if not hasattr(self, '_precomputed_index'):
                self._precomputed_index = {}
            cache_key = id(store)
            if cache_key not in self._precomputed_index:
                self._precomputed_index[cache_key] = {
                    path: idx for idx, path in enumerate(store['paths'])
                }
            return store['values'][self._precomputed_index[cache_key][file]]

        if isinstance(store, dict):
            return store[file]

        return store[int(osp.splitext(file)[0])]
        
    def __getitem__(self, index):
        
        file = self.image_list[index]
        
        img = pil_loader(osp.join(self.image_root,file))
        if self.transform is not None:
            img = self.transform(img)
        
        latent = self._lookup_precomputed(self.latents, file)
        pred = self._lookup_precomputed(self.preds, file)
        
        return img,latent,pred
    
    def __len__(self):
        return len(self.image_list)


class ImageDataset(Dataset):
    def __init__(self,
                 index_file,
                 image_root,
                 train=True,
                 transform=None):
        
        self.index_file=index_file
        self.image_root = image_root
        self.transform = transform
        
       
        # process the attributes
        df = pd.read_csv(self.index_file, index_col=None)
        df = df[df['split'].values.astype(bool) == (not train)]
        
        self.image_list = df['path'].values
    
    def __getitem__(self, index):
        file = self.image_list[index]
        img = pil_loader(osp.join(self.image_root,file))
        if self.transform is not None:
            img = self.transform(img)
            
        return img
    
    def __len__(self):
        return len(self.image_list)
