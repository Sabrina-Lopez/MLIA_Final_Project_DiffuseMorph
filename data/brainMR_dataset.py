from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np


class brainMRDataset(Dataset):
	def __init__(self, dataroot, split='test'):
		self.split = split
		self.dataroot = dataroot
		self.pairs = []

		# Determine file names based on split
		img_file = f'brain_{split}_image_final.npy'
		lbl_file = f'brain_{split}_label.npy'
		img_path = os.path.join(dataroot, img_file)
		lbl_path = os.path.join(dataroot, lbl_file)

		if os.path.exists(img_path):
			images = np.load(img_path)
		else:
			raise FileNotFoundError(f'Missing image file: {img_path}')

		self.images = images.astype(np.float32)
		self.data_len = images.shape[0]

		# Target padded size (divisible by 16 for 4 downsamplings)
		self.target_height = 112
		self.target_width = 80

	def __len__(self):
		return self.data_len

	def _pad(self, arr):
		h, w = arr.shape
		pad_h = self.target_height - h
		pad_w = self.target_width - w
		if pad_h < 0 or pad_w < 0:
			raise ValueError('Unexpected larger image size than target.')
		pt = pad_h // 2
		pb = pad_h - pt
		pl = pad_w // 2
		pr = pad_w - pl
		return np.pad(arr, ((pt, pb), (pl, pr)), mode='constant', constant_values=0)

	def __getitem__(self, index):
		# Get image pair from the preloaded numpy array
		# self.images has shape (N, 2, H, W)
		# image_pair will have shape (2, H, W)
		image_pair = self.images[index]

		# Moving is the first image, fixed is the second
		moving = image_pair[0].astype(np.float32)
		fixed = image_pair[1].astype(np.float32)

		# Normalize to [0,1] range for processing
		if moving.max() > 0:
			moving = moving / moving.max()
		if fixed.max() > 0:
			fixed = fixed / fixed.max()

		# Pad images to the target size (e.g., 112x80)
		moving = self._pad(moving)
		fixed = self._pad(fixed)

		# Create 3-channel RGB versions for visualization BEFORE transform
		# These should be float arrays in the [0, 255] range
		moving_rgb = np.repeat(moving[:, :, np.newaxis], 3, axis=-1) * 255.0
		fixed_rgb = np.repeat(fixed[:, :, np.newaxis], 3, axis=-1) * 255.0

		# Add a channel dimension for the grayscale images for the transform
		moving = moving[:, :, np.newaxis]
		fixed = fixed[:, :, np.newaxis]

		# Apply augmentations and normalize to [-1, 1] for the model
		moving, fixed = Util.transform_augment([moving, fixed], split=self.split, min_max=(-1, 1))

		# Create file info
		fileInfo = [f'brain_{self.split}_{index}_M.png', f'brain_{self.split}_{index}_F.png']
		
		return {'M': moving, 'F': fixed, 'MC': moving_rgb, 'FC': fixed_rgb, 'nS': 7, 'P': fileInfo, 'Index': index}

