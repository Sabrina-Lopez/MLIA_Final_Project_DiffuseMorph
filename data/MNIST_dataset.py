from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io
import random


class MNISTDataset(Dataset):
	def __init__(self, dataroot, split='test'):
		self.split = split
		self.dataroot = dataroot
		self.all_images = []
		self.images_by_class = {}

		# Walk the directory and populate all_images and images_by_class
		digit_dirs = [d for d in sorted(os.listdir(dataroot)) if os.path.isdir(os.path.join(dataroot, d))]
		for digit_dir in digit_dirs:
			digit_class = digit_dir.split('_')[0] # e.g., '0_affined' -> '0'
			digit_path = os.path.join(dataroot, digit_dir)
			self.images_by_class[digit_class] = []
			for root, _, files in os.walk(digit_path):
				for f in files:
					if f.lower().endswith('.png'):
						path = os.path.join(root, f)
						self.all_images.append((path, digit_class))
						self.images_by_class[digit_class].append(path)

		"""
		print(f"Total images found: {len(self.all_images)}")
		print(f"All images: {self.all_images}")
		print(f"Image classes: {self.images_by_class}")
		"""
		
		self.data_len = len(self.all_images)
		self.target_height = 32
		self.target_width = 32

	def __len__(self):
		return self.data_len

	def _pad(self, arr):
		h, w = arr.shape
		pad_h = self.target_height - h
		pad_w = self.target_width - w
		pt = max(pad_h // 2, 0)
		pb = max(pad_h - pt, 0)
		pl = max(pad_w // 2, 0)
		pr = max(pad_w - pl, 0)
		return np.pad(arr, ((pt, pb), (pl, pr)), mode='constant', constant_values=0)

	def __getitem__(self, index):
		# Get the moving image based on the global index
		moving_path, moving_class = self.all_images[index]

		# Get the list of candidates for the fixed image (same class)
		candidate_paths = self.images_by_class[moving_class]
		
		# Select a random fixed image from the same class
		fixed_path = random.choice(candidate_paths)
		
		# To be safe, ensure we don't pair an image with itself
		if moving_path == fixed_path and len(candidate_paths) > 1:
			while moving_path == fixed_path:
				fixed_path = random.choice(candidate_paths)

		imgX = io.imread(moving_path, as_gray=True).astype(np.float32)
		imgY = io.imread(fixed_path, as_gray=True).astype(np.float32)

		if imgX.max() > 0:
			imgX /= imgX.max()
		if imgY.max() > 0:
			imgY /= imgY.max()

		imgX = self._pad(imgX)
		imgY = self._pad(imgY)

		imgX_rgb = np.repeat(imgX[:, :, np.newaxis], 3, axis=-1) * 255.0
		imgY_rgb = np.repeat(imgY[:, :, np.newaxis], 3, axis=-1) * 255.0

		imgX = imgX[:, :, np.newaxis]
		imgY = imgY[:, :, np.newaxis]

		imgX, imgY = Util.transform_augment([imgX, imgY], split=self.split, min_max=(-1, 1))

		fileInfo = [os.path.basename(moving_path), os.path.basename(fixed_path)]
		
		return {'M': imgX, 'F': imgY, 'MC': imgX_rgb, 'FC': imgY_rgb, 'nS': 7, 'P':fileInfo, 'Index': index}