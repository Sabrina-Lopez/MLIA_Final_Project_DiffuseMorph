from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io


class GoogleDrawDataset(Dataset):
	def __init__(self, dataroot, split='test'):
		self.split = split
		self.dataroot = dataroot
		self.pairs = []

		# Categories assumed as subdirectories
		categories = [c for c in sorted(os.listdir(dataroot)) if os.path.isdir(os.path.join(dataroot, c))]
		all_images_paths = []
		for c in categories:
			cat_path = os.path.join(dataroot, c)
			for root, _, files in os.walk(cat_path):
				for f in files:
					if f.lower().endswith('.png'):
						all_images_paths.append(os.path.join(root, f))

		# Find max dimensions and calculate target size
		max_h, max_w = 0, 0
		print("Scanning GoogleDraw dataset to determine optimal dimensions")
		for path in all_images_paths:
			try:
				h, w = io.imread(path, as_gray=True).shape
				if h > max_h:
					max_h = h
				if w > max_w:
					max_w = w
			except Exception as e:
				print(f"Warning: Could not read {path}. Skipping. Error: {e}")
		
		# Round up to the nearest multiple of 16 for U-Net compatibility
		self.target_height = ((max_h + 15) // 16) * 16
		self.target_width = ((max_w + 15) // 16) * 16
		
		print(f"GoogleDraw Dataset: Max dims found ({max_h}, {max_w}). Target size set to ({self.target_height}, {self.target_width}).")

		# Build consecutive pairs
		for i in range(0, len(all_images_paths) - 1, 2):
			self.pairs.append([all_images_paths[i], all_images_paths[i + 1]])

		self.data_len = len(self.pairs)

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
		dataX_path, dataY_path = self.pairs[index]
		imgX = io.imread(dataX_path, as_gray=True).astype(np.float32)
		imgY = io.imread(dataY_path, as_gray=True).astype(np.float32)

		# Normalize to [0, 1] range
		if imgX.max() > 0:
			imgX /= imgX.max()
		if imgY.max() > 0:
			imgY /= imgY.max()

		# Pad to the dynamically calculated target size
		imgX = self._pad(imgX)
		imgY = self._pad(imgY)

		# Create RGB versions for visualization
		imgX_rgb = np.repeat(imgX[:, :, np.newaxis], 3, axis=-1) * 255.0
		imgY_rgb = np.repeat(imgY[:, :, np.newaxis], 3, axis=-1) * 255.0

		# Add channel dimension for transform
		imgX = imgX[:, :, np.newaxis]
		imgY = imgY[:, :, np.newaxis]

		imgX, imgY = Util.transform_augment([imgX, imgY], split=self.split, min_max=(-1, 1))

		fileInfo = [os.path.basename(dataX_path), os.path.basename(dataY_path)]
		
		return {'M': imgX, 'F': imgY, 'MC': imgX_rgb, 'FC': imgY_rgb, 'nS': 7, 'P':fileInfo, 'Index': index}

