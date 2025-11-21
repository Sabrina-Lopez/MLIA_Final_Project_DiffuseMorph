from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io


class MNISTDataset(Dataset):
	def __init__(self, dataroot, split='test'):
		self.split = split
		self.dataroot = dataroot
		self.pairs = []

		digit_dirs = [d for d in sorted(os.listdir(dataroot)) if os.path.isdir(os.path.join(dataroot, d))]
		all_images = []
		for d in digit_dirs:
			digit_path = os.path.join(dataroot, d)
			# Traverse type subdirectories
			for root, _, files in os.walk(digit_path):
				for f in files:
					if f.lower().endswith('.png'):
						all_images.append(os.path.join(root, f))

		# Build consecutive pairs
		for i in range(0, len(all_images) - 1, 2):
			self.pairs.append([all_images[i], all_images[i + 1]])

		self.data_len = len(self.pairs)
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
		dataX, dataY = self.pairs[index]
		imgX = io.imread(dataX, as_gray=True).astype(np.float32)
		imgY = io.imread(dataY, as_gray=True).astype(np.float32)

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

		fileInfo = [os.path.basename(dataX), os.path.basename(dataY)]
		
		return {'M': imgX, 'F': imgY, 'MC': imgX_rgb, 'FC': imgY_rgb, 'nS': 7, 'P':fileInfo, 'Index': index}