import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from torchvision.transforms import ColorJitter
from PIL import Image

class Img_Parser():
	def __init__(self):
		self.ColorTransform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)

	def parse_record(self, record, augmentation_level):
		"""Parse a record to an image and perform data preprocessing.

		Args:
			record: An array of shape [3072,]. One row of the x_* matrix.
			training: A boolean. Determine whether it is in training mode.

		Returns:
			image: An array of shape [3, 32, 32].
		"""
		
		# Reshape from [depth * height * width] to [depth, height, width].
		image = record.reshape((3, 32, 32))

		image = self.preprocess_image(image, augmentation_level)

		return image

	def preprocess_image(self, image, augmentation_level):
		"""Preprocess a single image of shape [height, width, depth].

		Args:
			image: An array of shape [3, 32, 32].
			training: A boolean. Determine whether it is in training mode.

		Returns:
			image: An array of shape [3, 32, 32]. The processed image.
		"""
		
		if augmentation_level > 0:
			# If augmentation_level is 1, either apply color jitter random cropping, with equal probability
			# If augmentation_level is 2, apply both
			augment_type = np.random.binomial(1,0.5,size=1)
			if augment_type == 1 or augmentation_level >= 2:
				# plt.imshow(np.moveaxis(image, 0, -1))
				# plt.savefig('image.png')
				# plt.show()
				# Color jitter
				image = np.moveaxis(image, 0, -1)
				PIL_image = Image.fromarray(image, mode='RGB')
				PIL_image = self.ColorTransform(PIL_image)
				image = np.array(PIL_image)
				image = np.moveaxis(image, 2, 0)
				# image = np.round(image + np.expand_dims(color_jitter_vals,(1)),0).astype(int)
			if augment_type == 0 or augmentation_level >= 2:
				# Resize the image to add four extra pixels on each side.
				image = np.pad(image, ((0,), (4,), (4,)))
				if augmentation_level >= 3:
					noisy_pixels = np.random.choice(255, size=(3,40,40))
					noisy_pixels[:,4:-4,4:-4] = 0
					image = image + noisy_pixels
				# Randomly crop a [32, 32] section of the image.
				crop_point_row = np.random.randint(0,9)
				crop_point_col = np.random.randint(0,9)
				image = image[:,crop_point_row:crop_point_row+32,crop_point_col:crop_point_col+32]
			
			# Insert noisy pixels
			if augmentation_level >= 3:
				pixel_noise_prob = 0.02
				no_noise_prob = 1 - pixel_noise_prob
				each_color_prob = pixel_noise_prob/8
				noise_mask = np.random.choice(9, size=(32,32), p=([no_noise_prob] + [each_color_prob]*8))
				image[:, noise_mask == 1] = np.repeat(np.array([235, 64, 52])[:,np.newaxis], (noise_mask == 1).sum(), axis=1)
				image[:, noise_mask == 2] = np.repeat(np.array([60, 232, 215])[:,np.newaxis], (noise_mask == 2).sum(), axis=1)
				image[:, noise_mask == 3] = np.repeat(np.array([247, 252, 78])[:,np.newaxis], (noise_mask == 3).sum(), axis=1)
				image[:, noise_mask == 4] = np.repeat(np.array([116, 238, 21])[:,np.newaxis], (noise_mask == 4).sum(), axis=1)
				image[:, noise_mask == 5] = np.repeat(np.array([240, 0, 255])[:,np.newaxis], (noise_mask == 5).sum(), axis=1)
				image[:, noise_mask == 6] = np.repeat(np.array([0, 30, 255])[:,np.newaxis], (noise_mask == 6).sum(), axis=1)
				image[:, noise_mask == 7] = np.repeat(np.array([0, 0, 0])[:,np.newaxis], (noise_mask == 7).sum(), axis=1)
				image[:, noise_mask == 8] = np.repeat(np.array([255, 255, 255])[:,np.newaxis], (noise_mask == 8).sum(), axis=1)

			# Randomly flip the image horizontally.
			if np.random.binomial(1, 0.5) == 1:
				image = np.flip(image,axis=2)
		else:
			pass

		# Normalize
		mean = np.mean(image, axis=(1,2))
		# Enable the mean array to be broadcasted to the shape of the image
		mean = np.expand_dims(mean, axis=(1,2))
		image = image - mean
		stdev = np.sqrt(np.sum(np.power(image,2), axis=(1,2))/(32*32)) + 0.1 # To avoid division by zero
		image = image/stdev[:, np.newaxis, np.newaxis]
		

		return image
