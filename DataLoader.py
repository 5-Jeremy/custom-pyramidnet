import os
import pickle
import numpy as np
from torchvision.datasets import ImageFolder
from math import ceil

# This includes the train/val split
def load_training_images(train_ratio=0.9, classes='all'):
	train_dataset = ImageFolder('./cifar10/train')
	
	if classes == 'all':
		x = [np.moveaxis(np.asarray(train_dataset[i][0]), -1, 0) for i in range(len(train_dataset))]
		y = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
		# Make sure the classes are evenly represented in the training and validation sets
		x_train = []
		y_train = []
		x_valid = []
		y_valid = []
		for i in range(10):
			x_train.append(x[i*5000:ceil((i+train_ratio)*5000)])
			y_train.append(y[i*5000:ceil((i+train_ratio)*5000)])
			x_valid.append(x[ceil((i+train_ratio)*5000):(i+1)*5000])
			y_valid.append(y[ceil((i+train_ratio)*5000):(i+1)*5000])
		x_train = np.concatenate(x_train)
		y_train = np.concatenate(y_train)
		x_valid = np.concatenate(x_valid)
		y_valid = np.concatenate(y_valid)
	else:
		if type(classes) is not list:
			raise ValueError("Expected classes to be a list containing two valid class indices")
		x_0 = [np.moveaxis(np.asarray(train_dataset[i][0]), -1, 0) for i in range(5000*classes[0],5000*(classes[0]+1)) if train_dataset[i][1] == classes[0]]
		x_1 = [np.moveaxis(np.asarray(train_dataset[i][0]), -1, 0) for i in range(5000*classes[1],5000*(classes[1]+1)) if train_dataset[i][1] == classes[1]]
		num_train = ceil(train_ratio*5000)
		x_train = x_0[:num_train] + x_1[:num_train]
		y_train = np.array([classes[0]]*num_train + [classes[1]]*num_train) 
		x_valid = x_0[num_train:] + x_1[num_train:]
		y_valid = np.array([classes[0]]*(5000-num_train) + [classes[1]]*(5000-num_train))

	return x_train, y_train, x_valid, y_valid

def load_testing_images(data_dir):
	"""Load the images in private testing dataset.

	Args:
		data_dir: A string. The directory where the testing images
		are stored.

	Returns:
		x_test: An numpy array of shape [N, 3072].
			(dtype=np.float32)
	"""
	# Without applying any transformations, you get a set of PIL images
	test_dataset = ImageFolder('./cifar10/test')

	return test_dataset