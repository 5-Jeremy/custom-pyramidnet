# This script downloads the images for the train and test splits of the CIFAR-10 dataset. 
# It should only be run once
# After running this script, the cifar10 directory containing the images will be created
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

download_url("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", '.')
tarfile.open('./cifar10.tgz', 'r:gz').extractall(path='./')
os.remove('./cifar10.tgz')