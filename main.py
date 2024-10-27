import torch
import os, argparse, datetime
import numpy as np
from Model import MyModel
from DataLoader import load_training_images, load_testing_images
from ImageUtils import Img_Parser

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Select 'train' to train a base model from scratch. Select 'validate' to load existing base model checkpoints and run inference on the validation data. Select 'Confusion matrix' to load a base model checkpoint and generate a confusion matrix based on the training and validation data")
parser.add_argument("--model_config", default="DefaultModelConf.yaml", help="YAML file containing the config options for the model architecture")
parser.add_argument("--training_config", default="TrainingConf.yaml", help="YAML file containing the config options for the training process")
parser.add_argument("--data_config", default="DataConf.yaml", help="YAML file containing the config options for loading data and checkpoints as well as saving outputs")
args = parser.parse_args()

if __name__ == '__main__':
	data_configs = OmegaConf.load(f'conf/{args.data_config}')
	mode = args.mode
	data_dir = data_configs.data_dir
	if not data_dir.endswith('/'):
		raise ValueError("Expected data_dir to be a directory name ending with /")
	# save_dir is the base directory for all outputs. Each time this script is run, a new directory will be created in 
	# 	save_dir to store the output of that run. The directory will be labeled using the date and time, as well as the
	# 	type of operation being performed
	save_dir = data_configs.save_dir
	if not save_dir.endswith('/'):
		raise ValueError("Expected save_dir to be a directory name ending with /")
	
	output_dir = save_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + mode + '/'
	os.makedirs(output_dir, exist_ok=True)
	config_logfile = open(output_dir + 'configurations.txt', 'w')
	
	model_configs = OmegaConf.load(f'conf/{args.model_config}')
	config_logfile.write(OmegaConf.to_yaml(model_configs))

	model = MyModel(model_configs, output_dir)
	model.logfile = open(output_dir + '{}_log.txt'.format(mode), 'w')
	model.LOG("Mode: {}".format(mode))

	training_configs = OmegaConf.load(f'conf/{args.training_config}')
	config_logfile.write(OmegaConf.to_yaml(training_configs))
	config_logfile.close()

	eval_chkpt = data_configs.eval_chkpt

	if mode == 'train':
		x_train, y_train, x_valid, y_valid = load_training_images(train_ratio=data_configs["train_split_ratio"])

		model.train(x_train, y_train, training_configs)
		model.evaluate(x_valid, y_valid, output_dir + 'chkpts/', post_training=True)

	elif mode == 'validate':
		# Use the validation data set aside using train_valid_split; make sure that train_split_ratio is consistent with the training run
		x_train, y_train, x_valid, y_valid = load_training_images(train_ratio=data_configs["train_split_ratio"], classes="all")
		model.evaluate(x_valid, y_valid, eval_chkpt)
	
	elif mode == 'confusion_matrix':
		x_train, y_train, x_valid, y_valid = load_training_images(train_ratio=data_configs["train_split_ratio"])
		# Load the model checkpoint once to avoid reloading it for each confusion matrix
		model.load_checkpoint(eval_chkpt)
		model.LOG("Restored model parameters from {}".format(eval_chkpt))
		# Make separate confusion matrices for the training, validation, and test data
		model.LOG('Get Confusion Matrix For Training Data')
		confusion_matrix_train = model.get_confusion_matrix(x_train, y_train)
		np.save(output_dir + 'confusion_matrix_train.npy', confusion_matrix_train)
		model.LOG("=======================================================")
		model.LOG('Get Confusion Matrix For Validation Data')
		confusion_matrix_valid = model.get_confusion_matrix(x_valid, y_valid)
		np.save(output_dir + 'confusion_matrix_valid.npy', confusion_matrix_train)
		model.LOG("=======================================================")
		model.LOG('Get Confusion Matrix For Test Data')
		test_dataset = load_testing_images(data_dir)
		x_test = [np.moveaxis(np.asarray(test_dataset[i][0]), -1, 0) for i in range(len(test_dataset))]
		y_test = [test_dataset[i][1] for i in range(len(test_dataset))]
		confusion_matrix_test = model.get_confusion_matrix(x_test, y_test)
		np.save(output_dir + 'confusion_matrix_test.npy', confusion_matrix_train)

	elif mode == 'test':
		if eval_chkpt is None or eval_chkpt.endswith('/'):
			raise ValueError("eval_chkpt needs to be a single checkpoint file when generating predictions on test data.")
		x_test = load_testing_images(data_dir)
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test, eval_chkpt)
		np.save(output_dir + 'predictions.npy', predictions)
	
	model.logfile.close()
