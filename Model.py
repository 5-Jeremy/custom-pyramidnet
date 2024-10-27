### YOUR CODE HERE
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import BaseNetwork
from ImageUtils import Img_Parser

from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

	def __init__(self, configs, output_dir):
		self.configs = configs
		self.output_dir = output_dir
		self.network = BaseNetwork(configs).cuda()
		self.img_parser = Img_Parser()
		total_size = 0
		for p in self.network.parameters():
			total_size += p.nelement()
		print("Number of params in network: ", total_size)
		
		if configs["activation_type"] == 'PReLU':
			self.using_PReLU = True
		else:
			self.using_PReLU = False

	def train(self, x_train, y_train, configs):
		print('Train Network')
		self.network.train()

		CELoss = nn.CrossEntropyLoss()
		
		if self.using_PReLU:
			# Get the optimizer groups with their respective weight decay values (this allows us to set the weight decay of the PReLU parameters to zero)
			non_prelu_weights = set()
			prelu_weights = set()
			for mn, m in self.network.named_modules():
				for pn, p in m.named_parameters():
					fpn = '%s.%s' % (mn, pn) if mn else pn
					non_prelu_weights.add(fpn)
					if pn.endswith('prelu') or ('prelu' in pn):
						prelu_weights.add(fpn)
			non_prelu_weights = non_prelu_weights - prelu_weights
			param_dict = {pn: p for pn, p in self.network.named_parameters()}
			optim_groups = [
				{"params": [param_dict[pn] for pn in sorted(list(non_prelu_weights))], "weight_decay": configs["weight_decay"]},
				{"params": [param_dict[pn] for pn in sorted(list(prelu_weights))], "weight_decay": 0.0},
			]
			optimizer = torch.optim.SGD(optim_groups, lr=configs["learning_rate"], momentum=0.9, nesterov=True)
		else:
			optimizer = torch.optim.SGD(self.network.parameters(), lr=configs["learning_rate"], weight_decay=configs["weight_decay"], momentum=0.9, nesterov=True)
		
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs["lr_step_size"], gamma=configs["lr_decay"])

		max_epoch = configs['max_epoch']
		batch_size = configs['batch_size']

		# Determine how many batches in an epoch
		num_samples = len(x_train)
		num_batches = num_samples // batch_size
		augmentation_level = 0
		for epoch in range(1, max_epoch+1):
			start_time = time.time()
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = [x_train[i] for i in shuffle_index]
			curr_y_train = y_train[shuffle_index]
			# Set the learning rate for this epoch
			if epoch > 1:
				scheduler.step()
			# After 100 epochs have passed, freeze the PReLU weights so that the rest of the model weights can adapt to them
			if self.using_PReLU and epoch == 101:
				optimizer.param_groups[1]['lr'] = 0
			
			for i in range(num_batches):
				batch_x = curr_x_train[i*batch_size:(i+1)*batch_size]
				batch_x_list = [self.img_parser.parse_record(batch_x[j], augmentation_level=augmentation_level) for j in range(batch_size)]
				batch_x = np.stack(batch_x_list)
				batch_x = torch.tensor(batch_x, dtype=torch.float32).cuda()
				batch_y = curr_y_train[i*batch_size:(i+1)*batch_size]
				batch_y = torch.tensor(batch_y).type(torch.LongTensor).cuda()
				batch_out = self.network(batch_x)
				loss = CELoss(batch_out, batch_y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
			duration = time.time() - start_time
			self.LOG('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
			
			if epoch == configs["augmentation_phase_1_start_epoch"]:
				augmentation_level = 1
			if epoch == configs["augmentation_phase_2_start_epoch"]:
				augmentation_level = 2
			if epoch == configs["augmentation_phase_3_start_epoch"]:
				augmentation_level = 3
			
			if epoch % 10 == 0:
				chkpts_path = self.output_dir + 'chkpts/'
				os.makedirs(chkpts_path, exist_ok=True)
				torch.save(self.network.state_dict(), chkpts_path + 'model-%d.ckpt'%(epoch))
				print("Checkpoint has been created.")

	def unbatched_eval_loop(self, x):
		num_samples = len(x)
		preds = []
		for i in tqdm(range(num_samples)):
			input = np.expand_dims(self.img_parser.parse_record(x[i], augmentation_level=0), axis=0)
			input_tensor = torch.from_numpy(input).type(torch.float32).cuda()
			model_output = self.network(input_tensor)
			preds.append(torch.argmax(model_output).cpu())
		return np.array(preds, dtype=int)
	
	def batched_eval_loop(self, x, batch_size):
		num_samples = len(x)
		preds = torch.empty((num_samples,))
		num_batches = num_samples // batch_size
		for i in tqdm(range(num_batches)):
			batch_x = x[i*batch_size:(i+1)*batch_size]
			batch_x_list = [self.img_parser.parse_record(batch_x[j], augmentation_level=0) for j in range(batch_size)]
			batch_x = np.stack(batch_x_list)
			batch_x = torch.tensor(batch_x, dtype=torch.float32).cuda()
			batch_out = self.network(batch_x)
			batch_preds = torch.argmax(batch_out,dim=1)
			preds[batch_size*i:batch_size*(i+1)] = batch_preds
		return np.array(preds, dtype=int)

	def evaluate(self, x, y, chkpts_path='eval_chkpts/', post_training=False):
		self.network.eval()
		print('Validate Network')

		# If chkpts_path is a directory, perform the evaluation on each of the checkpoints in the directory
		# If chkpts_path is a file, perform the evaluation on that single checkpoint
		# By default, if we are doing a training run then we set it to evaluate the checkpoints created during training
		if chkpts_path[-1] == '/':
			if post_training:
				checkpoints = sorted([os.path.join(chkpts_path, chkpt_name) for chkpt_name in os.listdir(chkpts_path)], key=lambda x: int(x.split('-')[-1].split('.')[0]))
			else:
				checkpoints = sorted([os.path.join(chkpts_path, chkpt_name) for chkpt_name in os.listdir(chkpts_path)])
		else:
			checkpoints = [chkpts_path]
		
		for checkpointfile in checkpoints:
			self.load_checkpoint(checkpointfile)
			self.LOG("Restored model parameters from {}".format(checkpointfile))

			num_samples = len(x)
			batch_size = 25
			if num_samples % batch_size != 0:
				self.LOG("Warning: the default batch size of 25 does not divide the number of validation samples evenly. Reverting to batch size of 1.")
				preds = self.unbatched_eval_loop(x)
			else:
				# Perform batch inference
				preds = self.batched_eval_loop(x, batch_size)
			self.LOG('Validation accuracy: {:.4f}'.format(np.sum(preds==y)/num_samples))
	
	# Gets the confusion matrix over the given set of data
	def get_confusion_matrix(self, x, y):
		self.network.eval()
		n_class = self.configs["num_classes"]
		confusion_matrix = np.zeros((n_class,n_class))
		# Perform batch inference
		num_samples = len(x)
		batch_size = 25
		if num_samples % batch_size != 0:
			self.LOG("Warning: the default batch size of 25 does not divide the number of samples evenly. Reverting to batch size of 1.")
			preds = self.unbatched_eval_loop(x)
		else:
			# Perform batch inference
			preds = self.batched_eval_loop(x, batch_size)
		for j in range(preds.shape[0]):
			# Row index is the predicted class, column index is the ground-truth class
			confusion_matrix[preds[j],y[j]] += 1
		confusion_matrix = confusion_matrix.astype(int)
		confusion_matrix_str = np.array2string(confusion_matrix, separator=', ')
		self.LOG(confusion_matrix_str)
		return confusion_matrix

	def predict_prob(self, dataset, checkpointfile):
		print('Generate Predictions on Test Data')
		self.network.eval()
		self.load_checkpoint(checkpointfile)
		preds = []

		# Convert dataset from PIL images to numpy arrays, while extracting the labels
		# x is the data, y is the labels
		# After converting a PIL image to a numpy array, its shape is [32, 32, 3]; but we want a shape of [3, 32, 32]
		x = [np.moveaxis(np.asarray(dataset[i][0]), -1, 0) for i in range(len(dataset))]
		y = [dataset[i][1] for i in range(len(dataset))]

		# Perform batch inference
		num_samples = len(x)
		batch_size = 50
		if num_samples % batch_size != 0:
			self.LOG("Warning: the default batch size of 50 does not divide the number of samples evenly. Reverting to batch size of 1.")
			preds = self.unbatched_eval_loop(x)
		else:
			# Perform batch inference
			preds = self.batched_eval_loop(x, batch_size)
		correct_preds = np.sum(preds == np.array(y))
		accuracy = correct_preds / num_samples
		self.LOG('Test accuracy: {:.4f}'.format(accuracy))
		return preds
	
	# Pass in the strings corresponding to the checkpoint filenames; they should both be in the eval_chkpts directory
	def compare_checkpoints(self, checkpoint1, checkpoint2):
		self.LOG("Comparing checkpoints {} and {}".format(checkpoint1, checkpoint2))
		params_1 = torch.load('eval_chkpts/'+checkpoint1, map_location="cpu")
		params_2 = torch.load('eval_chkpts/'+checkpoint2, map_location="cpu")
		keys_1 = set(params_1.keys())
		keys_2 = set(params_2.keys())
		if keys_1 == keys_2:
			self.LOG("The two checkpoints have the same keys.")
		else:
			self.LOG("The two checkpoints have different keys.")
		common_keys = keys_1.intersection(keys_2)
		self.LOG("Number of keys in common: {}".format(len(common_keys)))
		num_not_equal = 0
		for key in common_keys:
			if not torch.equal(params_1[key], params_2[key]):
				self.LOG("The tensors corresponding to key {} are not equal.".format(key))
				num_not_equal += 1
		self.LOG("Number of keys which are not equal: {}".format(num_not_equal))
		if num_not_equal == 0:
			self.LOG("The parameters in common between the two checkpoints are identical.")
		self.LOG("End of comparison.")
	
	def load_checkpoint(self, checkpointfile):
		checkpoint_params = torch.load(checkpointfile, map_location="cpu")
		try:
			self.network.load_state_dict(checkpoint_params, strict=True)
		except RuntimeError as e:
			model.LOG(f"Error loading checkpoint")
			# Avoid a wall of text in the case the checkpoint has unexpected discriminators or unexpectedly lacks them
			if len(str(e)) > 1000:
				model.LOG("Truncating error message due to length")
				model.LOG(str(e)[:1000])
			else:
				model.LOG(str(e))
			exit()
		self.LOG("Loaded base model parameters from {}".format(checkpointfile))

	def LOG(self, message):
		print(message)
		self.logfile.write(message + '\n')