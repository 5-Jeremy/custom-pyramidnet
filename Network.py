import torch
from torch.functional import Tensor
import torch.nn as nn
import math
import time
from copy import deepcopy

""" This class defines the network architecture.
"""
class BaseNetwork(nn.Module):
	def __init__(self, configs):
		super(BaseNetwork, self).__init__()
		self.num_classes = configs["num_classes"]
		activation_type = configs["activation_type"]
		self.first_num_filters = configs["first_num_filters"]
		depth = configs["depth"]
		widening_factor = configs["widening_factor"]
		dropout_prob = configs["dropout_prob"]

		self.layer_depth = int((depth - 2) / 9)
		self.featuremap_add_rate = widening_factor / (3*self.layer_depth*1.0)

		block_fn = bottleneck_block

		self.start_layer = nn.Conv2d(in_channels=3, out_channels=self.first_num_filters, kernel_size=3, stride=1, padding=1, bias=False)
		self.first_BN_relu = batch_norm_relu_layer(self.first_num_filters, activation_type, eps=1e-5, momentum=0.997)
		self.stack_layers = nn.ModuleList()
		block_configs = {'type': bottleneck_block, 'activation_type': activation_type}
		for i in range(3):
			strides = 1 if i == 0 else 2
			filters = self.first_num_filters + self.featuremap_add_rate + self.featuremap_add_rate*(self.layer_depth)*i
			self.stack_layers.append(self.build_layer(filters, self.layer_depth, strides, block_configs))
		end_filters = int(round(filters + self.featuremap_add_rate*self.layer_depth))*4
		self.last_BN_relu = batch_norm_relu_layer(end_filters, activation_type, eps=1e-5, momentum=0.997)
		self.dropout = nn.Dropout2d(p=dropout_prob)
		self.global_avg_pool = nn.AvgPool2d(kernel_size=8)
		self.fc_layer = nn.Linear(end_filters, self.num_classes)

		# Weight initialization following the method of "Delving Deep into Rectifiers: Surpassing Human-Level Performance 
			# on ImageNet Classification" by He et al.
		conv_count = 0 # For counting the number of convolutional layers in the final network
		for layer in self.modules():
			if isinstance(layer, nn.Conv2d):
				conv_count += 1
				num_weights = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
				layer.weight.data.normal_(0, math.sqrt(2. / num_weights))
			elif isinstance(layer, nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
	
	def forward(self, inputs):
		x = self.start_layer(inputs)
		x = self.first_BN_relu(x)
		for i in range(3):
			x = self.stack_layers[i](x)
		x = self.last_BN_relu(x)
		x = self.dropout(x)
		x = self.global_avg_pool(x)
		x = x.view(x.size(0), -1)
		outputs = self.fc_layer(x)
		return outputs
	
	def build_layer(self, in_filters, depth, stride, block_configs):
		if stride != 1:
			proj_shortcut = nn.AvgPool2d((2,2), stride = (2,2), ceil_mode=True)
		else:
			proj_shortcut = None

		blocks = []
		block_fn = block_configs['type']
		# The filters passed in as arguments to block_fn correspond to the number of bottleneck filters, which is 1/4 the number of input filters
		blocks.append(block_fn(in_filters, self.featuremap_add_rate, proj_shortcut, stride, block_configs, stride==1))
		curr_in_filters = in_filters + self.featuremap_add_rate
		for i in range(1,depth):
			# The number of input and output feature maps are 4x what they would have been for a non-bottleneck block; that is, 4x the number of filters 
			# output by the previous block and 4x that amount plus the add rate
			# Note that we only need to pass in the number of input feature maps because the block itself applies the add rate to compute the 
			# number of output feature maps
			prev_in_filters = curr_in_filters
			curr_in_filters += self.featuremap_add_rate
			blocks.append(block_fn(prev_in_filters, self.featuremap_add_rate, None, 1, block_configs))
		return nn.Sequential(*blocks)

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
	""" Perform batch normalization then relu.
	"""
	def __init__(self, num_features, activation_type, eps=1e-5, momentum=0.997) -> None:
		super(batch_norm_relu_layer, self).__init__()
		self.batchNorm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
		if activation_type == 'GELU':
			self.activation = nn.GELU()
		elif activation_type == 'PReLU':
			self.activation = nn.PReLU()
		else:
			self.activation = nn.ReLU()
	
	def forward(self, inputs: Tensor) -> Tensor:
		return self.activation(self.batchNorm(inputs))

class bottleneck_block(nn.Module):
	def __init__(self, filters, add_rate, projection_shortcut, strides, block_configs, is_first_block=False) -> None:
		super(bottleneck_block, self).__init__()
		activation_type = block_configs['activation_type']
		self.projection_shortcut = projection_shortcut
		if is_first_block:
			input_channels = 16
		else:
			input_channels = int(round(filters))*4
		bottleneck_channels = int(round(filters))
		output_channels = int(round(filters + add_rate))*4

		# In PyramidNet, there are activation functions after the first and second convolutions
		self.input_BN = nn.BatchNorm2d(input_channels, eps=1e-5, momentum=0.997) # Before in_conv but not in the shortcut path
		self.BN_ReLU_1 = batch_norm_relu_layer(bottleneck_channels, activation_type)  # Before middle_conv
		self.BN_ReLU_2 = batch_norm_relu_layer(bottleneck_channels, activation_type) # Before out_conv
		self.output_BN = nn.BatchNorm2d(output_channels, eps=1e-5, momentum=0.997) # Before adding the residual

		self.in_conv = nn.Conv2d(in_channels=input_channels, out_channels=bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.middle_conv = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, stride=strides, padding=1, bias=False)
		self.out_conv = nn.Conv2d(in_channels=bottleneck_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0, bias=False)
	
	def forward(self, inputs: Tensor) -> Tensor:
		x = self.input_BN(inputs)
		x = self.in_conv(x)
		x = self.BN_ReLU_1(x)
		x = self.middle_conv(x)
		x = self.BN_ReLU_2(x)
		x = self.out_conv(x)
		x = self.output_BN(x)

		if self.projection_shortcut is not None:
			residual_term = self.projection_shortcut(inputs)
			featuremap_size = residual_term.size()[2:4]
		else:
			residual_term = inputs
			featuremap_size = x.size()[2:4]

		if x.size()[1] != residual_term.size()[1]:
			padding = torch.autograd.Variable(torch.zeros(x.size()[0], x.size()[1] - residual_term.size()[1], featuremap_size[0], featuremap_size[1], dtype=torch.float32, device='cuda')) 
			residual_term = torch.cat((residual_term, padding), 1)

		out = x + residual_term
		return out