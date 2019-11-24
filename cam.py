import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


class FeatureExtractor():
	def __init__(self, model, target_layers):
		self.model = model
		# 定义预输出梯度的层数名称
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []
		for name, module in self.model._modules.items():
			if name == 'fc':
				x = x.view(x.size(0), -1)
			x = module(x)

			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
		return outputs, x 


class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		return target_activations, output # target_activations 表示为梯度list


def preprocess_image(img):
	test_transform =  transforms.Compose([
					transforms.Resize((896, 896)),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					])

	preprocessed_img = test_transform(img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad=True)
	return input


def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = 0.5*heatmap + 0.5 *np.float32(img) / 255
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index=None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype=np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		cam = cv2.resize(cam, (896, 896))

		return cam


if __name__ == '__main__':
	# load model
	model = models.resnet18(pretrained=True)
	for param in model.parameters():
	    param.requires_grad = False

	model.fc = nn.Sequential(
		nn.Linear(512, 128),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(128, 5)
	)

	model.load_state_dict(torch.load('best_resnet50_448.pkl'))

	grad_cam = GradCam(model=model, target_layer_names = ["layer4"], use_cuda=False)

	path = 'IDRiD_001.png'
	img = Image.open(path)
	input = preprocess_image(img)
	img_ori = transforms.Resize((896, 896))(img)

	mask = grad_cam(input, index=None)

	img_cv2 = cv2.cvtColor(np.array(img_ori), cv2.COLOR_RGB2BGR)
	show_cam_on_image(img_cv2, mask)


