### PREAMBLE ##################################################################
import torch
import torch.nn as nn
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 80, kernel_size = 5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size = 5)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)

        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


# the path for the network to load
pathNets = ''
fileToLoad = 'model_trained.pt'

### VISUALIZE FILTERS #########################################################
# load the network into the variable 'net'
#net = torch.load(os.path.join('', 'model_trained.pt'))
net = torch.load("model_trained.pt")
net.load_state_dict(torch.load("model_trained.pt"))
## figuring out dimensions of filters, layers, etc.
for k, v in net.items():
	print(k)

# the filters are of size 3x4x4, and there are 64 of them
# I think this is for the first conv. layer?
print(net["model.model.0.weight"].size())

# plot just one channel of every filter
for jj in range(64):
	# get jj-th filter, which is 3x4x4
	temp = np.floor((net["model.model.0.weight"][jj,:,:,:])*255);
	# save the first (0th) channel in the variable 'img'
	img = np.zeros((4,4))
	img = temp[0,:,:].numpy()
	# plot grayscale image of that channel in the jj-th filter
	plt.figure() 
	plt.imshow(img, vmin=-25, vmax=25, cmap='gray')
	fig = plt.gcf()
	fig.savefig("./filter"+str(jj)+".png")
