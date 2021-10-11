import numpy as np
import torch
from model import VAE
import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from train.mnist_network import mnist_SimpleMlp
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('pdf')

from random import seed
import random
seed(1)

class mnist:
    def __init__(self, CUDA, op):
        # The bounds in NN-space
        self.x_min = 0
        self.x_max = 1
        self.mean = 0.1307
        self.std = 0.3081
        self.batch_size = 128
        self.z_size = 8
        self.CUDA = CUDA
        self.x = None
        self.y = None
        self.x_latent = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.g_model = None
        self.load_model()
        if op == 'before':
            self.load_before_data()
        else:
            self.load_op3_data()


    def load_before_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=tform1)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=tform1)
        op_data = ConcatDataset([train_data, test_data])
        data_loader = torch.utils.data.DataLoader(op_data, batch_size=self.batch_size, shuffle=False)

        if self.CUDA:
            self.x = torch.zeros(len(op_data), 1, 28, 28).cuda()
            self.y = torch.zeros(len(op_data), dtype=torch.long).cuda()
            self.x_latent = torch.zeros(len(op_data), self.z_size).cuda()
            self.y_pred = torch.zeros(len(op_data), dtype=torch.long).cuda()

        else:
            self.x = torch.zeros(len(op_data), 1, 28, 28)
            self.y = torch.zeros(len(op_data), dtype=torch.long)
            self.x_latent = torch.zeros(len(op_data), self.z_size)
            self.y_pred = torch.zeros(len(op_data), dtype=torch.long)

        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                if self.CUDA:
                    data, target = data.float().cuda(), target.long().cuda()
                else:
                    data, target = data.float(), target.long()

                (mean, logvar), x_reconstructed = self.g_model.forward(self.data_resize(data,32))
                self.x_latent[(idx * self.batch_size):((idx + 1) * self.batch_size), :] = mean
                self.x[(idx * self.batch_size):((idx + 1) * self.batch_size), :, :, :] = data
                self.y[(idx * self.batch_size):((idx + 1) * self.batch_size)] = target

                data = data.view(-1, 784)
                pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)
                self.y_pred[(idx * self.batch_size):((idx + 1) * self.batch_size)] = pred

        print("Training set score: %f" % accuracy_score(np.array(self.y_pred.cpu()[:len(train_data)]), np.array(self.y.cpu()[:len(train_data)])))
        print("Test set score: %f" % accuracy_score(np.array(self.y_pred.cpu()[len(train_data):]), np.array(self.y.cpu()[len(train_data):])))




    def data_normalization(self, x_input):
        # x_input = x_input.view(-1,784)
        return (x_input - self.mean) / self.std

    def data_resize(self,x_input,img_size):
        transform = transforms.Compose([transforms.Resize((img_size, img_size))])
        return transform(x_input)

    def load_model(self):
        # Create model and load trained parameters
        self.model = mnist_SimpleMlp()
        self.model.load_state_dict(torch.load('./data/mnist_simplemlp.pickle'))
        self.model.eval()

        self.g_model = VAE(
            label='mnist',
            image_size=32,
            channel_num=1,
            kernel_num=32,
            z_size=self.z_size,
        )

        _ = utils.load_checkpoint(self.g_model, './data')
        self.g_model.eval()

        if self.CUDA:
            self.model.cuda()
            self.g_model.cuda()



