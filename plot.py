import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import density
import torch
import time
import os
from pathlib import Path
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
from torchvision.utils import save_image
from pytorchyolo import detect, models
from pytorchyolo import train
from pytorchyolo.utils.utils import to_cpu, load_classes, rescale_boxes, non_max_suppression, xywh2xyxy, box_iou
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.test import _create_data_loader, _create_validation_data_loader
from pytorchyolo.train import _create_train_data_loader
from module import VAE
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from multi_level import multilevel_uniform, greyscale_multilevel_uniform

def plot_2d_density(kde,file_name):
    # Create meshgrid
    x1, x2 = np.mgrid[-5:5:100j, -5:5:100j]
    positions = np.vstack([x1.ravel(), x2.ravel()])
    f = np.reshape(np.exp(kde.score_samples(positions.T)), x1.shape)
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    w = ax.plot_surface(x1, x2, f,cmap='gist_earth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    plt.savefig(file_name,bbox_inches='tight',dpi = 200)


if torch.cuda.is_available():
    device = torch.device("cuda")
    CUDA = True
else:
    device = torch.device("cpu")
    CUDA = False

# load the UWV dataset
# Get data configuration
data_config = parse_data_config("data/custom.data")
train_path = data_config['valid']
op_path = data_config['op_data']
class_names = load_classes(data_config["names"])

# Load vqvae model
NUM_CHANNEL = 3
IMAGE_SIZE = 256
BATCH_SIZE = 64
HIDDEN_DIM = 256
Z_DIM = 4
vae_weights_path = 'data/uwv_vae.pt'

# Load test dataloader
train_loader = _create_data_loader(
    train_path,
    BATCH_SIZE,
    IMAGE_SIZE,
    n_cpu = 8)


op_loader = _create_data_loader(
    op_path,
    BATCH_SIZE,
    IMAGE_SIZE,
    n_cpu = 8)

vae_model = VAE(NUM_CHANNEL, HIDDEN_DIM, Z_DIM).to(device)
vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
vae_model.eval()

x_train = []

with torch.no_grad():
    for idx, (_, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        latent_data = vae_model.encode(data)
        latent_data = torch.flatten(latent_data, start_dim=1)
        latent_data  = latent_data.detach().cpu()

        #  preds = model((data + 1)/2)
        #  detections = non_max_suppression(preds, conf_thres, nms_thres)

        x_train.append(latent_data)

x_op = []

with torch.no_grad():
    for idx, (_, data, target) in enumerate(op_loader):
        data, target = data.to(device), target.to(device)

        latent_data = vae_model.encode(data)
        latent_data = torch.flatten(latent_data, start_dim=1)
        latent_data  = latent_data.detach().cpu()

        #  preds = model((data + 1)/2)
        #  detections = non_max_suppression(preds, conf_thres, nms_thres)

        x_op.append(latent_data)

x_op = np.vstack(x_op)
x_train.append(x_op)
x_train = np.vstack(x_train)



pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_train)
weight = np.linspace(0,5,num = len(x_pca))
weight = np.exp(weight)
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_pca,sample_weight=weight)
op = kde.score_samples(x_pca[len(x_train)-len(x_op):])
op = np.exp(op)



# for i in range(len(x_train)-len(x_op),len(x_train)+1):
#     x = x_pca[:i]
#     weight = np.linspace(0,5,num = len(x))
#     weight = np.exp(weight)
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x,sample_weight=weight)
#     plot_2d_density(kde, 'kde_plot/{}.png'.format(i))



home_dict = 'output/'
file_paths = sorted(os.listdir(home_dict))
file_paths = [home_dict+p for p in file_paths]

p_set = []

for i in range(len(file_paths)):
    x_sample_z, x_class, p = torch.load(file_paths[i])
    p_set.append(p)

pfd = []
for i in range(len(p_set)):
    op_i = op[:i+1]
    op_i = op_i / sum(op_i)
    pfd.append(sum(op_i*p_set[:i+1]))


np.save('unreliability.npy',pfd)
plt.figure(figsize=(10, 6))
plt.xlabel('time step')
plt.ylabel('unrealibility')
plt.plot(pfd)
plt.savefig('unreliability.png', bbox_inches='tight',dpi = 200)



np.save('robustness.npy',p_set)

plt.figure(figsize=(10, 6))
plt.xlabel('time step')
plt.ylabel('lg_p')
plt.scatter(np.array(range(len(p_set))),p_set,c='#1f77b4',edgecolors='black')
plt.savefig('robustness.png', bbox_inches='tight',dpi = 200)

# for i in range(len(p_set)):
#     plt.plot(p_set[:i])
#     plt.savefig('robust_plot/{}.png'.format(i))


