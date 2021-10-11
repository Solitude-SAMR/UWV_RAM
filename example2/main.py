
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = True

from copy import deepcopy
import sys
from mnist import mnist
from cifar10 import cifar10
from op_learning import op_learning
import multiprocessing
from sklearn.neighbors import KernelDensity

import math
import os
import time
import pickle
import numpy as np
if CUDA:
  cuda_id = '0'
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
import torch.distributions as dist
import torchvision
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from plnn.model import load_and_simplify2
import utils

def cm2inch(value):
  return value/2.54

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
  with multiprocessing.Pool(thread_count) as p:
    return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def main(test_model, op, k = 10000, count_mh_steps = 100, count_particles = 1000):
  # Fix random seed for reproducibility

  seed = 5
  np.random.seed(seed)
  torch.manual_seed(seed)

  # load MNIST class
  # load cifar10 class
  # load gtsrb class
  if test_model == 'mnist':
    loader = mnist(CUDA,op)
    robust_test = greyscale_multilevel_uniform
  elif test_model == 'cifar10':
    loader = cifar10(CUDA,op)
    robust_test = multilevel_uniform
  else:
    print('please choose a model from mnist, cifar10!')
    sys.exit(1)

  if op == 'before':
    x_op = loader.x
    y_op = loader.y
    y_op_pred = loader.y_pred
    x_latent = loader.x_latent
    print('Prior to the operational testing, running with the existing data.')
  else:
    raise Exception("Please define an Operational Dataset")


  # r-separation to decide cell size
  # nns, ret = utils.get_nearest_oppo_dist(np.array(x_op.cpu()), np.array(y_op.cpu()), np.inf, n_jobs=10)
  # ret = np.sort(ret)
  # print(ret.min(), ret.mean())

  # learn op in latent space
  # use grid search cross-validation to optimize the bandwidth
  # params = {'bandwidth': np.logspace(-1, 1, 20)}
  # grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, n_jobs=10)
  # grid.fit(np.array(x_latent.cpu()))
  
  # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
  
  # # use the best estimator to compute the kernel density estimate
  # kde = grid.best_estimator_

  # MNIST bandwidth = 0.26366508987303583
  # CIFAR10 bandwidth = 0.5455594781168519
  kde = KernelDensity(kernel='gaussian', bandwidth=0.5455594781168519).fit(np.array(x_latent.cpu()))
  log_density = kde.score_samples(np.array(x_latent.cpu()))
  density =  10 ** log_density
  # density = density /sum(density)

  # # get the first k samples with largest op
  # idx = np.argsort(density)[-k:]
  # op_samples = density[idx]
  # x_samples = x_op[torch.tensor(idx)]
  # y_samples = y_op[torch.tensor(idx)]
  # y_samples_pred = y_op_pred[torch.tensor(idx)]


  np.save('op_cell.npy', density)

  # torchvision.utils.save_image(input_points[100:120], 'output/real_samples.png')

  # set parameters for multi-level splitting
  v = 2
  rho = 0.1
  debug= True
  stats=True
  sigma = 0.1

  print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

  # create empty list to save the sampling results
  cell_lambda = []
  max_vals = []
  levels = []

  # total samples n and failure observations k
  sample_count = 0
  sample_fail = 0

  # verify the probability of failure for each cell
  for idx in range(len(x_op)):
    print('--------------------------------')
    sample_count += 1
    x_class = y_op[idx]
    x_pred = y_op_pred[idx]
    x_sample = x_op[idx]
    print(f'cell {idx}, label {x_class}')

    
    if x_class != x_pred:
      cell_lambda.append(1)
      print('Mis-prediction cell')
      continue


    def prop(x_input):
      # x_input = x_input.view(-1, 784)
      x_input = loader.data_normalization(x_input)
      y_pred = loader.model(x_input)
      y_diff = torch.cat((y_pred[:, :x_class], y_pred[:, (x_class + 1):]), dim=1) - y_pred[:, x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff  # .max(dim=1)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = robust_test(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                             count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')


    cell_lambda.append(10 ** (lg_p))
    max_vals.append(max_val)
    levels.append(l)

    if idx % 200 == 0:
      np.save('pfd_cell.npy', np.array(cell_lambda))

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)

  np.save('pfd_cell.npy', np.array(cell_lambda))


if __name__ == "__main__":
    main('cifar10', 'before', k = 10000, count_mh_steps = 100, count_particles = 1000)








