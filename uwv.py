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




def sort_label(n_data, labels):
  results = []
  for i in range(n_data):
    results.append(labels[labels[:,0] == i])
  return results

# def s_adv_copy(y_pred, real_classes):
#   advs = []
#   y_pred = torch.log10(y_pred)
#   for cla in real_classes:
#     y_p = y_pred - y_pred[:,cla]
#     y_p[:,cla] = float("-Inf")
#     adv,_ = torch.max(y_p,dim=1)
#     advs.append(adv)
#   return torch.cat(advs)

# def s_mis_detect_copy(y_pred_batch, y_real, image_size):
#   y_diff = []
#   real_bbx = xywh2xyxy(y_real[:,2:]) * image_size
#   real_classes = y_real[:,1].int()
#   for y_pred in y_pred_batch:
#     if len(y_pred) != len(y_real):
#       y_diff.append(torch.tensor(1).cuda())
#     else:
#       pred_bbx = y_pred[:,:4]
#       iou = 0.3 - box_iou(real_bbx, pred_bbx)
#       adv = s_adv(y_pred[:,6:], real_classes)
#       iou_f = torch.sort(torch.flatten(iou))[len(y_real)-1]
#       adv_f = torch.sort(torch.flatten(adv))[len(y_real)-1]
#       if iou_f < 0 and adv_f < 0:
#         y_diff.append(adv_f)
#       else:
#         y_diff.append(torch.tensor(1).cuda())
#   return y_diff

def s_adv(y_pred, real_classes):
  y_pred = torch.log10(y_pred)
  y_p = y_pred - torch.gather(y_pred,1,real_classes)
  y_p = y_p[y_p.nonzero(as_tuple=True)]
  return torch.max(y_p)

def s_mis_detect(y_pred_batch, y_real):
  y_diff = []
  real_classes = y_real[:,[1]].to(dtype = torch.int64)
  for y_pred in y_pred_batch:
    if len(y_pred) != len(y_real):
      y_diff.append(torch.tensor(1))
    else:
      advs = s_adv(y_pred[:,6:], real_classes)
      y_diff.append(advs) 
  return y_diff



def main(start_zero):

  if torch.cuda.is_available():
    device = torch.device("cuda")
    CUDA = True
  else:
    device = torch.device("cpu")
    CUDA = False

   # create empty list to save the sampling results
  x_latent = []
  cell_lambda = []
  output_dict = 'output'

  if start_zero:
    dir = output_dict
    for f in os.listdir(dir):
      os.remove(os.path.join(dir, f))

  arr = os.listdir(output_dict + '/')
  for f_p in arr:
    result = torch.load(output_dict + '/'+f_p, map_location = device)
    x_latent.append(result[0])
    cell_lambda.append(result[2])
 
  n_previous = len(arr)

  # Load the YOLO model
  conf_thres = 0.3
  nms_thres = 0.3

  model = models.load_model(
    "data/yolov3.cfg",
    "data/yolov3.pth").to(device)

  model.eval()

  # Load vqvae model
  NUM_CHANNEL = 3
  IMAGE_SIZE = 256
  BATCH_SIZE = 64
  HIDDEN_DIM = 256
  Z_DIM = 4
  vae_weights_path = 'data/uwv_vae.pt'

  vae_model = VAE(NUM_CHANNEL, HIDDEN_DIM, Z_DIM).to(device)
  vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
  vae_model.eval()


  # load the UWV dataset
  # Get data configuration
  data_config = parse_data_config("data/custom.data")
  op_data_path = data_config['op_data']


  # Load test dataloader
  dataloader = _create_data_loader(
      op_data_path,
      BATCH_SIZE,
      IMAGE_SIZE,
      n_cpu = 8)

  x = []
  y = []
  

  with torch.no_grad():
    for idx, (_, data, target) in enumerate(dataloader):
      data, target = data.to(device), target.to(device)

      latent_data = vae_model.encode(data)
      latent_data = torch.flatten(latent_data, start_dim=1)
      latent_data  = latent_data.detach().cpu()

      #  preds = model((data + 1)/2)
      #  detections = non_max_suppression(preds, conf_thres, nms_thres)

      x.append(data)
      y.extend(sort_label(len(data),target))
      x_latent.append(latent_data)

  # operational data
  x = (torch.cat(x, dim=0) + 1)/2
  x_latent = np.vstack(x_latent)
  pca = PCA(n_components=4)
  x_pca = pca.fit_transform(x_latent)
  kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_pca)
  op =kde.score_samples(x_pca)
  op = op / sum(op)

  cell_op = op[:len(cell_lambda)]
  cell_op = cell_op.tolist()
  op = op[len(cell_lambda):]


  # images = (x[indices[:32]].cpu().data + 1)/2

  # save_image(
  #         images,
  #         'vqae_samples.png',
  #         nrow=4
  # )

  # set parameters for multi-level splitting
  v = 2
  rho = 0.1

  debug= True
  stats=False
  sigma = 0.08
  count_particles = 300
  count_mh_steps = 100
  print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)


  # total samples n and failure observations k
  sample_count = 0
  sample_fail = 0

  # verify the probability of failure for each cell
  for idx in range(len(x)):
    print('--------------------------------')
    sample_count += 1
    x_class = y[idx]
    x_sample = x[idx] 
    x_sample_z = x_latent[idx]
    x_op = op[idx]
    print(f'cell {idx}, label ', x_class[:,1])

  
    if x_class.numel() == 0:
      print('No detection in the Image, skipping.')
      continue

    def prop(x_input):
      x_input = torch.split(x_input, 100)
      y_diff = []
      with torch.no_grad():
        for x_batch in x_input:
          y_batch = model(x_batch)
          y_batch = non_max_suppression(y_batch, conf_thres, nms_thres)
          # property function s_mis_detect >= 0
          y_diff_batch = s_mis_detect(y_batch, x_class)
          y_diff.extend(y_diff_batch)
      return torch.stack(y_diff)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = multilevel_uniform(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                              count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')

    torch.save((x_sample_z, x_class, 10 ** lg_p), 'output/sample_{}.pt'.format(idx+n_previous))

    cell_lambda.append(min(10 ** (lg_p),1.0))
    cell_op.append(x_op)
    
    if debug:
      print('lg_p', lg_p, 'max_val', max_val)

    print()#
    reliability = torch.sum(torch.FloatTensor(cell_lambda) * torch.FloatTensor(cell_op/sum(cell_op)))
    print('yoloV3 probability of failure is ', reliability.item())


if __name__ == "__main__":
    main(start_zero = True)
