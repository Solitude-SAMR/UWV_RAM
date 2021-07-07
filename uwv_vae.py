import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import tqdm
import time
from module import VAE, weights_init
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.test import _create_data_loader
from pytorchyolo.train import _create_train_data_loader
from pytorchyolo.utils.utils import load_classes


def train(data_loader, model, optimizer, epoch, device):
    train_loss = []
    recon_loss = []
    kld_loss = []
    model.train()
    start_time = time.time()
    for batch_idx, (_, x, _)  in enumerate(tqdm.tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        x = x.to(device)

        optimizer.zero_grad()
        x_tilde, kl_d = model(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)

        loss = loss_recons + 0.001 * kl_d
        loss.backward()
        
        optimizer.step()
        train_loss.append(loss.item())
        recon_loss.append(loss_recons.item())
        kld_loss.append(kl_d.item())

    print('Train Completed!\tLoss: {:7.6f}   Reconstruction Loss: {:7.6f}   KLD Loss: {:7.6f}  Time: {:5.3f} s'.format(
        np.asarray(train_loss).mean(0),
        np.asarray(recon_loss).mean(0),
        np.asarray(kld_loss).mean(0),
        time.time() - start_time
    ))
       
def test(data_loader, model, device):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        loss_recons, loss_kld = 0., 0.
        for batch_idx, (_, x, _) in enumerate(tqdm.tqdm(data_loader, desc="Validating")):
            x = x.to(device)
            x_tilde, kl_d = model(x)
            loss_recons += F.mse_loss(x_tilde, x)
            loss_kld += kl_d

        loss_recons /= len(data_loader)
        loss_kld /= len(data_loader)   

    print('Validation Completed!\tReconstruction Loss: {:7.6f} Time: {:5.3f} s'.format(
        np.array(loss_recons.item()),
        time.time() - start_time
    ))
    return loss_recons.item(), loss_kld.item()
        

def generate_reconstructions(model, data_loader, device, DATASET):
    model.eval()
    _, x, _ = data_loader.__iter__().next()
    x = x[:4].to(device)
    x_tilde, kl_div = model(x)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1)/2

    save_image(
        images,
        'vae_reconstructions_{}.png'.format(DATASET),
        nrow=4
    )

def main():

    IMAGE_SIZE = 256
    BATCH_SIZE = 64
    N_EPOCHS = 100
    DATASET = 'uwv'  
    NUM_CHANNEL = 3
    HIDDEN_DIM = 256
    Z_DIM = 4
    LR = 1e-3
    weight_decay = 0

    save_filename = 'data/'
    weights_path = None
    # 'data/uwv_vae.pt'
    if not os.path.exists(save_filename):
        os.makedirs(save_filename)
    
    # load the UWV dataset
    # Get data configuration
    data_config = parse_data_config("data/custom.data")
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training dataloader
    train_loader = _create_data_loader(
        train_path,
        batch_size = BATCH_SIZE,
        img_size = IMAGE_SIZE,
        n_cpu=8)

    valid_loader = _create_data_loader(
        valid_path,
        batch_size = BATCH_SIZE,
        img_size = IMAGE_SIZE,
        n_cpu=8)


    model = VAE(NUM_CHANNEL, HIDDEN_DIM, Z_DIM).to(device)
    # load from last time trained file
    if weights_path != None:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    best_loss = -1.
    LAST_SAVED = -1
    for epoch in range(N_EPOCHS):
        train(train_loader, model, optimizer, epoch, device)
        loss, _ = test(valid_loader, model, device)

        generate_reconstructions(model, valid_loader, device, DATASET)

        if (epoch == 0) or (loss < best_loss):
            print("Saving model!\n")
            best_loss = loss
            LAST_SAVED = epoch
            with open('{0}/{1}_vae.pt'.format(save_filename,DATASET), 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            print("Not saving model! Last saved: {}\n".format(LAST_SAVED))
       

if __name__ == '__main__':
    import os
    import multiprocessing as mp
    main()