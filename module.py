import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from functions import vq, vq_st

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()



# class VAE(nn.Module):
#     def __init__(self, input_dim, dim, z_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dim, dim//64, 4, 2, 1),
#             nn.BatchNorm2d(dim//64),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//64, dim//32, 4, 2, 1),
#             nn.BatchNorm2d(dim//32),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//32, dim//16, 4, 2, 1),
#             nn.BatchNorm2d(dim//16),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//16, dim//8, 4, 2, 1),
#             nn.BatchNorm2d(dim//8),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//8, dim//4, 4, 2, 1),
#             nn.BatchNorm2d(dim//4),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//4, dim//2, 5, 1, 0),
#             nn.BatchNorm2d(dim//2),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim//2, dim, 3, 1, 0),
#             nn.BatchNorm2d(dim)
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim//2, 3, 1, 0),
#             nn.BatchNorm2d(dim//2),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//2, dim//4, 5, 1, 0),
#             nn.BatchNorm2d(dim//4),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//4, dim//8, 4, 2, 1),
#             nn.BatchNorm2d(dim//8),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//8, dim//16, 4, 2, 1),
#             nn.BatchNorm2d(dim//16),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//16, dim//32, 4, 2, 1),
#             nn.BatchNorm2d(dim//32),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//32, dim//64, 4, 2, 1),
#             nn.BatchNorm2d(dim//64),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(dim//64, input_dim, 4, 2, 1),
#             nn.Tanh()
#         )

#         self.fc_mu = nn.Linear(1024, z_dim)
#         self.fc_var = nn.Linear(1024, z_dim)
#         self.decoder_input = nn.Linear(z_dim, 1024)
#         # self.apply(weights_init)

#     def encode(self, input):
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)
#         # print(result.size())

#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return mu, log_var

#     def decode(self, z):
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         result = result.view(-1, 256, 2, 2)
#         result = self.decoder(result)
#         return result

#     def reparameterize(self, mu, logvar):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu


#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)

#         kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
#         x_tilde = self.decode(z)
#         return x_tilde, kl_div

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu

    

# class VQEmbedding(nn.Module):
#     def __init__(self, K, D):
#         super().__init__()
#         self.embedding = nn.Embedding(K, D)
#         self.embedding.weight.data.uniform_(-1./K, 1./K)

#     def forward(self, z_e_x):
#         z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
#         latents = vq(z_e_x_, self.embedding.weight)
#         return latents

#     def straight_through(self, z_e_x):
#         z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
#         z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
#         z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

#         z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
#             dim=0, index=indices)
#         z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
#         z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

#         return z_q_x, z_q_x_bar


# class ResBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 3, 1, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 1),
#             nn.BatchNorm2d(dim)
#         )

#     def forward(self, x):
#         return x + self.block(x)


# class VectorQuantizedVAE(nn.Module):
#     def __init__(self, input_dim, dim, K=512):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 4, 2, 1),
#             ResBlock(dim),
#             ResBlock(dim),
#         )

#         self.codebook = VQEmbedding(K, dim)

#         self.decoder = nn.Sequential(
#             ResBlock(dim),
#             ResBlock(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
#             nn.Tanh()
#         )

#         self.apply(weights_init)

#     def encode(self, x):
#         z_e_x = self.encoder(x)
#         latents = self.codebook(z_e_x)
#         return latents

#     def decode(self, latents):
#         z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
#         x_tilde = self.decoder(z_q_x)
#         return x_tilde

#     def forward(self, x):
#         z_e_x = self.encoder(x)
#         z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
#         x_tilde = self.decoder(z_q_x_st)
#         return x_tilde, z_e_x, z_q_x


# class GatedActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x, y = x.chunk(2, dim=1)
#         return torch.tanh(x) * torch.sigmoid(y)


# class GatedMaskedConv2d(nn.Module):
#     def __init__(self, mask_type, dim, kernel, residual=True, n_classes = None):
#         super().__init__()
#         assert kernel % 2 == 1, print("Kernel size must be odd")
#         self.mask_type = mask_type
#         self.residual = residual

#         if n_classes != None:
#             self.class_cond_embedding = nn.Embedding(
#                 n_classes, 2 * dim
#             )

#         kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
#         padding_shp = (kernel // 2, kernel // 2)
#         self.vert_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

#         kernel_shp = (1, kernel // 2 + 1)
#         padding_shp = (0, kernel // 2)
#         self.horiz_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.horiz_resid = nn.Conv2d(dim, dim, 1)

#         self.gate = GatedActivation()

#     def make_causal(self):
#         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
#         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

#     def forward(self, x_v, x_h, h):
#         if self.mask_type == 'A':
#             self.make_causal()

#         h_vert = self.vert_stack(x_v)
#         h_vert = h_vert[:, :, :x_v.size(-1), :]

#         h_horiz = self.horiz_stack(x_h)
#         h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
#         v2h = self.vert_to_horiz(h_vert)

#         if h == None:
#             out_v = self.gate(h_vert)
#             out = self.gate(v2h + h_horiz)
#         else:
#             h = self.class_cond_embedding(h)
#             out_v = self.gate(h_vert + h[:, :, None, None])
#             out = self.gate(v2h + h_horiz + h[:, :, None, None])


#         if self.residual:
#             out_h = self.horiz_resid(out) + x_h
#         else:
#             out_h = self.horiz_resid(out)

#         return out_v, out_h


# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
#         super().__init__()
#         self.dim = dim

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Building the PixelCNN layer by layer
#         self.layers = nn.ModuleList()

#         # Initial block with Mask-A convolution
#         # Rest with Mask-B convolutions
#         for i in range(n_layers):
#             mask_type = 'A' if i == 0 else 'B'
#             kernel = 7 if i == 0 else 3
#             residual = False if i == 0 else True

#             self.layers.append(
#                 GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
#             )

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)

#     def forward(self, x, label):
#         shp = x.size() + (-1, )
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

#         x_v, x_h = (x, x)
#         for i, layer in enumerate(self.layers):
#             x_v, x_h = layer(x_v, x_h, label)

#         return self.output_conv(x_h)

#     def generate(self, label = None, shape=(64, 64), batch_size=64):
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )

#         with torch.no_grad():
#             for i in range(shape[0]):
#                 for j in range(shape[1]):
#                     logits = self.forward(x, label)
#                     probs = F.softmax(logits[:, :, i, j], -1)
#                     x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
#         return x

#     def density_prob(self, x, label, shape=(64,64)):

#         param = next(self.parameters())
#         logits = self.forward(x, label)

#         results = torch.zeros(len(x), device=param.device)

#         with torch.no_grad():
#             for i in range(shape[0]):
#                 for j in range(shape[1]):
#                     latent_code = x[:, i, j].view(-1, 1)  
#                     probs = F.softmax(logits[:, :, i, j], -1)
#                     probs = torch.gather(probs, 1, latent_code).view(-1)
#                     results = results + torch.log10(probs)

#         return results
