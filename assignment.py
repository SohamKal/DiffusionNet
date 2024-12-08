import torch

from models import VAE
from models import DDPM
from models import DDIM
from models import LDDPM
from models import UNet
from models import VarianceScheduler


def prepare_ddpm() -> DDPM:
    """
    EXAMPLE OF INITIALIZING DDPM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: define the configurations of the Variance Scheduler
    network = UNet(
        in_channels=1,
        down_channels=(64, 128, 256, 512),
        up_channels=(512, 256, 128, 64),
        time_emb_dim=128,
        num_classes=10
    )
    
    var_scheduler = VarianceScheduler(
        beta_start=1e-4,
        beta_end=0.02,
        num_steps=1000,
        interpolation='cosine'
    )
    
    return DDPM(network=network, var_scheduler=var_scheduler)


def prepare_ddim() -> DDIM:
    """
    EXAMPLE OF INITIALIZING DDIM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: define the configurations of the Variance Scheduler
    network = UNet(
        in_channels=1,
        down_channels=(64, 128, 256, 512),
        up_channels=(512, 256, 128, 64),
        time_emb_dim=128,
        num_classes=10
    )
    
    var_scheduler = VarianceScheduler(
        beta_start=1e-4,
        beta_end=0.02,
        num_steps=1000,
        interpolation='cosine'  # Using cosine schedule for better results
    )
    
    return DDIM(network=network, var_scheduler=var_scheduler)
def prepare_vae() -> VAE:
    """
    EXAMPLE OF INITIALIZING VAE. Feel free to change the following based on your needs and implementation.
    """
    in_channels = 1  # FashionMNIST is grayscale
    height = width = 32  # Padded dimensions
    mid_channels = [32, 64, 128]  # 3 downsampling layers
    latent_dim = 128  # Size of latent space
    num_classes = 10  # Number of FashionMNIST classes
    
    vae = VAE(
        in_channels=in_channels,
        height=height,
        width=width,
        mid_channels=mid_channels,
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    
    return vae


def prepare_lddpm() -> LDDPM:
    """
    EXAMPLE OF INITIALIZING LDDPM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: vae configs (NOTE: it should be exactly the same config as used in prepare_vae() function)
    in_channels = 1
    mid_channels = [64, 128, 256, 512]
    height = width = 32
    latent_dim = 1
    num_classes = 10
    vae = VAE(in_channels=in_channels,
              mid_channels=mid_channels,
              height=height,
              width=width,
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    # NOTE: DO NOT remove the following line
    vae.load_state_dict(torch.load('checkpoints/VAE.pt'))

    # variance scheduler configs
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # TODO: diffusion unet configs (NOTE: not more than 2 down sampling layers)
    ddpm_in_channels = latent_dim
    down_channels = [256, 512, 1024]
    up_channels = [1024, 512, 256]
    time_embed_dim = 128

    # TODO: defining the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # TODO: defining the UNet for the diffusion model
    network = UNet(in_channels=ddpm_in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim)
    
    lddpm = LDDPM(network=network, vae=vae, var_scheduler=var_scheduler)

    return lddpm

