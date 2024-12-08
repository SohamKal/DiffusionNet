import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple



class VarianceScheduler:
      def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000, interpolation='cosine'):
        self.num_steps = num_steps
        self.device = torch.device('cpu')
        
        # Beta schedule
        if interpolation == 'cosine':
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:  # quadratic
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps)**2
        
        # Calculate parameters
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        
        # For recovery - using consistent naming
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alpha_bar.roll(1)) / (1 - self.alpha_bar)
        self.posterior_variance[0] = self.betas[0]
        
        # Additional parameters 
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)
    
      def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)  # Fixed name
        self.posterior_variance = self.posterior_variance.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.sqrt_one_minus_alphas = self.sqrt_one_minus_alphas.to(device)
        return self
        
      def add_noise(self, x, time_step):
        """Add noise to input tensor"""
        # Move time_step to correct device
        time_step = time_step.to(self.device)
        
        # Get values from precomputed tensors (already on correct device)
        sqrt_alpha = self.sqrt_alpha_bar[time_step]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bar[time_step]
        
        # Generate noise on same device as input
        noise = torch.randn_like(x, device=x.device)
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)
        
        # Add noise
        noisy_input = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        
        return noisy_input, noise





class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        # Frequency values
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1))
        # Compute angle rates
        angle_rates = time[:, None] * freqs[None, :]
        # Combine sine and cosine embeddings
        embeddings = torch.cat([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)

        return embeddings
def sinusoidal_embedding(n, d):
    """Returns the standard positional embedding"""
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding

class MyBlock(nn.Module):
    """
    MyBlock is a building block for the U-Net architecture, designed to process input features
    and incorporate time embeddings. It can operate in both upsampling and downsampling modes.

    Attributes:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        time_emb_dim (int): Dimension of the time embedding.
        up (bool): Whether the block is used for upsampling (True) or downsampling (False).
        time_mlp (nn.Linear): Linear layer for time embeddings.
        conv (nn.Conv2d): Convolutional layer for feature processing.
        transform (nn.Module): Either Conv2D (downsampling) or ConvTranspose2D (upsampling).
        norm (nn.BatchNorm2d): Normalization layer for stability.
        relu (nn.ReLU): Activation function.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        """
        Initializes the MyBlock with given input and output channels, time embedding dimensions,
        and mode (upsampling or downsampling).
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            time_emb_dim (int): Dimension of the time embedding.
            up (bool): Whether this block is used for upsampling.
        """
        super().__init__()  # Fixed init call
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)  
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)  
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  
        
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        Forward pass of the MyBlock.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
            t (torch.Tensor): Time embeddings of shape (B, time_emb_dim).

        Returns:
            torch.Tensor: Transformed feature map after applying convolution,
                          normalization, time embedding, and up/downsampling.
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.relu(h)
        
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  
        
        h = h + time_emb
        return self.transform(h)

class UNet(nn.Module):
    """
    U-Net architecture for processing 2D image data, integrating time embeddings
    and class embeddings for conditional tasks like diffusion models.

    Attributes:
        in_channels (int): Number of input channels.
        down_channels (tuple): Number of channels in each downsampling block.
        up_channels (tuple): Number of channels in each upsampling block.
        time_emb_dim (int): Dimension of the time embedding.
        num_classes (int): Number of classes for class conditioning.
    """
    def __init__(self, in_channels=1,
                 down_channels=(64, 128, 256, 512),
                 up_channels=(512, 256, 128, 64),
                 time_emb_dim=128,
                 num_classes=10):
        """
        Initializes the U-Net with the specified parameters.

        Args:
            in_channels (int): Number of input channels (e.g., grayscale=1, RGB=3).
            down_channels (tuple): Number of channels in each downsampling block.
            up_channels (tuple): Number of channels in each upsampling block.
            time_emb_dim (int): Dimension of the time embedding.
            num_classes (int): Number of classes for conditional tasks.
        """
        super().__init__()  

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.ReLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim),
        )

        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            MyBlock(down_channels[0], down_channels[1], time_emb_dim),
            MyBlock(down_channels[1], down_channels[2], time_emb_dim),  
            MyBlock(down_channels[2], down_channels[3], time_emb_dim) 
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1),
            nn.BatchNorm2d(down_channels[3]),
            nn.ReLU(),
            nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1),
            nn.BatchNorm2d(down_channels[3]),
            nn.ReLU()
        )

        self.ups = nn.ModuleList([
            MyBlock(up_channels[0], up_channels[1], time_emb_dim, up=True),
            MyBlock(up_channels[1], up_channels[2], time_emb_dim, up=True),  
            MyBlock(up_channels[2], up_channels[3], time_emb_dim, up=True) 
        ])

        self.output = nn.Conv2d(up_channels[-1], in_channels, 1)

    def forward(self, x, timestep, class_idx):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
            timestep (torch.Tensor): Time step embedding for diffusion models.
            class_idx (torch.Tensor): Class labels for conditional generation.

        Returns:
            torch.Tensor: Final output after U-Net processing.
        """
        class_emb = self.class_embedding(class_idx)
        t = self.time_mlp(timestep)
        emb = t + class_emb

        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, emb)
            residual_inputs.append(x)

        x = self.bottleneck(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)  
            x = up(x, emb)

        return self.output(x)


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32, 64, 128], 
                 latent_dim: int=128, 
                 num_classes: int=10) -> None:
        super().__init__()
        
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.final_h = height // (2 ** len(mid_channels))
        self.final_w = width // (2 ** len(mid_channels))
        
        self.mid_size = [mid_channels[-1], self.final_h, self.final_w]
        self.flatten_size = mid_channels[-1] * self.final_h * self.final_w
        
        self.class_emb = nn.Embedding(num_classes, 32)
        
        encoder_layers = []
        curr_channels = in_channels
        
        for channels in mid_channels:
            encoder_layers.extend([
                nn.Conv2d(curr_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ])
            curr_channels = channels
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and logvar networks
        self.mean_net = nn.Sequential(
            nn.Linear(self.flatten_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(self.flatten_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder input
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, self.flatten_size),
            nn.ReLU()
        )
        
        # Decoder
        decoder_layers = []
        curr_channels = mid_channels[-1]
        
        for channels in reversed(mid_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose2d(curr_channels, channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ])
            curr_channels = channels
            
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(curr_channels, in_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        encoded_flat = encoded.flatten(start_dim=1)
        
        # Get label embedding and concatenate
        label_emb = self.class_emb(label)
        encoded_with_label = torch.cat([encoded_flat, label_emb], dim=1)
        
        # Get mean and logvar
        mean = self.mean_net(encoded_with_label)
        logvar = self.logvar_net(encoded_with_label)
        
        # Sample from latent distribution
        sample = self.reparameterize(mean, logvar)
        
        # Decode
        out = self.decode(sample, label)
        
        return out, mean, logvar


    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)  # Convert logvar to std
        eps = torch.randn_like(std)     # Sample from standard normal
        sample = mean + eps * std       # Reparameterization trick
        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # Sample from standard Normal distribution
        noise = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode the noise based on the given labels
        out = self.decode(noise, labels)

        return out

    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        label_emb = self.class_emb(labels)
        
        # Concatenate sample with label embedding
        z_with_label = torch.cat([sample, label_emb], dim=1)
        
        # Pass through decoder input network
        x = self.decoder_input(z_with_label)
        
        # Reshape to match decoder input dimensions using stored mid_size
        x = x.view(-1, self.mid_size[0], self.mid_size[1], self.mid_size[2])
        
        # Pass through decoder
        out = self.decoder(x)
        
        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
    
    def to(self, device):
        """Override to() to handle variance scheduler"""
        super().to(device)
        self.var_scheduler.to(device)
        return self
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=device)
        
        # Add noise
        noisy_input, noise = self.var_scheduler.add_noise(x, t)
        
        # Estimate noise using network
        estimated_noise = self.network(noisy_input, t, label)
        
        # Compute loss
        loss = F.mse_loss(estimated_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
      device = noisy_sample.device
      timestep = timestep.to(device)
    
    # Get beta and alpha values for current timestep
      beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
      alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
      alpha_bar_t = self.var_scheduler.alpha_bar[timestep].view(-1, 1, 1, 1)
    
    # Calculate posterior mean coefficient
      coef1 = 1 / torch.sqrt(alpha_t)
      coef2 = beta_t / (torch.sqrt(1 - alpha_bar_t))
    
    # Calculate posterior mean
      mean = coef1 * (noisy_sample - coef2 * estimated_noise)
    
    # Add noise only if t > 0 (checking min value of the batch)
      if timestep.min() > 0:
        noise = torch.randn_like(noisy_sample, device=device)
        # Calculate posterior variance
        variance = beta_t
        sample = mean + torch.sqrt(variance) * noise
      else:
        sample = mean
        
      return sample
    
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        # Initialize random noise
        sample = torch.randn(num_samples, 1, 32, 32, device=device)
        
        # Handle labels
        if labels is not None:
            assert len(labels) == num_samples, 'Number of labels must match number of samples'
            labels = labels.to(device)
        elif self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        
        # Generate samples
        for t in reversed(range(self.var_scheduler.num_steps)):
            # Create timestep batch
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Estimate noise
            estimated_noise = self.network(sample, timesteps, labels)
            
            # Recover sample
            sample = self.recover_sample(sample, estimated_noise, timesteps)
            
            # Optional: clip values to [-1, 1]
            sample = torch.clamp(sample, -1.0, 1.0)
        
        return sample




class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
    
    def to(self, device):
        super().to(device)
        self.var_scheduler.to(device)
        return self
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=device)
        
        # Add noise
        noisy_input, noise = self.var_scheduler.add_noise(x, t)
        
        # Estimate noise
        estimated_noise = self.network(noisy_input, t, label)
        
        # Use L2 loss for better stability
        loss = F.mse_loss(estimated_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
      device = noisy_sample.device
      timestep = timestep.to(device)
    
    # Get beta and alpha values for current timestep
      beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
      alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
      alpha_bar_t = self.var_scheduler.alpha_bar[timestep].view(-1, 1, 1, 1)
    
    # Calculate posterior mean coefficient
      coef1 = 1 / torch.sqrt(alpha_t)
      coef2 = beta_t / (torch.sqrt(1 - alpha_bar_t))
    
    # Calculate posterior mean
      mean = coef1 * (noisy_sample - coef2 * estimated_noise)
    
    # Add noise only if t > 0 (checking min value of the batch)
      if timestep.min() > 0:
        noise = torch.randn_like(noisy_sample, device=device)
        # Calculate posterior variance
        variance = beta_t
        sample = mean + torch.sqrt(variance) * noise
      else:
        sample = mean
        
      return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        # Start from random noise
        x = torch.randn(num_samples, 1, 32, 32, device=device)
        
        # Handle labels
        if labels is not None:
            assert len(labels) == num_samples
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        
        # Use fewer timesteps for faster sampling
        timesteps = 100  # Reduced from 1000 for efficiency
        step_size = self.var_scheduler.num_steps // timesteps
        timesteps = range(self.var_scheduler.num_steps - 1, -1, -step_size)
        
        for t in timesteps:
            # Create timestep batch
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Estimate noise
            estimated_noise = self.network(x, t_batch, labels)
            
            # Recover sample
            x = self.recover_sample(x, estimated_noise, t_batch)
            
            # Clip values for stability
            x = torch.clamp(x, -1.0, 1.0)
        
        return x

    
