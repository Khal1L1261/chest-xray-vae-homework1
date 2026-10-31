"""Convolutional VAE for 64 x 64 grayscale radiographs."""
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        blocks = []
        for a, b in zip([1, 32, 64, 128], [32, 64, 128, 256]):
            blocks.extend([nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()])
        self.encoder = nn.Sequential(*blocks, nn.Flatten())
        self.mu = nn.Linear(4096, latent_dim)
        self.logvar = nn.Linear(4096, latent_dim)
        self.project = nn.Linear(latent_dim, 4096)
        blocks = []
        for a, b in zip([256, 128, 64, 32], [128, 64, 32, 1]):
            blocks.append(nn.ConvTranspose2d(a, b, 4, 2, 1))
            if b != 1:
                blocks.append(nn.ReLU())
        self.decoder = nn.Sequential(*blocks)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-20, 20)

    def decode(self, z):
        return self.decoder(self.project(z).view(-1, 256, 4, 4))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z), mu, logvar

    def generate(self, z):
        return self.decode(z).sigmoid()


def objective(logits, x, mu, logvar):
    # Pixel-summed BCE and latent-summed KL, averaged over samples.
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='none').flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)
    return (recon + kl).mean(), recon.mean(), kl.mean()
