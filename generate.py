"""Generate fresh prior samples from a trained checkpoint."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from model import VAE

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, default=Path('results/best.pt'))
p.add_argument('--output', type=Path, default=Path('samples.png'))
p.add_argument('--count', type=int, default=64)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--device', default='cuda')
a = p.parse_args()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if a.count < 1: p.error('--count must be positive')
state = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
model = VAE(state['config']['latent_dim']).to(a.device)
model.load_state_dict(state['model']); model.eval()
z = torch.randn(a.count, model.latent_dim, generator=torch.Generator().manual_seed(a.seed)).to(a.device)
a.output.parent.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    save_image(model.generate(z), a.output, nrow=8)
print(a.output)
