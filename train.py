"""Reproducible full-data training, best-validation checkpoint and real logs."""
import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from model import VAE, objective


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    seed_all(a.seed)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA PyTorch or explicitly pass --device cpu.')
    a.output.mkdir(parents=True, exist_ok=True)
    data = {s: torch.load(a.data / f'{s}.pt', weights_only=True).float().div(255)
            for s in ['train', 'val']}
    loaders = {s: DataLoader(TensorDataset(x), batch_size=a.batch_size, shuffle=s == 'train',
                            num_workers=0, pin_memory=device.type == 'cuda') for s, x in data.items()}
    model = VAE(a.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_count=len(data['train']), val_count=len(data['val']))
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    best, start = float('inf'), 1
    if a.resume:
        state = torch.load(a.output / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch'] + 1, state['best']
        torch.set_rng_state(state['rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    fixed_z = torch.randn(64, a.latent_dim, generator=torch.Generator().manual_seed(123)).to(device)
    fields = ['epoch', 'train_total', 'train_bce', 'train_kl', 'val_total', 'val_bce', 'val_kl', 'seconds']
    with (a.output / 'history.csv').open('a' if a.resume else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not a.resume: writer.writeheader()
        for epoch in range(start, a.epochs + 1):
            t = time.perf_counter(); row = {'epoch': epoch}
            for split, loader in loaders.items():
                model.train(split == 'train'); sums = np.zeros(3)
                # Fixed validation sampling gives a consistent Monte Carlo ELBO comparison.
                with (torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []) if split == 'val' else nullcontext()):
                    if split == 'val': torch.manual_seed(2024)
                    for (x,) in loader:
                        x = x.to(device, non_blocking=True)
                        with torch.set_grad_enabled(split == 'train'):
                            logits, mu, logvar = model(x)
                            values = objective(logits, x, mu, logvar)
                            if split == 'train':
                                optimizer.zero_grad(set_to_none=True)
                                values[0].backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                                optimizer.step()
                        sums += np.array([v.item() for v in values]) * len(x)
                row.update({f'{split}_{k}': float(v / len(data[split]))
                            for k, v in zip(['total', 'bce', 'kl'], sums)})
            row['seconds'] = time.perf_counter() - t
            improved = row['val_total'] < best
            best = min(best, row['val_total'])
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best': best, 'config': config, 'rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
            torch.save(state, a.output / 'last.pt')
            if improved: torch.save(state, a.output / 'best.pt')
            writer.writerow(row); f.flush()
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(a.output / 'best.pt', map_location=device, weights_only=False)['model'])
    model.eval()
    with torch.no_grad():
        save_image(model.generate(fixed_z), a.output / 'generated.png', nrow=8)
        x = data['val'][:8].to(device)
        rec = model(x, sample=False)[0].sigmoid()
        save_image(torch.cat([x, rec]), a.output / 'reconstructions.png', nrow=8)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((a.output / 'history.csv').open()))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, key, title in zip(axes, ['total', 'bce', 'kl'], ['Negative ELBO', 'Reconstruction BCE', 'KL divergence']):
        for split in ['train', 'val']:
            ax.plot([int(r['epoch']) for r in rows], [float(r[f'{split}_{key}']) for r in rows], label=split)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('Nats / image'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(a.output / 'loss_curves.png', dpi=200)


if __name__ == '__main__':
    main()
