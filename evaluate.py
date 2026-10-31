"""Held-out FID/IS using Ignite, with explicit torchvision Inception features.

Adapted from https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
FID uses 2048-D pooled features, NOT the 1000-class probability vectors.
This is torchvision-FID, not TensorFlow/clean-fid; compare only same protocols.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from ignite.metrics import FID, InceptionScore
from model import VAE, objective


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=31415)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    state = torch.load(a.output / 'best.pt', map_location=device, weights_only=False)
    model = VAE(state['config']['latent_dim']).to(device)
    model.load_state_dict(state['model']); model.eval()
    real = torch.load(a.data / 'test.pt', weights_only=True).float().div(255)
    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device).eval()
    classifier = net.fc
    net.fc = nn.Identity()
    fid = FID(num_features=2048, feature_extractor=nn.Identity(), device='cpu')
    score = InceptionScore(num_features=1000, feature_extractor=nn.Identity(), device='cpu')
    probabilities = []
    sums = np.zeros(3)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    def features(x):
        x = F.interpolate(x.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear',
                          align_corners=False, antialias=True)
        return net((x - mean) / std)

    for i in range(0, len(real), a.batch_size):
        x = real[i:i + a.batch_size].to(device)
        z = torch.randn(len(x), model.latent_dim, device=device)
        fake = model.generate(z)
        rf, ff = features(x), features(fake)
        probs = classifier(ff).softmax(1).cpu()
        fid.update((ff.cpu(), rf.cpu())); score.update(probs)
        probabilities.append(probs)
        logits, mu, logvar = model(x)
        values = objective(logits, x, mu, logvar)
        sums += np.array([v.item() for v in values]) * len(x)
        print(f'Evaluated {min(i+a.batch_size,len(real))}/{len(real)}', flush=True)
    probs = torch.cat(probabilities)
    split_values = []
    for chunk in torch.tensor_split(probs, 10):
        metric = InceptionScore(num_features=1000, feature_extractor=nn.Identity())
        metric.update(chunk); split_values.append(float(metric.compute()))
    print('Computing 2048-dimensional covariance square root...', flush=True)
    results = {'fid': float(fid.compute()), 'inception_score': float(score.compute()),
               'is_10split_mean': float(np.mean(split_values)), 'is_10split_std': float(np.std(split_values)),
               'num_real': len(real), 'num_generated': len(real), 'seed': a.seed,
               'checkpoint_epoch': state['epoch'],
               'test_total': float(sums[0]/len(real)), 'test_bce': float(sums[1]/len(real)),
               'test_kl': float(sums[2]/len(real)),
               'protocol': 'torchvision Inception_V3_Weights.IMAGENET1K_V1; 2048 pool features; grayscale repeated RGB; 299 bilinear antialias; ImageNet normalization; transform_input=False',
               'caveats': 'Small held-out set; ImageNet domain mismatch; not clean-fid; split IS differs from full-set IS.'}
    (a.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
