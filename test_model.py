"""Small meaningful numerical check before spending GPU time."""
import torch
from model import VAE, objective

torch.set_num_threads(2)
torch.manual_seed(7)
m = VAE(64)
x = torch.rand(4, 1, 64, 64)
logits, mu, logvar = m(x)
assert logits.shape == x.shape and mu.shape == (4, 64)
loss, bce, kl = objective(logits, x, mu, logvar)
assert torch.isfinite(loss) and kl >= 0
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
zero = torch.zeros(4, 64)
_, _, zero_kl = objective(logits, x, zero, zero)
assert zero_kl == 0
m.eval()
assert torch.equal(m(x, sample=False)[0], m(x, sample=False)[0])
print('PASS: shapes, finite backward gradients, zero-prior KL, deterministic reconstruction')
