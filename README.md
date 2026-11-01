# Homework 1: Chest X-ray VAE

**Author:** Pengxuan Zhang  
**Repository:** https://github.com/Khal1L1261/chest-xray-vae-homework1  
**Report:** [Two-page NeurIPS-format PDF](report/main.pdf)

A convolutional variational autoencoder implemented in PyTorch for unconditional chest X-ray generation. The project includes data preparation, training, sampling, FID and Inception Score evaluation, recorded experimental results, and a two-page report.

## Setup and reproduction

Use Python 3.10 and, preferably, an NVIDIA GPU. Run these commands from the project root:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python test_model.py
python prepare_data.py --output data
python train.py --data data --output results --epochs 40
python evaluate.py --data data --output results
```

On Linux, activate the environment with `source .venv/bin/activate` instead. Training and evaluation default to CUDA; pass `--device cpu` explicitly to use a CPU.

Allow several gigabytes of disk space for the dataset. If automatic downloading fails, download the ZIP from the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), then run:

```powershell
python prepare_data.py --archive path/to/chest-xray.zip --output data
```

The original train/validation/test splits are preserved. The test set is used only for final evaluation, not checkpoint selection. Duplicate archive directory trees are excluded; a SHA256 manifest records source files and exact cross-split overlaps. This check does not establish patient-level independence.

To resume an interrupted run from `results/last.pt`, pass `--resume` and set `--epochs` to the desired total number of epochs. Generate new samples with:

```powershell
python generate.py --checkpoint results/best.pt --output samples.png --seed 123
```

## Model and training

| Setting | Value |
| --- | --- |
| Input | 64 x 64 grayscale, intensities in [0, 1] |
| Preprocessing | Bilinear resize with aspect-ratio-preserving black padding; no augmentation |
| Encoder | Four stride-2 convolutional layers; channels 32, 64, 128, 256 |
| Latent space | 64-dimensional diagonal Gaussian with reparameterization |
| Decoder | Linear projection followed by four transposed convolutions |
| Objective | Pixel-summed BCE plus latent-summed KL, averaged over the batch; beta = 1 |
| Optimizer | Adam, learning rate 0.0002 |
| Batch size / epochs | 64 / 40 |
| Training seed | 42 |
| Checkpoint selection | Lowest validation negative ELBO with a fixed validation sampling seed |

## Evaluation protocol

`evaluate.py` generates 624 images from the standard normal prior and compares them with all 624 held-out test images, using seed 31415. It uses Ignite's `FID` and `InceptionScore` implementations with explicit torchvision Inception-v3 ImageNet feature extraction:

- FID uses 2,048-dimensional pooled features; IS uses 1,000-class softmax probabilities.
- Both real and generated grayscale images are repeated into RGB, resized to 299 x 299 with antialiased bilinear interpolation, and normalized using ImageNet means and standard deviations. The network's internal input transform is disabled.
- Full-set IS and the mean and standard deviation of ten split scores are reported. The split standard deviation is not a confidence interval.

The FID feature configuration differs from the linked tutorial's default configuration. Scores should not be compared directly with clean-fid or implementations using different weights or preprocessing. ImageNet features do not establish medical or anatomical validity.

## Recorded results

The completed run used 5,216 training images, 16 validation images, and 624 test images. The best checkpoint was selected at epoch 40.

| Metric | Value |
| --- | ---: |
| FID | 243.7402 |
| Full-set Inception Score | 2.03045 |
| Ten-split IS, mean +/- SD | 1.98224 +/- 0.08104 |
| Test negative ELBO, nats/image | 1694.0012 |

Recorded training and validation loops took approximately 57.5 seconds in total, excluding setup, data preparation, checkpoint writes, and evaluation. Samples show broad chest silhouettes but remain blurred, with some distorted shapes. Only one training seed was evaluated, and the small validation and test sets limit the strength of the conclusions.

## Project files

| File | Purpose |
| --- | --- |
| `model.py` | VAE architecture and loss |
| `prepare_data.py` | Dataset download, preprocessing, and split manifest |
| `train.py` | Training, checkpoint selection, curves, and sample grids |
| `evaluate.py` | Held-out FID, IS, and loss evaluation |
| `generate.py` | Prior sampling from a saved checkpoint |
| `test_model.py` | Shape, gradient, KL, and deterministic reconstruction checks |
| `results/history.csv` | Per-epoch training and validation measurements |
| `results/config.json` | Hyperparameters and device information |
| `results/metrics.json` | Held-out results and evaluation protocol |
| `results/data_manifest.json` | Source filenames, hashes, and overlap counts |
| `results/generated.png` | Prior samples |
| `results/reconstructions.png` | Validation originals above posterior-mean reconstructions |
| `results/loss_curves.png` | Negative ELBO, BCE, and KL curves |
| `report/main.pdf` | Two-page report |

GitHub includes source code, the report, and lightweight results. Dataset files and `.pt` checkpoints are excluded from Git. Training produces `results/best.pt` and `results/last.pt`; the complete local project ZIP also includes the recorded checkpoints.

## References

- [Chest X-Ray Images (Pneumonia), Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [PyTorch-Ignite: GAN evaluation using FID and IS](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)
- [Kingma and Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
