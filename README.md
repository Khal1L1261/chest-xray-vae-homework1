# Homework 1: Chest X-ray VAE

Author: **Pengxuan Zhang**  
Code: https://github.com/Khal1L1261/chest-xray-vae-homework1  
Report: [Two-page PDF](report/main.pdf)

本项目实现图片作业中的 PyTorch VAE、真实胸片训练、生成样例、FID/IS 评估及两页 NeurIPS 格式英文报告。实验结果见 `results/`；不要将随机初始化模型或示例数字作为训练结果。

GitHub 包含源代码、报告及轻量实验结果；数据和 `.pt` 模型权重不进入 Git。训练命令可重新生成权重，本地完整交付 ZIP 保留已有权重。

## 环境与复现

Python 3.10，推荐 NVIDIA GPU。以下命令从本项目根目录执行；Linux 使用 `source .venv/bin/activate` 激活环境。

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

下载可能需要数 GB 磁盘空间；数据不包含在提交压缩包中。网络无法直接下载时，从指定 Kaggle 页面下载 ZIP，并运行 `python prepare_data.py --archive <ZIP路径> --output data`。不要把测试集用于选择超参数或 checkpoint。

当前会话的虚拟环境和数据缓存放在工作目录的 `work/venv`、`work/data`；交付项目保持独立，不依赖写死的本机路径。中断训练后可以增加目标总 epoch 数并传入 `--resume` 恢复 `last.pt`。默认拒绝静默回退至 CPU；如需要，明确传 `--device cpu`。

## 实现与评价协议

- `model.py`: 4 层卷积编码器和转置卷积解码器，64 维高斯潜变量、重参数化采样。
- 输入：64×64 灰度、保留长宽比的黑边补齐、像素范围 [0,1]；不做随机增强。
- 目标：逐像素求和的 BCE + 逐潜变量求和的 KL，再取 batch 均值；beta=1。
- `prepare_data.py`: 保留 Kaggle train/val/test 划分；排除压缩包的重复目录；保存源文件 SHA256 清单与跨划分完全重复数。该检查不等同于患者级去重。
- `train.py`: Adam、学习率 0.0002、batch=64、seed=42、40 epochs；按固定随机种子的验证 ELBO 选最佳模型。
- `evaluate.py`: 从标准正态先验生成与完整测试集等量的新图；评估 seed=31415。使用 Ignite FID 和 InceptionScore，显式提供 torchvision Inception-v3 ImageNet 权重的 2048 维池化特征及 1000 类 softmax 概率。
- FID/IS 输入：灰度复制为 RGB，双线性抗锯齿插值至 299×299，ImageNet mean/std 标准化，关闭 Inception 内部重复变换。真实与生成数据使用相同处理。
- 同时报告完整集合 IS 和十等分 IS 的均值/标准差；后者不是置信区间。FID 受样本量、权重及预处理影响，不能直接与 clean-fid 或其他协议比较。ImageNet 特征也不能验证医学有效性。

## 输出

本次完整运行：40 epochs，最佳 checkpoint 为第 40 轮；FID 243.7402，完整集合 IS 2.03045，十等分 IS 1.98224 ± 0.08104。测试负 ELBO 1694.0012 nats/image。全部 40 轮训练与验证循环合计约 58 秒（以 history.csv 为准；不含安装、数据准备、保存模型和评估）。图像有胸部轮廓但细节模糊，不应夸大生成质量。

`results/history.csv` 为真实逐轮训练记录，`config.json` 保存参数与设备；`best.pt` 和 `last.pt` 为模型及优化器 checkpoint；`metrics.json` 保存独立测试指标；`generated.png` 为先验采样，`reconstructions.png` 上排为验证原图，下排为后验均值重建。`loss_curves.png` 显示负 ELBO、BCE 和 KL。

报告采用 `report/neurips_2024.sty` 和 `report/main.tex`，可用 `tectonic main.tex` 或 `pdflatex main.tex` 编译。需提交的报告与代码由用户检查后上传 Gradescope；本项目不会自动提交。

完成新实验后运行 `python build_report.py` 可从结果 JSON/CSV 更新报告内容。单独生成新样例：`python generate.py --checkpoint results/best.pt --output samples.png --seed 123`。报告作者为 Pengxuan Zhang。

## 来源

- 数据：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- 作业指定评价教程：https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- VAE：https://arxiv.org/abs/1312.6114
- NeurIPS 样式副本：https://github.com/NeurIPS-Creative-AI/2024/blob/main/assets/neurips_2024.sty

注意：原始验证集很小，单次实验与有限测试样本存在不确定性；报告应据实说明低分辨率和生成模糊等现象。
