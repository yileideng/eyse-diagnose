# ODIR 多标签分类与病灶定位

本项目旨在对 ODIR 眼底图像进行多标签分类，并生成无监督的病灶注意力热图。代码库覆盖数据预处理、模型训练、评估工具以及针对双眼（左/右眼）图像的可视化流程。

## 主要功能
- **端到端多标签训练**：基于Ctran骨干，结合注意力池化与双眼分支。
- **稳健的预处理**：执行以眼底为中心的裁剪、同态滤波、CLAHE 增强以及 Albumentations 随机增强。
- **无监督病灶定位**：将特征库异常检测与类别 Grad-CAM 热图融合，并配合视网膜掩码抑制背景。
- **实验自动化**：提供多标签分层划分、mixup/cutmix、EMA 权重跟踪与详细指标日志。
- **工具与测试**：包含训练曲线绘制脚本及基于 pytest 的数据集/模型单元测试。

## 仓库结构
- `configs/`：分类器训练及路径配置的 YAML 文件。
- `datasets/`：PyTorch 数据集、预处理工具和增广方案。
- `docs/`：设计笔记（如 `overhaul_plan.md`）。
- `model/`：模型定义（EfficientViT 分类器、EfficientFormer 变体、CTran、Grad-CAM 等）。
- `pipelines/`：面向用户的流程脚本，核心是生成热图的 `unsupervised_loc.py`。
- `train/`：训练脚本 `train_multilabel.py` 及自定义损失函数。
- `scripts/`：如 `plot_history.py` 等独立辅助脚本，用于可视化 `history.json`。
- `tests/`：基于 pytest 的冒烟测试，覆盖数据集与模型加载。
- `Training_data/`：存放 ODIR 左/右眼原始图像的大体积目录。
- `work/`：生成的中间产物（缓存、定位结果、伪框等）。
- 根目录下的 `train.py` 与 `test.py`：历史遗留的 DenseNet/CTran 实验脚本，仅供参考。

## 数据准备
1. 将 ODIR 眼底图像放入 `Training_data/`，并确保 `total_data.csv` 包含 `ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus, N, D, G, C, A, H, M, O` 等列。
2. 在 `configs/classifier.yaml` 中配置 `img_dir`、`csv_path` 及可选的 `cache_dir`（用于保存预处理后的 PNG）。
3. （可选）先运行一次短训练以预热缓存；数据集类会自动填充 `work/cache/`。

## 环境安装
建议使用 Python 3.9+ 新建虚拟环境，并安装核心依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install albumentations==1.4.8 timm==1.0.8 opencv-python-headless==4.10.0.84
pip install pandas==2.2.2 scikit-learn==1.5.1 iterstrat==0.2.6 matplotlib==3.9.2 tqdm==4.66.5
```

请根据本地 CUDA 版本调整对应 wheel，确保 `numpy` 与 PyTorch 兼容。

## 训练分类器
```bash
python train/train_multilabel.py --config configs/classifier.yaml --output exp/classifier_run1
```

该脚本会执行多标签分层划分，应用 mixup/cutmix，并以 Asymmetric Loss + Logit-Adjusted BCE 进行训练。`--output` 目录下将生成：

- `best.pt`：包含优化器与训练轨迹的最佳 EMA 权重快照。
- `history.json`：逐 epoch 的训练/验证指标与耗时。
- `config_resolved.yaml`：实际使用的最终配置。
- `final_metrics.json`：验证集（及可选测试集）汇总指标。
- `val_preds.npy`、`val_targets.npy`、`test_preds.npy`、`test_targets.npy`：保存的预测概率与标签。

如需续训，可指定 `--resume path/to/checkpoint.pt`。

### 关键配置项
- `model.backbone`：可选 `lightweight` 自定义 CNN 或 `efficientformer_l1/l3/l7` 等 EfficientFormer 变体。
- `optimization.mixup_alpha` / `optimization.cutmix_alpha`：控制样本混合强度。
- `optimization.ema_decay`：大于 0 时启用 EMA 权重。
- `data.resample_prob`：训练阶段对少数标签进行过采样。
- `logging.val_interval` 与 `optimization.early_stop_patience`：验证频率与早停策略。

## 检验模型
训练完成后，可直接复用保存的概率文件，或按下面方式重新评估：

```bash
python train/train_multilabel.py --config configs/classifier.yaml --output exp/eval_run --resume exp/classifier_run1/best.pt
```

该命令会加载 EMA 权重并重新计算指标。若需分析旧的 CTran 检查点，可使用保留的 `test.py`，但不属于推荐流程。

## 无监督病灶定位
```bash
python pipelines/unsupervised_loc.py `
  --config configs/classifier.yaml `
  --checkpoint best_model.pth `
  --split val `
  --output work/unsup_loc_run1 `
  --prob_thr 0.3 `
  --heat_thr 0.6 `
  --overlay_alpha 0.5 `
  --smooth_sigma 1.5 `
  --max_labels 4
```

流程概述：
1. 为左右眼对构建视网膜掩码，仅保留面积最大的两个轮廓。
2. 利用健康样本特征库生成异常图（PaDiM 思路），并与 Grad-CAM 结果按类别融合。
3. 对融合热图执行按概率加权的最大化、阈值二值化，并对眼外区域做遮罩。
4. 每个样本输出 `XXXXX_heat_overlay.png`、`XXXXX_heat_mask.png`、`XXXXX_heatmap.npy`、`XXXXX_activated.npy`、`XXXXX_amap.npy` 等文件。

可通过 `--heat_thr` 调整病灶锐度，`--overlay_alpha` 控制叠加透明度，`--prob_thr` 过滤低置信度类别。如若裁剪过紧，可在脚本中调整 `prepare_eye_pair` 的 `margin_ratio`。

## 常用脚本
- `scripts/plot_history.py`：读取 `history.json` 绘制训练曲线（支持 `--output` 保存 PNG）。
- `scripts/test_api.py`：预留的集成测试占位脚本，可按需扩展。

## 运行测试
提供基础回归测试，确保数据集与模型接口稳定：

```bash
pytest tests -q
```

## 可清理的生成文件
以下目录/文件为自动生成，可在打包或提交前清理：

- `work/cache/`、`work/det_yolo/`、`work/pseudo_boxes/`、`work/test*/` 等结果目录。
- 各处的 `__pycache__/` 与 `.pytest_cache/`。
- IDE 配置 `.vscode/`。
- 体积较大的模型权重：`best_model.pth`、`yolo11n.pt`、`yolov8n.pt`（建议另行存放）。
- 空的实验占位：`exp/`、`runs/`。




## 致谢
- 数据来源：`ODIR（Ocular Disease Intelligent Recognition）挑战赛`。



