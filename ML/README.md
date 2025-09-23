# ODIR 多标签分类与病灶定位

本项目针对 ODIR 眼底图像数据集，采用 CTran 多标签分类模型，并结合异常检测 + Grad-CAM 生成平滑的病灶热图。仓库提供数据预处理、模型训练/测试、无监督定位及可视化等完整流程。

## 目录概览
- `configs/`：训练与推理所需的 YAML 配置。
- `datasets/`：ODIR 数据集读取、预处理与增广实现。
- `model/`：CTran 及相关模块、Grad-CAM 等组件。
- `pipelines/`：推理与可视化脚本（例如 `unsupervised_loc.py`）。
- `utils/`：当前使用的训练脚本 `train.py`、测试脚本 `test.py` 以及损失函数、辅助函数等。
- `work/`：运行时生成的缓存与输出。
- 其它目录（如 `scripts/`、`tests/`）提供可选工具与单元测试。

## 数据准备
1. 将 ODIR 左右眼图像存放于 `Training_data/`，并确保 `total_data.csv` 至少包含 `ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus, N, D, G, C, A, H, M, O` 等列。
2. 根据实际路径修改 `configs/classifier.yaml` 中的 `img_dir`、`csv_path` 以及（可选）`cache_dir`，用于保存预处理后的 PNG。
3. 如需加速后续训练，可提前运行一次预处理/训练，以便自动填充 `work/cache/`。

## 环境安装
建议使用 Python 3.9+ 创建虚拟环境，并安装以下依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install albumentations==1.4.8 timm==1.0.8 opencv-python-headless==4.10.0.84
pip install pandas==2.2.2 scikit-learn==1.5.1 iterstrat==0.2.6 matplotlib==3.9.2 tqdm==4.66.5
```

若使用 GPU，请确保 CUDA 版本与 `torch` 匹配，`numpy` 会随依赖自动安装。

## 模型训练（`utils/train.py`）
`utils/train.py` 为当前默认的训练脚本，内部封装了数据划分、CTran 模型构建、损失与优化流程。

```bash
python utils/train.py   --config configs/classifier.yaml   --output exp/ctrans_run1   --epochs 200   --batch-size 12
```

常用参数说明：
- `--config`：指定数据与模型的配置文件。
- `--output`：训练输出目录，将保存最佳模型、日志与指标。
- `--resume`：加载历史检查点继续训练。
- 其它参数（学习率、权重衰减、EMA 等）可在脚本或配置中调整。

训练结束后，`exp/ctrans_run1/` 内会包含：
- `best_model.pth`：权重与优化器状态。
- `history.json`：逐 epoch 指标。
- `final_metrics.json`、`val_preds.npy` 等结果文件。

## 模型测试（`utils/test.py`）
使用相同的数据拆分与预处理，在验证/测试集上重新评估模型：

```bash
python utils/test.py   --config configs/classifier.yaml   --checkpoint exp/ctrans_run1/best_model.pth   --split val   --output work/eval_run1
```

脚本会计算多标签指标（准确率、精确率、召回率、F1、AUC 等），并可按需保存预测概率、混淆矩阵或可视化结果。若指定 `--split test`，将评估测试集。

## 无监督病灶定位
`pipelines/unsupervised_loc.py` 基于训练好的 CTran 分类器，生成平滑的病灶热图并自动标注中文/英文疾病名称：

```bash
python pipelines/unsupervised_loc.py   --config configs/classifier.yaml   --checkpoint exp/ctrans_run1/best_model.pth   --split val   --output work/unsup_loc_run1   --prob_thr 0.3   --heat_thr 0.6   --overlay_alpha 0.5   --smooth_sigma 1.5   --max_labels 4   --crop_margin 0.05
```

输出说明：
- `XXXXX_heat_overlay.png`：叠加热图与疾病标签的原图，可显示多种疾病概率。
- `XXXXX_heat_mask.png`：平滑后的激活掩码。
- `XXXXX_heatmap.npy`、`XXXXX_activated.npy`、`XXXXX_amap.npy`：供进一步分析的数值文件。

常见可调参数：
- `--prob_thr`：过滤低置信度疾病。
- `--smooth_sigma`：控制热图与掩码平滑程度。
- `--crop_margin`：调节裁剪时的留白，防止眼球被截断或压扁。

## 实用工具
- `scripts/plot_history.py`：读取 `history.json` 绘制训练/验证曲线，支持 `--output` 保存图像。
- `scripts/test_api.py`：接口/管线测试示例，可按需扩展。

如有问题或建议，欢迎提交 Issue 或 Pull Request。
