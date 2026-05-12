# 街景字符识别 — 方法说明与命令行手册

本文档说明本仓库中 **命令行工具** 的用途、参数与推荐执行顺序，并给出 **后续训练计划**（YOLO 检测方案为主，可与 Baseline 对照）。

---

## 1. 方案概览

| 路线 | 说明 | 主要入口 |
|------|------|----------|
| **Baseline（分类）** | 整图 + 四位分类头（0–9 与「空」），见 `README.md` | `baseline.py`、`baseline.ipynb` |
| **YOLO（检测）** | 将每个数字视为检测框，类别 0–9；推理时按框 **从左到右** 排序拼成门牌字符串 | `prepare_yolo_dataset.py` → `train_yolo.py` → `eval_yolo_sequence.py` / `predict_yolo_submit.py` |

赛题评分中的 **整图 Acc** 与 YOLO 路线对齐时，应以 **`eval_yolo_sequence.py` 输出的 Sequence accuracy** 为准（整串与标注完全一致才算对），而不是仅看 Ultralytics 训练日志里的 mAP。

---

## 2. 环境与依赖

### 2.1 Python 与 PyTorch

- 建议使用 **Python 3.10+**。
- **GPU 训练**：安装与显卡驱动匹配的 **CUDA 版 PyTorch**，使 `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`。
- **仅 CPU**：可安装 CPU 版 PyTorch；训练会较慢，建议减小 `batch`、`epochs` 或使用 `--fraction` 做冒烟测试。

### 2.2 依赖安装

YOLO 相关（推荐国内镜像，避免 `ultralytics` / 大包下载超时）：

```bash
pip install -r requirements-yolo.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Baseline 另需：`torchvision`、`pandas`、`tqdm` 等（见 `baseline.py` 顶部 import；若缺包按报错补装即可）。

### 2.3 数据目录约定

解压后天池数据应包含（路径与 `baseline.py` 一致）：

- `dataset/mchar_train/`、`dataset/mchar_val/`、`dataset/mchar_test_a/`：PNG 图像  
- `dataset/mchar_train.json`、`dataset/mchar_val.json`：标注  
- `dataset/mchar_sample_submit_A.csv`：提交样例  

若尚未下载，可使用仓库中的 `mchar_data_list_0515.csv` 与 `baseline.ipynb` 首格下载逻辑。

---

## 3. 设备选择（CPU / GPU 自动切换）

模块 **`yolo_device.py`** 为训练与推理脚本统一解析设备：

| `--device` 取值 | 行为 |
|------------------|------|
| `auto`（默认） | 若 `torch.cuda.is_available()` 为真，使用 **GPU 0**；否则 **CPU** |
| `cpu` | 强制 CPU |
| `0`、`1`、`cuda:1`、`0,1` 等 | 原样传给 Ultralytics（多卡、指定卡） |

**训练脚本额外行为**：在 CPU 上会自动关闭 **AMP**（混合精度），避免部分环境下兼容性问题；GPU 上默认开启 AMP。

---

## 4. 命令行功能说明（按推荐顺序）

以下命令均在 **项目根目录** 执行（`Street_Character_Recognition`）。

### 4.1 `prepare_yolo_dataset.py` — 生成 YOLO 格式数据

**作用**：读取 `mchar_train.json` / `mchar_val.json`，在 `dataset/yolo/` 下生成：

- `images/train`、`images/val`：指向原图的硬链接（失败则复制）  
- `labels/train`、`labels/val`：每张图对应一个 `.txt`，行为 `class x_center y_center width height`（归一化到 0–1），类别 0–9 对应数字 `'0'`–`'9'`  

**命令**（无子命令行参数）：

```bash
python prepare_yolo_dataset.py
```

**前置条件**：上述 JSON 与 `mchar_train` / `mchar_val` 图像目录已就绪。  
**输出**：与 `svhn_digits.yaml` 中 `path: dataset/yolo` 一致，供 `train_yolo.py` 使用。

---

### 4.2 `download_yolo_weights.py` — 下载预训练权重到项目根目录

**作用**：从 Hugging Face / GitHub 镜像依次尝试，将官方 **YOLO11 检测** 权重下载到 **当前项目根目录**，并校验最小文件体积，避免半截文件。

**支持的文件名**：`yolo11n.pt`、`yolo11s.pt`、`yolo11m.pt`、`yolo11l.pt`、`yolo11x.pt`

**命令示例**：

```bash
# 下载全部五个检测权重（已存在且体积合格则跳过）
python download_yolo_weights.py

# 只下载或补全某一个
python download_yolo_weights.py yolo11m.pt

# 强制重新下载（若文件被占用可能无法替换，需先关闭占用程序）
python download_yolo_weights.py --force yolo11m.pt
```

训练时通过 `train_yolo.py --model .\yolo11m.pt` 等形式指向本地文件即可，减少对在线下载的依赖。

---

### 4.3 `train_yolo.py` — Ultralytics YOLO 训练

**作用**：使用 `svhn_digits.yaml` 指向的数据集训练 **10 类数字检测** 模型；默认使用余弦学习率、warmup、`close_mosaic` 等与常见 Ultralytics 实践一致的参数。

**常用命令**：

```bash
# 自动选设备：有 GPU 用 GPU 0，否则 CPU
python train_yolo.py

# 指定本地权重与 GPU
python train_yolo.py --model .\yolo11m.pt --device 0

# 强制 CPU（笔记本调试、无 CUDA 环境）
python train_yolo.py --device cpu --batch 4 --workers 0

# 快速冒烟：只用部分训练图（例如 2%）
python train_yolo.py --fraction 0.02 --epochs 1 --device cpu --workers 0
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolo11m.pt` | 预训练权重文件名或绝对路径 |
| `--data` | `svhn_digits.yaml` | 数据集 YAML |
| `--epochs` | `120` | 训练轮数 |
| `--imgsz` | `640` | 训练输入边长 |
| `--batch` | `16` | 批大小；设为 **-1** 时交给 Ultralytics 自动 batch |
| `--device` | `auto` | 见第 3 节 |
| `--workers` | `8` | DataLoader 进程数；Windows 报错时可试 **0** |
| `--project` | `runs/detect` | 训练输出根目录 |
| `--name` | `mchar_digits` | 本次实验子目录名 |
| `--patience` | `40` | 早停耐心值 |
| `--seed` | `42` | 随机种子 |
| `--fraction` | `1.0` | 使用训练集比例（0–1），用于快速验证管线 |

**输出**：最优权重路径一般为：

`runs/detect/<name>/weights/best.pt`

（若修改了 `--project` 或 `--name`，请按终端最后一行提示为准。）

---

### 4.4 `eval_yolo_sequence.py` — 验证集「整串准确率」

**作用**：对验证集每张图做检测，将框按 **x 中心从左到右** 排序，拼成数字字符串，与 `mchar_val.json` 中的 `label` 顺序拼接结果比较，统计 **完全一致** 的比例（与赛题整串 Acc  spirit 一致）。

**命令示例**：

```bash
python eval_yolo_sequence.py --weights runs\detect\mchar_digits\weights\best.pt

python eval_yolo_sequence.py --weights runs\detect\mchar_digits\weights\best.pt --device 0 --conf 0.3
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | （必填） | `best.pt` 或 `last.pt` 路径 |
| `--val-json` | `dataset/mchar_val.json` | 验证标注 JSON |
| `--val-img` | `dataset/mchar_val` | 验证图像目录 |
| `--imgsz` | `640` | 推理尺寸（可与训练一致或略大） |
| `--conf` | `0.25` | 置信度阈值，可在验证集上微调以提升整串准确率 |
| `--device` | `auto` | 见第 3 节 |

---

### 4.5 `predict_yolo_submit.py` — 测试集提交 CSV

**作用**：对 `mchar_test_a` 下 PNG 逐张预测，输出天池格式 CSV（`file_name`, `file_code`）。

**命令示例**：

```bash
python predict_yolo_submit.py --weights runs\detect\mchar_digits\weights\best.pt --out yolo_submit.csv --device 0
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | （必填） | 训练得到的权重路径 |
| `--test-dir` | `dataset/mchar_test_a` | 测试图像目录 |
| `--out` | `yolo_submit.csv` | 输出 CSV 路径 |
| `--imgsz` | `640` | 推理尺寸 |
| `--conf` | `0.25` | 置信度阈值，可与验证集上最优 `--conf` 对齐 |
| `--device` | `auto` | 见第 3 节 |

---

### 4.6 `baseline.py` — 原 ResNet50 四位分类 Baseline

**作用**：下载数据（若缺失）、构建 `DigitsDataset`、训练 `DigitsResnet50`、在验证集评估、并对测试集写出 `result.csv`（具体逻辑见脚本内 `Config` 与 `Trainer`）。

**命令**：

```bash
python baseline.py
```

**说明**：超参数在脚本内 `class Config` 中（如 `epoches`、`batch_size`）；默认会尝试 `resnet50(pretrained=True)`，需能访问 torchvision 预训练或已缓存权重。该路线与 YOLO 路线独立，可并行作为对照实验。

---

### 4.7 配置文件 `svhn_digits.yaml`

**作用**：告诉 Ultralytics 数据集根目录与类别数、类别名。一般无需改；若数据放在其他盘符，可修改其中 `path:` 为绝对路径或调整相对路径。

---

## 5. 推荐端到端流程（YOLO）

1. 准备数据与 JSON（见 2.3）。  
2. `python prepare_yolo_dataset.py`  
3. `python download_yolo_weights.py`（或手动放置 `.pt`）  
4. `python train_yolo.py --model .\yolo11m.pt --device 0`（按显存调整 `--batch`、`--model`）  
5. `python eval_yolo_sequence.py --weights ...\best.pt`  
6. 在验证集上对 `--conf` 做小幅网格搜索（如 0.15–0.45）。  
7. `python predict_yolo_submit.py --weights ...\best.pt --conf <最优> --out yolo_submit.csv`  

---

## 6. 后续训练计划（建议分阶段执行）

以下计划假设 **已具备 GPU 与 CUDA 版 PyTorch**；若仅有 CPU，将各阶段 epoch 减半或仅用 `--fraction` 验证流程，正式结果仍以 GPU 全量训练为准。

### 阶段 A：基线复现与指标对齐（约 0.5–1 天）

- 使用 **`yolo11m.pt`**，`--imgsz 640`，`--epochs 120`，`--batch` 按显存设为 8–16。  
- 跑通 `eval_yolo_sequence.py`，记录 **Sequence accuracy** 作为基线。  
- 确认 `predict_yolo_submit.py` 能生成完整行数的 CSV。

### 阶段 B：模型与分辨率（约 1–2 天）

- 显存允许时依次尝试 **`yolo11l.pt` → `yolo11x.pt`**，其它超参不变，对比验证集整串准确率。  
- 显存仍有余量时尝试 **`--imgsz 800`**（门牌数字较小，有时有收益），注意同步调小 `batch` 或启用 `batch=-1`。

### 阶段 C：推理与阈值（约 0.5 天）

- 固定 `best.pt` 与 `imgsz`，在验证集上扫描 **`--conf`**（如 0.15、0.2、0.25、0.3、0.35、0.4）。  
- 将最优 `conf` 用于 `predict_yolo_submit.py` 生成最终提交文件。

### 阶段 D：训练时长与早停（按需）

- 若验证曲线仍在上升，可将 **`--epochs` 提到 180–250**，保持 **`--patience 40`** 避免过拟合无效拖时。  
- 可多随机种子（如 `--seed 42` 与 `123`）训练 2 次，取验证集最优权重或做简单模型集成（需自行写脚本融合多模型预测）。

### 阶段 E（可选）：与 Baseline 融合

- 保留 **`baseline.py`** 一路结果；若时间与算力允许，可对 YOLO 与 ResNet 的字符串级预测做 **投票或规则融合**（例如仅在高置信度区域采用某一模型），并在报告中说明。

---

## 7. 常见问题

| 现象 | 建议 |
|------|------|
| 权重下载失败 / SSL 中断 | 使用 `download_yolo_weights.py` 或浏览器从 Hugging Face / GitHub Release 手动下载到项目根目录 |
| `WinError 5` / 文件被占用 | 关闭预览、杀毒实时扫描或其它占用 `*.pt` 的进程后再 `--force` 下载 |
| Windows 上 DataLoader 报错 | `train_yolo.py --workers 0` |
| 训练很快但整串 Acc 低 | 检查 `prepare_yolo_dataset.py` 是否已重新跑过；提高 `imgsz` 或换更大 `--model`；调 `--conf` |
| CUDA 不可用 | 检查 PyTorch 是否为 CUDA 构建；`--device cpu` 仍可跑通全流程 |

---

## 8. 参考链接

- 赛题与数据说明：见仓库根目录 `README.md`  
- Ultralytics YOLO：<https://github.com/ultralytics/ultralytics>  
- 文档：<https://docs.ultralytics.com/>  

---

*文档版本与仓库脚本保持一致；若修改了脚本默认参数，请以 `python <script>.py --help` 为准。*
