# 云上 GPU 训练 — Shell 命令汇总

适用于 **Linux + GPU** 环境（如 ModelScope / PAI-DSW、AutoDL、远程服务器等）。在实例的 **Terminal**（JupyterLab → New → Terminal）中执行。

以下用 **`PROJECT`** 表示项目根目录（含 `train_yolo.py`、`dataset/` 等）。请按实际上传路径修改，例如：

```bash
export PROJECT=~/Street_Character_Recognition
cd "$PROJECT"
```

---

## 0. 打开终端

在 JupyterLab：**File → New → Terminal**，或 Launcher 里点 **Terminal**。

---

## 1. 确认 GPU 与 PyTorch

```bash
nvidia-smi

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

第二行应出现 **`cuda True`** 和显卡名称。

---

## 2. 进入项目目录

**若尚未克隆仓库：**

```bash
git clone https://github.com/<你的用户名>/Street_Character_Recognition.git
export PROJECT=~/Street_Character_Recognition   # 与 clone 目录一致
cd "$PROJECT"
```

**已有代码时：**

```bash
export PROJECT=~/Street_Character_Recognition   # 改成你的路径
cd "$PROJECT"
pwd
ls
```

---

## 3. 安装 Python 依赖

```bash
cd "$PROJECT"
pip install -U pip
pip install -r requirements-yolo.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若镜像已含 `ultralytics`，可跳过或仅执行 `pip install -r requirements-yolo.txt` 做补齐。

---

## 3.5 从 Git 克隆后复原 `dataset/` 与 `*.pt`

通过 **`git clone`** 拉到的通常只有代码：仓库 **不包含** `dataset/`、**`*.pt`**、`runs/` 等（见根目录 `.gitignore`）。在 GPU 实例上训练前，在 **`$PROJECT`** 下按顺序执行下面三步。

### A. 数据集目录与 JSON

目标布局（与 `README`、`prepare_yolo_dataset.py` 一致）：

- `dataset/mchar_train/`、`dataset/mchar_val/`、`dataset/mchar_test_a/`（PNG）
- `dataset/mchar_train.json`、`dataset/mchar_val.json`

**来源：** 在天池赛题页下载完整数据（README 顶部赛题链接），解压后把上述目录与文件拷到实例上的 **`$PROJECT/dataset/`**（可用 scp、rsync、平台「上传」、挂载网盘等）。

若压缩包已在实例本机，示例（**解压结构因压缩包而异**，若多套一层目录请再 `mv` 对齐）：

```bash
cd "$PROJECT"
mkdir -p dataset
# unzip -q ~/downloads/mchar_xxx.zip -d dataset
# 或：tar -xzf ~/downloads/mchar_xxx.tar.gz -C dataset
```

### B. 预训练权重 `*.pt`（项目根目录）

```bash
cd "$PROJECT"
python download_yolo_weights.py
# 只下流水线常用权重时可缩小范围，例如：
# python download_yolo_weights.py yolo11m.pt yolo11l.pt yolo11x.pt
```

### C. 生成 `dataset/yolo/`（数据就位后执行一次）

```bash
cd "$PROJECT"
python prepare_yolo_dataset.py
```

**自检（与下一节一致）：**

```bash
cd "$PROJECT"
ls dataset/mchar_train 2>/dev/null | head -n 3
test -f dataset/mchar_train.json && test -f dataset/mchar_val.json && echo "dataset json OK"
test -f yolo11m.pt && echo "weights OK" || echo "缺少 yolo11m.pt，请再运行 download_yolo_weights.py"
test -d dataset/yolo/images/train && echo "yolo layout OK" || echo "尚未 prepare_yolo_dataset，请运行上一节命令 C"
```

完成后再进入 **第 4 节** 做完整检查、**第 7 节** 开始训练。

---

## 4. 数据与目录检查

若刚完成 **第 3.5 节**，本节用于复查。训练需要（与 `README` / `baseline.py` 一致）：

- `dataset/mchar_train/`、`dataset/mchar_val/`、`dataset/mchar_test_a/`（PNG）
- `dataset/mchar_train.json`、`dataset/mchar_val.json`
- `svhn_digits.yaml`（仓库内已有）

检查示例：

```bash
cd "$PROJECT"
ls dataset/mchar_train  | head
test -f dataset/mchar_train.json && echo "train json OK"
```

若数据在网盘/OSS，请先在平台内下载/挂载到上述相对路径。

---

## 5. 生成 YOLO 格式数据（仅需一次）

> **仅从 Git 克隆、且已在第 3.5 节执行过** `prepare_yolo_dataset.py` **时可跳过本节。**

```bash
cd "$PROJECT"
python prepare_yolo_dataset.py
```

生成 `dataset/yolo/images/{train,val}` 与 `labels/{train,val}`。

---

## 6. 下载预训练权重（推荐，减少在线拉取失败）

> **仅从 Git 克隆、且已在第 3.5 节执行过** `download_yolo_weights.py` **时可跳过本节。**

```bash
cd "$PROJECT"
python download_yolo_weights.py
# 或只下部分：
# python download_yolo_weights.py yolo11m.pt yolo11l.pt yolo11x.pt
```

---

## 7. 一键训练 + 验证集置信度搜索 + 自动换更大模型（推荐）

在 **全量数据** 上训练，并在验证集上对 `conf` 搜索 **整串 Sequence accuracy**；若低于阈值会依次尝试更大 backbone（默认 `l`、`x`，见脚本参数）。

```bash
cd "$PROJECT"
python run_yolo_gpu_pipeline.py \
  --device auto \
  --require-gpu \
  --workers 8 \
  --batch 16 \
  --epochs 100
```

**显存不足（OOM）** 时减小 batch，例如：

```bash
python run_yolo_gpu_pipeline.py --require-gpu --workers 8 --batch 8 --epochs 100
```

**不训练最大模型 `x`（省显存/时间）：**

```bash
python run_yolo_gpu_pipeline.py --require-gpu --no-mega-upgrade
```

**快速冒烟（小数据、短 epoch，仅测流程）：**

```bash
python run_yolo_gpu_pipeline.py --quick --workers 4
```

### 流水线 Stage 1 断点续训（`--resume-train`）

Stage 1 训练中断或想**在同一 run 目录**里接着训时，指向该 run 的 **`weights/last.pt`**。续训完成后仍会执行 **验证集 conf 搜索**，并按阈值决定是否训练更大模型（`l` / `x`）。

| 场景 | 命令 |
|------|------|
| **同一 run 续 epoch**（中断后继续） | `--resume-train runs/detect/<run>/weights/last.pt` |
| **从旧 best 开新 run 再训**（新目录） | 不用 `--resume-train`；用 **`--primary-model runs/detect/<旧run>/weights/best.pt`**（会新建带时间戳的 run 名） |

示例（将 `<run>` 换成 `runs/detect/` 下实际文件夹名）：

```bash
cd "$PROJECT"
python run_yolo_gpu_pipeline.py \
  --require-gpu \
  --resume-train runs/detect/<run>/weights/last.pt \
  --workers 8 \
  --batch 16 \
  --epochs 100
```

说明：

- **`--epochs`** 为 Ultralytics **目标总 epoch**（从 checkpoint 接着涨到该值；若只想再多训 N 轮，设为 **已完成 epoch + N**）。
- **`--resume-train` 与 `--quick` 不要一起用**（quick 会改 epoch/数据比例，与续训意图冲突）。
- 仅续训、跳过 Stage 1 之后的 conf/升级时，请直接用 **§8** 的 `train_yolo.py --resume`。

### 流水线「作弊训练」（`--cheat-train`）

使用仓库内 **`svhn_digits_cheat.yaml`**：`train` 同时包含 **`images/train`** 与 **`images/val`**，验证集图像参与梯度更新。**流水线里对 `mchar_val` 的整串准确率与 conf 搜索会严重偏高**，不能当真实泛化或赛题公平分数；**测试集无标签**，作弊与否不改变测试集本身，但作业/报告若要求诚实汇报，须明确写明数据使用方式。

Stage 1 / 升级 / mega 各阶段调用 `train_yolo.py` 时均会带上 **`--cheat`**。`yolo_pipeline_summary.json` 中会有 **`"cheat_train": true`**。

```bash
cd "$PROJECT"
python run_yolo_gpu_pipeline.py \
  --require-gpu \
  --cheat-train \
  --workers 8 \
  --batch 16 \
  --epochs 100
```

**输出：** 终端末尾 JSON + 文件 **`runs/yolo_pipeline_summary.json`**  
其中 **`best_weights`**、**`best_conf`**、**`best_sequence_accuracy`** 为全局最优；若用了续训，JSON 里会有 **`resume_train`** 字段记录 `last.pt` 路径。

---

## 8. 仅手动训练（不用流水线时）

```bash
cd "$PROJECT"
python train_yolo.py --model yolo11m.pt --device auto --epochs 120 --batch 16 --workers 8
```

权重默认在：`runs/detect/mchar_digits/weights/best.pt`（若改过 `--name` 则目录名会变）。

### 在旧 run 的 `best.pt` 上再训一轮（新实验名）

用 **`--continue-from`** 指向上一阶段的 **`best.pt`**，并用 **`--name`** 新建子目录（避免与旧 run 混淆或覆盖）。**`--model` 在此模式下不使用**。

```bash
cd "$PROJECT"
python train_yolo.py \
  --continue-from runs/detect/<旧run目录名>/weights/best.pt \
  --name mchar_r2 \
  --device auto \
  --epochs 80 \
  --batch 16 \
  --workers 8
```

新权重输出在：**`runs/detect/mchar_r2/weights/best.pt`**。

### 同一 run 断点续训（`last.pt`）

与中断前相同的 **`--project` / `--name`**，或写全路径：

```bash
python train_yolo.py --resume --device auto --batch 16 --workers 8
# 或：python train_yolo.py --resume runs/detect/<你的run目录名>/weights/last.pt --device auto
# 与下面等价（路径为 last.pt 时）：
# python train_yolo.py --continue-from runs/detect/<你的run目录名>/weights/last.pt --device auto
```

**`--resume` 与 `--continue-from` 不要同时使用。**

### 作弊训练（`--cheat`，仅本地实验）

与默认 **`svhn_digits.yaml`** 不同，**`--cheat`** 使用 **`svhn_digits_cheat.yaml`**：训练集包含 **训练图 + 验证集图**，验证集上的 `eval_yolo_sequence` 与训练日志里的 val **不可当作公平准确率**。

```bash
cd "$PROJECT"
python train_yolo.py --cheat --model yolo11m.pt --device auto --epochs 100 --batch 16 --workers 8
```

---

## 9. 验证集整串准确率（与赛题口径一致）

流水线结束后，用 `summary.json` 里的权重与 `conf`；或手动指定：

```bash
cd "$PROJECT"
python eval_yolo_sequence.py \
  --weights runs/detect/<你的run目录名>/weights/best.pt \
  --conf 0.25 \
  --device auto
```

将 `<你的run目录名>` 换成实际文件夹名；**`--conf`** 可与 `yolo_pipeline_summary.json` 中的 **`best_conf`** 对齐。

---

## 10. 测试集提交 CSV

```bash
cd "$PROJECT"
python predict_yolo_submit.py \
  --weights runs/detect/<你的run目录名>/weights/best.pt \
  --conf 0.25 \
  --out yolo_submit.csv \
  --device auto
```

---

## 11. 常用调参（可选）

| 目的 | 示例 |
|------|------|
| 更大输入分辨率 | 在 `train_yolo.py` 中加 `--imgsz 800`（显存占用上升） |
| 换更大 backbone | `--primary-model yolo11l.pt` 或流水线里已自动升级 |
| 提高「继续训更大模型」的门槛 | `run_yolo_gpu_pipeline.py --min-acc 0.90` |
| DataLoader  workers | Linux 一般可用 `8`；报错时试 `--workers 4` |

---

## 12. 与本机 Windows 的区别

- 使用 **`/`** 路径，无 `C:\`。  
- 不要用 **`run_yolo_full_gpu.bat`**（仅 Windows）；云上直接用 **`python run_yolo_gpu_pipeline.py ...`**。  
- 必须在 **云实例 Terminal** 里执行；本机无 NVIDIA 时 `cuda` 会一直为 `False`。

---

## 13. 故障排查

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: ultralytics` | 执行第 3 节 `pip install -r requirements-yolo.txt` |
| `Missing svhn_digits.yaml` / 无 `dataset/yolo` | 确认在 `$PROJECT` 根目录；先跑 `prepare_yolo_dataset.py` |
| `Model weight not found` | 运行 `download_yolo_weights.py` 或上传 `.pt` 到项目根目录 |
| CUDA OOM | 减小 `--batch`，或 `--no-mega-upgrade`，或换 `yolo11s.pt` |
| `--require-gpu` 立即退出 | 当前 Python 未识别 GPU，检查 `nvidia-smi` 与 `torch.cuda.is_available()` |
| 流水线接着上次训练 | **`--resume-train runs/detect/<run>/weights/last.pt`**（§7）；仅手动续训见 §8 **`train_yolo.py --resume`** |
| **`--cheat` / `--cheat-train`** | 验证集已进训练集，**勿**把 val 整串 Acc 当真实分数；见 §7 / §8 |
| `--resume-train` 报错找不到文件 | 确认路径为 **`last.pt`**（不是 `best.pt`）；中断后若从未保存过 checkpoint，需重新 Stage 1 |

---

*与仓库内 `METHOD.md`、`run_yolo_gpu_pipeline.py`、`train_yolo.py` 保持一致；参数以 `python <脚本>.py --help` 为准。*
