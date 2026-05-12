# 本地 CPU 完整训练指南

在 **无 NVIDIA GPU** 或 **仅使用 CPU 版 PyTorch** 时，仍可按本文完成 **YOLO 检测方案** 的全流程：数据准备 → 训练 → 验证集整串准确率 →（可选）测试集 CSV。

**重要说明**：全量数据 + 默认 100 epoch 在 CPU 上可能耗时 **数天**。建议先用 **`--quick`** 或 **`--fraction`** 跑通流程，再决定是否开长时间全量训练。

---

## 0. 环境要求

- **Python 3.10+**（你当前为 3.12 亦可）
- **CPU 版 PyTorch** 即可：`pip install torch torchvision`（或清华镜像安装 `+cpu` 轮子）
- 项目依赖：

```powershell
cd C:\Users\LJTsc\Desktop\Street_Character_Recognition
pip install -r requirements-yolo.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 1. 数据准备

确保已解压天池数据，目录与 `README` 一致：

- `dataset\mchar_train\`、`dataset\mchar_val\`、`dataset\mchar_test_a\`
- `dataset\mchar_train.json`、`dataset\mchar_val.json`

生成 YOLO 标注与链接（只需执行一次）：

```powershell
cd C:\Users\LJTsc\Desktop\Street_Character_Recognition
python prepare_yolo_dataset.py
```

---

## 2. 预训练权重

将 `yolo11m.pt`（及可选 `yolo11l.pt`）放在项目根目录，或运行：

```powershell
python download_yolo_weights.py yolo11m.pt
```

CPU 上 **不建议** 使用 `yolo11x.pt` 全量训练（极慢且占内存）；流水线在 CPU 预设下会 **跳过 mega 阶段**。

---

## 3. 推荐：一键流水线（含 conf 搜索 + 自动换更大模型）

已增加 **`--cpu-preset`**：固定 **`device=cpu`**，将 **`batch` 限制为 ≤4**、**`workers`≤2**，**验证集 conf 搜索** 默认只用 **前 5000 张**（减轻 CPU 上数万次推理负担）；**不跑 `yolo11x`**。全量训练 epoch 仍由 **`--epochs`** 控制（默认 100）。

```powershell
cd C:\Users\LJTsc\Desktop\Street_Character_Recognition
python run_yolo_gpu_pipeline.py --cpu-preset
```

**更省时间（先验证能跑通）：**

```powershell
python run_yolo_gpu_pipeline.py --quick
```

若本机已装 CUDA 版 PyTorch 且希望**冒烟也强制走 CPU**，请用：

```powershell
python run_yolo_gpu_pipeline.py --cpu-preset --quick
```

**全量训练但缩短 epoch（例如 30 轮）：**

```powershell
python run_yolo_gpu_pipeline.py --cpu-preset --epochs 30
```

**显式指定设备（与 preset 二选一亦可）：**

```powershell
python run_yolo_gpu_pipeline.py --device cpu --batch 2 --workers 0 --no-mega-upgrade --val-max-samples 5000
```

Windows 上 DataLoader 若报错，把 **`--workers 0`**。

**不要使用** **`--require-gpu`**（在 CPU 上会直接退出）。

---

## 4. 流水线输出

结束后查看：

- 终端末尾 **JSON**
- 文件 **`runs\yolo_pipeline_summary.json`**

其中 **`best_weights`**、**`best_conf`**、**`best_sequence_accuracy`** 为当前策略下的最优结果。

若使用了 **`--cpu-preset` 的 5000 张子集做 conf 搜索**，建议在全验证集上 **再算一次** 官方口径准确率（见下一节）。

---

## 5. 验证集整串准确率（赛题口径）

将上一步得到的权重与 `best_conf` 代入：

```powershell
python eval_yolo_sequence.py --weights "runs\detect\<run名>\weights\best.pt" --conf 0.25 --device cpu
```

把 **`<run名>`** 换成实际目录名；**`--conf`** 与 `summary.json` 里 **`best_conf`** 对齐。

---

## 6. 仅训练（不用流水线）

```powershell
python train_yolo.py --device cpu --batch 2 --workers 0 --epochs 100 --model yolo11m.pt
```

`train_yolo.py` 在 CPU 上会自动 **关闭 AMP**。

---

## 7. 测试集提交 CSV

```powershell
python predict_yolo_submit.py --weights "runs\detect\<run名>\weights\best.pt" --conf 0.25 --out yolo_submit.csv --device cpu
```

---

## 8. 参数与耗时建议

| 项目 | CPU 建议 |
|------|----------|
| `--batch` | `2`～`4`，OOM 再减 |
| `--workers` | Windows 常用 **`0`**；Linux 可 `2` |
| `--epochs` | 全量可从 `30`～`100` 权衡；可先 `--fraction 0.1` 试训 |
| 更大模型 | 优先 **`yolo11n.pt` / `yolo11s.pt`**；`m` 已较慢；`l`/`x` 仅在有耐心时 |
| `--no-upgrade` / `--no-mega-upgrade` | 跳过后续大模型阶段，节省时间 |

---

## 9. 与 GPU 流程的差异

| 项目 | CPU（本文） | GPU（见 `SHELL.md`） |
|------|-------------|----------------------|
| 设备 | `--device cpu` 或 `--cpu-preset` | `--device auto` 或 `--require-gpu` |
| 批大小 | 小 | 可 8～16+ |
| 验证 conf 搜索 | 建议子集或 `--cpu-preset` 默认 5000 | 可全量 1 万张 |
| 一键批处理 | 用下面 `run_yolo_cpu.bat`（可选） | `run_yolo_full_gpu.bat` |

---

## 10. 可选：本地一键批处理

在项目根目录创建或运行（可自行新建 `run_yolo_cpu.bat`）：

```bat
@echo off
cd /d "%~dp0"
python run_yolo_gpu_pipeline.py --cpu-preset
pause
```

（脚本名仍为 `run_yolo_gpu_pipeline.py`，仅通过参数区分 CPU/GPU。）

---

## 11. 常见问题

| 现象 | 处理 |
|------|------|
| 极慢 / 想中断 | `Ctrl+C`；已保存的 `last.pt` 可在 `train_yolo.py` 里用 `resume`（需查 Ultralytics 文档）或减小 epoch 重跑 |
| 内存不足 | 减小 `--batch`、`--imgsz`，或换 `yolo11n.pt` |
| `cuda: False` | CPU 训练为预期；勿加 `--require-gpu` |

---

*与 `METHOD.md`、`SHELL.md`、`run_yolo_gpu_pipeline.py`（含 `--cpu-preset`）一致。*
