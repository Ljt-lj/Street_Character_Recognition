# 课堂练习3 — 街景字符识别实验报告

| 项目   | 内容                                                                        |
| ---- | ------------------------------------------------------------------------- |
| 学号   | 23307130167                                                               |
| 姓名   | 李江涛                                                                       |
| 代码仓库 | `[Street_Character_Recognition](https://github.com/Ljt-lj/Street_Character_Recognition.git)`                                            |

---
## 1. 方案概述

本仓库实现两条路线，**主方案为 YOLO 目标检测**（利用框标注），Baseline 作对照。

| 路线            | 思路                                      | 主要脚本                                                                  |
| ------------- | --------------------------------------- | --------------------------------------------------------------------- |
| **Baseline**  | 整图 ResNet50 特征 + 4 个分类头（0–9 +「空」），未用检测框 | `baseline.py`                                                         |
| **YOLO（主方案）** | 每位数字一类检测（10 类）；推理按框 **x 中心从左到右** 拼串     | `prepare_yolo_dataset.py` → `train_yolo.py` → `eval_yolo_sequence.py` |

**整串准确率**统一用 `eval_yolo_sequence.py` 的 **Sequence accuracy** 统计.

---

## 2. 运行环境

| 项目      | 配置                                                                                 |
| ------- | ---------------------------------------------------------------------------------- |
| 操作系统    | Windows 10 / Linux（ModelScope DSW 云 GPU）                                           |
| Python  | 3.10+（本地 3.12 已测）                                                                  |
| 深度学习框架  | PyTorch（GPU 实例为 CUDA 版；本地无显卡时为 CPU 版）                                              |
| 检测训练    | Ultralytics YOLO11（`requirements-yolo.txt`：`ultralytics>=8.3.0`）                   |
| 显卡（云训练） | NVIDIA A10 等（`nvidia-smi` + `torch.cuda.is_available()==True`）                     |
| 依赖安装    | `pip install -r requirements-yolo.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |

**数据与权重路径**：`dataset/mchar_{train,val,test_a}/` + JSON；预训练 `yolo11m.pt` 等由 `download_yolo_weights.py` 置于项目根目录（`.gitignore` 忽略 `*.pt`）。

---

## 3. 模型设计

### 3.1 主方案：YOLO11 数字检测

- **类别**：10 类，对应数字 `0`–`9`；每张图多个框。
- **数据格式**：`prepare_yolo_dataset.py` 将 JSON 转为 YOLO `class cx cy w h`（归一化），目录 `dataset/yolo/`。
- **配置**：`svhn_digits.yaml`（`train` / `val` 严格划分）。
- **默认骨干**：`yolo11m.pt`；流水线 `run_yolo_gpu_pipeline.py` 在验证集整串 Acc 低于阈值时自动尝试 `**yolo11l` / `yolo11x`**。
- **推理拼串**（`yolo_metrics.py`）：对每张图取检测框，按 **x 中心升序** 拼接类别 ID 为字符串；空检测视为 `""`。

### 3.2 评测与流水线

- **单模型评测**：`python eval_yolo_sequence.py --weights <best.pt> --conf <阈值> --device auto`
- **一键流程**：`run_yolo_gpu_pipeline.py` = 训练 → 验证集 **conf 网格搜索**（0.2–0.4）→ 按 Acc 选优 → 可选更大模型；结果写入 `runs/yolo_pipeline_summary.json`。

---

## 4. 损失函数与训练目标

### 4.1 YOLO

- **损失**：由 Ultralytics 内置（**box** + **cls** + **dfl**），训练日志中可见；验证阶段报告 **mAP50 / mAP50-95** 与 P、R。
- **本仓库显式设置**（`train_yolo.py`）：`cos_lr=True`，`warmup_epochs=3`，`close_mosaic=10`，`patience=40`（早停），GPU 上 **AMP** 开启，CPU 上关闭。
- **与赛题对齐的指标**：整串 **Sequence accuracy**（非 mAP），需在训练后用 `eval_yolo_sequence.py` 单独计算。

---

## 5. 数据增强与调参

### 5.1 YOLO 增强与超参

- **增强**：沿用 Ultralytics 默认（含 Mosaic 等）；`close_mosaic=10` 在最后 10 个 epoch 关闭 Mosaic 利于收敛。
- **主要超参**：

| 参数              | 推荐值                          | 说明             |
| --------------- | ---------------------------- | -------------- |
| `imgsz`         | 640（可试 800）                  | 门牌数字较小，提分辨率常有利 |
| `epochs`        | 80–120                       | 流水线默认 100      |
| `batch`         | 32（OOM 则 16）                 | 与显存相关          |
| `conf`（推理）      | 0.2–0.4 搜索（best-conf：0.325）  | 对整串 Acc 影响大    |
| `primary-model` | yolo11m → l → x（实际采用yolo11l） | 流水线自动升级        |

### 5.2 调参记录（此处保留第一次训练与最优训练）

| 实验编号  | 模型      | imgsz | epochs | conf  | 验证集 Sequence Acc | 备注                  |
| ----- | ------- | ----- | ------ | ----- | ---------------- | ------------------- |
| Exp-1 | yolo11m | 640   | 80     | 0.325 | 0.853            | 默认流水线               |
| Exp-2 | yolo11l | 640   | 80     | 0.325 | 0.9384           | 主结果（`mchar_digits`） |

---

## 6. 方案创新性与工程改进

1. **检测 + 排序拼串**：直接利用赛题框标注，避免 Baseline 仅 4 位、不用框的缺陷。
2. **赛题口径评估**：`yolo_metrics.py` 统一「左→右」拼串与 `eval_yolo_sequence.py`，与 mAP 解耦。
3. **端到端流水线**：`run_yolo_gpu_pipeline.py` 集成训练、**conf 搜索**、未达标自动换更大模型，并输出 JSON 摘要。
4. **可复现工程**：`prepare_yolo_dataset.py`、权重下载脚本、`yolo_device.py` 自动 CPU/GPU；`SHELL.md` / `METHOD.md` 分环境说明。
5. **训练续跑与热启动**：`--resume` / `--continue-from` / 流水线 `--resume-train`。

---

## 7. 实验结果

**指标定义**：验证集 **Sequence accuracy** = 预测整串与 `mchar_val.json` 完全一致图片数 / 总图片数。

**本次主结果）**：

| 指标               | 结果                                          |
| ---------------- | ------------------------------------------- | 
| 验证集 Sequence Acc | **0.9384**                                  | 
| 最优权重             | `runs/detect/mchar_digits/weights/best.pt`  | 
| 最优 conf          | **0.325**                                   |

**图1 — 训练过程（loss / mAP 等，Ultralytics 汇总）**
![1778997301738](images/PROJECT1/1778997301738.png)
此为主结果流水线中模型yolo11l.pt的20轮epoch，总训练轮次为4阶段流水线（80轮）。

**图2 — 验证集整串准确率（`eval_yolo_sequence.py`，conf=0.325）**
![1778997404628](images/PROJECT1/1778997404628.png)

验证集 Sequence accuracy 0.9384

复现验证集命令：

```bash
python eval_yolo_sequence.py \
  --weights runs/detect/mchar_digits/weights/best.pt \
  --conf 0.325 --device auto
```

**图3 — 验证集检测可视化（预测框，节选）**

验证集预测可视化 val_batch0_pred
![1778997535543](images/PROJECT1/1778997535543.png)

同目录另有 `val_batch0_labels.jpg`（标注）、`train_batch0.jpg`（训练 batch 样例）、`confusion_matrix.png` 等，均位于 `**runs/detect/mchar_digits/`**。

---

## 9. 实验困难与解决方案

| 困难                | 现象                     | 处理                                                         |
| ----------------- | ---------------------- | ---------------------------------------------------------- |
| 训练中断              | 实例断开                   | `train_yolo.py --resume` 或流水线 `--resume-train .../last.pt` |
| mAP 高、整串 Acc 低    | 漏检/多检/conf 不当          | 调 `--conf`；增大 `imgsz`；换 yolo11l/x                          |
| CUDA OOM          | 训练崩溃                   | 减小 `batch`；`--no-mega-upgrade`                             |

---

## 10. 参考文献与资源

1. 赛题与数据说明：仓库 `README.md`；天池 [https://tianchi.aliyun.com/competition/entrance/531795](https://tianchi.aliyun.com/competition/entrance/531795)
2. Ultralytics YOLO：[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. SVHN 原始论文与数据集背景（赛题采样来源）
4. 开源 OCR 参考：PaddleOCR、OpenOCR（README 改进思路）
5. 本仓库CPU方法手册：`METHOD.md`；GPU上复现：`SHELL.md`

---

## 附录：复现命令摘要

```bash
# 数据与 YOLO 标注
python prepare_yolo_dataset.py
python download_yolo_weights.py yolo11m.pt

# GPU 全流程（推荐）
python run_yolo_gpu_pipeline.py --require-gpu --workers 8 --batch 16 --epochs 100

# 验证集整串 Acc
python eval_yolo_sequence.py --weights runs/detect/<run>/weights/best.pt --conf 0.25 --device auto

# 测试集提交
python predict_yolo_submit.py --weights runs/detect/<run>/weights/best.pt --conf 0.25 --out yolo_submit.csv
```

