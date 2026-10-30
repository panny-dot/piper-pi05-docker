# PI0.5 全参数微调环境迁移交接说明

交付日期：2026 年 10 月 30 日  
镜像版本：`piper-pi05-full:20261030`

## 一、交付内容

本交付包用于迁移 PI0.5 全参数微调环境，包含 Docker 镜像、配套启动与构建源码、PI0.5 基座及 tokenizer。数据集由接收方另行准备。

训练框架采用 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 的 PyTorch PI0.5 实现。基座来源为 [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)，固定版本为 `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`。配套 Docker 文件及 `run.sh` 为本项目新增的封装与启动文件。

| 项目 | 交付版本与配置 |
|---|---|
| 镜像名称 | `piper-pi05-full:20261030` |
| 容器系统 | Ubuntu 24.04 |
| Python | 3.12 |
| LeRobot | 0.6.1 源码快照 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| Transformers | 5.5.4 |
| 训练方式 | 全参数微调，关闭 PEFT / LoRA |
| 显存相关配置 | bfloat16、开启梯度检查点、关闭 compile |
| 数据接口 | LeRobot 数据集，三路 RGB、14 维状态和动作、30Hz |

## 二、交付目录

```text
PI05全参数微调_20261030/
├── README.md                              # 本交接说明
├── SHA256SUMS                             # 交付文件完整性校验清单
├── piper-pi05-full_20261030.tar             # Docker 镜像
├── source/                                # 配套启动与构建源码
│   ├── run.sh                             # 宿主机训练启动入口
│   ├── scripts/run_container.py            # Docker 调用与目录挂载
│   ├── Dockerfile                         # 镜像构建定义
│   ├── build.sh                           # 镜像构建入口
│   ├── export.sh                          # 镜像导出入口
│   ├── constraints.txt                    # 核心依赖版本
│   ├── requirements-torch.txt             # PyTorch 安装依赖
│   ├── container/                         # 容器内训练入口与配置
│   ├── vendor/                            # LeRobot 源码快照
│   ├── evidence/                          # 源码清单与依赖版本记录
│   └── LICENSE.lerobot                     # LeRobot 许可证
├── models/
│   └── pi05_base/                         # PI0.5 原始基座及配置
└── assets/
    └── tokenizer/                         # 本地 tokenizer
```

镜像归档约 8.74GB，基座权重约 14.47GB，交付文件合计约 23.23GB。部署时需为镜像导入、数据集、缓存和训练检查点另行预留空间。

## 三、目标机器要求

- Linux x86_64。
- 已安装 Docker，执行用户具备 Docker 使用权限。
- 已安装兼容 CUDA 12.8 的 NVIDIA 驱动，并配置 NVIDIA Container Toolkit。
- 宿主机安装 Python 3，用于运行配套启动脚本。
- GPU 显存与磁盘空间能够满足本轮训练需求。

**资源限制：**全参数训练的显存开销包含模型、梯度和优化器状态，不能仅按权重文件大小估算；32GB 显存下已有首次 AdamW 更新显存不足的记录。训练入口采用单训练进程，未集成 CPU 卸载或多卡参数分片；开放多张 GPU 不会自动启用分布式训练。正式训练需使用资源足够且经过验证的配置。

## 四、导入镜像

### 1. 进入交付目录

将交付文件夹完整复制到目标机器，进入其实际存放目录。例如：

```bash
cd '/data/PI05全参数微调_20261030'
```

其中 `/data/PI05全参数微调_20261030` 为示例路径，按实际位置替换。后续命令均在交付根目录执行。

### 2. 校验交付文件

```bash
sha256sum -c SHA256SUMS
```

全部文件显示 `OK` 后，继续导入。

### 3. 导入镜像

```bash
docker load -i piper-pi05-full_20261030.tar
```

完成后显示：

```text
Loaded image: piper-pi05-full:20261030
```

此步骤将训练软件环境导入 Docker。宿主机上的启动脚本、模型和 tokenizer 由交付包提供，导入镜像不会生成这些文件。

## 五、准备训练输入

基座及 tokenizer 已分别放在 `models/pi05_base/` 和 `assets/tokenizer/`，迁移时保留整个目录结构。

数据集需另行准备为完整 LeRobot 格式，保留 `meta/`、`data/`、`videos/` 的相对结构。本入口面向 Piper 双臂数据：

| 数据项 | 要求 |
|---|---|
| 图像 | `observation.images.front`、`observation.images.left`、`observation.images.right`，RGB，480×640 |
| 状态 | `observation.state`，14 维，从臂实际反馈 |
| 动作 | `action`，14 维，主臂示教的绝对目标 |
| 频率与单位 | 30Hz；关节使用弧度，夹爪使用米 |

模型、数据和 tokenizer 以只读方式挂载到容器；训练输出与缓存保存到宿主机。容器运行默认断网，输入资料应提前完整准备。

## 六、启动训练

仍在交付根目录执行，先设置数据集和输出目录。下面两项示例路径需按目标机器修改：

```bash
DATA='/data/piper_box_master_hold_30ms_full_20261014'
OUT='/data/pi05_runs/full_from_base'
```

然后启动训练：

```bash
bash ./source/run.sh train \
  --dataset "$DATA" \
  --model "$PWD/models/pi05_base" \
  --tokenizer "$PWD/assets/tokenizer" \
  --output "$OUT" \
  --batch-size 1 \
  --steps 30000 \
  --save-freq 5000 \
  --workers 4
```

上述命令会直接从交付的原始基座开始全参数微调。30000 步、batch size 为 1、每 5000 步保存一次均为示例设置，需按本轮训练计划调整；训练步数不代表效果保证。

`run.sh` 负责调用 Docker、使用 GPU 并建立目录挂载。它依赖同目录下的 `scripts/run_container.py`，交接时应保留完整 `source/` 目录。

当前入口会新建优化器和学习率调度器，并使用全部数据训练；未配置独立验证集，也不执行原训练进度的断点恢复。

## 七、训练结果保存

训练配置、指标记录及检查点均保存在 `OUT` 指定的宿主机目录。检查点包含模型与训练状态，需为多次保存预留磁盘空间。用于后续部署时，应保留选定检查点的完整模型和处理器文件。

