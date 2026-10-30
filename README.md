# Piper PI0.5 Docker 训练环境

基于 LeRobot 的 PI0.5 全参数微调环境封装，提供 Docker 构建文件、训练启动入口和部署交接说明。

## 内容

- `source/`：Dockerfile、依赖配置、训练入口及 LeRobot 0.6.1 源码快照。
- `docs/DEPLOYMENT.md`：离线交付包的部署与训练说明。

`run.sh` 和 Docker 包装文件为本项目新增内容。训练实现来自 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，模型原作者项目为 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。LeRobot 许可证保存在 `source/LICENSE.lerobot`。

## 部署

源码和文档位于本仓库，Docker 镜像通过 [v20261030 镜像附件](https://github.com/panny-dot/piper-pi05-docker/releases/tag/v20261030) 提供。模型权重、tokenizer 和数据集另行准备，不在本次 GitHub 发布中上传。

镜像以 gzip 分卷附件提供。使用具有仓库访问权限的 GitHub 账号登录 `gh` 后，执行：

```bash
gh release download v20261030 \
  --repo panny-dot/piper-pi05-docker \
  --dir ./image-20261030

bash ./image-20261030/RESTORE.sh /data/pi05-image
docker load -i /data/pi05-image/piper-pi05-full_20261030.tar
```

其中 `/data/pi05-image` 为示例还原目录，需按目标机器修改并确保可写、空间充足。脚本会校验下载分卷并还原 Docker 镜像归档，不启动训练。

也可在 Release 页面下载全部附件，放在同一文件夹后执行 `RESTORE.sh`。分卷必须全部下载，不能只下载其中一个。

使用已构建镜像时，按 `docs/DEPLOYMENT.md` 准备完整交付目录，导入 `piper-pi05-full:20261030` 后，通过包内 `source/run.sh train` 启动训练。

需要重新构建时，在本仓库根目录执行：

```bash
bash ./source/build.sh
```

构建需要网络访问；构建后的镜像仍需挂载本地模型、tokenizer 和数据集。

## 运行要求

Linux x86_64、Docker、宿主机 Python 3、兼容 CUDA 12.8 的 NVIDIA 驱动及 NVIDIA Container Toolkit。

全参数训练需容纳模型、梯度和优化器状态。32GB 显存下已有首次 AdamW 更新显存不足的记录；当前入口未集成 CPU 卸载或多卡参数分片。镜像导入成功不代表目标硬件具备完整训练能力。
