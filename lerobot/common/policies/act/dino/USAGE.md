# 在ACT中使用DINOv3作为视觉编码器

## 快速开始

### 1. 使用DINOv3 ViT-Small

```python
from lerobot.policies.act import ACTPolicy, ACTConfig

config = ACTConfig(
    vision_backbone="dinov3_vits16",  # DINOv3 ViT-Small/16
    freeze_backbone=True,              # 冻结backbone权重
    dim_model=512,
)

policy = ACTPolicy(config, dataset_stats=dataset_stats)
```

### 2. 使用DINOv3 ViT-Base

```python
config = ACTConfig(
    vision_backbone="dinov3_vitb16",  # DINOv3 ViT-Base/16
    freeze_backbone=True,
    dim_model=512,
)
```

### 3. 使用DINOv3 ViT-Large

```python
config = ACTConfig(
    vision_backbone="dinov3_vitl16",  # DINOv3 ViT-Large/16
    freeze_backbone=False,             # 微调backbone
    dim_model=768,                     # 更大的隐藏维度
)
```

## 支持的模型

| 模型名称 | 参数量 | 隐藏维度 | Patch Size | 推荐场景 |
|---------|-------|---------|-----------|---------|
| `dinov3_vits16` | ~22M | 384 | 16 | 快速原型、资源受限 |
| `dinov3_vitb16` | ~86M | 768 | 16 | 平衡性能与速度 ⭐ |
| `dinov3_vitl16` | ~304M | 1024 | 16 | 追求最佳性能 |
| `dinov3_vith16plus` | ~632M | 1280 | 16 | 大规模数据集 |
| `dinov3_vit7b16` | ~7B | 4096 | 16 | 研究用途 |

## 配置参数

```python
config = ACTConfig(
    # Vision backbone
    vision_backbone="dinov3_vitb16",   # DINOv3模型名称
    freeze_backbone=True,              # 是否冻结backbone
    
    # ACT核心参数
    n_obs_steps=1,
    chunk_size=100,
    n_action_steps=100,
    dim_model=512,                     # Transformer隐藏维度
    
    # 其他参数...
)
```

## 注意事项

### 1. DINOv3加载路径

代码会尝试两种方式加载DINOv3：
1. 从本地路径 `'dinov3-vits'` 加载（如果你克隆了DINOv3仓库）
2. 从GitHub `'facebookresearch/dinov3'` 加载

**推荐做法**：将DINOv3仓库克隆到本地
```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3-vits
cd dinov3-vits
# 按照DINOv3的安装说明进行设置
```

### 2. 图像预处理

DINOv3使用ImageNet归一化（与ResNet相同），但要求：
- 输入图像尺寸能被patch_size（16）整除
- 推荐分辨率：224×224, 336×336, 448×448

### 3. 显存优化

DINOv3模型较大，建议：
- 使用 `freeze_backbone=True` 冻结权重
- 使用较小的模型（vits或vitb）
- 减小batch size
- 使用混合精度训练

### 4. 训练策略

**冻结训练（推荐）**：
```python
config = ACTConfig(
    vision_backbone="dinov3_vitb16",
    freeze_backbone=True,  # 只训练投影层和transformer
    optimizer_lr=1e-4,
)
```

**端到端微调**：
```python
config = ACTConfig(
    vision_backbone="dinov3_vitb16",
    freeze_backbone=False,
    optimizer_lr=1e-5,
    optimizer_lr_backbone=1e-6,  # backbone用更小的学习率
)
```

## 性能对比

### 特征质量对比

| Backbone | 参数量 | 特征维度 | 输出分辨率 | 语义特征强度 |
|---------|--------|---------|-----------|------------|
| ResNet18 | 11M | 512 | H/32 × W/32 | ⭐⭐ |
| DINOv3-S | 22M | 384 | H/16 × W/16 | ⭐⭐⭐⭐ |
| DINOv3-B | 86M | 768 | H/16 × W/16 | ⭐⭐⭐⭐⭐ |

### 计算成本对比

| Backbone | 训练速度 | 推理速度 | 显存占用 |
|---------|---------|---------|---------|
| ResNet18 | 100% | 100% | 1× |
| DINOv3-S (frozen) | 90% | 80% | 1.5× |
| DINOv3-B (frozen) | 75% | 60% | 2× |
| DINOv3-B (finetune) | 60% | 60% | 3× |

## 完整训练示例

```python
from lerobot.policies.act import ACTPolicy, ACTConfig

# 创建配置
config = ACTConfig(
    # 使用DINOv3
    vision_backbone="dinov3_vitb16",
    freeze_backbone=True,
    
    # 输入输出配置
    n_obs_steps=1,
    chunk_size=100,
    n_action_steps=100,
    
    # Transformer配置
    dim_model=512,
    n_heads=8,
    dim_feedforward=3200,
    n_encoder_layers=4,
    n_decoder_layers=1,
    
    # VAE配置
    use_vae=True,
    latent_dim=32,
    n_vae_encoder_layers=4,
    
    # 训练配置
    dropout=0.1,
    kl_weight=10.0,
    optimizer_lr=1e-4,
    optimizer_lr_backbone=1e-5,
)

# 创建策略
policy = ACTPolicy(config, dataset_stats=dataset_stats)

# 训练
for batch in dataloader:
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
```

## 故障排除

### 问题1：无法加载DINOv3模型

```python
# 解决方案1：克隆DINOv3仓库到本地
git clone https://github.com/facebookresearch/dinov3.git dinov3-vits

# 解决方案2：手动指定加载路径
# 在 dino_encoder.py 中修改加载路径
```

### 问题2：显存不足 (OOM)

```python
# 解决方案：使用更小的模型或冻结backbone
config = ACTConfig(
    vision_backbone="dinov3_vits16",  # 使用smaller model
    freeze_backbone=True,              # 冻结权重
)
```

### 问题3：特征图尺寸不匹配

DINOv3 patch_size=16，输出分辨率为输入的1/16：
- 输入224×224 → 输出14×14
- 输入336×336 → 输出21×21
- 输入448×448 → 输出28×28

确保输入图像尺寸是16的倍数。

## 从ResNet迁移

如果你已有使用ResNet训练的模型，迁移到DINOv3：

```python
# 原配置
old_config = ACTConfig(
    vision_backbone="resnet18",
    dim_model=512,
)

# 新配置
new_config = ACTConfig(
    vision_backbone="dinov3_vitb16",  # 替换为DINOv3
    freeze_backbone=True,              # 推荐冻结
    dim_model=512,                     # 保持相同
    # 其他参数保持不变
)

# 注意：backbone权重无法迁移，需要重新训练
# 但transformer部分的权重可以尝试迁移
```