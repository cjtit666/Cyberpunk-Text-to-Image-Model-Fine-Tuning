# 赛博朋克风格微调项目运行指南

## 项目概述

本项目基于 min-DALLE 模型，实现了赛博朋克风格的图像生成微调。通过 LoRA 技术对模型进行参数高效微调，使生成的图像具有赛博朋克风格特征。

## 项目结构

```
1项目三代码-运行版/
├── files/              # 模型文件目录
│   ├── merges.txt      # BPE 合并规则
│   └── vocab.json      # 词表文件
├── min_dalle_stub/     # 核心模型代码
│   ├── text_tokenizer.py         # 文本分词器
│   ├── dalle_bart_encoder.py     # BART 文本编码器
│   ├── dalle_bart_decoder.py     # BART 图像解码器
│   └── vqgan_detokenizer.py      # VQGAN 图像解码
├── cyberpunk_images/   # 训练数据图像
├── cyberpunk_lora/     # LoRA 权重目录
├── results/            # 生成结果目录
│   ├── comparison/     # 对比图像
│   ├── visualization/  # 可视化数据
│   └── validation/     # 验证图像
├── experiment_records/ # 实验记录
├── README_CYBERPUNK.md # 运行指南
├── min-DALLE.ipynb     # 原始 Jupyter 笔记本
├── requirements.txt    # 依赖文件
├── run_min_dalle.py    # 原始模型运行脚本
└── train_cyberpunk.py  # 赛博朋克风格微调脚本
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

运行原始的 `run_min_dalle.py` 脚本，会自动下载所需的模型文件：

```bash
python run_min_dalle.py
```

## 运行步骤

### 1. 训练赛博朋克风格模型

```bash
python train_cyberpunk.py
```

**训练过程包括：**
- 自动生成 50 张赛博朋克风格训练图像
- 对训练图像进行质量评估和过滤
- 使用 LoRA 技术对模型进行微调（约 3-4 小时）
- 每轮生成验证图像，监控训练效果
- 保存 LoRA 权重到 `cyberpunk_lora` 目录

### 2. 生成对比图像

训练完成后，脚本会自动生成 30 组对比图像，保存到 `results/comparison` 目录：
- `original_*.png`：原始模型生成的图像
- `cyberpunk_*.png`：微调后模型生成的图像
- `comparison_*.png`：并排对比图像（包含输入文本）

### 3. 查看可视化数据

脚本会生成以下可视化数据：
- `results/visualization/loss_curve.png`：训练损失曲线
- `results/visualization/training_config.json`：训练配置信息
- `results/comparison_results.json`：对比结果记录
- `experiment_records/experiment_record.json`：完整实验记录

## 结果分析

### 1. 图像对比

打开 `results/comparison` 目录，查看生成的 30 组对比图像：
- 原始模型生成的图像：普通风格
- 微调模型生成的图像：赛博朋克风格（霓虹灯光、未来感、dystopian 元素）

### 2. 训练过程

查看 `results/visualization/loss_curve.png` 文件，分析训练损失曲线：
- 观察损失是否稳定下降
- 评估训练是否充分

### 3. 实验记录

查看 `experiment_records/experiment_record.json` 文件，包含以下信息：
- 训练数据配置
- 模型参数设置
- 训练超参数
- 生成配置
- 结果统计

## 论文撰写建议

### 1. 引言
- 介绍 DALL-E 模型的原理
- 说明风格微调的意义
- 提出赛博朋克风格作为目标风格的原因

### 2. 方法
- 描述 LoRA 微调技术
- 详细说明数据处理流程（自动生成 + 质量过滤）
- 展示模型微调的具体实现
- 分析训练策略和参数选择

### 3. 实验结果
- 展示对比图像（原始 vs 微调）
- 分析损失曲线和训练过程
- 讨论微调效果和局限性
- 提供定量和定性评估

### 4. 结论
- 总结微调方法的有效性
- 提出未来改进方向
- 讨论在其他风格上的应用潜力

## 技术特点

1. **参数高效**：使用 LoRA 技术，仅训练少量参数
2. **数据自生成**：自动生成赛博朋克风格训练数据，无需外部数据集
3. **质量控制**：实现图像质量评估和过滤，确保训练数据质量
4. **实时验证**：每轮生成验证图像，监控训练效果
5. **详细记录**：保存完整的实验记录，方便论文撰写
6. **多样化对比**：生成 30 组不同场景的对比图像

## 注意事项

1. **显存要求**：推荐使用至少 8GB GPU 显存
2. **训练时间**：在 RTX 3060 上约需 3-4 小时
3. **数据生成**：首次运行会生成 50 张训练图像（约 15-20 分钟）
4. **结果可复现性**：由于生成过程的随机性，每次运行的结果会略有不同
5. **生成参数**：可在 `generate_image` 函数中调整生成参数，获得不同风格效果

## 扩展建议

1. **尝试其他风格**：修改 `cyberpunk_prompts` 列表，可尝试其他风格的微调
2. **调整微调参数**：修改 LoRA 的 rank 和 alpha 值，探索不同参数对结果的影响
3. **增加评估指标**：可添加 FID、IS 等图像质量评估指标
4. **构建 Web 界面**：创建简单的 Web 应用，允许用户输入提示词并生成赛博朋克风格图像
5. **多风格融合**：尝试融合多种风格，创建更丰富的生成效果

项目完成后，你将拥有一个功能完整的风格微调系统，可作为毕设论文的核心内容，并为实习面试提供有力的技术展示。