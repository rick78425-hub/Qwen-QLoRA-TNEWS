# Qwen2.5-1.5B QLoRA 中文新闻分类微调

基于 Qwen2.5-1.5B-Instruct 和 CLUE TNEWS 数据集，使用 4-bit NF4 量化与 LoRA 参数高效微调，实现中文新闻文本生成式多分类任务。在单卡 6GB 显存环境下完成完整训练、模型保存、Adapter 权重排错及验证集评估。

## 1. 项目概述

本项目旨在探索有限 GPU 显存条件下，利用开源大语言模型完成中文新闻分类任务的可行性。

与传统分类模型直接输出类别 ID 不同，本项目将新闻文本分类转化为指令微调任务：输入新闻标题，由 Qwen 模型根据任务指令生成对应的中文类别名称。

项目围绕完整的大模型微调流程展开，主要包括数据集处理、指令数据构建、Chat Template、Label Mask、4-bit 量化加载、LoRA 参数注入、模型训练、Adapter 保存与加载、批量推理及分类效果评估。

项目不仅关注模型微调后的准确率，也重点处理了训练标签与预测结果映射、Adapter 权重加载异常及生成式分类结果解析等工程问题。

## 2. 数据集与任务定义

### 2.1 数据集

采用 CLUE 基准中的 TNEWS 中文新闻分类数据集。

| 数据划分 | 样本数量 | 用途 |
| --- | ---: | --- |
| Train | 53,360 | 模型训练 |
| Validation | 10,000 | 训练过程评估及最终模型评估 |
| Test | 10,000 | 原始数据集测试划分，本项目未将其作为最终指标来源 |

任务包含实验中使用的 16 个类别标签：

体育、其他、军事、农业、国际、娱乐、房产、故事、教育、文化、旅游、汽车、游戏、科技、股票、财经。

### 2.2 标签与指令构造

将原始新闻文本及数字标签转换为中文指令对话格式。

输入包含：
- System：定义中文新闻分类任务及允许输出的类别。
- User：提供待分类的新闻标题。
- Assistant：生成对应的中文新闻类别。

使用 Hugging Face Chat Template 构造模型输入，并通过 Label Mask 将 Prompt 部分的标签设置为 -100，仅对 Assistant 的目标回答计算训练损失。

这样可以让模型重点学习新闻文本与目标类别之间的映射关系。

## 3. 模型与微调方案

### 3.1 基础模型

- Base Model：Qwen/Qwen2.5-1.5B-Instruct
- 模型框架：PyTorch、Transformers
- 微调框架：PEFT
- 量化工具：bitsandbytes

### 3.2 QLoRA 量化与参数配置

采用 4-bit NF4 量化加载基础模型，减少模型权重的显存占用；在 Transformer 注意力层注入 LoRA 适配器，仅训练低秩增量参数，不对基础模型进行全参数更新。

| 配置项 | 参数 |
| --- | --- |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj、k_proj、v_proj、o_proj |
| Task Type | CAUSAL_LM |
| Optimizer Batch Size | 4 |
| Gradient Accumulation | 4 |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Training Precision | BF16 |

训练共完成 10,005 个优化器更新步，耗时约 7 小时 53 分钟。

最终 Adapter 参数量约 1,746 万字节（约 16.65 MiB），与基础模型权重分离保存，推理时需要加载匹配的基础模型及 Adapter。

## 4. 核心工程实现

### 4.1 指令数据预处理

基于 TNEWS 原始数据构造 System、User、Assistant 三部分消息。

使用 Chat Template 生成完整输入序列，并根据 Prompt 与 Assistant 回答的 Token 边界构造 Labels。

Prompt 部分使用 -100 屏蔽损失，避免模型学习复述输入指令，使训练目标集中在类别回答上。

### 4.2 低显存模型微调

使用 BitsAndBytesConfig 加载 4-bit NF4 量化模型，通过 PEFT 注入 LoRA 适配器。

训练过程中只更新 LoRA 参数，降低训练所需的显存和可训练参数规模，使 1.5B 模型能够在单卡 6GB 显存环境下完成微调。

### 4.3 生成式推理与结果解析

构造统一的新闻分类 Prompt，通过模型生成中文类别名称。

对生成结果进行清洗和类别映射，处理额外标点、前缀和非标准输出，最终将模型生成结果转换为可计算的分类标签。

### 4.4 Adapter 权重保存与加载排错

项目中曾出现 Adapter 保存后参数名称前缀不一致的问题，导致重新加载时权重无法正确匹配。

针对该问题，检查 Adapter 配置、参数名称及张量映射关系，通过参数名称前缀修正和权重逐张量核验完成修复。

最终修复结果：
- 成功匹配 224 个 Adapter 权重张量。
- 修复后最大绝对误差为 0。
- 重新加载 Adapter 后完成推理验证。

这一过程验证了基础模型、Adapter 配置和实际权重之间的一致性。

## 5. 完整实验结果

### 5.1 Base 与 QLoRA 对比

在相同的 10,000 条 Validation 样本上，对基础模型和微调模型进行批量生成式分类评估。

| 评价指标 | Base Model | QLoRA Model |
| --- | ---: | ---: |
| Accuracy | 56.30% | 58.96% |
| Macro F1 | 约 0.50 | 0.54 |
| Weighted F1 | 约 0.57 | 0.59 |

Accuracy 从 56.30% 提升至 58.96%，提高 2.66 个百分点。

### 5.2 分类效果分析

- 体育类 F1 为 0.76，表现相对较好。
- 汽车、游戏、房产、教育等类别取得中等偏好的分类效果。
- 故事类 Recall 为 0.81，但 Precision 仅为 0.33，存在一定的过预测现象。
- 股票类仅有 45 条真实验证样本，分类结果容易受到小样本影响。
- 其他类 Support 为 0，当前评估结果无法反映模型在该类别上的真实分类能力。

Macro F1 为 0.54，低于 Weighted F1 的 0.59，说明类别不均衡对模型整体表现存在影响。

### 5.3 实验结论

QLoRA 微调后模型的整体分类准确率有所提升，说明模型能够通过少量可训练参数学习新闻主题分类任务。

但模型在部分少数类别上的识别能力仍然有限，生成式分类的输出解析和标签映射也会影响最终评估结果，因此当前结果更适合作为有限显存条件下大模型参数高效微调的阶段性实验。

## 6. 代码结构

```text
Qwen-QLoRA-TNEWS/
├── configs/
│   └── training.json
├── results/
│   ├── base_classification_report.txt
│   ├── evaluation_summary.json
│   └── qlora_classification_report.txt
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── repair_adapter.py
│   └── verify_adapter.py
├── .gitignore
├── requirements.txt
└── README.md
```

各模块职责：

| 文件 | 主要功能 |
| --- | --- |
| preprocess.py | 数据加载、指令构造及 Token 预处理 |
| train.py | 量化模型加载、LoRA 注入及训练 |
| inference.py | Adapter 加载与生成式分类 |
| evaluate.py | 批量推理及分类指标评估 |
| repair_adapter.py | Adapter 参数名称映射与修复 |
| verify_adapter.py | Adapter 权重一致性检查 |
| training.json | 保存微调参数配置 |
| results/ | 保存原始实验结果及分类报告 |

## 7. 环境配置与运行说明

### 7.1 环境依赖

主要依赖：
- Python
- PyTorch
- Transformers
- PEFT
- bitsandbytes
- Accelerate
- Datasets
- Scikit-learn
- Safetensors

安装依赖：

```bash
pip install -r requirements.txt
```

需要根据本地 GPU、CUDA 和操作系统选择兼容的 PyTorch 与 bitsandbytes 版本。

### 7.2 数据与模型

数据集来源：

https://huggingface.co/datasets/clue/clue

基础模型：

https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

本仓库不包含原始数据集、基础模型权重及训练后的 Adapter 权重，需要根据实际运行环境下载或配置。

### 7.3 查看命令帮助

```bash
python -m src.train --help

python -m src.evaluate --help
```

训练脚本支持小样本冒烟测试：

```bash
python -m src.train --output outputs/smoke --smoke
```

该模式仅使用 32 条训练样本、16 条验证样本及一个优化器更新步，用于检查训练流程，不代表完整训练复现。

完整训练命令：

```bash
python -m src.train --output outputs/qlora
```

该命令会启动完整训练，预计需要较长时间及兼容的 GPU 环境，不建议仅为查看项目而直接运行。

使用已有 Adapter 进行评估：

```bash
python -m src.evaluate --adapter /path/to/adapter
```

检查 Adapter 权重：

```bash
python -m src.verify_adapter --adapter /path/to/adapter
```

## 8. 公开版代码验证记录

2026-11-24，完成 Notebook 代码整理后的独立运行验证：

- 训练与评估命令入口检查通过。
- 4-bit Qwen 基础模型加载成功。
- 原实验 Adapter 加载成功。
- 224 个权重张量全部匹配，最大绝对误差为 0。
- 在 20 条验证样本上完成推理冒烟测试，Accuracy 为 70%。

需要区分：上述 20 条样本仅用于验证重构后代码的推理流程，不是项目的正式实验结果。正式 Accuracy 58.96% 来自原始完整实验中的 10,000 条验证样本。

本次代码整理未重新执行 53,360 条样本的完整微调，也未重新进行 10,000 条样本的完整评估。

## 9. 项目局限与后续改进

1. 类别分布不均衡，少数类别的分类效果仍有提升空间。
2. 生成式分类可能产生非标准输出，需要进一步完善输出约束和解析机制。
3. 当前实验尚未提供完整混淆矩阵，需要进一步分析类别间的混淆关系。
4. Adapter 权重加载与修复流程已经验证，但仍需在不同 Transformers、PEFT 版本及硬件环境下进行兼容性测试。
5. 后续可以尝试类别均衡采样、不同 LoRA 参数配置、目标模块消融及更稳定的分类输出约束。

## 10. 项目说明

本项目为个人大语言模型微调实验，重点在于 QLoRA 参数高效微调、低显存模型加载、指令数据构造、Adapter 权重排错及生成式分类评估。

项目展示的是实际完成的离线实验与代码整理结果，不代表已经完成生产级模型部署或线上服务验证。
