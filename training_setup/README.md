# LLaMA Factory 训练配置指南

本目录包含使用 LLaMA Factory 微调模型的配置文件。我们直接使用 `data/wuxia_vernacular_pairs.jsonl` 中的配对文件进行训练。

## 1. 准备工作

确保你已经安装了 LLaMA Factory。如果还没安装，请参考以下步骤：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e "."
```

## 2. 数据配置

我们已经在 `data/dataset_info.json` 中配置了数据映射：
- 输入 (`prompt`): `vernacular_text` (白话文)
- 输出 (`response`): `wuxia_text` (武侠文)

## 3. 启动训练

在项目根目录下 (`c:\Users\caoya\source\repos\data-pre`) 运行以下命令：

```powershell
llamafactory-cli train training_setup/wuxia_sft.yaml
```

**注意：**
- 配置文件 `training_setup/wuxia_sft.yaml` 中使用的是 `Qwen/Qwen2.5-7B-Instruct` 模型。如果你的显存较小 (如 < 16GB)，可能需要调整 `finetuning_type` 为 `lora` 并启用 `quantization_bit: 4` (如果支持)。
- `per_device_train_batch_size` 默认为 2，显存不足可设为 1。
- `dataset_dir` 设为 `data`，确保 `dataset_info.json` 在该目录下。

## 4. 导出模型

训练完成后，可以使用以下命令导出模型或进行推理：

```powershell
llamafactory-cli chat training_setup/wuxia_sft.yaml
```
(注意：推理时可能需要修改 yaml 中的 `do_train` 为 `false`)
