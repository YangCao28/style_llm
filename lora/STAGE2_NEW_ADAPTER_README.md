# Stage2 指令微调修复方案

## 🔍 问题诊断

你的模型"完全学习不到instruct"的根本原因：

### 当前问题
1. **只有一个 LoRA adapter**：Stage1 训练的 adapter 针对风格注入设计（连续文本补全）
2. **训练方式错误**：直接在 Stage1 adapter 上继续训练指令数据
3. **能力冲突**：风格补全（续写）vs 指令遵循（问答）使用相同的权重

### 为什么失效
- Stage1 学到的是：**给定文本 → 续写更多相似风格的文本**
- Stage2 需要的是：**给定指令 → 生成回复 → 停止**
- 在同一个 adapter 上训练会导致**灾难性遗忘**或**能力混淆**

## ✅ 解决方案

使用 **双 adapter 架构**：

```
Base Model (Qwen)
    ├── Adapter 1: style (Stage1训练，冻结) → 风格注入能力
    └── Adapter 2: instruct (Stage2训练) → 指令遵循能力
```

### 优势
1. **能力隔离**：两种不同的能力使用不同的参数
2. **无冲突**：Style adapter 冻结，保留风格能力；Instruct adapter 新训练，学习指令
3. **推理时叠加**：两个 adapter 同时激活，模型既有风格又听指令

## 🚀 使用方法

### 1. 使用新脚本重新训练 Stage2

```bash
python -m lora.stage2_with_new_adapter \
    --config lora/stage2_new_adapter_config.json
```

### 2. 测试新模型

修改测试脚本加载两个 adapters：

```python
# 加载 base model
model = AutoModelForCausalLM.from_pretrained(
    "stage2_instruct_new_adapter",  # 包含两个 adapters
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 两个 adapters 都会自动激活
```

### 3. 验证效果

运行测试：
```bash
python -m lora.test_stage2_instruction \
    --model_name_or_path stage2_instruct_new_adapter \
    --preset elegant_style
```

期望输出：
- ✅ 遵循指令格式（system/user/assistant）
- ✅ 保持雅致文学风格
- ✅ 生成完毕后停止（不继续对话）

## 📊 对比

### 旧方案（单 adapter）
```
Stage1: [Base Model] + [LoRA-style] → 风格续写
                           ↓ 继续训练（错误！）
Stage2: [Base Model] + [LoRA-style*] → 风格丢失或指令失效
```

### 新方案（双 adapter）
```
Stage1: [Base Model] + [LoRA-style] → 风格续写
                           ↓ 冻结
Stage2: [Base Model] + [LoRA-style(frozen)] + [LoRA-instruct] → 两种能力都有
```

## 🔧 关键配置

### stage2_new_adapter_config.json
```json
{
  "base_model_name": "Qwen/Qwen3-8B-Base",
  "stage1_adapter_path": "stage1_style_injection/checkpoint-531",
  "lora_r": 64,
  "lora_alpha": 128
}
```

- `lora_r`: 新 adapter 的秩（64 足够）
- `lora_alpha`: 缩放因子（128 = 2×r）

## 📝 技术细节

### Adapter 管理
```python
# 训练时
model.set_adapter("instruct")  # 只训练新的 instruct adapter
# style adapter 自动冻结但保持激活

# 推理时
model.eval()  # 两个 adapters 都激活
# 风格能力来自 style adapter
# 指令能力来自 instruct adapter
```

### Labels 分割
```python
# ✅ 正确：只对 assistant 回复计算 loss
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

# ❌ 错误：对整个文本计算 loss（旧版本的问题）
labels = full_ids
```

## 🎯 预期结果

训练后的模型应该：

1. **理解指令**：能区分 system/user/assistant 角色
2. **生成回复**：根据 user 提示生成内容
3. **保持风格**：回复内容具有雅致文学风格（来自 Stage1）
4. **正确停止**：生成 `<|im_end|>` 后停止，不继续对话

## 🐛 如果还不work

检查：
1. Stage1 checkpoint 是否正确（确认有风格能力）
2. 数据格式是否正确（conversations 格式）
3. Tokenizer 的 special tokens 配置
4. 训练 loss 是否下降

查看训练日志：
```bash
# 应该看到类似输出
[step 5] loss = 2.1234
[step 10] loss = 1.8765
...
```

如果 loss 不下降 → 检查数据和 labels
如果 loss 下降但推理不对 → 检查 adapter 加载和激活
