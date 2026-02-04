# LoRA 叠加机制详解

## 什么是 LoRA 叠加？

当模型包含多个 LoRA adapters 时（如 Stage2 的 `style` + `instruct`），推理时这些 adapters 会**自动叠加**（相加）应用到基座模型上。

## 🔑 关键概念：两个 Adapter 在不同文件夹

**重要**：Stage1 和 Stage2 的 adapters 是分开存储的：
- **Style adapter**（Stage1训练得到）：`stage1_style_injection/checkpoint-531/`
- **Instruct adapter**（Stage2训练得到）：`stage2_instruct_new_adapter/`

**为什么分开？**
- Stage1 训练后保存了 style adapter
- Stage2 训练时加载 Stage1 的 style adapter（冻结），添加新的 instruct adapter，训练后**只保存 instruct adapter**
- 因此测试 Stage2 时需要**分别加载两个 adapters**，让 PEFT 库自动叠加它们

**目录结构**：
```
项目目录/
├── stage1_style_injection/
│   └── checkpoint-531/
│       ├── adapter_config.json
│       └── adapter_model.safetensors  # Style adapter 权重
└── stage2_instruct_new_adapter/
    ├── adapter_config.json
    └── adapter_model.safetensors      # Instruct adapter 权重
```

## 数学原理

### 单个 LoRA
```
W_modified = W_base + ΔW
ΔW = B @ A  (rank-r 矩阵)
```

### 多个 LoRA 叠加
```
W_modified = W_base + ΔW_style + ΔW_instruct
           = W_base + (B_style @ A_style) + (B_instruct @ A_instruct)
```

**关键**：多个 LoRA 的效果是**相加的**，不是替换或覆盖。

## 在我们的项目中

### Stage1：单个 Adapter
```
Base Model (Qwen3-8B-Base)
    └── LoRA-style (风格注入)
```

输出 = Base + style

### Stage2：双 Adapter 叠加
```
Base Model (Qwen3-8B-Base)
    ├── LoRA-style (风格注入，冻结)
    └── LoRA-instruct (指令遵循，训练)
```

输出 = Base + style + instruct

## PEFT 库的默认行为

### 自动叠加
```python
# 加载包含多个 adapters 的模型
model = PeftModel.from_pretrained(base_model, "stage2_instruct_new_adapter")

# 推理时，所有 adapters 自动叠加
output = model.generate(...)  # 自动应用 style + instruct
```

### 手动控制（高级用法）

**启用特定 adapter：**
```python
# 只使用 style adapter
model.set_adapter("style")
output = model.generate(...)  # 只有风格，没有指令能力

# 只使用 instruct adapter
model.set_adapter("instruct")
output = model.generate(...)  # 只有指令，没有风格

# 启用所有 adapters（默认）
model.enable_adapters()
output = model.generate(...)  # style + instruct 叠加
```

**禁用所有 adapters：**
```python
model.disable_adapters()
output = model.generate(...)  # 纯基座模型，无任何 adapter
```

## 测试脚本的处理

### 参数设计

测试脚本使用简洁统一的参数设计，根据提供的参数**自动判断模式**：

| 参数组合 | 模式 | 说明 |
|---------|------|------|
| `--base_model` | 基座模式 | 测试纯基座模型 |
| `--lora_model` | 单adapter | 测试一个adapter（如Stage1） |
| `--style_adapter` + `--instruct_adapter` | 双adapter叠加 | 测试两个adapter叠加（Stage2） |

**自动base model检测优先级**：
1. 命令行参数 `--base_model`（最高优先级）
2. adapter的 `adapter_config.json` 中的 `base_model_name_or_path`
3. 当前目录下的 `Qwen3-8B-Base/`

### 当前实现
```python
# test_stage2_instruction.py
model = PeftModel.from_pretrained(base_model, lora_path)

# 检查 adapters
adapters = list(model.peft_config.keys())
print(f"Adapters: {adapters}")

# 如果有多个，提示用户
if len(adapters) > 1:
    print("All adapters will be stacked during inference")
```

### 输出示例
```
Mode: Dual LoRA (Stacked Adapters)

Style adapter:    stage1_style_injection/checkpoint-531
Instruct adapter: stage2_instruct_new_adapter
Base: Qwen3-8B-Base (from style adapter config)

✓ Loaded style adapter
✓ Loaded instruct adapter

🔗 Stacking adapters:
  ✓ style
  ✓ instruct

✓ All adapters will be stacked during inference
  Formula: W = W_base + ΔW_style + ΔW_instruct
```

**解释**：
- 同时指定 `--style_adapter` 和 `--instruct_adapter` 自动启用双adapter模式
- PEFT 库会分别加载两个adapter文件夹的权重并自动叠加
- 推理时效果：`W = W_base + ΔW_style + ΔW_instruct`
- `--base_model` 参数可选（优先级：命令行 > adapter_config.json > 当前目录）

## 训练时的叠加

### Stage2 训练配置
```python
# stage2_with_new_adapter.py

# 1. 加载 base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen3-8B-Base")

# 2. 加载 Stage1 的 style adapter
model = PeftModel.from_pretrained(base_model, "stage1_checkpoint", adapter_name="style")

# 3. 添加新的 instruct adapter
model.add_adapter("instruct", lora_config)

# 4. 冻结 style，只训练 instruct
model.set_adapter("instruct")
for name, param in model.named_parameters():
    if "style" in name:
        param.requires_grad = False
```

**训练时**：
- Forward: Base + style + instruct（style 冻结，但参与前向传播）
- Backward: 只更新 instruct 的参数

**推理时**：
- Forward: Base + style + instruct（两个 adapter 都生效）

## 验证叠加效果

### 方法1：对比测试
```bash
# 测试基座模型
python -m lora.test_stage2_instruction \
    --base_model Qwen3-8B-Base \
    > base_output.txt

# 测试 Stage1（只有 style）
python -m lora.test_stage2_instruction \
    --lora_model stage1_style_injection/checkpoint-531 \
    > stage1_output.txt

# 测试 Stage2（style + instruct 叠加）
python -m lora.test_stage2_instruction \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter \
    > stage2_output.txt

# 对比结果
diff base_output.txt stage1_output.txt
diff stage1_output.txt stage2_output.txt
```

**预期结果**：
- `base_output.txt`: 基础输出，无风格，不听指令（原始 Qwen3-8B-Base）
- `stage1_output.txt`: 有雅致风格，但不听指令（可能会续写而不是回答）
- `stage2_output.txt`: 有风格 + 听指令 + 正确停止（style 和 instruct 完美叠加）

### 方法2：检查参数
```python
# 检查模型包含的 adapters
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name)

# 输出：
# base_model.model.model.layers.0.self_attn.q_proj.lora_A.style
# base_model.model.model.layers.0.self_attn.q_proj.lora_B.style
# base_model.model.model.layers.0.self_attn.q_proj.lora_A.instruct
# base_model.model.model.layers.0.self_attn.q_proj.lora_B.instruct
```

## 常见问题

### Q1: Adapters 叠加会冲突吗？

**A**: 不会！因为：
1. 它们是**相加**关系，不是覆盖
2. Style 和 Instruct 学习的是**不同方面**的能力
3. 训练时 style 被冻结，保证了能力分离

### Q2: 能否选择性使用 adapter？

**A**: 可以！使用 `model.set_adapter("style")` 或 `model.set_adapter("instruct")`

### Q3: 叠加顺序重要吗？

**A**: 数学上不重要（加法交换律），但：
- 训练顺序重要：先 style 后 instruct
- 加载顺序最好保持一致

### Q4: 如何确认叠加生效？

**A**: 看输出效果：
- ✅ 既有雅致文学风格（来自 style）
- ✅ 又能遵循指令格式（来自 instruct）
- ✅ 生成后正确停止（来自 instruct）

## 技术细节

### Adapter 的存储结构

**实际情况**（两个adapter分开存储）：
```
项目目录/
├── stage1_style_injection/checkpoint-531/
│   ├── adapter_config.json
│   └── adapter_model.safetensors    # Style adapter 权重
└── stage2_instruct_new_adapter/
    ├── adapter_config.json
    └── adapter_model.safetensors    # Instruct adapter 权重
```

**加载时行为**：
- Stage2 测试时，分别从两个目录加载adapter
- PEFT 库在前向传播时自动叠加两个adapter的效果

### Forward 过程
```python
# 伪代码
def forward(x):
    # 基座模型
    h = base_linear(x)
    
    # 叠加所有启用的 adapters
    for adapter in active_adapters:
        lora_A = adapter.lora_A
        lora_B = adapter.lora_B
        h = h + lora_B(lora_A(x))
    
    return h
```

## 完整测试流程

### 1. 测试基座模型（baseline）
```bash
python -m lora.test_stage2_instruction \
    --base_model Qwen3-8B-Base \
    --preset elegant_style
```

### 2. 测试 Stage1（仅风格 adapter）
```bash
python -m lora.test_stage2_instruction \
    --lora_model stage1_style_injection/checkpoint-531 \
    --preset elegant_style
```

### 3. 测试 Stage2（风格 + 指令双 adapters 叠加）
```bash
# ⚠️ 正确方式：分别指定两个adapter的路径
python -m lora.test_stage2_instruction \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter \
    --preset elegant_style

# 可选：手动指定base model（覆盖自动检测）
python -m lora.test_stage2_instruction \
    --base_model /path/to/Qwen3-8B-Base \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter \
    --preset elegant_style
```

### 4. 对比分析
观察三个输出的区别：
- **Base**: 现代白话，不遵循改写指令
- **Stage1**: 雅致风格，但可能继续续写（不听指令）
- **Stage2**: 雅致风格 + 遵循指令 + 生成后停止 ✅（两个adapter完美叠加）

---

## 最佳实践

1. **训练时**：明确冻结不需要更新的 adapter
2. **推理时**：默认启用所有 adapters（自动叠加）
3. **调试时**：逐个测试每个 adapter 的效果
4. **部署时**：确保所有 adapters 都被正确加载

## 快速验证命令

### 一键测试所有模式
```bash
# 创建测试脚本
cat > test_all.sh << 'EOF'
#!/bin/bash

echo "=== 测试基座模型 ==="
python -m lora.test_stage2_instruction --base_model Qwen3-8B-Base --preset elegant_style

echo -e "\n\n=== 测试 Stage1 (style only) ==="
python -m lora.test_stage2_instruction --lora_model stage1_style_injection/checkpoint-531 --preset elegant_style

echo -e "\n\n=== 测试 Stage2 (style + instruct stacked) ==="
python -m lora.test_stage2_instruction \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter \
    --preset elegant_style
EOF

chmod +x test_all.sh
./test_all.sh
```

### Windows PowerShell 版本
```powershell
# 测试基座模型
Write-Host "=== 测试基座模型 ===" -ForegroundColor Cyan
python -m lora.test_stage2_instruction --base_model Qwen3-8B-Base --preset elegant_style

# 测试 Stage1
Write-Host "`n`n=== 测试 Stage1 (style only) ===" -ForegroundColor Cyan
python -m lora.test_stage2_instruction --lora_model stage1_style_injection/checkpoint-531 --preset elegant_style

# 测试 Stage2（双adapter叠加）
Write-Host "`n`n=== 测试 Stage2 (style + instruct stacked) ===" -ForegroundColor Cyan
python -m lora.test_stage2_instruction `
    --style_adapter stage1_style_injection/checkpoint-531 `
    --instruct_adapter stage2_instruct_new_adapter `
    --preset elegant_style
```

---

## 参考资料

- PEFT 文档: https://huggingface.co/docs/peft
- LoRA 论文: https://arxiv.org/abs/2106.09685
- Multi-Adapter 实践: https://github.com/huggingface/peft/tree/main/examples
- 本项目测试脚本: [test_stage2_instruction.py](test_stage2_instruction.py)
- 使用说明: [TEST_USAGE.md](TEST_USAGE.md)
