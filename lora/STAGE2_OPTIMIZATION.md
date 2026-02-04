# Stage2 训练优化说明

## 优化内容

### 1. 正确的模型加载流程

**之前的问题**：
- 脚本可能没有明确从纯净的base model开始加载
- 保存时会把style和instruct两个adapter混在一起

**优化后的流程**：
```python
# Step 1: 加载纯净的 base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-8B-Base")

# Step 2: 加载 Stage1 的 style adapter（FROZEN）
model = PeftModel.from_pretrained(
    base_model,
    "stage1_style_injection/checkpoint-531",
    adapter_name="style"
)

# Step 3: 添加新的 instruct adapter（TRAINABLE）
lora_config = LoraConfig(r=64, lora_alpha=128, ...)
model.add_adapter("instruct", lora_config)

# Step 4: 冻结 style adapter，只训练 instruct
model.set_adapter("instruct")
for name, param in model.named_parameters():
    if "style" in name:
        param.requires_grad = False
```

### 2. 独立保存 instruct adapter

**关键修改**：使用 `selected_adapters` 参数只保存 instruct adapter

```python
# ✅ 只保存 instruct adapter（不保存 style）
model.save_pretrained(
    args.output_dir,
    selected_adapters=["instruct"]  # 关键参数
)
```

**结果**：
- Style adapter 保持在：`stage1_style_injection/checkpoint-531/`
- Instruct adapter 独立保存在：`stage2_instruct_new_adapter/`

### 3. 推理时使用两个 adapter

训练完成后，推理时需要同时加载两个adapter：

```bash
python -m lora.test_stage2_instruction \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter
```

## 文件结构

训练后的目录结构：
```
lora/
├── stage1_style_injection/
│   └── checkpoint-531/          # Style adapter（风格注入）
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
└── stage2_instruct_new_adapter/  # Instruct adapter（指令遵循）
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── adapter_info.json         # 记录如何使用
```

## 训练命令

```bash
python -m lora.stage2_with_new_adapter --config lora/stage2_config.json
```

## 工作原理

### LoRA 堆叠机制

$$W_{final} = W_{base} + \Delta W_{style} + \Delta W_{instruct}$$

- **$W_{base}$**：基础模型权重（Qwen2.5-8B-Base）
- **$\Delta W_{style}$**：风格adapter（Stage1训练，推理时冻结但激活）
- **$\Delta W_{instruct}$**：指令adapter（Stage2训练，新增）

### 训练过程

1. **Forward pass**：两个adapter都激活，模型同时具有风格和指令能力
2. **Backward pass**：只有instruct adapter的梯度会更新
3. **保存**：只保存instruct adapter到独立文件夹

### 推理过程

```python
# 加载顺序
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-8B-Base")
model = PeftModel.from_pretrained(base_model, style_adapter_path)
model.load_adapter(instruct_adapter_path)

# 两个adapter同时激活
model.enable_adapters()
```

## 优化优势

1. ✅ **清晰的模块化**：风格和指令分离，方便管理
2. ✅ **灵活性**：可以单独更新instruct adapter，不影响style
3. ✅ **可组合性**：可以尝试不同的instruct adapter配合同一个style
4. ✅ **存储效率**：不重复保存style adapter

## 测试验证

### 测试 Base Model（基线）
```bash
python -m lora.test_stage2_instruction \
    --base_model Qwen/Qwen2.5-8B-Base
```
预期：生成不相关的内容（未训练）

### 测试 Stage1（只有风格）
```bash
python -m lora.test_stage2_instruction \
    --lora_model stage1_style_injection/checkpoint-531
```
预期：有文学风格但不遵循指令（Stage1实际测试结果）

### 测试 Stage2（风格 + 指令）
```bash
python -m lora.test_stage2_instruction \
    --style_adapter stage1_style_injection/checkpoint-531 \
    --instruct_adapter stage2_instruct_new_adapter
```
预期：既有风格又遵循指令（正确改写故事）

## 相关文档

- [LORA_STACKING.md](LORA_STACKING.md) - LoRA堆叠机制详解
- [STAGE2_FIX_README.md](STAGE2_FIX_README.md) - Stage2修复说明
