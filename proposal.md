# 能够进行“白话文到武侠风”改写的基座模型微调技术报告
**——基于 Decoder-only 架构的配对学习与掩码策略深度解析**

## 摘要
在自然语言处理（NLP）领域，文本风格迁移（Text Style Transfer, TST）一直是一个极具挑战性的前沿课题。特别是将现代白话文转换为具有高度文学性、半文半白且富含特定文化意象的“武侠风”文本，不仅要求模型具备强大的语义理解能力，更要求其掌握深层的文体映射逻辑。本研究报告旨在深入探讨如何利用 **Decoder-only** 架构的大语言模型（LLM），通过 **监督微调（Supervised Fine-Tuning, SFT）** 和先进的 **掩码（Masking）技术**，实现这一复杂的风格重写任务。本报告将从 Decoder-only 模型的配对学习机制、基于翻译任务 Prompt 的微调方案、以及实现“掩码预测”的高级技巧三个维度展开详尽的论述，并结合 Qwen2.5、DeepSeek-V3 等前沿开源模型的特性，提供一套完整的技术落地路线图。

---

## 1. 理论基础：Decoder-only 架构在序列配对任务中的学习机制

虽然历史上序列到序列（Seq2Seq）的任务（如翻译、风格迁移）多由 Encoder-Decoder 架构（如 Transformer-Big, BART, T5）主导，但随着 GPT 系列的崛起，Decoder-only 架构已证明其在处理此类任务上的卓越能力。理解 Decoder-only 模型如何学习“白话 $\leftrightarrow$ 武侠”的配对关系，是构建高效微调策略的前提。

### 1.1 因果注意力与条件概率建模

Decoder-only 模型的核心在于 **因果语言建模（Causal Language Modeling, CLM）**。其本质是基于前文预测下一个 token。在处理“白话文输入”与“武侠风输出”这一配对（Pair）关系时，模型并非像 Encoder-Decoder 那样将输入和输出视为两个独立的编码空间，而是将其拼接为一个连续的序列。

**序列拼接与注意力流：**

假设白话文输入序列为 $X = \{x_1, x_2,..., x_m\}$，目标武侠风输出序列为 $Y = \{y_1, y_2,..., y_n\}$。在微调阶段，我们将它们通过特定的分隔符（Separator）或模版（Prompt Template）拼接成单一序列 $S$：

$$S = \{ x_1, x_2, \dots, x_m, \texttt{<SEP>}, y_1, y_2, \dots, y_n, \texttt{<EOS>} \}$$

其中 $\texttt{<SEP>}$ 代表分隔符（如 ChatML 中的 `<|im_start|>assistant`），$\texttt{<EOS>}$ 代表结束符。

在模型内部，**自注意力机制（Self-Attention）配合因果掩码（Causal Mask）** 起到了关键作用。因果掩码是一个上三角矩阵（Upper Triangular Matrix），其作用是确保位置 $t$ 的 token 只能“看到”位置 $j \leq t$ 的信息。

1.  **编码阶段（Context Encoding）**：当模型处理序列的前半部分（即白话文 $X$）时，尽管它也在计算 $P(x_t | x_{<t})$，但这一阶段的主要功能是构建 Key-Value Cache (KV Cache)。这些 KV 对实际上是对白话文语义、句法结构的稠密向量表示。
2.  **解码阶段（Conditional Generation）**：当模型开始生成序列的后半部分（即武侠文 $Y$）时，因果掩码允许 $y_t$ 对 $X$ 中的所有 token 以及 $Y$ 中已生成的 $y_{<t}$ 进行注意力计算。

**数学上的配对学习：**

Decoder-only 模型学习 Pair 关系的本质，是最大化条件概率 $P(Y|X)$。虽然形式上是联合概率 $P(S)$ 的一部分，但通过对输入部分 $X$ 进行 Loss Masking（损失掩码）（详见第3章），我们强制模型专注于优化以下目标：

$$\mathcal{L}_{\text{SFT}} = - \sum_{i=1}^{n} \log P(y_i | X, y_{<i}; \theta)$$

在此公式中，模型学习到的不再是简单的“下一个词”，而是一种映射函数 $f: \mathcal{S}_{\text{modern}} \rightarrow \mathcal{S}_{\text{wuxia}}$。模型内部的注意力头（Attention Heads）会逐渐分化，特定的头会专门负责“回看”输入中的语义锚点（如“生气”），并激活前馈神经网络（FFN）中存储的武侠词汇（如“怒发冲冠”、“无名火起”）。这种机制使得 Decoder-only 模型能够极其出色地掌握跨文体的映射逻辑。

### 1.2 语义空间与文体空间的解耦与重构

在深度学习的潜空间（Latent Space）中，风格迁移可以被视为将语义向量从一个分布域投影到另一个分布域的过程。

*   **语义不变性（Content Preservation）**：白话文“他拿起杯子喝了一口酒”包含的核心实体（人、动作、物体）必须保留。Decoder-only 模型通过深层网络的上下文理解能力，将这些实体编码为高维语义向量。
*   **风格特异性（Style Specificity）**：武侠风格的实现依赖于特定的 **词汇选择（Lexical Choice）** 和 **句法重组（Syntactic Reordering）**。例如，上述句子在武侠风中可能变为“他端起酒碗，仰头一饮而尽，豪气干云”。模型通过 SFT 学习到的权重更新，实际上是在调整输出层的概率分布，使其在面对特定语义向量时，倾向于采样属于“武侠子空间”的 token。

Qwen2.5 和 DeepSeek-V3 等模型由于在海量中文语料上进行了预训练，其参数空间中已经包含了“白话”和“武侠”两个子流形（Sub-manifold）。微调的过程，本质上是建立连接这两个子流形的“虫洞”，或者说是一条低阻力的推理路径。

---

## 2. 监督微调（SFT）实战：加入翻译 Prompt 的微调方案

为了让 Decoder-only 模型明确执行“改写”任务，而非仅仅是续写或对话，我们需要通过精心设计的 Prompt Engineering 将其转化为一个 **指令跟随（Instruction Following）** 任务。

### 2.1 翻译任务 Prompt 的设计原则

在微调数据的构建中，Prompt 不仅是输入的容器，更是任务定义的锚点。我们将“白话文-武侠”Pair 包装成一种 **翻译（Translation）或重写（Rewriting）** 的指令格式。

**推荐的数据结构（ChatML格式）：**

目前主流的 Decoder-only 模型（如 Qwen, Llama-3, DeepSeek）都经过了指令微调，对结构化的对话格式（System, User, Assistant）非常敏感。

*   **System Prompt（系统提示词）**：定义模型的角色和风格约束。这是确立“武侠风”基调的关键。
*   **User Prompt（用户提示词）**：承载白话文输入。
*   **Assistant Response（模型回答）**：承载武侠文输出。

**Prompt 模版示例：**

| 角色 (Role) | 内容 (Content) | 设计意图 |
| :--- | :--- | :--- |
| **System** | 你是一位通晓古今的武侠小说大师，继承了金庸、古龙的笔法。你的任务是将现代白话文改写为地道的武侠风格。要求：用词古雅，多用四字成语，描写需体现江湖气息，注重动作的凌厉与意境的苍凉。 | **风格注入**：明确界定输出的分布空间，激活模型内部的武侠相关参数。 |
| **User** | 把这句话改成武侠风：他非常生气，用力拍了一下桌子，桌子就碎了。 | **语义输入**：提供待转换的 Source Content。 |
| **Assistant** | 他只觉胸中一股无名火起，怒喝一声，右掌运劲重重拍落。“咔嚓”一声巨响，那张硬木方桌竟被他这一掌震得四分五裂，木屑纷飞。 | **目标输出**：Ground Truth，用于计算损失。 |

### 2.2 数据集构建与增强策略

高质量的 Pair 数据是微调成功的核心。由于现成的“白话-武侠”平行语料库极其稀缺，我们需要构建数据飞轮。

**逆向合成法（Back-Translation Synthesis）：**

1.  **步骤一：收集武侠语料。** 利用 GitHub 上的金庸、古龙全集语料库，进行分句和清洗。这些是天然的高质量 Target（$Y$）。
2.  **步骤二：利用强模型生成白话文。** 使用 GPT-4o、DeepSeek-V3 或 Qwen-72B-Instruct 作为“教师模型”，输入武侠原句，要求其“翻译成现代大白话，去除所有修辞”。这将生成对应的 Source（$X$）。
3.  **步骤三：构建 Pair。** 将生成的白话文（$X$）与原始武侠文（$Y$）配对，形成 $\{X, Y\}$ 数据集。这种方法的优势在于 Target 端（武侠风）是人类大师的原作，保证了风格的纯正。

**风格分类器过滤：**

*   训练一个基于 BERT 的二分类器（Ancient/Wuxia vs. Modern/Vernacular）。
*   对合成的 Pair 进行校验，确保 $X$ 被分类为现代文，$Y$ 被分类为武侠文，剔除风格区分不明显的样本。

### 2.3 微调参数配置建议

针对 Qwen2.5 或 DeepSeek 系列模型进行 SFT 时，以下参数配置对风格迁移任务尤为重要：

*   **Learning Rate (学习率)**：建议设置较小，如 `1e-5` 至 `5e-5`。风格迁移需要精细调整，过大的学习率可能破坏模型的通用语言能力，导致输出逻辑混乱。
*   **Epochs**：3-5 个 Epoch 通常足够。过拟合会导致模型只会背诵训练集中的具体句子，而丧失泛化改写能力。
*   **Max Length**：武侠描写通常包含大段的环境渲染和心理活动，建议上下文长度设为 2048 或 4096。
*   **Packing**：启用 `packing=True`，将多个短样本拼接成一个长序列进行训练，极大提高训练效率。

---

## 3. 核心技巧：Decoder-only 实现“掩码预测”与 Loss Masking

用户在提问中提到了“masking来微调”，这是一个非常专业且关键的点。在 Decoder-only 模型的微调中，“Masking”主要有两种截然不同的含义和应用场景：

1.  **Loss Masking (Instruction Masking)**：在 SFT 训练中，屏蔽掉输入部分（Prompt）的损失计算，只对输出部分（Response）计算梯度。这是 SFT 的标准操作。
2.  **Fill-In-the-Middle (FIM) / Span Corruption**：一种类似于 BERT“掩码预测”的预训练目标，允许模型根据上下文填充中间缺失的片段。这对于风格润色和局部改写极其有效。

### 3.1 Loss Masking：让模型“只学输出，不学输入”

在标准的 Causal Language Modeling (CLM) 中，模型会对序列中的每一个 token 计算损失。如果我们直接训练 `<System + User + Assistant>` 的拼接序列，模型会同时学习如何生成 System Prompt 和 User Prompt。

这在风格迁移中是极度有害的。如果模型学习了 User 部分的分布（即白话文），它可能会混淆源风格和目标风格。我们需要模型做的是：看到白话文（Source），生成武侠文（Target）。

**实现技巧：**

在构建 Data Collator 时，我们需要对 `labels` 张量进行处理：

*   Input IDs: `[System, User, Assistant]`
*   Labels: `[-100, -100, Assistant]`

将 Assistant 回复之前的所有 token 的 Label 设置为 `-100`（PyTorch 中 CrossEntropyLoss 的默认 ignore_index）。这样，反向传播时，只有 Assistant 部分的预测误差会产生梯度更新。

**代码逻辑示意（基于 Transformers）：**

```python
def mask_user_labels(input_ids, tokenizer):
    labels = input_ids.clone()
    # 找到分隔符的位置，例如 <|im_start|>assistant
    sep_token_id = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
    
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        # 寻找分割点
        sep_idx = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_idx) > 0:
            start_gen_idx = sep_idx[-1] + 1 # 从 assistant 标签后开始计算 loss
            labels[i, :start_gen_idx] = -100 # 掩盖前面的所有 token
    return labels
```

通过这种 Loss Masking，模型被强制建立从“未掩盖的输入上下文”到“目标输出”的单向映射，这是学习 Pair 关系最高效的方式。

### 3.2 Fill-In-the-Middle (FIM)：实现真正的“掩码填空”

用户提到的“掩码预测”可能还指向一种更高级的用法——**FIM（Fill-In-the-Middle）**。虽然 Decoder-only 本质是单向的，但通过 FIM 变换，我们可以让模型具备类似 BERT 的“完形填空”能力。

**原理：**

FIM 通过将文档重新排列为 `Prefix - Suffix - Middle` 的形式，让模型先看到前文（Prefix）和后文（Suffix），然后生成中间缺失的内容（Middle）。

$$\text{Input: } \texttt{<|fim\_prefix|>} \ X_{\text{pre}} \ \texttt{<|fim\_suffix|>} \ X_{\text{suf}} \ \texttt{<|fim\_middle|>}$$

$$\text{Target: } X_{\text{mid}}$$

**在武侠改写中的应用：**

FIM 非常适合局部风格润色。例如，你可能有一段文字，开头和结尾的武侠味很浓，但中间一句太白话。

*   **原始**：“李寻欢咳嗽了两声，[拿出一个小瓶子喝了一口]，苍白的脸上泛起红晕。”
*   **FIM 任务**：将中间部分 Mask 掉，要求模型填充。
*   **模型生成**：“[从怀中摸出那个贴身的酒囊，仰首灌入喉中]”

**实施方案：**

1.  在微调数据中，可以按一定比例（如 20%）混入 FIM 格式的数据。
2.  选取高质量的纯武侠语料（如金庸小说段落）。
3.  随机 Mask 掉其中的形容词短语、动作描写或成语。
4.  构造 PSM（Prefix-Suffix-Middle）格式的样本进行训练。

这能增强模型对武侠 **行文气韵（Coherence）** 的掌握，使其生成的文字在上下文衔接上更加自然。

---

## 4. 进阶：利用激活导向（Activation Steering）增强风格

除了微调权重，还可以通过 **激活导向（Activation Steering）** 在推理阶段干预模型行为。这是一种无需重新训练即可调整模型风格的轻量级技术。

**原理：**

在模型的 Transformer 层中，存在代表“武侠风格”的激活方向（Vector）。我们可以通过对比“武侠输入”和“白话输入”的激活差异来提取这个向量。

$$V_{\text{style}} = \mathbb{E}[Act(Wuxia)] - \mathbb{E}[Act(Vernacular)]$$

**操作：**

在推理时，将这个向量 $V_{\text{style}}$ 乘以一个系数 $\alpha$，叠加到模型的隐藏层状态（Hidden States）上。

$$h_l' = h_l + \alpha \cdot V_{\text{style}}$$

这可以作为一种“外挂”，在 SFT 的基础上进一步强化武侠风味，或者允许用户通过调整 $\alpha$ 来控制“含侠量”（风格强度）。对于 Qwen 等模型，可以在第 10-20 层之间注入此向量效果最佳。

---

## 5. 总结与推荐方案

针对您“微调基座模型以完成白话文到武侠风改写”的需求，本报告的结论性建议如下：

1.  **模型选择**：首选 **Qwen2.5-7B-Instruct** 或 **Qwen2.5-14B-Instruct**。它们中文能力极强，且对 FIM 和指令微调支持完善。
2.  **数据策略**：构建“逆向合成”数据集。利用强模型将金庸、古龙小说“翻译”回白话文，以此构建 $\{白话 \rightarrow 武侠\}$ 的高质量 Pair。
3.  **微调方案**：采用 **LoRA SFT**。
    *   使用标准的 ChatML Prompt 结构。
    *   必须实施 **Loss Masking**，将 System 和 User 部分的 Label 设为 -100，强迫模型专注于学习从白话 Context 生成武侠 Token 的概率分布。
4.  **增强技巧**：引入 **FIM（Fill-In-the-Middle）** 任务。在训练数据中混合 20% 的纯武侠文本完形填空任务，使用 `<|fim_prefix|>` 等特殊 token，提升模型对武侠语感的连贯性掌握。

通过这一套组合拳，Decoder-only 模型将不仅仅是在做翻译，而是在深层语义空间建立起了通往“江湖”的桥梁，实现神形兼备的风格重塑。

---

## 6. 数据与表格支撑

### 表1：推荐的基座模型与特性对比

| 模型名称 | 参数量 | 优势 (针对武侠改写) | 适用场景 | FIM 支持 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | 7B | 中文语料占比极高，不仅懂成语，更懂典故；128k 上下文适合长篇改写。 | 消费级显卡 (24G VRAM) 微调与推理 | 原生支持 |
| **DeepSeek-V3** | MoE | 极强的推理与文学创作能力，成本极低；API 价格优势明显。 | 大规模生产环境，或作为教师模型生成数据 | 支持 |
| **Yi-34B-Chat** | 34B | 200k 上下文，预训练数据包含大量高质量中文书籍，文学感强。 | 需要极高质量输出，且算力充足 | 需自定义 |

### 表2：微调数据格式与 Masking 策略对照

| 数据部分 | 内容示例 (缩略) | Masking 处理 (Labels) | 目的 |
| :--- | :--- | :--- | :--- |
| **System** | `<|im_start|>system\n你是金庸风格...<|im_end|>\n` | **-100** (Masked) | 设定风格基调 |
| **User** | `<|im_start|>user\n改写：他很生气。<|im_end|>\n` | **-100** (Masked) | 提供语义输入 |
| **Assistant** | `<|im_start|>assistant\n他怒发冲冠...<|im_end|>\n` | **原样保留** (Computed) | 计算 Loss，优化生成 |

### 表3：常用武侠风格评测指标

| 指标类型 | 具体指标 | 说明 | 工具/方法 |
| :--- | :--- | :--- | :--- |
| **语义保留** | BERTScore / SIM | 确保改写后的武侠文没有偏离原意。 | 现代文 Embedding 模型计算余弦相似度。 |
| **风格强度** | Style Classifier Accuracy | 判断输出是否属于“武侠”类。 | 训练一个二分类器 (Bert-Base)，区分金庸语料与现代语料。 |
| **流畅度** | Perplexity (PPL) | 衡量文本生成的自然程度。 | 使用在武侠语料上预训练的小模型 (如 GPT-2) 计算 PPL。 |

---

## 引用的著作与资源

1.  **Mask Your User Tokens** - Yoni Gottesman [链接](https://yonigottesman.github.io/2024/05/13/mask-user-tokens.html)
2.  **StableMask: Refining Causal Masking in Decoder-only Transformer** [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/yin24a/yin24a.pdf)
3.  **The Best Open Source LLMs for Mandarin Chinese in 2026** - SiliconFlow
4.  **CAT-LLM: Prompting Large Language Models with Text Style Definition** [arXiv](https://arxiv.org/html/2401.05707v1)
5.  **huayi-dou/The-speaker-identification-corpus-of-Jin-Yong-novels** [GitHub](https://github.com/huayi-dou/The-speaker-identification-corpus-of-Jin-Yong-novels)
6.  **SFT Trainer - Hugging Face** [文档](https://huggingface.co/docs/trl/main/en/sft_trainer)
7.  **Fill-in-the-Middle (FIM) in Language Models** [Emergent Mind](https://www.emergentmind.com/topics/fill-in-the-middle-fim)
8.  **Activation Steering in LLMs** [Emergent Mind](https://www.emergentmind.com/topics/activation-steering-method)
