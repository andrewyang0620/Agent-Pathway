# B2B Sales Policy Copilot — 学习路径
## Week 1：环境搭建 + AI 开发基础认知

> **本周目标**：在正式写任何 AI 代码之前，先把地基打稳。
> 结束时你应该能做到：本地环境跑通、第一个 API 调用成功、理解 LLM 的基本工作边界。

---

## 目录

1. [整体项目路线图](#0-整体路线图先看一眼)
2. [开发环境搭建](#1-开发环境搭建)
3. [LLM 核心概念：你必须真正理解的部分](#2-llm-核心概念你必须真正理解的部分)
4. [第一个 API 调用](#3-第一个-api-调用)
5. [Python 工具链：AI 开发必用的几个库](#4-python-工具链ai-开发必用的几个库)
6. [本周项目骨架初始化](#5-本周项目骨架初始化)
7. [本周 Checklist](#6-本周-checklist)
8. [预告：Week 2 要做什么](#7-预告week-2-要做什么)

---

## 0. 整体路线图：先看一眼

```
Week 1  ── 环境 + 认知基础          ← 你在这里
Week 2  ── LLM + Prompt + Structured Output
Week 3  ── Embedding + Retrieval + 手写 RAG
Week 4  ── LangChain RAG pipeline + Citation
Week 5  ── PDF/DOCX/CSV ingestion + SQL Tool
Week 6  ── Agent + Tool Calling + Eval
          ↓
      最终项目：B2B Sales Policy & Operations Copilot
```

每一周都是下一周的地基，**不要跳**。

---

## 1. 开发环境搭建

### 1.1 Python 版本

推荐 Python **3.11**（稳定，主流 AI 库兼容性最好）。

```bash
# 检查版本
python --version

# 推荐用 pyenv 管理多版本（可选但好习惯）
pyenv install 3.11.9
pyenv local 3.11.9
```

### 1.2 虚拟环境

**每个项目独立虚拟环境，这是规矩。**

```bash
# 在你的项目目录下
python -m venv .venv

# 激活（macOS / Linux）
source .venv/bin/activate

# 激活（Windows）
.venv\Scripts\activate

# 确认激活成功
which python   # 应该指向 .venv 里的 python
```

### 1.3 核心依赖（Week 1 只装这些）

```bash
pip install openai anthropic python-dotenv ipykernel jupyter
```

| 库 | 用途 |
|---|---|
| `openai` | OpenAI API client，也兼容很多其他模型 |
| `anthropic` | Claude API client（Week 2+ 用） |
| `python-dotenv` | 读取 `.env` 文件里的 API key，**别把 key 硬编码** |
| `ipykernel` + `jupyter` | notebook 环境，调试 LLM 输出最方便 |

### 1.4 API Key 管理

```bash
# 在项目根目录创建 .env 文件
touch .env
```

```ini
# .env 文件内容
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

```bash
# .gitignore 里加上这一行，永远不要提交 key
echo ".env" >> .gitignore
```

> ⚠️ **关键习惯**：API key 泄漏是真实事故。`.env` + `.gitignore` 是最低限度的保护。

### 1.5 IDE 推荐

**VS Code** + 以下插件：
- `Python` (Microsoft)
- `Jupyter`
- `Pylance`
- `GitLens`（可选，但好用）

---

## 2. LLM 核心概念：你必须真正理解的部分

这一节不是让你背定义，是让你**建立正确的心智模型**，后面的一切都从这里生长出来。

---

### 2.1 Token：LLM 的基本计量单位

LLM 不读"字"，也不读"词"，它读的是 **token**。

```
"Hello, world!"  →  ["Hello", ",", " world", "!"]  → 4 tokens
"你好世界"        →  ["你", "好", "世", "界"]        → 4 tokens（中文 token 效率较低）
```

**为什么你要理解 token：**
- 计费按 token 算
- Context window（上下文窗口）的限制按 token 算
- 切分文档（chunking，Week 3 的核心）要按 token 切，不能按字符切

**动手感受一下：**
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 用的 tokenizer

text = "客户 A 这单最多能给多少折扣？"
tokens = enc.encode(text)
print(f"Token 数量: {len(tokens)}")
print(f"Token IDs: {tokens}")
```

---

### 2.2 Context Window：LLM 的短期记忆上限

**Context Window** = 模型在一次对话中能"看到"的最大 token 数量。

```
┌─────────────────────────────────────────────┐
│              Context Window                  │
│                                             │
│  System Prompt  │  History  │  Your Input   │
│  (policy docs)  │  (turns)  │  (question)   │
│                                             │
│  ← 超过这个范围的内容，模型完全看不到 →        │
└─────────────────────────────────────────────┘
```

| 模型 | Context Window |
|---|---|
| GPT-4o | 128k tokens ≈ 100k 中文字 |
| Claude 3.5 Sonnet | 200k tokens |
| GPT-3.5 Turbo | 16k tokens |

**对你的项目的含义**：
- pricing_policy.pdf 可能有 50 页，远超一次能塞进去的量
- 这就是为什么需要 **RAG**（Week 3）——先检索，再回答，而不是把整个文档塞进去

---

### 2.3 Temperature：控制输出的"随机性"

```python
# temperature = 0 → 确定性最高，每次输出几乎一样
# temperature = 1 → 随机性高，创意感强
# temperature = 0.7 → 常用默认值，平衡

response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,   # 对于判断类任务，用 0
    messages=[...]
)
```

**你的项目要用 temperature=0**。
原因：你的系统在判断"这笔订单是否违反毛利政策"，这不是创意写作，需要可复现的确定性结论。

---

### 2.4 消息格式：role 的含义

OpenAI / Anthropic 的 API 都用同一套消息结构：

```python
messages = [
    {
        "role": "system",    # 系统指令，定义 AI 的行为边界和角色
        "content": "你是一个 B2B 销售合规顾问。..."
    },
    {
        "role": "user",      # 用户输入
        "content": "客户 A 这单能给 12% 折扣吗？"
    },
    {
        "role": "assistant", # AI 的上一轮回答（多轮对话时需要传入）
        "content": "根据定价政策..."
    },
    {
        "role": "user",      # 用户继续提问
        "content": "那 10% 呢？"
    }
]
```

**关键认知**：LLM **没有内置记忆**。每次 API 调用都是无状态的。
多轮对话靠的是你把历史消息手动塞回 messages 数组。

---

### 2.5 LLM 能做什么 / 不能做什么

| 能做（善用） | 不能做（别指望） |
|---|---|
| 理解自然语言、归纳、推理 | 实时查数据库（需要 tool calling） |
| 按格式输出结构化结果 | 精确计算（会算错，用代码算） |
| 引用上下文中提供的信息 | 记住上一次对话（无状态） |
| 根据规则做判断 | 访问外部系统（需要 agent） |

> 这张表很重要。你的 Copilot 项目里，**LLM 负责理解和推理，数据库 / 文件系统负责存储和检索**。两者职责不混。

---

## 3. 第一个 API 调用

### 3.1 最简单的调用

新建 `week1_hello.py`：

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # 读取 .env 文件

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",  # 便宜，适合测试
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "你是一个 B2B 销售合规顾问，专门帮助销售团队判断报价是否符合公司政策。"
        },
        {
            "role": "user",
            "content": "客户问能不能给 15% 的折扣，我们公司最高折扣是 10%，我应该怎么回复？"
        }
    ]
)

print(response.choices[0].message.content)
```

运行：
```bash
python week1_hello.py
```

---

### 3.2 理解 Response 结构

```python
# 打印完整的 response 对象，看清楚结构
import json
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
```

关键字段：
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "..."        // 模型的回答
    },
    "finish_reason": "stop"   // stop = 正常结束 / length = 被截断
  }],
  "usage": {
    "prompt_tokens": 87,      // 你输入用了多少 token
    "completion_tokens": 156, // 模型输出用了多少 token
    "total_tokens": 243       // 总计，计费依据
  }
}
```

---

### 3.3 封装一个可复用的调用函数

这是你整个项目会一直用到的基础函数，现在就养成好习惯：

```python
# utils/llm.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(
    user_message: str,
    system_prompt: str = "你是一个有帮助的助手。",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    """
    最基础的 LLM 调用封装。
    返回 assistant 的文字回复。
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content


# 测试
if __name__ == "__main__":
    result = chat(
        user_message="客户要求 12% 折扣，标准上限是 10%，这单能批吗？",
        system_prompt="你是 B2B 销售合规顾问，用简洁的中文回答。"
    )
    print(result)
```

---

### 3.4 用 Jupyter Notebook 调试（推荐）

```bash
jupyter notebook
```

在 notebook 里调用 LLM 有一个巨大优势：**可以反复修改 prompt，立刻看到效果，不用每次重跑整个脚本**。

Week 2 开始，prompt engineering 的所有实验都建议在 notebook 里做。

---

## 4. Python 工具链：AI 开发必用的几个库

### 4.1 Pydantic：结构化数据验证

Week 2 的 structured output 会大量用到。先装，先熟悉概念。

```bash
pip install pydantic
```

```python
from pydantic import BaseModel
from typing import Optional

class DiscountDecision(BaseModel):
    approved: bool
    max_discount_pct: float
    reason: str
    policy_reference: Optional[str] = None

# 这个模型定义了你的 AI 输出应该长什么样
# Week 2 会详细讲怎么让 LLM 输出符合这个格式的 JSON
```

---

### 4.2 Rich：让终端输出好看（调试神器）

```bash
pip install rich
```

```python
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown

# 直接 print 任何 Python 对象都会有颜色和格式
print({"approved": True, "max_discount": 0.08})

# 打印 LLM 的 markdown 回答时特别有用
md = Markdown("## 结论\n建议折扣不超过 **8%**")
print(Panel(md, title="AI 回答", border_style="blue"))
```

---

### 4.3 Loguru：比 print 更好的日志

```bash
pip install loguru
```

```python
from loguru import logger

logger.info("开始查询客户 A 的折扣上限")
logger.warning("当前毛利率接近下限，建议谨慎")
logger.error("无法连接数据库")

# 日志自动带时间戳和级别，调试 agent 流程时救命
```

---

### 4.4 依赖文件管理

每次安装新库后，更新 requirements.txt：

```bash
pip freeze > requirements.txt
```

或者用更现代的方式（推荐）：

```bash
pip install uv  # 极速包管理工具，越来越主流

uv pip freeze > requirements.txt
```

---

## 5. 本周项目骨架初始化

现在搭好整个项目的目录结构，后面每一周都往里填内容。

```
b2b-sales-copilot/
│
├── .env                    # API keys（不提交）
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   ├── policies/           # Week 5: PDF/DOCX 政策文档
│   ├── contracts/          # Week 5: 供应商合同
│   └── tables/             # Week 5: CSV 订单/库存/客户数据
│
├── utils/
│   ├── __init__.py
│   └── llm.py              # 今天写的基础调用封装
│
├── notebooks/
│   ├── week1_basics.ipynb  # 本周实验
│   └── week2_prompts.ipynb # 下周用
│
├── week1/
│   └── hello.py
│
└── tests/                  # Week 6: eval set
```

初始化命令：

```bash
mkdir -p b2b-sales-copilot/{data/{policies,contracts,tables},utils,notebooks,week1,tests}
cd b2b-sales-copilot
touch utils/__init__.py utils/llm.py
git init
echo ".env" >> .gitignore
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

---

## 6. 本周 Checklist

完成以下所有项，才算 Week 1 真正结束：

### 环境
- [ ] Python 3.11 安装并激活
- [ ] 虚拟环境创建并激活
- [ ] `openai`, `anthropic`, `python-dotenv`, `pydantic`, `rich`, `loguru` 全部安装成功
- [ ] `.env` 文件配置好 API key
- [ ] `.gitignore` 包含 `.env` 和 `.venv/`

### 概念理解（能用自己的话解释）
- [ ] 什么是 Token，为什么比字符数更重要
- [ ] Context Window 的上限对你的项目意味着什么
- [ ] 为什么 temperature 在你的项目里应该接近 0
- [ ] LLM 的无状态性：为什么多轮对话要手动传 history

### 代码
- [ ] `week1/hello.py` 运行成功，有实际输出
- [ ] `utils/llm.py` 封装完成，`chat()` 函数可调用
- [ ] Jupyter Notebook 可以正常打开，在里面调用 API 成功
- [ ] 项目目录结构按上面搭好

### 加分项（不强制，但推荐）
- [ ] 用 tiktoken 数一个你自己写的句子有多少 token
- [ ] 对比 temperature=0 和 temperature=1 对同一个问题的不同输出
- [ ] 用 Rich 的 Panel 打印一个格式化的 AI 回答

---

## 7. 预告：Week 2 要做什么

下周进入正题：**LLM 基础概念 + Prompt Engineering + Structured Output**。

你会学到：

**Prompt Engineering（直接影响项目质量）**
- Zero-shot / Few-shot / Chain-of-Thought 的区别和选择时机
- 如何写 system prompt 让模型"扮演"合规顾问
- 如何用 prompt 引导模型给出有引用的结构化结论

**Structured Output（项目的骨架）**
- 让 LLM 输出 JSON 而不是散文
- 用 Pydantic + OpenAI Function Calling 做输出验证
- 你最终的 `DiscountDecision`、`PolicyViolation` 等数据结构会在这里定型

**本周写的 `utils/llm.py` 下周会继续扩展。**

---

> 💡 **学习建议**：每一周的 notebook 里要有真实的实验记录——不只是能跑通的代码，还有你改过的 prompt、观察到的模型行为差异、失败的尝试。
> 这些记录本身就是你项目最有价值的一部分，面试时讲"我发现 temperature 高了之后判断结果开始飘"远比"我实现了 RAG"有说服力。

---

*Week 1 / 6 完成后继续 →* **[Week 2：LLM 基础 + Prompt Engineering + Structured Output]**
