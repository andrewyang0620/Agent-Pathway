# Week 2：LLM 基础 + Prompt Engineering + Structured Output

> **本周目标**：掌握让 LLM "听话"的核心技术。
> 结束时你应该能做到：写出稳定产出结构化判断的 prompt，并用 Pydantic 验证 LLM 输出，为整个 Copilot 项目奠定输入/输出的数据契约。

---

## 目录

1. [Prompt Engineering：不是玄学，是工程](#1-prompt-engineering不是玄学是工程)
2. [System Prompt 设计：给模型一个角色和边界](#2-system-prompt-设计给模型一个角色和边界)
3. [Zero-shot / Few-shot / Chain-of-Thought](#3-zero-shot--few-shot--chain-of-thought)
4. [Structured Output：让 LLM 输出 JSON](#4-structured-output让-llm-输出-json)
5. [Pydantic + Function Calling：工程级输出验证](#5-pydantic--function-calling工程级输出验证)
6. [本周项目任务：DiscountDecision Demo](#6-本周项目任务discountdecision-demo)
7. [常见 Prompt 失败模式](#7-常见-prompt-失败模式)
8. [本周 Checklist](#8-本周-checklist)
9. [预告：Week 3](#9-预告week-3)

---

## 1. Prompt Engineering：不是玄学，是工程

很多人把 prompt engineering 理解成"换个措辞、多试几次"。这是错的。

正确理解：**Prompt 是你和 LLM 之间的接口规范。**

就像写函数签名一样，你要定义：
- 输入的格式和含义
- 输出的格式和约束
- 处理逻辑的边界条件

---

### 1.1 Prompt 的基本结构

一个完整的工程级 prompt 通常包含以下部分：

```
[角色定义]      → 告诉模型它是谁
[任务说明]      → 告诉模型要做什么
[约束和规则]    → 告诉模型不能做什么
[输出格式]      → 告诉模型怎么输出
[上下文/数据]   → 提供模型做判断需要的信息
[具体问题]      → 当前这次的实际请求
```

**不好的 prompt：**
```
判断这单能不能给折扣
```

**好的 prompt：**
```
你是 Acme 公司的销售合规顾问。

任务：判断一笔报价是否符合公司定价政策，并给出折扣建议。

规则：
- 标准品类折扣上限：10%
- 战略客户（年采购额 > 100 万）可申请特批，上限 15%
- 折扣后毛利率不得低于 25%

你必须基于以下信息做判断，不得凭空推断：
[订单信息]
客户：客户 A
年采购额：85 万
当前报价毛利率：30%
请求折扣：12%

输出 JSON 格式，字段：approved (bool), max_discount_pct (float), reason (str), risk_level (str)
```

差距不在聪明，在严谨。

---

### 1.2 Prompt 的核心原则

**原则一：具体优于模糊**

```python
# 模糊
"分析这个订单"

# 具体
"判断该订单在给予 12% 折扣后，毛利率是否仍高于公司最低要求 25%。
 如果不满足，计算最大允许折扣率。"
```

**原则二：格式要求放在最后**

LLM 在生成时是从左到右的，格式要求放最后能让它在内容生成完后才"收尾"，
放最前反而容易被遗忘。

```python
# 不好
"输出 JSON。判断客户 A 的折扣申请是否合规。"

# 好
"判断客户 A 的折扣申请是否合规。最后以 JSON 格式输出结论。"
```

**原则三：给模型"退路"**

如果信息不足，告诉模型该怎么处理，否则它会乱猜：

```python
"如果提供的信息不足以做出判断，在 reason 字段中说明缺少哪些信息，
 将 approved 设为 false，risk_level 设为 'insufficient_data'。"
```

**原则四：用分隔符区分数据和指令**

```python
system_prompt = "你是销售合规顾问..."

user_message = """
请判断以下订单：

<order_info>
客户：客户 A
请求折扣：12%
当前毛利：30%
</order_info>

<policy>
标准折扣上限：10%
最低毛利要求：25%
</policy>
"""
```

用 XML tag 或 `---` 分隔，防止 LLM 把数据和指令混淆。

---

## 2. System Prompt 设计：给模型一个角色和边界

### 2.1 System Prompt 的作用

System prompt 是整个对话的"宪法"，它定义模型的：
- 身份（它是谁）
- 能力边界（它能做什么）
- 行为约束（它不该做什么）
- 输出标准（它怎么回答）

### 2.2 为你的项目写第一版 System Prompt

新建 `week2/system_prompts.py`：

```python
COPILOT_SYSTEM_PROMPT = """
你是 Acme 公司的 B2B 销售合规与运营顾问，代号 Copilot。

## 你的职责
- 判断销售报价是否符合公司定价政策
- 识别订单中的毛利风险
- 根据合同条款回答采购和付款相关问题
- 分析库存与销售数据，提供补货建议

## 你的判断原则
1. 所有判断必须基于提供的政策文件或数据，不得凭空推断
2. 如果信息不足，明确说明缺少什么，而不是猜测
3. 输出结论时必须注明依据来源（哪条政策 / 哪个数据）
4. 对于高风险判断，必须标记并说明不确定性

## 你不会做的事
- 在没有政策依据的情况下批准超标折扣
- 在数据缺失时给出确定性结论
- 忽略风险因素只给出用户想听的答案

## 输出语言
使用中文回答，专业术语可保留英文缩写（如 SKU、SLA、KPI）。
"""
```

### 2.3 System Prompt 的迭代方式

System prompt 不是写一次就完的，它应该随着你发现模型的问题而更新：

```
发现问题 → 分析是 prompt 问题还是模型能力问题 → 修改 prompt → 对比前后输出 → 记录
```

在 notebook 里做这件事，**保留每一个版本和对应的输出**，这就是你的"prompt 版本历史"。

---

## 3. Zero-shot / Few-shot / Chain-of-Thought

### 3.1 Zero-shot：直接给任务

```python
messages = [
    {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
    {"role": "user", "content": "客户 A 申请 12% 折扣，标准上限 10%，能批吗？"}
]
```

**适用场景**：任务描述清晰、模型能力足够时。
**你的项目**：Week 6 之前大部分场景都用 zero-shot，够了。

---

### 3.2 Few-shot：给例子

当模型的输出格式或推理方式不符合预期时，直接给它看例子。

```python
FEW_SHOT_EXAMPLES = """
## 示例判断

### 示例 1
输入：
- 客户：客户 X（普通客户，年采购 60 万）
- 请求折扣：8%
- 折扣后毛利：28%

输出：
{
  "approved": true,
  "max_discount_pct": 8.0,
  "reason": "折扣在标准上限 10% 以内，折扣后毛利 28% 高于最低要求 25%。",
  "risk_level": "low"
}

### 示例 2
输入：
- 客户：客户 Y（普通客户，年采购 40 万）
- 请求折扣：13%
- 折扣后毛利：22%

输出：
{
  "approved": false,
  "max_discount_pct": 9.5,
  "reason": "请求折扣 13% 超过标准上限 10%，且折扣后毛利 22% 低于最低要求 25%。最大可批折扣为 9.5%（毛利恰好达到 25% 下限）。",
  "risk_level": "high"
}
"""
```

把这段加到 user message 里，或者作为 system prompt 的一部分。

**核心价值**：few-shot 不只是"让模型看例子"，而是**校准模型的推理路径**。
当你发现模型经常漏掉毛利计算这一步，加一个涉及毛利的 few-shot 例子，比修改文字描述有效得多。

---

### 3.3 Chain-of-Thought（CoT）：让模型"先想再答"

**问题**：直接让模型输出 JSON，它有时候会跳过推理步骤，结论错误但格式正确。

**解决方案**：让模型先用自然语言推理，再输出结论。

```python
COT_PROMPT = """
请按以下步骤判断：

步骤 1：确认客户类型（普通 / 战略）
步骤 2：查找对应的折扣上限
步骤 3：计算折扣后的毛利率
步骤 4：对比最低毛利要求
步骤 5：给出最终判断和最大允许折扣

完成以上分析后，最后输出 JSON 格式的结论。
```

**CoT 的本质**：你在强迫模型把中间步骤显式输出，而不是在"脑内"跳过。
这对于复杂判断任务的准确率提升非常显著（20-40%）。

**你的项目里什么时候用 CoT**：
- 涉及多步计算时（毛利率计算、折扣叠加）
- 需要引用多条政策时
- Week 5+ 涉及 SQL 结果 + 政策文档联合判断时

---

### 3.4 三种方式的选择矩阵

| 场景 | 推荐方式 |
|---|---|
| 简单是/否判断，信息完整 | Zero-shot |
| 模型输出格式不稳定 | Few-shot |
| 多步推理，容易出错 | Chain-of-Thought |
| 复杂业务规则，且要稳定输出 | Few-shot + CoT |

---

## 4. Structured Output：让 LLM 输出 JSON

### 4.1 为什么必须要结构化输出

你的 Copilot 最终要：
- 把结论显示在 UI 上（需要字段分离）
- 把引用来源单独列出
- 把风险等级传给下游系统

如果模型输出的是一段话，你得用正则去解析，这是灾难。

目标：**LLM 的输出从一开始就是可以直接用 `json.loads()` 解析的字符串。**

---

### 4.2 方法一：Prompt 强制（最简单，但不稳定）

```python
user_message = """
...（订单信息）...

请只输出 JSON，不要有任何其他文字，格式如下：
{
  "approved": true 或 false,
  "max_discount_pct": 数字,
  "reason": "字符串",
  "risk_level": "low" 或 "medium" 或 "high"
}
"""
```

问题：模型有时候会在 JSON 前后加 ```json ``` 标记，或者加一句"以下是判断结果："。

**加一个解析保险：**

```python
import json
import re

def extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，容错处理"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取第一个 { ... } 块
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法从以下文本中提取 JSON:\n{text}")
```

---

### 4.3 方法二：OpenAI JSON Mode（推荐）

```python
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    response_format={"type": "json_object"},  # 关键行
    messages=[
        {"role": "system", "content": "你是合规顾问。所有回答以 JSON 格式输出。"},
        {"role": "user", "content": "..."}
    ]
)

result = json.loads(response.choices[0].message.content)
```

**注意**：开启 JSON Mode 时，system prompt 里必须提到 "JSON"，否则 API 会报错。

---

### 4.4 方法三：OpenAI Structured Outputs（最严格，推荐用于生产）

这是 2024 年推出的功能，**保证输出严格符合你定义的 schema**，不会有多余字段，不会有类型错误。

```python
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

client = OpenAI()

class DiscountDecision(BaseModel):
    approved: bool
    max_discount_pct: float
    reason: str
    risk_level: Literal["low", "medium", "high", "insufficient_data"]
    policy_references: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",         # 注意：需要 gpt-4o 及以上
    temperature=0,
    messages=[
        {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
        {"role": "user", "content": "..."}
    ],
    response_format=DiscountDecision,  # 直接传 Pydantic 类
)

decision = response.choices[0].message.parsed  # 直接是 DiscountDecision 对象
print(decision.approved)           # True / False
print(decision.max_discount_pct)   # 8.0
print(decision.risk_level)         # "low"
```

**这才是工程级的正确做法**：输出直接是你的 Python 对象，类型安全，无需解析。

---

## 5. Pydantic + Function Calling：工程级输出验证

### 5.1 为你的项目定义数据契约

新建 `week2/schemas.py`，这个文件会一直用到 Week 6：

```python
# week2/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

# ── 折扣判断 ──────────────────────────────────────────
class DiscountDecision(BaseModel):
    """销售折扣合规判断结果"""
    approved: bool = Field(description="是否批准该折扣申请")
    max_discount_pct: float = Field(
        ge=0, le=100,
        description="最大允许折扣百分比，例如 8.0 表示 8%"
    )
    reason: str = Field(description="判断依据，必须引用具体政策或数据")
    risk_level: Literal["low", "medium", "high", "insufficient_data"]
    policy_references: list[str] = Field(
        default=[],
        description="引用的政策条款，例如 ['pricing_policy §3.2']"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="判断置信度，0-1 之间"
    )

# ── 政策违规检测 ──────────────────────────────────────
class PolicyViolation(BaseModel):
    """订单政策违规记录"""
    order_id: str
    violation_type: Literal[
        "discount_exceeded",
        "margin_below_threshold",
        "unauthorized_bundle",
        "contract_term_mismatch"
    ]
    severity: Literal["warning", "critical"]
    description: str
    recommended_action: str

# ── 批量检查结果 ──────────────────────────────────────
class BatchCheckResult(BaseModel):
    """批量订单检查结果"""
    total_orders: int
    violations: list[PolicyViolation]
    clean_orders: int
    summary: str

# ── 供应商合同查询结果 ────────────────────────────────
class ContractQueryResult(BaseModel):
    """供应商合同条款查询结果"""
    vendor_name: str
    payment_terms: Optional[str] = None
    delivery_sla_days: Optional[int] = None
    key_clauses: list[str] = Field(default=[])
    source_document: str
    page_reference: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
```

**为什么现在就定义这些 schema？**

因为这些数据结构是你整个项目的骨架。Week 3 的 RAG 输出要填进这些字段，Week 5 的 SQL 查询结果也要和这些字段对齐。先定义契约，后面的组件才有接口可以对接。

---

### 5.2 完整的结构化调用封装

扩展 `utils/llm.py`：

```python
# utils/llm.py（更新版）
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Type, TypeVar
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

T = TypeVar("T", bound=BaseModel)

def chat(
    user_message: str,
    system_prompt: str = "你是一个有帮助的助手。",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    """基础文字对话，返回字符串"""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content


def structured_chat(
    user_message: str,
    response_schema: Type[T],
    system_prompt: str = "你是一个有帮助的助手。所有回答以 JSON 格式输出。",
    model: str = "gpt-4o",   # structured output 需要 gpt-4o
    temperature: float = 0,
) -> T:
    """
    结构化输出调用，返回 Pydantic 对象。
    保证输出严格符合 response_schema 定义的格式。
    """
    response = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=response_schema,
    )
    return response.choices[0].message.parsed
```

---

### 5.3 验证它能工作

```python
# week2/test_structured.py
import sys
sys.path.append("..")

from utils.llm import structured_chat
from week2.schemas import DiscountDecision
from week2.system_prompts import COPILOT_SYSTEM_PROMPT

ORDER_INFO = """
<order_info>
客户：科技股份有限公司（普通客户，年采购额 75 万元）
请求折扣：12%
产品品类：标准硬件
当前报价毛利率：29%
折扣后预计毛利率：22%
</order_info>

<policy>
- 标准品类折扣上限：10%
- 战略客户（年采购 > 100 万）可申请特批上限：15%
- 折扣后毛利率不得低于：25%
</policy>

请判断该折扣申请是否合规，并给出最大允许折扣。
"""

decision = structured_chat(
    user_message=ORDER_INFO,
    response_schema=DiscountDecision,
    system_prompt=COPILOT_SYSTEM_PROMPT,
)

print(f"批准：{decision.approved}")
print(f"最大折扣：{decision.max_discount_pct}%")
print(f"风险等级：{decision.risk_level}")
print(f"置信度：{decision.confidence}")
print(f"理由：{decision.reason}")
print(f"政策引用：{decision.policy_references}")
```

预期输出大致是：
```
批准：False
最大折扣：7.5%
风险等级：high
置信度：0.95
理由：请求折扣 12% 超过标准品类上限 10%；即便在 10% 折扣下，毛利率仍低于 25% 最低要求...
政策引用：['pricing_policy §3.1', 'pricing_policy §2.3']
```

---

## 6. 本周项目任务：DiscountDecision Demo

把以上所有内容组装成一个完整的命令行 demo：

新建 `week2/discount_demo.py`：

```python
"""
Week 2 Demo：折扣合规判断器
输入：自然语言订单描述
输出：结构化判断结果
"""
import sys
sys.path.append("..")

from utils.llm import structured_chat, chat
from week2.schemas import DiscountDecision
from week2.system_prompts import COPILOT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

POLICY_CONTEXT = """
<policy_rules>
- 标准品类（硬件/软件）折扣上限：10%
- 服务类折扣上限：15%  
- 战略客户（年采购 > 100 万）特批上限：15%（需要销售总监签字）
- 折扣后毛利率不得低于：25%
- 单笔订单超过 50 万且折扣 > 8%：需要区域经理审批
</policy_rules>
"""

def analyze_discount_request(order_description: str) -> DiscountDecision:
    """分析折扣申请是否合规"""
    
    user_message = f"""
{FEW_SHOT_EXAMPLES}

{POLICY_CONTEXT}

<current_request>
{order_description}
</current_request>

请按以下步骤分析（Chain-of-Thought）：
1. 确认客户类型和对应折扣上限
2. 计算折扣后毛利率
3. 检查是否需要额外审批
4. 给出最终判断和最大允许折扣

完成分析后输出结构化结论。
"""
    
    return structured_chat(
        user_message=user_message,
        response_schema=DiscountDecision,
        system_prompt=COPILOT_SYSTEM_PROMPT,
    )


def display_decision(decision: DiscountDecision, order_desc: str):
    """用 Rich 格式化显示判断结果"""
    
    # 颜色映射
    risk_color = {"low": "green", "medium": "yellow", "high": "red", "insufficient_data": "grey50"}
    status_text = "✅ 批准" if decision.approved else "❌ 拒绝"
    color = risk_color.get(decision.risk_level, "white")
    
    # 主结论
    console.print(Panel(
        f"[bold]{status_text}[/bold]\n"
        f"最大允许折扣：[cyan]{decision.max_discount_pct}%[/cyan]\n"
        f"风险等级：[{color}]{decision.risk_level.upper()}[/{color}]\n"
        f"置信度：{decision.confidence * 100:.0f}%",
        title="📊 合规判断结果",
        border_style=color
    ))
    
    # 判断依据
    console.print(Panel(
        decision.reason,
        title="📋 判断依据",
        border_style="blue"
    ))
    
    # 政策引用
    if decision.policy_references:
        console.print("\n[bold]📎 引用来源：[/bold]")
        for ref in decision.policy_references:
            console.print(f"  • {ref}")


def main():
    # 测试案例
    test_cases = [
        {
            "name": "案例 1：普通客户超标折扣",
            "order": "客户：华东制造有限公司（年采购额 60 万，普通客户）\n产品：标准硬件设备\n请求折扣：12%\n当前毛利率：30%"
        },
        {
            "name": "案例 2：战略客户特批申请",
            "order": "客户：上海集成电路（年采购额 150 万，战略客户）\n产品：定制软件服务\n请求折扣：13%\n当前毛利率：35%"
        },
        {
            "name": "案例 3：毛利临界情况",
            "order": "客户：北京贸易公司（年采购额 80 万，普通客户）\n产品：标准软件许可\n请求折扣：8%\n当前毛利率：27%"
        },
    ]
    
    for case in test_cases:
        console.rule(f"[bold]{case['name']}[/bold]")
        console.print(f"\n[dim]订单信息：\n{case['order']}[/dim]\n")
        
        decision = analyze_discount_request(case["order"])
        display_decision(decision, case["order"])
        console.print("\n")


if __name__ == "__main__":
    main()
```

运行：
```bash
python week2/discount_demo.py
```

---

## 7. 常见 Prompt 失败模式

这是你在调试时最容易踩的坑，提前记住：

### 失败模式 1：过于依赖模型"理解"

```python
# 错误：指望模型自己知道"高风险"的定义
"判断这单是否高风险"

# 正确：显式定义
"高风险定义：折扣后毛利率 < 25%，或请求折扣超过标准上限 50% 以上。请按此定义判断。"
```

### 失败模式 2：在 JSON 模式下让模型写很长的 reason

模型在 JSON mode 下，长字符串里容易出现未转义的引号，导致解析失败。

```python
# 给 reason 加长度提示
"reason 字段限制在 100 字以内，简洁表达核心依据。"
```

### 失败模式 3：没有处理模型"编造"政策的情况

```python
# 在 system prompt 里加
"你只能引用用户提供的政策信息。如果政策文件中没有相关规定，
明确说明'未找到相关政策依据'，不得自行制定规则。"
```

### 失败模式 4：Few-shot 例子和实际格式不一致

如果你的 few-shot 例子里 `risk_level` 是 `"HIGH"`, 但 schema 要求的是 `"high"`，
模型会跟着例子走，然后 Pydantic 验证失败。**例子的格式必须和 schema 完全一致。**

### 失败模式 5：温度设置不当

```python
# 对于合规判断：temperature=0（确定性）
# 对于生成摘要/总结：temperature=0.3（稍微灵活）
# 永远不要用 temperature>0.5 做规则判断
```

---

## 8. 本周 Checklist

### 概念
- [ ] 能解释 Zero-shot / Few-shot / CoT 的区别和各自适用场景
- [ ] 知道为什么要用 XML tag 分隔数据和指令
- [ ] 理解为什么 temperature=0 对判断类任务更好
- [ ] 能说出 JSON Mode 和 Structured Outputs 的区别

### 代码
- [ ] `week2/system_prompts.py` 包含 `COPILOT_SYSTEM_PROMPT` 和 `FEW_SHOT_EXAMPLES`
- [ ] `week2/schemas.py` 包含 `DiscountDecision`、`PolicyViolation`、`ContractQueryResult`
- [ ] `utils/llm.py` 新增 `structured_chat()` 函数
- [ ] `week2/discount_demo.py` 可以运行，三个测试案例都有输出
- [ ] `extract_json()` 工具函数写好（兜底用）

### 实验记录（Notebook）
- [ ] 在 `notebooks/week2_prompts.ipynb` 里对比了至少两个不同 prompt 版本的输出
- [ ] 记录了一个 CoT 前后准确率差异的对比实验
- [ ] 记录了一个 few-shot 修复输出不稳定问题的案例

### 加分项
- [ ] 给 `DiscountDecision` 加一个 `needs_approval_from` 字段，表示需要哪级主管审批
- [ ] 写一个 `batch_check()` 函数，输入多个订单，返回 `BatchCheckResult`
- [ ] 测试模型在信息不足时能否正确输出 `insufficient_data` 而不是瞎猜

---

## 9. 预告：Week 3

下周进入 RAG 的核心：**Embedding + Retrieval + 手写 Minimal RAG**。

你会学到：

**Embedding 是什么**
- 文本 → 向量，语义相似 = 向量相近
- 为什么这比关键词搜索更适合政策文档查询

**Chunking 策略**
- 不能把整个 PDF 塞进一个 chunk
- 按什么切？切多长？chunk 之间要不要 overlap？
- 这些决策直接影响你的 RAG 质量

**手写 Minimal RAG（不用任何框架）**
- 用 `numpy` 做向量相似度计算
- 用字典模拟向量数据库
- 理解 retrieve → augment → generate 三步的数据流

**本周的 `schemas.py` 下周会用到**：RAG 检索结果会被填进 `ContractQueryResult`。

---

*Week 2 / 6 完成后继续 →* **[Week 3：Embedding + Retrieval + 手写 Minimal RAG]**
