# Week 3：Embedding + Chunking + Retrieval + 手写 Minimal RAG

> **本周目标**：从零理解 RAG 的底层机制，不借助任何框架，手写完整的检索-增强-生成流程。
> 结束时你应该能做到：把一份 policy 文档切成 chunks，向量化，存入内存索引，用余弦相似度检索，把结果喂给 LLM 生成带引用的答案。

---

## 目录

1. [RAG 是什么，为什么需要它](#1-rag-是什么为什么需要它)
2. [Embedding：把文本变成向量](#2-embedding把文本变成向量)
3. [Chunking：切文档的学问](#3-chunking切文档的学问)
4. [相似度检索：余弦相似度](#4-相似度检索余弦相似度)
5. [手写 Minimal RAG：完整流程](#5-手写-minimal-rag完整流程)
6. [本周 Mock 数据准备](#6-本周-mock-数据准备)
7. [本周项目任务：Policy RAG Demo](#7-本周项目任务policy-rag-demo)
8. [RAG 质量的常见问题](#8-rag-质量的常见问题)
9. [本周 Checklist](#9-本周-checklist)
10. [预告：Week 4](#10-预告week-4)

---

## 1. RAG 是什么，为什么需要它

### 1.1 问题从哪里来

上周你写的 `DiscountDecision` demo，政策规则是直接硬编码在 prompt 里的：

```python
POLICY_CONTEXT = """
- 标准品类折扣上限：10%
- 战略客户特批上限：15%
- 折扣后毛利率不得低于：25%
"""
```

这在真实项目里完全不可行，原因有三：

**原因一：文档太长，塞不进去。**
真实的 `pricing_policy.pdf` 可能有 40 页，`margin_rulebook.docx` 可能有 20 页。
GPT-4o 的 context window 是 128k token，但：
- 把所有文档全塞进去，每次调用成本极高
- 检索质量随 context 长度下降（"lost in the middle" 问题）
- 文档会更新，硬编码维护成本无法接受

**原因二：你不知道哪段话是相关的。**
用户问"客户 A 能给多少折扣"，相关的可能只是 pricing_policy 第 3.2 节，不是整个文档。

**原因三：需要引用来源。**
你的 Copilot 要求输出 `policy_references`，这意味着系统必须知道答案来自哪里。

### 1.2 RAG 的核心思路

```
传统方式：
用户问题 → [整个文档塞进 prompt] → LLM → 答案

RAG 方式：
用户问题 → [语义检索，只找相关段落] → LLM → 带引用的答案
               ↑
           这一步是本周的核心
```

**RAG 三步：**

```
1. Retrieve  → 从文档库里找到最相关的几段话
2. Augment   → 把这几段话加进 prompt
3. Generate  → LLM 基于这几段话生成答案
```

看起来简单，但每一步都有很多工程决策。本周逐一拆开。

---

## 2. Embedding：把文本变成向量

### 2.1 直觉理解

Embedding 就是把一段文字映射到高维空间里的一个点（向量）。

语义相近的文字，在这个空间里距离也近：

```
"折扣上限是多少？"         → [0.12, -0.45, 0.87, ...]   ←─┐ 距离很近
"最高可以打几折？"         → [0.14, -0.43, 0.85, ...]   ←─┘
"供应商的付款周期是多久？" → [-0.67, 0.23, -0.12, ...]  ←── 距离很远
```

这就是为什么 embedding 比关键词搜索更适合做文档检索：
关键词搜索找不到"最高可以打几折"和"折扣上限"之间的语义关联。

### 2.2 调用 OpenAI Embedding API

```python
# week3/embeddings.py
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    把一段文本转成 embedding 向量。
    text-embedding-3-small：1536 维，便宜，够用。
    text-embedding-3-large：3072 维，更准，贵 5 倍。
    """
    text = text.replace("\n", " ")  # 换行符会影响质量，统一替换
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    批量 embedding，比循环调用便宜很多。
    OpenAI 支持一次传入多个文本。
    """
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    # 按原始顺序返回
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ── 快速验证 ─────────────────────────────────────────
if __name__ == "__main__":
    texts = [
        "标准品类折扣上限为 10%",
        "最高折扣不得超过一成",
        "供应商付款周期为 net 30",
    ]
    
    embeddings = get_embeddings_batch(texts)
    
    print(f"向量维度: {len(embeddings[0])}")  # 1536
    
    # 验证：前两句语义相近，第三句应该和前两句差很多
    from numpy.linalg import norm
    
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (norm(a) * norm(b)))
    
    print(f"句1 vs 句2（应该接近 1）: {cosine_sim(embeddings[0], embeddings[1]):.4f}")
    print(f"句1 vs 句3（应该较小）:   {cosine_sim(embeddings[0], embeddings[2]):.4f}")
```

预期输出大概是：
```
向量维度: 1536
句1 vs 句2（应该接近 1）: 0.8923
句1 vs 句3（应该较小）:   0.3241
```

### 2.3 Embedding 的关键认知

**一次 embedding，永久复用。**
文档的 embedding 只需要算一次，存下来。
查询的 embedding 每次实时计算。
不要每次查询都重新 embed 整个文档库。

**embedding 模型要一致。**
文档用 `text-embedding-3-small` 算的，查询也必须用 `text-embedding-3-small`。
混用不同模型的向量做相似度没有意义。

**向量维度 ≠ 质量上限。**
`text-embedding-3-small` 的 1536 维对你的项目完全够用。
不要因为"大的更好"就无脑用 3072 维，五倍的成本不值得。

---

## 3. Chunking：切文档的学问

### 3.1 为什么不能把整篇文档当一个 chunk

假设你把整个 `pricing_policy.pdf` 的内容拼成一个字符串，算出一个 embedding。

问题：这个向量是整篇文档的"平均语义"，它和任何具体问题的相似度都会很平庸。
你需要的是精确匹配：用户问折扣政策，返回的是折扣章节，不是整篇文档。

**chunk 要足够小**，才能有精确的语义定位。
**chunk 也不能太小**，否则上下文不完整，LLM 看不懂。

### 3.2 三种 Chunking 策略

**策略一：固定大小切割（Fixed-size Chunking）**

```python
def chunk_by_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    按 token 数量固定切割，带 overlap（重叠）。
    overlap 的作用：防止一句话正好被切断，相邻 chunk 共享一段上下文。
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    tokens = enc.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap  # overlap：下一个 chunk 从这里开始
    
    return chunks
```

**优点**：简单，可控，适合没有明显结构的文档。
**缺点**：可能在句子中间切断，语义不完整。

---

**策略二：按分隔符切割（Separator-based Chunking）**

```python
def chunk_by_separator(text: str, separators: list[str] = ["\n\n", "\n", "。"]) -> list[str]:
    """
    按自然分隔符切割，优先按段落，其次按句子。
    适合有明显段落结构的政策文档。
    """
    chunks = [text]
    
    for sep in separators:
        new_chunks = []
        for chunk in chunks:
            parts = chunk.split(sep)
            new_chunks.extend([p.strip() for p in parts if p.strip()])
        chunks = new_chunks
    
    # 过滤太短的 chunk（可能是标题或空行）
    return [c for c in chunks if len(c) > 50]
```

**优点**：语义边界更自然，适合政策/合同类文档。
**缺点**：段落长度不均匀，有些 chunk 会很长。

---

**策略三：递归切割（Recursive Chunking）—— 推荐**

```python
def chunk_recursive(
    text: str,
    chunk_size: int = 500,      # 目标 token 数
    overlap: int = 50,
    separators: list[str] = ["\n\n", "\n", "。", "；", " "]
) -> list[str]:
    """
    递归切割：先按最大分隔符切，如果还是太长，换下一级分隔符继续切。
    这是 LangChain RecursiveCharacterTextSplitter 的核心逻辑。
    本周手写它，Week 4 直接用 LangChain 的实现。
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    def token_len(s: str) -> int:
        return len(enc.encode(s))
    
    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or token_len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        sep = seps[0]
        parts = text.split(sep)
        
        results = []
        current = ""
        
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            
            if token_len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                # 如果单个 part 还是太长，用下一级分隔符继续切
                if token_len(part) > chunk_size:
                    results.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part.strip()
        
        if current:
            results.append(current)
        
        return results
    
    raw_chunks = _split(text, separators)
    
    # 加入 overlap：每个 chunk 后面附上下一个 chunk 的前几个 token
    if overlap == 0:
        return raw_chunks
    
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i < len(raw_chunks) - 1:
            next_chunk = raw_chunks[i + 1]
            next_tokens = enc.encode(next_chunk)[:overlap]
            overlap_text = enc.decode(next_tokens)
            result.append(chunk + " " + overlap_text)
        else:
            result.append(chunk)
    
    return result
```

**这是你项目里实际用的策略**。Week 4 换成 LangChain 时，底层逻辑完全一样，只是你不用自己维护这段代码了。

### 3.3 Chunk 大小怎么选

| 场景 | 推荐 chunk size | 理由 |
|------|----------------|------|
| 政策文档（条款密集） | 300–500 tokens | 每条条款独立，不要混在一起 |
| 合同文档（段落较长） | 500–800 tokens | 保留完整的法律表述 |
| FAQ 类文档 | 200–300 tokens | 一问一答通常很短 |
| 技术文档 | 400–600 tokens | 平衡上下文完整性 |

**你的项目：用 400 tokens，overlap 40 tokens。**

---

## 4. 相似度检索：余弦相似度

### 4.1 余弦相似度原理

两个向量的余弦相似度 = 它们夹角的余弦值。

```
相似度 = 1.0   → 方向完全一致，语义极度相近
相似度 = 0.0   → 垂直，语义不相关
相似度 = -1.0  → 方向相反（在 embedding 里极少出现）
```

公式：

```
cos(a, b) = (a · b) / (|a| × |b|)
```

代码：

```python
import numpy as np

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### 4.2 批量检索：从向量库找 Top-K

```python
def retrieve_top_k(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[str],
    k: int = 3,
    threshold: float = 0.3,    # 低于这个分数的 chunk 不返回，过滤噪声
) -> list[dict]:
    """
    给定一个 query 向量，从 chunk 库里找最相关的 k 个。
    返回按相似度降序排列的结果，附带原始文本和分数。
    """
    query = np.array(query_embedding)
    corpus = np.array(chunk_embeddings)   # shape: (n_chunks, embedding_dim)
    
    # 矩阵运算一次性算出所有相似度，比 for 循环快很多
    norms = np.linalg.norm(corpus, axis=1) * np.linalg.norm(query)
    scores = corpus @ query / norms       # shape: (n_chunks,)
    
    # 取 Top-K
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= threshold:
            results.append({
                "chunk": chunks[idx],
                "score": score,
                "index": int(idx),
            })
    
    return results
```

### 4.3 为什么不用欧氏距离

欧氏距离（L2 distance）衡量的是向量的绝对距离，受向量长度影响。

余弦相似度衡量的是方向，不受向量长度影响。

对于 embedding 向量，方向才是语义，长度没有意义。
所以 embedding 检索**永远用余弦相似度**，不用欧氏距离。

---

## 5. 手写 Minimal RAG：完整流程

现在把所有零件组装起来。新建 `week3/minimal_rag.py`：

### 5.1 内存向量库

```python
# week3/vector_store.py
"""
手写的极简内存向量库。
不用 FAISS，不用 ChromaDB，用字典 + numpy。
目的：理解向量库的核心数据结构。
Week 4 换成真正的向量库后，你会明白它们替你做了什么。
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
import os


@dataclass
class Chunk:
    """向量库里存储的最小单元"""
    id: str                          # 唯一标识，例如 "pricing_policy_§3.2_chunk_0"
    text: str                        # 原始文本
    embedding: list[float]           # 向量
    metadata: dict = field(default_factory=dict)  # 来源、页码、章节等


class InMemoryVectorStore:
    """
    内存向量库：存、检索、持久化。
    这就是 FAISS / ChromaDB 最核心的功能，stripped down。
    """
    
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._embeddings_matrix: Optional[np.ndarray] = None  # 缓存矩阵，加速检索
    
    def add(self, chunk: Chunk):
        """添加一个 chunk"""
        self.chunks.append(chunk)
        self._embeddings_matrix = None  # 清除缓存，等下次检索时重建
    
    def add_batch(self, chunks: list[Chunk]):
        """批量添加"""
        self.chunks.extend(chunks)
        self._embeddings_matrix = None
    
    def _build_matrix(self):
        """把所有 embedding 建成矩阵，用于向量化相似度计算"""
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.array([c.embedding for c in self.chunks])
    
    def search(self, query_embedding: list[float], k: int = 3, threshold: float = 0.3) -> list[dict]:
        """
        语义检索：返回 Top-K 最相关 chunk
        """
        if not self.chunks:
            return []
        
        self._build_matrix()
        
        query = np.array(query_embedding)
        
        # 批量余弦相似度
        norms = np.linalg.norm(self._embeddings_matrix, axis=1) * np.linalg.norm(query)
        scores = self._embeddings_matrix @ query / norms
        
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_idx:
            score = float(scores[idx])
            if score >= threshold:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "score": round(score, 4),
                    "metadata": chunk.metadata,
                })
        
        return results
    
    def save(self, path: str):
        """持久化到 JSON 文件（生产环境用真正的向量数据库，这里只是为了调试方便）"""
        data = [
            {
                "id": c.id,
                "text": c.text,
                "embedding": c.embedding,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"向量库已保存：{len(self.chunks)} 个 chunk → {path}")
    
    def load(self, path: str):
        """从文件恢复"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [
            Chunk(id=d["id"], text=d["text"], embedding=d["embedding"], metadata=d["metadata"])
            for d in data
        ]
        self._embeddings_matrix = None
        print(f"向量库已加载：{len(self.chunks)} 个 chunk ← {path}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __repr__(self):
        return f"InMemoryVectorStore({len(self.chunks)} chunks)"
```

### 5.2 文档索引 Pipeline

```python
# week3/indexer.py
"""
把文档 → chunks → embeddings → 向量库
这是 RAG 的"离线"阶段，只需要跑一次（或文档更新时重跑）。
"""
import sys
sys.path.append("..")

from week3.vector_store import InMemoryVectorStore, Chunk
from week3.chunking import chunk_recursive
from week3.embeddings import get_embeddings_batch
import hashlib
import os


def ingest_document(
    text: str,
    doc_id: str,
    metadata: dict = None,
    chunk_size: int = 400,
    overlap: int = 40,
) -> list[Chunk]:
    """
    把一篇文档切成 chunks 并 embed。
    metadata 可以包含：source_file, section, page_range 等。
    """
    metadata = metadata or {}
    
    # 1. 切割
    raw_chunks = chunk_recursive(text, chunk_size=chunk_size, overlap=overlap)
    print(f"  [{doc_id}] 切割完成：{len(raw_chunks)} 个 chunk")
    
    # 2. 批量 embedding
    texts = [c for c in raw_chunks]
    embeddings = get_embeddings_batch(texts)
    print(f"  [{doc_id}] Embedding 完成")
    
    # 3. 组装 Chunk 对象
    chunks = []
    for i, (text_chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
        chunk_id = f"{doc_id}_chunk_{i:03d}"
        chunk = Chunk(
            id=chunk_id,
            text=text_chunk,
            embedding=embedding,
            metadata={
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "char_count": len(text_chunk),
            }
        )
        chunks.append(chunk)
    
    return chunks


def build_index(documents: list[dict], save_path: str = None) -> InMemoryVectorStore:
    """
    批量索引多个文档，返回向量库。
    
    documents 格式：
    [
        {"id": "pricing_policy", "text": "...", "metadata": {"source": "pricing_policy.pdf"}},
        {"id": "margin_rulebook", "text": "...", "metadata": {"source": "margin_rulebook.docx"}},
    ]
    """
    store = InMemoryVectorStore()
    
    for doc in documents:
        print(f"\n正在索引：{doc['id']}")
        chunks = ingest_document(
            text=doc["text"],
            doc_id=doc["id"],
            metadata=doc.get("metadata", {}),
        )
        store.add_batch(chunks)
    
    print(f"\n索引完成：共 {len(store)} 个 chunk")
    
    if save_path:
        store.save(save_path)
    
    return store
```

### 5.3 RAG 查询 Pipeline

```python
# week3/rag_pipeline.py
"""
RAG 的"在线"阶段：接收用户问题，检索，生成带引用的答案。
"""
import sys
sys.path.append("..")

from week3.vector_store import InMemoryVectorStore
from week3.embeddings import get_embedding
from week2.schemas import ContractQueryResult
from utils.llm import chat, structured_chat


RAG_SYSTEM_PROMPT = """
你是 Acme 公司的 B2B 销售合规顾问。

你的回答必须严格基于下方提供的政策文档片段（Context）。
- 如果 Context 中有明确答案，直接引用并给出结论。
- 如果 Context 中信息不足，明确说明"政策文档中未找到相关规定"。
- 不得凭自己的知识补充未在 Context 中出现的政策内容。
- 每个关键结论后面必须注明来源（chunk_id 或文档名）。
"""


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    把检索到的 chunk 格式化成 prompt 里的 context 块。
    """
    if not retrieved_chunks:
        return "<context>\n未找到相关政策文档内容。\n</context>"
    
    parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", chunk["chunk_id"])
        score = chunk["score"]
        parts.append(
            f"[来源 {i+1}: {source} | 相关度: {score:.2f}]\n{chunk['text']}"
        )
    
    return "<context>\n" + "\n\n---\n\n".join(parts) + "\n</context>"


def rag_query(
    question: str,
    store: InMemoryVectorStore,
    k: int = 3,
    threshold: float = 0.3,
    verbose: bool = False,
) -> dict:
    """
    完整的 RAG 查询。
    返回：answer（字符串）、sources（引用来源列表）、retrieved_chunks（原始检索结果）
    """
    
    # Step 1: Embed 问题
    query_embedding = get_embedding(question)
    
    # Step 2: 检索
    retrieved = store.search(query_embedding, k=k, threshold=threshold)
    
    if verbose:
        print(f"\n[检索结果] 问题：{question}")
        for r in retrieved:
            print(f"  - {r['chunk_id']} | 相关度: {r['score']:.4f}")
            print(f"    {r['text'][:80]}...")
    
    # Step 3: 组装 prompt
    context = format_context(retrieved)
    
    user_message = f"""
{context}

问题：{question}

请基于以上 Context 回答，并在答案末尾列出引用的来源编号。
如果 Context 不足以回答，请明确说明。
"""
    
    # Step 4: LLM 生成
    answer = chat(
        user_message=user_message,
        system_prompt=RAG_SYSTEM_PROMPT,
        model="gpt-4o",
        temperature=0,
    )
    
    # Step 5: 整理来源
    sources = [
        {
            "chunk_id": r["chunk_id"],
            "source": r["metadata"].get("source", r["chunk_id"]),
            "score": r["score"],
            "preview": r["text"][:100] + "...",
        }
        for r in retrieved
    ]
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
    }
```

---

## 6. 本周 Mock 数据准备

这周开始需要真实的文档，用 mock 数据替代。新建 `data/policies/mock_policies.py`：

```python
# data/policies/mock_policies.py
"""
Mock 政策文档内容。
真实项目里这些来自 PDF/DOCX，Week 5 会实现真正的解析。
本周先用纯文本。
"""

PRICING_POLICY = """
Acme 公司定价与折扣管理政策
版本：v2.3  生效日期：2024年1月1日

第一章 总则

1.1 目的
本政策旨在规范 Acme 公司销售团队的报价与折扣行为，确保公司整体毛利率目标的实现，
同时为销售团队提供清晰的操作指引。

1.2 适用范围
本政策适用于所有 B2B 销售业务，包括直销、渠道销售和战略合作项目。

第二章 客户分级

2.1 客户分级定义
- 标准客户：年采购额低于 100 万元人民币
- 优质客户：年采购额在 100 万至 500 万元之间
- 战略客户：年采购额超过 500 万元，或由 CEO 批准的重要战略合作客户

2.2 客户升级机制
客户分级每季度评估一次，以过去 12 个月的实际采购额为准。
新客户默认为标准客户，达标后次季度起升级。

第三章 折扣政策

3.1 标准折扣上限
各产品品类的标准折扣上限如下：
- 硬件设备类：最高 10%
- 软件许可类：最高 12%
- 专业服务类：最高 15%
- 维保服务类：最高 8%

3.2 客户分级折扣调整
在标准折扣基础上，按客户级别可叠加以下特批上限：
- 优质客户：在标准上限基础上额外 3%（需区域经理审批）
- 战略客户：在标准上限基础上额外 5%（需销售副总裁审批）

3.3 最终折扣绝对上限
无论客户级别和产品品类，单笔订单折扣后最终售价不得低于成本价的 115%（即最低毛利率 13%）。

第四章 毛利率管控

4.1 最低毛利率要求
- 单笔订单最低毛利率：25%（低于此值需要财务总监特批）
- 季度整体毛利率目标：32%
- 战略客户项目毛利率下限：20%（需要 CEO 审批）

4.2 毛利率计算方式
毛利率 = (售价 - 成本) / 售价 × 100%
成本包括：产品直接成本、运输成本、安装调试成本。
不包括：销售费用、管理费用。

4.3 低毛利订单审批流程
毛利率 20%-25%：区域经理 + 财务经理联合审批
毛利率 15%-20%：销售副总裁 + CFO 联合审批
毛利率低于 15%：CEO 特批，需提交书面战略理由

第五章 审批权限

5.1 审批层级
- 折扣 ≤ 标准上限：销售代表自主决定，无需审批
- 折扣超标准上限：区域经理审批
- 折扣超优质客户上限：销售副总裁审批
- 单笔订单金额 > 200 万：无论折扣如何，均需销售副总裁知会

5.2 紧急审批
如客户有紧急下单需求，可通过销售运营系统发起紧急审批，
要求 4 小时内响应。紧急审批不得用于规避常规审批流程。
"""

MARGIN_RULEBOOK = """
Acme 公司毛利管控手册
版本：v1.8  最后更新：2024年3月

一、毛利率基准

1.1 产品线毛利基准
公司各产品线的目标毛利率如下：
- 核心硬件（服务器、存储）：28%-35%
- 网络设备：30%-38%
- 软件产品（标准版）：45%-60%
- 软件产品（定制版）：35%-50%
- 专业服务：40%-55%
- 售后维保：50%-65%

1.2 低于基准的处理
当报价毛利率低于对应产品线基准区间下限时，系统将自动标记为"低毛利预警"。
销售代表需在报价备注中说明原因，并获得对应层级审批后方可提交。

二、Bundle 定价规则

2.1 产品组合定价
当客户采购包含多个产品线的组合方案时：
- 整体方案毛利率不得低于 25%
- 任意单个产品线毛利率不得低于 15%
- 高毛利产品（软件/服务）不得用于补贴硬件亏损

2.2 禁止行为
以下定价行为被明确禁止：
- 将软件产品以成本价打包进硬件方案以拉低竞争对手报价
- 跨季度拆分订单以规避大单审批
- 在未获批准前向客户承诺超标折扣

三、高风险订单识别

3.1 自动风险标记
以下情况系统将自动将订单标记为高风险：
- 折扣率超过对应品类标准上限
- 毛利率低于 25%
- 单笔金额超过 100 万且折扣 > 8%
- 客户首单且折扣超过 5%
- 同一客户 30 天内多笔订单累计折扣异常

3.2 高风险订单处理
高风险订单不得提交，必须先完成以下步骤：
1. 销售代表填写风险说明
2. 区域经理确认业务合理性
3. 财务团队验证毛利数据
4. 走完对应审批流程

四、季度结算规则

4.1 季度末特殊政策
每季度最后两周，允许销售团队对战略客户额外给予不超过 2% 的季末促销折扣，
但整体毛利率不得因此低于 22%。此政策需提前向销售副总裁申报。

4.2 年度大单优惠
对于签订年度框架协议的客户，可在年度协议层面给予额外 1%-3% 的框架折扣，
独立于单笔订单折扣计算。框架折扣需 CFO 审批。
"""

VENDOR_CONTRACT_ATLAS = """
供应商合同摘要：Atlas 科技有限公司
合同编号：VND-2024-ATLAS-001
签约日期：2024年2月15日  有效期至：2026年2月14日

一、供货范围
Atlas 科技作为 Acme 公司的一级认证供应商，提供以下产品的独家供货：
- Atlas X 系列企业级服务器
- Atlas S 系列存储设备
- 相关配件及备件

二、定价与结算

2.1 供货价格
合同期内供货价格按照双方确认的《价格附件 A》执行，价格每半年复议一次。
价格调整需提前 60 天书面通知，调整幅度不得超过上期价格的 ±8%。

2.2 付款条款
- 标准付款周期：货到验收后 Net 45 天
- 提前付款优惠：货到后 10 天内付款，享受 2% 现金折扣（2/10 Net 45）
- 逾期付款罚息：超过 Net 45 天后，每日 0.05% 罚息

2.3 结算货币
人民币结算，跨境采购部分按签单日当日中国人民银行汇率折算。

三、交货条款

3.1 交货周期 SLA
- 标准配置（目录内产品）：下单后 15 个工作日内交货
- 非标/定制配置：下单后 30 个工作日内交货，复杂项目另议
- 紧急采购（需书面确认）：7 个工作日内，附加 5% 紧急服务费

3.2 交货地点
Acme 指定仓库（华东：上海青浦；华南：广州南沙；华北：天津滨海）。
非上述地点需额外协商运费。

3.3 延迟交货处罚
超过 SLA 的延迟交货，每超过 1 个工作日，Acme 可扣除该批次货款的 0.3%，
最高扣款不超过该批次总价的 5%。

四、质量保证

4.1 质保期
- Atlas X 系列服务器：3 年整机质保
- Atlas S 系列存储：3 年质保，硬盘 1 年质保
- 配件：1 年质保

4.2 质量问题处理
到货后 5 个工作日内验收，发现质量问题须书面通知 Atlas。
Atlas 须在 3 个工作日内响应，10 个工作日内完成换货或修复。

五、保密与竞业

双方对合同价格、技术规格及业务信息负有保密义务，保密期为合同期满后 3 年。
"""


# 文档集合，供索引使用
ALL_DOCUMENTS = [
    {
        "id": "pricing_policy",
        "text": PRICING_POLICY,
        "metadata": {
            "source": "pricing_policy_v2.3.pdf",
            "doc_type": "policy",
            "version": "2.3",
        }
    },
    {
        "id": "margin_rulebook",
        "text": MARGIN_RULEBOOK,
        "metadata": {
            "source": "margin_rulebook_v1.8.pdf",
            "doc_type": "rulebook",
            "version": "1.8",
        }
    },
    {
        "id": "vendor_atlas_contract",
        "text": VENDOR_CONTRACT_ATLAS,
        "metadata": {
            "source": "VND-2024-ATLAS-001.pdf",
            "doc_type": "contract",
            "vendor": "Atlas 科技有限公司",
        }
    },
]
```

---

## 7. 本周项目任务：Policy RAG Demo

新建 `week3/demo.py`，把所有组件串起来：

```python
# week3/demo.py
"""
Week 3 Demo：Policy RAG
输入：自然语言问题
输出：基于检索到的政策文档片段，生成带引用的结构化答案
"""
import sys, os
sys.path.append("..")

from week3.indexer import build_index
from week3.rag_pipeline import rag_query
from data.policies.mock_policies import ALL_DOCUMENTS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()
INDEX_PATH = "week3/index_cache.json"


def get_or_build_index():
    """加载缓存的索引，或重新构建（首次运行需要调用 API，约需 1 分钟）"""
    from week3.vector_store import InMemoryVectorStore
    
    store = InMemoryVectorStore()
    
    if os.path.exists(INDEX_PATH):
        console.print("[dim]加载已有索引...[/dim]")
        store.load(INDEX_PATH)
    else:
        console.print("[yellow]首次运行，构建索引（需要调用 Embedding API）...[/yellow]")
        store = build_index(ALL_DOCUMENTS, save_path=INDEX_PATH)
    
    return store


def display_result(result: dict):
    """格式化显示 RAG 结果"""
    
    # 答案
    console.print(Panel(
        result["answer"],
        title=f"[bold]💬 回答[/bold]",
        border_style="green"
    ))
    
    # 来源
    if result["sources"]:
        table = Table(title="📎 引用来源", show_header=True, header_style="bold blue")
        table.add_column("来源文档", style="cyan", width=30)
        table.add_column("相关度", justify="center", width=8)
        table.add_column("内容预览", width=50)
        
        for src in result["sources"]:
            table.add_row(
                src["source"],
                f"{src['score']:.2f}",
                src["preview"]
            )
        
        console.print(table)


def main():
    store = get_or_build_index()
    console.print(f"\n[green]索引就绪：{len(store)} 个 chunk[/green]\n")
    
    # 测试问题集
    questions = [
        "软件产品的最高折扣是多少？优质客户有额外额度吗？",
        "Atlas 供应商的付款条款是什么？有提前付款优惠吗？",
        "一笔订单毛利率降到 22% 需要谁来审批？",
        "什么情况下订单会被系统标记为高风险？",
        "紧急采购的交货周期是多少天？会有附加费用吗？",
    ]
    
    for i, question in enumerate(questions, 1):
        console.rule(f"[bold]问题 {i} / {len(questions)}[/bold]")
        console.print(f"\n[bold cyan]❓ {question}[/bold cyan]\n")
        
        result = rag_query(question, store, k=3, verbose=False)
        display_result(result)
        console.print()
    
    # 交互模式
    console.rule("[bold]交互模式[/bold]")
    console.print("[dim]输入问题直接查询，输入 'q' 退出[/dim]\n")
    
    while True:
        question = input("❓ 你的问题：").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        
        result = rag_query(question, store, k=3, verbose=True)
        display_result(result)
        print()


if __name__ == "__main__":
    main()
```

运行：
```bash
python week3/demo.py
```

---

## 8. RAG 质量的常见问题

本周写完 demo 之后，跑一跑以下问题，观察系统在哪里表现不好，记录进 notebook。

### 问题一：检索到了但答案错

**现象**：相关 chunk 确实被检索回来了，但 LLM 生成的答案不准确。
**诊断**：打开 `verbose=True`，看检索结果的实际内容。
**常见原因**：chunk 太长，相关句子在 chunk 中间，LLM 没有聚焦到它。
**解法**：减小 chunk size，或者在 context 格式化时高亮关键段落。

### 问题二：检索回来的 chunk 不相关

**现象**：相似度分数看起来不低（比如 0.4），但内容和问题无关。
**诊断**：问题本身的语义太模糊，或文档里确实没有相关内容。
**解法**：提高 `threshold`（比如从 0.3 调到 0.5），过滤低质量检索结果；或对问题做改写（query rewriting，Week 4 会讲）。

### 问题三：答案没有引用来源

**现象**：LLM 给出了答案，但没有引用 context 里的来源编号。
**诊断**：prompt 里关于引用的指令不够强。
**解法**：在 prompt 里加 `"每一个关键判断后面必须用[来源 N]标注"`，并在 few-shot 里给一个带引用的例子。

### 问题四：问题和索引语言不一致

**现象**：文档是中文，问题也是中文，但偶尔检索质量差。
**原因**：`text-embedding-3-small` 是多语言模型，中文支持良好，但不如英文精确。
**解法**：对于纯中文项目，可以考虑 `text-embedding-3-large`，或者评估一下具体差距再决定。

---

## 9. 本周 Checklist

### 概念
- [ ] 能用自己的话解释为什么 RAG 比"把整个文档塞进 prompt"更好
- [ ] 理解余弦相似度为什么比欧氏距离更适合 embedding 检索
- [ ] 能解释 chunk overlap 的作用
- [ ] 知道 chunk size 的选择如何影响检索精度

### 代码
- [ ] `week3/embeddings.py`：单条和批量 embedding 都可以调用
- [ ] `week3/chunking.py`：`chunk_recursive()` 实现完成，能处理长文本
- [ ] `week3/vector_store.py`：`InMemoryVectorStore` 支持 add/search/save/load
- [ ] `week3/indexer.py`：`build_index()` 能把 mock 文档转成向量库
- [ ] `week3/rag_pipeline.py`：`rag_query()` 完整跑通，有格式化引用输出
- [ ] `week3/demo.py`：5 个测试问题全部有输出，交互模式可用
- [ ] `data/policies/mock_policies.py`：三份 mock 文档写好

### 实验记录（Notebook）
- [ ] 对比 chunk_size=200 和 chunk_size=600 对同一问题的检索质量差异
- [ ] 找一个检索失败的案例（检索结果和问题不相关），分析原因
- [ ] 测试 threshold 从 0.2 到 0.6 时检索结果数量和质量的变化
- [ ] 记录一个问题：LLM 忽略了 context，用"自己的知识"回答了（这是 RAG 的核心风险）

### 加分项
- [ ] 给 `InMemoryVectorStore` 加一个 `search_with_filter()` 方法，支持按 `metadata` 字段过滤（比如只搜 `doc_type == "policy"` 的 chunk）
- [ ] 实现 `chunk_with_metadata()`：切割时保留章节标题作为每个 chunk 的 metadata，让引用来源更精确（比如 `pricing_policy §3.2` 而不只是 `pricing_policy_chunk_014`）
- [ ] 写一个 `evaluate_retrieval()` 函数：给定问题和期望的 chunk_id，计算 Top-3 检索的命中率（这是 Week 6 eval 的雏形）

---

## 10. 预告：Week 4

下周用 **LangChain** 重写本周的整个 RAG pipeline，同时加入真正的 Citation 机制。

你会看到：

**LangChain 替你做了什么**
- `RecursiveCharacterTextSplitter` = 你本周手写的 `chunk_recursive()`
- `OpenAIEmbeddings` = 你的 `get_embeddings_batch()`
- `FAISS` 向量库 = 你的 `InMemoryVectorStore`（但快 100 倍）
- `RetrievalQAWithSourcesChain` = 你的 `rag_query()`

理解这个对应关系非常重要。你不是在"学框架"，而是在用框架替换你已经理解的组件。

**Citation 机制升级**
本周的引用是靠 prompt 指令让 LLM "自觉"标注来源，不可靠。
下周会用 LangChain 的 `source_documents` 机制做可靠的引用追踪，引用来源直接从检索结果里提取，不依赖 LLM 输出。

**Query Rewriting**
当用户的问题表述不清时，先让 LLM 改写问题，再检索。
这是提升 RAG 检索质量最简单有效的技巧之一。

**本周的 `InMemoryVectorStore` 和 `mock_policies.py` 下周继续用。**

---

*Week 3 / 6 完成后继续 →* **[Week 4：LangChain RAG Pipeline + Citation]**
