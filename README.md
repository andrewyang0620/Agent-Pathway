# Agent-Pathway

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-F59E0B?style=flat-square)

A structured, week-by-week learning repository documenting my path from LLM fundamentals to building a production-grade AI agent system. The end goal is a **B2B Sales Policy & Operations Copilot** — a decision support system that reasons over policy documents and structured business data to assist sales teams with compliance, pricing, and risk assessment.

This is a learning repository, not a polished product. It is organized to reflect real progression: concepts are introduced incrementally, earlier implementations are intentionally minimal, and later weeks refactor and extend what came before.

---

## Motivation

Most AI tutorials either stop at "hello world" API calls or jump straight into framework abstractions without explaining the underlying mechanics. This repository takes a different approach: each week builds on the last, every abstraction is introduced only after its manual equivalent has been implemented, and the target application is a real business problem rather than a toy demo.

The capstone project — a B2B Sales Copilot — is chosen specifically because it requires the full stack of modern AI engineering: document retrieval, structured data querying, multi-step reasoning, tool orchestration, citation, and evaluation. It is a decision support system, not a chatbot.

---

## Roadmap

| Week | Topic | Key Concepts |
|------|-------|-------------|
| **1** | Environment & Foundations | Python toolchain, API setup, token/context fundamentals |
| **2** | LLM Basics + Prompt Engineering + Structured Output | System prompts, few-shot, CoT, Pydantic schemas, JSON mode |
| **3** | Embeddings + Retrieval + Minimal RAG | Embedding vectors, cosine similarity, chunking, hand-built RAG |
| **4** | LangChain RAG Pipeline + Citation | LangChain abstractions, retrievers, citation-grounded answers |
| **5** | Document Ingestion + SQL Tool | PDF/DOCX/CSV parsing, metadata design, DuckDB, tool calling |
| **6** | Agent + Tool Calling + Evaluation | LangChain agents, multi-tool routing, eval set, faithfulness metrics |

---

## Target Application: B2B Sales Policy & Operations Copilot

The system accepts natural language queries from sales and operations teams and responds with structured, evidence-backed decisions. It is designed to answer questions like:

- *Can we offer this customer a 12% discount?*
- *Which orders in this batch violate minimum margin policy?*
- *What are the payment terms in Vendor B's contract?*
- *Which SKUs are below reorder threshold but still trending upward in sales?*

The system reasons over two heterogeneous source types simultaneously:

**Policy & contract documents** (PDF, DOCX) — pricing guidelines, margin rulebooks, vendor contracts, sales handbooks

**Structured business data** (SQL / CSV) — order records, customer history, inventory levels, quote margins

Every response includes a conclusion, a reasoning trace, source citations, and an explicit uncertainty signal when the available information is insufficient to support a confident judgment.

---

## Repository Structure

```
Agent-Pathway/
│
├── README.md
│
├── data/
│   ├── policies/          # Mock policy documents (PDF, DOCX)
│   ├── contracts/         # Mock vendor contracts
│   └── tables/            # Mock structured data (CSV)
│
├── utils/
│   ├── __init__.py
│   └── llm.py             # Base LLM call wrappers (chat, structured_chat)
│
├── notebooks/             # Experimental notebooks, one per week
│   ├── week1_basics.ipynb
│   ├── week2_prompts.ipynb
│   └── ...
│
├── week1/                 # Environment setup, first API call
├── week2/                 # Prompt engineering, Pydantic schemas, structured output demo
├── week3/                 # Embeddings, chunking, hand-built RAG (no frameworks)
├── week4/                 # LangChain RAG pipeline with citation
├── week5/                 # Document ingestion pipeline, SQL tool integration
├── week6/                 # Full agent, multi-tool routing, evaluation harness
│
└── tests/                 # Eval set and test cases (built in Week 6)
```

---

## Design Principles

**No framework before the manual version.** RAG is implemented by hand in Week 3 before LangChain is introduced in Week 4. Tool calling is understood at the API level before agents abstract it away.

**Schemas first.** Pydantic output schemas (`DiscountDecision`, `PolicyViolation`, `ContractQueryResult`) are defined in Week 2 and serve as the data contract throughout the project. Downstream components are built to produce and consume these types.

**Notebooks as lab notebooks.** Each weekly notebook captures prompt experiments, output comparisons, and failure analysis — not just working code. The reasoning process is part of the record.

**Evaluation is not an afterthought.** Week 6 introduces a structured eval set with metrics for citation accuracy, answer faithfulness, and tool routing correctness. The goal is to quantify whether a change to the system is actually an improvement.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| LLM APIs | OpenAI (GPT-4o), Anthropic (Claude) |
| Structured output | Pydantic v2, OpenAI Structured Outputs |
| RAG framework | LangChain (Week 4+) |
| Vector store | FAISS (local, Week 3–4), ChromaDB (Week 5+) |
| Structured data | DuckDB, pandas |
| Document parsing | pypdf, python-docx, pandas |
| Evaluation | Custom eval harness + RAGAS (Week 6) |
| Dev tooling | Jupyter, Rich, Loguru, python-dotenv |

---

## Getting Started

```bash
git clone https://github.com/<your-username>/Agent-Pathway.git
cd Agent-Pathway

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY and ANTHROPIC_API_KEY to .env
```

Each week's directory contains its own scripts and is self-contained relative to `utils/`. Start from `week1/` and follow the weekly guides.

---

## Status

| Week | Status |
|------|--------|
| Week 1 — Environment & Foundations | ✅ Complete |
| Week 2 — Prompt Engineering + Structured Output | ✅ Complete |
| Week 3 — Embeddings + Minimal RAG | 🔄 In progress |
| Week 4 — LangChain RAG + Citation | ⬜ Not started |
| Week 5 — Ingestion + SQL Tool | ⬜ Not started |
| Week 6 — Agent + Eval | ⬜ Not started |

---

## Notes

All business data in this repository is synthetic and generated for learning purposes. No real customer, vendor, or order information is used anywhere in the codebase.