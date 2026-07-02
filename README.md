# InsureAI — Multi-Agent Insurance Policy Assistant

InsureAI is an end-to-end insurance document intelligence system that lets users upload multiple PDF policy documents and ask natural-language questions about them. The system combines document ingestion, table-aware retrieval, hybrid search, multi-agent reasoning, LLM validation, Obervability and a production-ready API + frontend experience.

This project is designed for real-world insurance use cases such as:
- answering questions from uploaded PDFs
- querying an existing policy corpus
- checking coverage, exclusions, waiting periods, claims, and premiums
- comparing policy documents across insurers
- generating grounded, cited answers with observability and evaluation

---

## Why this project is special

This is not a simple chatbot over a few PDFs. It is a full RAG + agentic AI insurance QA platform with:
- custom ingestion for real insurance documents
- table-first and layout-aware extraction
- chunking enriched with table context, page metadata, and hierarchy clues
- hybrid retrieval using semantic + lexical signals
- context compression and answer grounding
- self-reflection, corrective RAG, validation, observability and evaluation
- multi-agent orchestration with LangGraph
- FastAPI backend and Streamlit frontend
- Docker-based deployment, Redis short term memory

The project was built around a very important insight: insurance documents are heavily table-driven. Therefore, the system is optimized not only for plain text, but especially for structured table content, coverage rows, exclusions, eligibility tables, hidden risks and policy clauses.

---

## Project overview

Users can:
1. Upload one or more PDF documents
2. Ingest them through a production pipeline
3. Extract text, tables, and page-level structure
4. Chunk content into retrieval-friendly units
5. Generate embeddings and index the chunks for semantic search
6. Ask questions in natural language
7. Receive grounded answers with citations and confidence scoring

The system also supports querying an existing knowledge base built from the prepared corpus, so users are not limited to only newly uploaded policies.

## Architecture overview

                    ┌──────────────────────────────┐
                    │        User Uploads PDF      │
                    └──────────────┬───────────────┘
                                   │
                         PDF Extraction + OCR
                                   │
                     Table Extraction + Cleaning
                                   │
                     Metadata Enrichment
                                   │
                        Table-aware Chunking
                                   │
                   Embeddings + FAISS Indexing
                                   │
                              BM25 Index
                                   │
                                   ▼
                         Hybrid Retrieval Engine
                     (Semantic + Lexical + Boosting)
                                   │
                              Reranking
                                   │
                          Context Compression
                                   │
                                   ▼
                      LangGraph Multi-Agent System

             Query Agent → Retrieval 
                               ├── Policy Agent
                               ├── Claim Agent
                               ├── Risk Agent
                               ├── Comparison Agent
                                   │
                          Recommendation Agent

                                   │
                           Self-RAG Reflection
                                   │
                         Response Validation
                                   │
                           Corrective RAG Loop
                                   │
                             Report Generator
                                   │
                              FastAPI Backend
                                   │
                           Streamlit Frontend

---

## Core architecture

The architecture is built around four layers:

### 1. Data ingestion and preprocessing
This layer handles:
- PDF upload
- text extraction
- OCR fallback
- table extraction
- document cleaning
- chunking
- embedding generation
- FAISS indexing

### 2. Retrieval layer
This layer handles:
- semantic search with embeddings
- lexical retrieval with BM25
- structural boosting for table and row data
- reranking
- neighbor expansion for related table rows
- context packing for downstream reasoning
- advance rag inculding , crag , self rag, LLM validation, obervation using LangSmith, and Evaluation

### 3. Agentic reasoning layer
This layer uses LangGraph and multiple specialist agents to interpret intent and answer questions precisely.

### 4. API and UI layer
This layer exposes:
- a FastAPI backend for queries and uploads
- a Streamlit frontend for chat-style interaction
- Redis-backed session memory
- containerized deployment via Docker Compose

---

## What data was used

A key strength of this project is that the data is not from Kaggle or a generic public dataset. The corpus was collected manually and curated specifically for insurance policy documents.

The dataset includes multiple insurance providers and document categories such as:
- Policy
- Coverage
- Exclusions
- Claim
- Brochure
- CIS
- PreAuth
- Proposal
- Policy Usage Guide

The documents come from insurers such as:
- Care Health
- HDFC ERGO
- ICICI Lombard
- Niva Bupa
- Star Health

The data includes both narrative text and, critically, many tabular sections that are common in insurance documents. That makes the project especially suited for retrieval over policy tables, eligibility grids, exclusions, waiting periods, and benefits.

---

## End-to-end pipeline

### Step 1: PDF ingestion
When a user uploads a PDF, the system:
- saves the file temporarily
- extracts page-level text using layout-aware parsing
- applies OCR fallback when needed
- extracts tables from each page
- cleans the content
- stores structured extracted output

This is handled primarily in:
- [backend/api/pdf_pipeline.py](backend/api/pdf_pipeline.py)
- [backend/app/utils/pdf_extractor.py](backend/app/utils/pdf_extractor.py)

### Step 2: Document understanding and enrichment
The extracted document is enriched with metadata such as:
- insurer name
- document type
- page number
- table data
- document-level context

This is important because policy documents are highly structured and often require context from section headings, document type, and table relationships.

### Step 3: Chunking
The content is chunked in a way that preserves document meaning and table structure.

The chunker:
- processes both text blocks and tables
- creates table-aware chunks
- maintains parent/row context for tabular sections
- preserves page information
- includes section and metadata context

This is implemented in:
- [backend/app/utils/chunker.py](backend/app/utils/chunker.py)

The chunking is especially valuable because insurance policies contain a lot of table-based rules. Instead of losing those rows as plain text, the pipeline preserves them as structured retrieval units.

### Step 4: Embedding and indexing
Each chunk is converted into embeddings using a local sentence-transformers model and stored in a FAISS vector index.

This is handled in:
- [backend/app/utils/embeddings.py](backend/app/utils/embeddings.py)

The index is persisted on disk and updated incrementally, so new uploads can be indexed without rebuilding everything from scratch.

### Step 5: Retrieval
Query handling uses a sophisticated retrieval pipeline that goes far beyond naive vector search.

The retrieval stack includes:
- semantic search using FAISS
- lexical search using BM25
- intent-aware boosting
- table-aware scoring
- related-row expansion
- reranking
- context packing

Key files:
- [backend/app/retrieval/hybrid_search.py](backend/app/retrieval/hybrid_search.py)
- [backend/app/retrieval/final.py](backend/app/retrieval/final.py)
- [backend/app/retrieval/reranker.py](backend/app/retrieval/reranker.py)
- [backend/app/retrieval/query_router.py](backend/app/retrieval/query_router.py)

#### Why this retrieval is smart
The system is designed to retrieve the right evidence from insurance documents, not just any similar text. It explicitly improves retrieval for:
- exclusions
- coverage details
- waiting periods
- claim-related clauses
- table rows and related table content

It also expands around table rows and groups related table chunks, which is extremely useful for policy questions that depend on surrounding rows rather than a single isolated sentence.

---

## Table-aware and context-aware retrieval

One of the strongest parts of this project is its support for table-centric information retrieval.

### What makes it special
The chunking and retrieval pipeline preserve:
- table identifiers
- row identifiers
- parent-row context
- page numbers
- section information
- document metadata

This means the system can retrieve not only the exact relevant row, but also the surrounding row context and related table content. For insurance policies, this is very important because a single row often only makes sense when read with its parent header or neighboring rows.

This is a major improvement over standard chunking approaches, which often break table information into poor, isolated fragments.

---

## Multi-agent reasoning system

This project is built as a LangGraph multi-agent workflow rather than a single prompt-to-answer pipeline. In this architecture, the user query moves through a graph of specialized nodes, where each node performs one job and passes the state forward.

### What is the graph?
The graph is the full execution workflow of the application. It is defined in [backend/agents_claude/orchestrator.py](backend/agents_claude/orchestrator.py).

In this graph:
- a node is one step or agent in the workflow
- an edge is the route from one node to the next
- the shared state is the evolving memory of the query, retrieved evidence, intent, confidence, intermediate analysis, and final results

So the workflow is not just “LLM answers query.” It is: understand → retrieve → reason → validate → generate → report.

### Step-by-step query flow
When a user sends a question, the graph executes in the following order:

1. Query Normalizer Node
   - This is the first node.
   - It receives the raw user question and rewrites it into a cleaner search query.
   - It corrects spelling, OCR noise, grammar, and unclear phrasing.
   - It also detects the intent of the query, such as coverage, exclusion, claim, waiting period, premium, comparison, or general policy question.
   - It extracts entities from the query, such as insurer name, policy category, treatment, sum insured, and other important terms.

2. Retrieval Node
   - The normalized query is passed to the retrieval agent.
   - This node searches the indexed insurance corpus using hybrid retrieval.
   - It returns relevant chunks from the policy documents, including page numbers, section information, chunk type, and table-aware context.
   - If the retrieval confidence is weak, this node can trigger a clarification path instead of continuing with weak evidence.

3. Conditional routing after retrieval
   - This is where the graph becomes truly multi-agent.
   - The next node is chosen based on the detected intent:
     - coverage / waiting period / premium / definition → Policy Analysis Agent
     - claim → Claim Eligibility Agent
     - exclusion → Risk Analysis Agent
     - comparison → Comparison Agent
   - If confidence is very low, the system can route to Human Review instead of forcing an incorrect answer.

4. Specialist reasoning nodes
   - Policy Analysis Agent: explains coverage, benefits, eligibility, and policy clauses.
   - Claim Eligibility Agent: handles claim-related reasoning such as settlement, claims process, reimbursement, and eligibility conditions.
   - Risk Analysis Agent: focuses on exclusions, limitations, waiting periods, and potential risks in the policy wording.
   - Comparison Agent: compares policies or insurers based on retrieved evidence.
   - Recommendation Agent: turns the analysis into actionable guidance and highlights what the user should focus on.

5. Answer Synthesis Node
   - After the specialist agents produce their reasoning, the workflow moves into the Advanced RAG layer.
   - This node uses the retrieved policy chunks to generate a grounded answer that is supported by the evidence.
   - It performs compression, self-reflection, validation, citation checking, and optional corrective retries.

6. Report Generator Node
   - The final node converts the reasoning and synthesized answer into a polished response for the user.
   - It formats the answer as a report with markdown, confidence, sources, and diagnostics.

### Human-in-the-loop flow
Human-in-the-loop is one of the strongest parts of this architecture.

- If the retrieval step does not find enough reliable evidence, or the confidence is too low, the system pauses and asks the user a clarifying question.
- The Human Review Node is triggered by the graph.
- The user’s response is then sent back into the Query Normalizer Node, restarting the flow with better context.
- This is extremely useful for ambiguous insurance questions where the same term may refer to different policy sections or different insurers.

This is implemented through LangGraph interrupt logic in [backend/agents_claude/orchestrator.py](backend/agents_claude/orchestrator.py).

---
## Architecture of Graph:

```
User Query
    │
    ▼
[Query Normalizer Agent]       ← LLM: fixes typos, detects intent, extracts entities
    │                            Fast path: regex for clean queries 
    ▼
[Retrieval Agent]              ← Hybrid Search (FAISS + BM25) + Reranker
    │                            Computes confidence score → triggers routing
    ├── confidence < 0.30 ──────→ [Human Review Node] → retrieval agent
    │
    ├── intent: coverage/waiting/premium/definition
    │       └──→ [Policy Analysis Agent]
    │                   └──→ [Risk Analysis Agent]
    │
    ├── intent: claim
    │       └──→ [Claim Eligibility Agent]
    │                   └──→ [Risk Analysis Agent]
    │
    ├── intent: exclusion
    │       └──→ [Risk Analysis Agent]
    │                   (also runs Policy Analysis for context)
    │
    └── intent: comparison / is_comparison=True
            └──→ [Comparison Agent]
                        │
                        ▼
              [Recommendation Agent]   ← Synthesizes all prior outputs
                        │
                        ▼
              [Report Generator]       ← Markdown + JSON + PDF-ready output
                        │
                        ▼
              confidence < 0.50 → [Human Review Node] -> retrievel agent
```

## Advanced RAG pipeline

The final answer is not produced by a simple prompt over raw text. It uses an Advanced RAG pipeline that is specifically designed for reliability, grounding, and quality control.

### How retrieval works
The retrieval stage is a multi-signal pipeline:

1. Semantic retrieval with FAISS
   - The system embeds the policy chunks and searches a FAISS vector index for semantically similar content.

2. Lexical retrieval with BM25
   - It also uses keyword-based retrieval so exact policy phrases, clause wording, and important terms are not missed.

3. Intent-aware scoring
   - The retrieval layer boosts chunks related to high-value insurance intents such as exclusions, waiting periods, coverage, claims, and definitions.

4. Table-aware retrieval
   - Since insurance documents heavily depend on table-based rules, the system preserves table identifiers, row numbers, and parent-row context.
   - This allows the system to retrieve not just a single cell or line, but also the surrounding table structure that makes it meaningful.

5. Reranking
   - After the initial candidates are collected, a reranker improves the rank order so the most useful evidence moves to the top.

6. Context packing
   - The top retrieved chunks are packed together into a compact evidence block so the LLM sees the most relevant context without flooding the prompt.

The retrieval logic is implemented in [backend/app/retrieval/hybrid_search.py](backend/app/retrieval/hybrid_search.py), [backend/app/retrieval/final.py](backend/app/retrieval/final.py), and [backend/app/retrieval/reranker.py](backend/app/retrieval/reranker.py).

### How the answer is generated
Once the evidence is retrieved, the Advanced RAG pipeline runs the following sequence:

1. Context compression
   - The retrieved chunks are compressed into a smaller, high-signal context block.
   - This keeps the prompt focused on the best evidence instead of overwhelming the LLM with too much irrelevant text.

2. Grounded answer generation
   - The LLM generates the answer using only the compressed evidence as context.

3. Self-RAG reflection
   - The system reflects on whether the retrieved evidence is actually relevant to the query.
   - It checks whether the answer is grounded in the retrieved chunks rather than being invented.

4. Response validation
   - The answer is validated for relevance, faithfulness, completeness, and citation correctness.
   - If the response is weak, the pipeline marks it for correction.

5. Citation checking
   - The answer is checked to ensure that any cited claim points to an existing retrieved chunk.
   - This reduces the risk of hallucinated sources.

6. Corrective RAG loop
   - If the first answer fails validation, the system enters CRAG.
   - It improves the retrieval process and retries the generation step.

### How CRAG works
CRAG is the corrective loop that activates when the first answer is not reliable enough.

When validation fails, the system inspects the reason codes such as:
- low_relevance
- low_faithfulness
- hallucinated_citation
- no_supporting_chunks
- retrieval_low_confidence

Based on these reasons, the system can:
- widen the retrieval window by pulling more candidate chunks
- rewrite the query to make it broader or more precise
- re-run retrieval and regeneration with better evidence

The loop is intentionally bounded so it does not run forever. It performs a few controlled retries and then stops gracefully.

This is implemented in [backend/app/rag/corrective_rag.py](backend/app/rag/corrective_rag.py) and called from [backend/app/rag/advanced_rag.py](backend/app/rag/advanced_rag.py).

### How Self-RAG works
Self-RAG is the reflection layer that checks the quality of the reasoning before the final answer is shown.

It evaluates:
- whether the retrieved context is relevant to the question
- whether the answer is supported by the context
- whether the response uses valid citations

If an LLM-based judge is available, it uses that. Otherwise it falls back to heuristic scoring. This makes the system practical even without expensive external evaluation services.

This is implemented in [backend/app/rag/self_rag.py](backend/app/rag/self_rag.py).

### Validation and evaluation
This project places a strong emphasis on quality control, which is one of the main reasons it is impressive for real-world AI deployments.

The system evaluates the answer using:
- faithfulness: whether the answer is grounded in retrieved evidence
- relevance: whether it aligns with the user’s question
- completeness: whether the answer covers the request sufficiently
- citation quality: whether every source tag is valid
- context precision and recall: whether the selected chunks are both precise and sufficient
- hallucination and correctness proxies: whether the answer drifts or makes unsupported claims

These checks are implemented in [backend/app/rag/response_validator.py](backend/app/rag/response_validator.py), [backend/app/rag/evaluation.py](backend/app/rag/evaluation.py), and [backend/app/rag/observability.py](backend/app/rag/observability.py).

### Why this is highly valuable 
This architecture is attractive because it combines:
- modular agent orchestration
- evidence-grounded answer generation
- retry logic when quality is low
- validation before final output
- observability for debugging and monitoring
- evaluation metrics for quality assurance

That is much more advanced than a simple chatbot and is closer to a production-grade AI reasoning system.

---



---

## API and frontend

### Backend API
The backend exposes a FastAPI server with endpoints for:
- health checks
- querying the agentic RAG system
- uploading PDFs for ingestion
- retrieving session history

Main file:
- [backend/api/main.py](backend/api/main.py)

### Frontend
A Streamlit frontend provides a polished chat interface where users can:
- upload PDF batches
- ask questions
- see answers and sources
- interact with the backend in real time

Main file:
- [streamlit_frontend.py](streamlit_frontend.py)

---

## Deployment

The project is containerized with Docker and Docker Compose.

### Services
- Redis for short-term session memory
- FastAPI backend
- Streamlit frontend

Deployment files:
- [docker-compose.yml](docker-compose.yml)
- [Dockerfile.api](Dockerfile.api)
- [Dockerfile.frontend](Dockerfile.frontend)

### Run locally
```bash
docker compose up --build
```

Then open:
- frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Tech stack

### Core
- Python
- FastAPI
- Streamlit
- LangChain
- LangGraph
- LangSmith
- Docker
- RAGAS
- DeepEval
- Redis
- FAISS
- Sentence Transformers

### Document processing
- PyMuPDF
- pdfplumber
- pytesseract
- Pillow

### ML / retrieval
- FAISS vector search
- BM25-style lexical scoring
- hybrid retrieval
- sentence-transformers embeddings
- reranking

### Deployment
- Docker
- Docker Compose

---

## Repository structure

- [backend/api](backend/api) — API entry points and upload pipeline
- [backend/app/retrieval](backend/app/retrieval) — hybrid retrieval, reranking, routing
- [backend/app/rag](backend/app/rag) — advanced RAG, self-RAG, CRAG, validation, evaluation
- [backend/agents_claude](backend/agents_claude) — LangGraph orchestrator and specialist agents
- [backend/app/utils](backend/app/utils) — PDF extraction, OCR, chunking, embedding utilities
- [data](data) — processed data, chunk storage, index artifacts
- [Dataset](Dataset) — source insurance document folders
- [streamlit_frontend.py](streamlit_frontend.py) — chat UI

---

## Key highlights

- Built for real insurance document QA rather than generic RAG
- Uses curated, manually collected policy data
- Designed around tables, structured rules, and policy rows
- Strong retrieval pipeline with table-aware and context-aware logic
- Multi-agent reasoning with LangGraph
- Advanced RAG quality controls including self-reflection, corrective retries, and validation
- Production-ready API and user interface
- Docker deployment support

---

## Summary

InsureAI is a complete, production-style insurance document assistant that can ingest policy PDFs, understand their structure, retrieve the right evidence from both text and tables, and answer user questions through a multi-agent reasoning pipeline. It is designed to be accurate, explainable, and deployable for real-world policy support workflows.

This project combines document intelligence, retrieval augmentation, agent orchestration, evaluation, and deployment into a single powerful system.

## Experimental evaluation

A lightweight local benchmark runner is included to generate free, portfolio-style metrics from the existing chunk corpus without relying on paid LLM judge APIs.

Run it with:

```bash
python -m backend.app.rag.benchmark_runner
```

This produces:
- [backend/data/benchmark_results.json](backend/data/benchmark_results.json) — machine-readable benchmark output
- [backend/data/benchmark_report.md](backend/data/benchmark_report.md) — Markdown report ready to paste into the README

The script reports:
- Recall@5, MRR, and nDCG@5 for retrieval quality
- Faithfulness, answer relevancy, context precision, context recall, and hallucination rate for RAG quality
- Average latency and a simple human-in-the-loop clarification rate

# Experimental Results

The system was evaluated on a curated set of insurance policy questions spanning coverage, exclusions, waiting periods, claims, premiums, comparisons, maternity, critical illness, co-payment, network hospitals, eligibility, hospitalization, day-care procedures, pre-existing diseases, and policy definitions.

# Benchmark Setup
- Queries: 58
- Categories: claim, comparison, copayment, coverage, critical_illness, daycare, definition, eligibility, exclusion, hospitalization, maternity, network_hospital, pre_existing, premium, waiting_period
- Chunk files: 45
- Strategies: naive, hybrid, table_aware, full_agentic

| Strategy | Recall@5 | MRR | nDCG@5 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Hallucination Rate | Avg Latency (ms) | HITL Clarification Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 0.862 | 0.735 | 0.764 | 0.655 | 0.563 | 0.971 | 1.000 | 0.345 | 58.8 | 0.345 |
| hybrid | 0.879 | 0.755 | 0.778 | 0.690 | 0.559 | 0.971 | 1.000 | 0.310 | 57.5 | 0.310 |
| table_aware | 0.862 | 0.739 | 0.762 | 0.672 | 0.554 | 0.971 | 1.000 | 0.328 | 63.0 | 0.328 |
| full_agentic | 0.845 | 0.753 | 0.766 | 0.707 | 0.548 | 0.966 | 1.000 | 0.293 | 71.5 | 0.293 |

### Discussion

The benchmark demonstrates the trade-offs introduced by progressively more sophisticated retrieval and reasoning strategies. Hybrid retrieval achieves the highest retrieval effectiveness, obtaining the best Recall@5 (0.879), nDCG@5 (0.778), and lowest average latency among the advanced retrieval approaches. The full agentic pipeline slightly sacrifices retrieval metrics due to additional reasoning and routing steps but achieves the highest faithfulness (0.707) and the lowest hallucination rate (0.293), indicating that the multi-agent workflow improves answer reliability even when retrieval rankings remain similar. These results suggest that the additional orchestration, validation, and corrective reasoning primarily improve answer quality rather than raw retrieval performance, which is consistent with the intended design of the system.
