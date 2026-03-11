# RAG Pipeline Technical Design

**Authors**: Zhao Lei, Wang Hao | **Date**: 2026-02-03 | **Version**: v2.0 (pre-review revision)

---

## 1. Architecture Overview

```
User uploads documents
    |
    v
doc-parser (parsing + structuring)
    |
    v
chunker (intelligent splitting)
    |
    |-- Embedding (bge-large-zh-v1.5) --> Qdrant (semantic index)
    |
    `-- Elasticsearch (full-text index)

User asks a question
    |
    v
Query Rewriter (optional)
    |
    |-- Qdrant semantic search (top-20)
    |
    `-- Elasticsearch BM25 search (top-20)
    |
    v
Reciprocal Rank Fusion
    |
    v
bge-reranker-v2-m3 (rerank, keep top-5)
    |
    v
LLM Gateway -> DeepSeek-V3 / Qwen2.5 (answer generation)
```

## 2. Document Parsing Layer

### 2.1 Parser Selection

After evaluation we chose **pdfplumber** as the primary PDF parser with **PyMuPDF (fitz)** as fallback:

| Engine | PDF text | Table extraction | Speed | Compatibility |
|--------|---------|------------------|-------|---------------|
| pdfplumber | Excellent | Excellent (built-in table detector) | Medium | Some CID mappings fail |
| PyMuPDF | Excellent | Medium (needs post-processing) | Fast (3x) | Highly compatible |
| Tika | Good | Good | Slow (JVM startup) | Widest format support |
| unstructured | Excellent | Good | Slow | API friendly but heavy |

Dual-engine policy: pdfplumber runs first; any segment with >10% garbled characters falls back to PyMuPDF. Garble rate dropped from 30% to 5% (see bug report on PDF gibberish).

### 2.2 Structured Output

All parsed documents return a unified JSON shape:

```json
{
  "doc_id": "uuid",
  "filename": "Contract_ACME_20260101.pdf",
  "format": "pdf",
  "pages": 42,
  "parse_engine": "pdfplumber+fitz",
  "parse_time_ms": 3200,
  "sections": [
    {
      "type": "heading",
      "level": 1,
      "text": "Chapter 1 General Provisions",
      "page": 1,
      "bbox": [72, 100, 540, 130]
    },
    {
      "type": "paragraph",
      "text": "This contract is entered into by Party A (XX) and Party B (YY)...",
      "page": 1,
      "bbox": [72, 140, 540, 300]
    },
    {
      "type": "table",
      "rows": [["Item", "Amount", "Note"], ["Design", "500k", "VAT incl."]],
      "page": 2,
      "bbox": [72, 50, 540, 200]
    }
  ],
  "metadata": {
    "author": "Zhang San",
    "created_at": "2026-01-01",
    "ocr_confidence": 0.92
  }
}
```

### 2.3 OCR Policy

For scanned PDFs we add an OCR preprocessing stage:

- **Detection**: if pdfplumber extracts <10 characters/page => treat as scanned.
- **OCR engines**: PaddleOCR v4 (Chinese-first) or Tesseract 5.x (multilingual).
- **Performance**: 2-3 seconds/page; a 100-page document takes 3-5 minutes (10-20x slower than text PDFs).
- **Configuration**: optional per upload, disabled by default.

Chen Jing requested surfacing OCR confidence so users can decide if manual review is needed.

## 3. Chunking Strategy

### 3.1 Experiment Setup

Per kickoff decision we compared chunking strategies on an internal test set (1,200 docs across contracts, manuals, research papers):

| Strategy | Description | F1 | MRR@5 | Avg length |
|----------|-------------|----|-------|------------|
| Fixed 512 | Fixed-length token splits | 0.74 | 0.68 | 512 |
| Recursive | LangChain RecursiveCharacterTextSplitter | 0.78 | 0.73 | 480 |
| Paragraph | Natural paragraph boundaries | 0.80 | 0.76 | 620 |
| **Hybrid** | Paragraph + table protection + overlap | **0.82** | **0.79** | 550 |
| Semantic | Embedding-based adaptive splits | 0.81 | 0.78 | 490 |

### 3.2 Final Strategy: Hybrid

Rules:

1. **Paragraph-first** - split by natural paragraphs (`\n\n`) targeting 512 tokens.
2. **Long paragraphs** - if >768 tokens, split at sentence boundaries.
3. **Short paragraphs** - merge consecutive sections <128 tokens until threshold.
4. **Table protection** - keep tables/code blocks intact as standalone chunks.
5. **Overlap** - 64-token overlap to preserve continuity.
6. **Heading inheritance** - prefix each chunk with its heading chain (e.g., `Chapter 1 > 1.1 Overview >`).

### 3.3 Per Document Type

| Type | F1 | MRR@5 | Challenge |
|------|----|-------|-----------|
| Contracts | 0.85 | 0.83 | Clause numbering must retain context |
| Manuals | 0.83 | 0.80 | Code snippets and config examples |
| Research reports | 0.78 | 0.74 | Equations and figure references |

Research reports lag because formulas/figures lose semantics in plain text chunks. Phase 2 will explore multimodal embeddings.

## 4. Retrieval & Reranking

### 4.1 Dual-Channel Retrieval

- **Semantic search**: Qdrant (top-20).
- **Full-text search**: Elasticsearch BM25 (top-20).
- **Fusion**: Reciprocal Rank Fusion with k=60.

Formula: `score(d) = sum_i 1 / (k + rank_i(d))`. Dual-channel improves MRR by 23% for exact-match queries like "Contract No. ZH-2026-0042".

### 4.2 Reranker Performance

bge-reranker-v2-m3 results:

| Setting | MRR@5 | Latency p99 (ms) |
|---------|-------|------------------|
| No reranker | 0.71 | - |
| Rerank top-10 | 0.79 | 45 |
| Rerank top-20 | 0.83 | 82 |
| Rerank top-50 | 0.84 | 195 |

Final config: rerank top-20 (only 0.01 less MRR than top-50 but 58% lower latency).

**Known issue**: reranker drops to 0.65 MRR on table-heavy docs because rows lack context. See Weekly Report W8 for mitigation.

## 5. LLM Generation

### 5.1 Prompt Template

```
You are an enterprise document assistant. Answer the user based on the reference content.

## Reference Content
{context}

## User Question
{question}

## Requirements
1. Only use the reference content; do not fabricate facts.
2. If the reference is insufficient, state it clearly.
3. Cite document name and page when quoting.
4. Use concise, professional language.
```

### 5.2 Token Quotas

Per CTO review (see `review_comments_0207.md`) we must enforce tenant-level token quotas:

- Monthly quota per tenant (input + output tokens).
- Real-time usage metrics and alerts at 80/90/100%.
- Overages trigger downgrade: switch to a smaller model or return retrieval-only results.
- Admin console can adjust quotas.

## 6. Open Issues

1. OCR quality for scanned PDFs remains inconsistent; need more real data (see `bug_report_parsing_garbled_text.md`).
2. Ultra-long documents (>100 pages) produce 500+ chunks, hurting retrieval. Proposal: add a summary index layer (per chapter summary, then drill down).
3. Table retrieval channel design TBD (ties to Weekly Report W8 reranker problem).
4. Multi-turn dialogue context retention strategy (cache prior chunk references?).

---

**Reviewer**: Wang Hao | **Approver**: Zhang Ming
