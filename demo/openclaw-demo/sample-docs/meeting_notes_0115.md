# DocMind Project Kickoff Meeting Notes

**Date**: 2026-01-15  
**Location**: Galaxy Tower 3F - Conference Room A  
**Participants**: Zhang Ming (PM), Li Wei (Frontend Lead), Wang Hao (Backend Lead), Zhao Lei (ML Engineer), Chen Jing (Product Manager), Liu Yang (QA Lead)

---

## 1. Project Background

The company is launching the "Intelligent Document Platform" (code name: DocMind). It targets enterprise customers and delivers a document intelligence platform powered by large language models. Primary capabilities:

1. **Document parsing and structuring** - automatically parse PDF, Word, PPT, and Excel files and extract headings, paragraphs, tables, images, and other structured elements.
2. **Intelligent summarization and Q&A** - implement document-level summarization and multi-turn Q&A with a RAG architecture.
3. **Cross-document relationship analysis** - build a knowledge graph across multiple documents to surface latent links.
4. **Collaboration and approval workflows** - integrate with customers' OA systems for online annotations, approvals, and version management.

Phase 1 focuses on the first two capabilities and targets a delivery date of 2026-04-30. Phase 2 (knowledge graph + OA integration) is not yet scheduled.

## 2. Technical Decisions

### 2.1 Backend Architecture

Wang Hao proposed a microservice architecture with the following services:

- **doc-parser** - Python service (unstructured + pdfplumber) that converts incoming documents into a unified structured JSON format.
- **doc-index** - Python service based on Elasticsearch + a vector database that serves both full-text and semantic retrieval.
- **llm-gateway** - Python gateway wrapping OpenAI/private models with routing, throttling, and fallback.
- **api-gateway** - Go service (Gin) responsible for auth, rate limiting, and routing.
- **user-service** - Go service managing RBAC roles and tenant isolation.

Services communicate via gRPC + protobuf. External APIs expose REST + JSON. Deployment: Docker Compose for dev, Kubernetes for production.

### 2.2 Vector Database

Zhao Lei evaluated three options:

| Option | Pros | Cons |
|--------|------|------|
| Milvus 2.4 | Active community, feature-rich, hybrid search support | Complex deployment (etcd/MinIO), heavy footprint |
| Qdrant 1.8 | Rust implementation, great performance, HTTP/gRPC | Younger ecosystem, limited docs for cluster mode |
| pgvector 0.6 | Reuses PostgreSQL stack, low ops cost | Scale-out performance uncertain, no native distributed mode |

He proposed benchmarking insertion throughput, query latency, memory footprint, and recall. Zhang Ming approved and asked for a decision report by end of January.

### 2.3 Frontend Architecture

Li Wei recommended Next.js 14 + TypeScript:

1. SSR improves SEO and first paint.
2. App Router gives flexible routing/layout nesting.
3. Server Components cut client JS payload for better performance.
4. The team already shipped three Next.js projects, so ramp-up is low.

UI library: Ant Design 5.x; charts: ECharts. Document preview must be custom: PDF rendering via pdf.js, Office preview TBD (options: LibreOffice conversion vs. OnlyOffice online editor).

Chen Jing asked for mobile usability. Li Wei suggested responsive design in phase 1 and evaluating native/Flutter apps in phase 2.

### 2.4 Model Choices

Zhao Lei shared the LLM plan:

**Summarization & QA models**
- Prefer DeepSeek-V3 API (low cost, strong on Chinese long-form, 128K context).
- Backup: Qwen2.5-72B for private deployments when data must stay on-prem.
- Dev/test: Qwen2.5-7B to keep GPU cost low while reusing the same API.

**Embedding models**
- Primary: bge-large-zh-v1.5 (1024 dims, top-5 on MTEB Chinese).
- Multilingual: multilingual-e5-large (100+ languages, 1024 dims).

**Reranker**
- bge-reranker-v2-m3 for bilingual cross-document reordering.

He emphasized chunking strategies as a major RAG quality lever--different document types (papers, contracts, manuals) need tailored splits. Zhang Ming asked Zhao Lei to finish experiments before the technical design review.

## 3. Milestones

| Milestone | Deadline | Deliverables | Owners |
|-----------|----------|-------------|--------|
| M1: Technical design review | 2026-02-07 | Technical spec, product prototype, selection report | Wang Hao, Zhao Lei |
| M2: Core feature complete | 2026-03-15 | Doc parsing + summarization/Q&A MVP | Entire team |
| M3: Integration testing | 2026-04-05 | Test report, bug fixes, perf tuning | Liu Yang |
| M4: UAT + launch | 2026-04-30 | Prod deployment, user manual, ops docs | Zhang Ming |

## 4. Risks

1. **Model quality risk** (High) - long-form Chinese summaries are hard, especially domain terminology. Requires careful prompt engineering and human evals.
2. **Format compatibility** (Medium) - enterprise documents vary widely; scanned PDFs require OCR, handwritten text is nearly impossible.
3. **Performance** (Medium) - 100+ page documents take time to parse; need async processing and progress tracking.
4. **Staffing** (Low) - frontend has only Li Wei + one junior dev; may need help if UI scope expands.
5. **Compliance** (Low) - some customers require onshore storage / MLPS Level 3; architecture must accommodate.

## 5. Action Items

| # | Item | Owner | Due |
|---|------|-------|-----|
| 1 | Finish vector DB benchmark (1M entries, four metrics) | Zhao Lei | 2026-01-31 |
| 2 | Draft technical spec (architecture, APIs, deployment) | Wang Hao | 2026-01-28 |
| 3 | Deliver product prototype in Figma (core flows) | Chen Jing | 2026-01-25 |
| 4 | Set up dev environment and CI/CD (GitHub Actions + Docker) | Wang Hao | 2026-02-01 |
| 5 | Complete chunking experiments (>=3 strategies x 3 doc types) | Zhao Lei | 2026-02-03 |
| 6 | Define QA plan (functional, perf, security) | Liu Yang | 2026-02-05 |
| 7 | Research client compliance requirements (MLPS, localization) | Zhang Ming | 2026-02-10 |

---

**Recorder**: Zhang Ming  
**Reviewer**: Chen Jing
