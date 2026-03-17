# DocMind Weekly Report W8 (Feb 17 - Feb 21)

**Author**: Zhang Ming | **Date**: 2026-02-21

---

## 1. Progress This Week

### Backend - Wang Hao

1. **doc-parser service**  
   - PDF accuracy 92% (1,200-doc test set); Word accuracy 95%.  
   - Dual-engine flow (pdfplumber + PyMuPDF fallback) merged to `dev`.  
   - Asynchronous pipeline uses Celery + Redis with up to 3 retries.  
   - Avg speed: 1.2 s/page (text PDF) vs. 3.5 s/page (scanned).

2. **Tenant isolation**  
   - Isolation design submitted 2/14; CTO approved via email.  
   - Approach: Qdrant payload filter (`tenant_id`) + forced tenant conditions in query layer.  
   - Document blobs encrypted with AES-256-GCM, keys managed via HashiCorp Vault.  
   - LLM call logs stored in a separate audit DB after PII masking.

3. **CI/CD**  
   - GitHub Actions: lint -> test -> build -> deploy (auto-deploy dev).  
   - Docker Compose dev stack validated (5 core services + Redis + PG + ES + Qdrant).  
   - Production Helm chart draft ready for M3 testing.

### ML / Retrieval - Zhao Lei

1. **RAG pipeline E2E**  
   - Full flow: upload -> parse -> chunk -> index -> retrieve -> rerank -> generate.  
   - 50-doc eval yields MRR@5 = 0.81 (close to offline 0.83).

2. **Summary index prototype**  
   - Qwen2.5-7B generates 100-200 word summaries per chapter.  
   - Two-stage retrieval: search summaries (top-5 chapters) -> drill down into chunks.  
   - 100-page query latency drops from 320 ms to 180 ms (-44%) with no loss in MRR.

3. **Reranker table issue**  
   - Table-heavy docs (financial reports) see MRR drop to 0.65.  
   - Root cause: rows like "Design Fee | 500k | VAT incl." lose context when isolated.  
   - Interim mitigation: keep table chunks intact and bypass reranker.  
   - Long-term options: (a) dedicated table retrieval channel (b) LLM-generated natural-language descriptions before indexing.

### Frontend - Li Wei

1. **Document list & upload**  
   - Drag/drop and batch upload with progress bars.  
   - List supports search/filter/pagination with responsive layout.  
   - Wired to doc-parser API (upload -> async parse -> WebSocket status updates).

2. **OnlyOffice integration blockers**  
   - Open-source (AGPL) lacks inline comments/collab.  
   - Commercial license quoted at CNY 150k/year (Document Server + 100 concurrent users).  
   - Chen Jing submitted a purchase request; decision due next week.  
   - Backup: pdf.js for PDF preview + LibreOffice server-side conversion.

3. **Q&A page**  
   - Multi-turn chat UI done; SSE streaming hookup in progress.  
   - Source tracing: clicking citations jumps to the original snippet.

### QA - Liu Yang

1. **Test plan**  
   - Functional: 126 cases (upload, parse, search, Q&A, auth).  
   - Performance: 32 cases (parse throughput, retrieval latency, QPS).  
   - Security: 28 cases (injection, privilege escalation, data leakage).  
   - Totals: 186 cases (P0: 42, P1: 85, P2: 59).

2. **Automation**  
   - API smoke tests via pytest + httpx (8 doc-parser cases running).  
   - Performance harness using Locust; scenarios pending stable APIs.  
   - P0 automation scheduled for next week.

## 2. Risk Update

| ID | Risk | Level | Status | Notes |
|----|------|-------|--------|-------|
| R1 | OnlyOffice license over budget | High | [Warning] Pending | CNY 150k/year; awaiting approval |
| R2 | Reranker underperforms on tables | Medium | [Warning] Investigating | MRR 0.65; need table channel |
| R3 | Vector DB selection | - | [Done] Closed | Qdrant deployed, stable single node |
| R4 | OCR quality variability | Medium | [Warning] Monitoring | Dual-engine cut garble to 5%; see bug report |
| R5 | Frontend bandwidth | Low | [Warning] Watch | Chat UI behind schedule, may need overtime |

## 3. Next Week Plan

| Owner | Tasks | Priority |
|-------|-------|----------|
| Wang Hao | Finish llm-gateway service + token quota system | P0 |
| Wang Hao | Deliver Prometheus + Grafana monitoring (CTO Blocker B3) | P0 |
| Zhao Lei | Resolve table reranker issue, produce evaluation memo | P1 |
| Zhao Lei | Integrate summary index into main pipeline | P1 |
| Li Wei | Complete chat UI (streaming + citation jumps) | P0 |
| Li Wei | Await OnlyOffice decision, prep fallback | P1 |
| Liu Yang | Kick off P0 API automation | P0 |
| Liu Yang | Partner with Zhao Lei on table-channel validation | P2 |

---

**CC**: CTO Zhou Hang and all functional leads
