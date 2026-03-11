# Technical Design Review Comments

**Date**: 2026-02-07 | **Reviewer**: CTO Zhou Hang | **Format**: On-site review (2 hours)

---

## 1. Overall Assessment

The plan is feasible with a clear architecture and solid experiments (chunking, vector DB benchmarks). However, several issues must be addressed before development.

**Decision: Conditionally Approved**

## 2. Blockers

### B1. Security & Tenant Isolation

The multi-tenant story is vague:

1. Qdrant collections are too coarse; how do we isolate tenants inside one collection? Payload filters with tenant IDs might work, but performance impact must be proven.
2. Raw documents and parsed results must be encrypted at rest (AES-256-GCM minimum).
3. LLM gateway logs may contain sensitive text; they need masking or segregated storage.

**Action**: Wang Hao to submit a tenant-isolation addendum (data-flow and security boundaries) by 2/14.

### B2. Cost Control

DeepSeek-V3 is cheap (~CNY 1 per million tokens), but without limits costs can explode. Based on 500 DAU x 10 Q&A/day x 8K tokens each:

- Daily tokens: 40 million
- Monthly cost: CNY 1,200 (acceptable)

But abusive tenants (bots, crawlers) could multiply usage.

**Action**: Implement tenant-level token tracking and quotas (see Technical Design Section 5.2).

### B3. Observability

No monitoring or alerting is defined. For an AI service this is non-negotiable.

**Action**: Integrate Prometheus + Grafana. Track at minimum:

- Doc parsing latency p50/p99
- Vector retrieval latency p50/p99
- Reranker latency p50/p99
- LLM latency p50/p99 + token usage
- Retrieval quality proxy (user feedback)
- Overall QPS and error rate

## 3. Suggestions (Non-blockers)

### S1. Summary Index

The chapter-level summary index (Technical Design Section 6.2) should ship in phase 1:

1. 100+ page documents are common for enterprise.
2. 500+ chunks hurt latency directly.
3. Implementation is lightweight (LLM-generated summary per chapter).

### S2. pgvector Backup Option

Qdrant is the right default, but keep pgvector as a lightweight alternative:

1. Small tenants (<100k docs) are fine with pgvector.
2. One less external component simplifies ops.
3. pgvector 0.7 roadmap is aggressive; retain the ability to upgrade later.

Implementation: add a retrieval adapter layer so Qdrant/pgvector can be swapped.

### S3. Document Preview

Prefer **OnlyOffice** over LibreOffice conversion:

1. OnlyOffice offers native web editing with inline comments.
2. LibreOffice-to-PDF often mangles complex layouts (merged cells, charts).
3. OnlyOffice provides an open-source Document Server (AGPL) plus commercial licenses.

**Note**: Pricing must be confirmed (Weekly Report W8 shows CNY 150k/year, above budget).

### S4. Failure Recovery

Parsing is asynchronous, so plan for retries:

1. Failed documents should be flagged for manual/auto retry.
2. Partial success (e.g., first 50 pages parsed, next 50 OCR timeout) should keep the successful portion.
3. Consider a lightweight queue (Redis + Celery or Postgres with advisory locks).

## 4. Collaboration

1. Define backend/frontend APIs using OpenAPI 3.0 so TS types can be generated.
2. Cross-review between backend and ML to avoid silos.
3. Hold a standing Friday 4 pm demo for all modules.

## 5. Conclusion

**Conditionally approved.** Once B1-B3 are resolved, Wang Hao can confirm via email; no second review required. Implement S1-S4 during development.

---

**Recorder**: Zhang Ming
