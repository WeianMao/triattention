# Vector Database Selection Report

**Author**: Zhao Lei | **Date**: 2026-01-29 | **Version**: v1.2

---

## 1. Context

Per the 1/15 kickoff decision we benchmarked Milvus 2.4, Qdrant 1.8, and pgvector 0.6 for DocMind's semantic search workload (1-5 million vectors, enterprise documents).

## 2. Test Environment

| Item | Spec |
|------|------|
| Server | 8C32G, 1 TB NVMe SSD |
| OS | Ubuntu 22.04 LTS |
| Dataset | 1 million 1024-d vectors (bge-large-zh-v1.5) |
| Source | Internal set (contracts/manuals/reports, 12k docs) |
| Index | Milvus: IVF_FLAT, Qdrant: HNSW, pgvector: ivfflat |
| Queries | top-10 nearest neighbors, batch size 1 and 100 |

## 3. Results

### 3.1 Core Metrics

| Metric | Milvus 2.4 | Qdrant 1.8 | pgvector 0.6 |
|--------|------------|------------|--------------|
| Insert throughput (10k/s) | 3.2 | **4.1** | 1.8 |
| Query latency p50 (ms) | 5 | **3** | 18 |
| Query latency p99 (ms) | 12 | **8** | 45 |
| Batch latency p99 (ms) | 85 | **52** | 320 |
| Memory (GB) | 6.8 | **4.2** | 2.1 (shared with PG) |
| Disk (GB) | 8.5 | **5.8** | 12.3 |
| Recall@10 | 0.95 | **0.96** | 0.93 |
| QPS @10 concurrency | 1,200 | **1,850** | 180 |

### 3.2 Feature Comparison

| Feature | Milvus 2.4 | Qdrant 1.8 | pgvector 0.6 |
|---------|------------|------------|--------------|
| Hybrid search (vector + scalar) | Native | Native (payload filter) | Manual JOIN |
| Distributed cluster | Native (Pulsar/Kafka) | Raft-based | Requires Citus |
| Multi-tenant isolation | Partition key | Collection + API key | Schema-level |
| Incremental indexing | Yes | Yes | Needs REINDEX |
| Backup & restore | Bulk insert/backup API | Snapshot API | pg_dump |
| SDK ecosystem | Python/Java/Go/Node | Python/Rust/Go/TS | Any PG client |

### 3.3 Ops Complexity

- **Milvus** - depends on etcd + MinIO + Pulsar/Kafka (>=5 containers). Upgrades require compatibility checks. Community is vibrant.
- **Qdrant** - single binary, no external deps. Cluster mode needs >=3 nodes. Docker image is only 80 MB. Docs are clear but lack best practices for HA.
- **pgvector** - PostgreSQL extension, so we can reuse existing PG workflows. Large-scale tuning is nontrivial (maintenance_work_mem, ivfflat lists, etc.).

## 4. Scaling to 5 Million Vectors

Extended with synthetic data:

| Metric | Milvus 2.4 | Qdrant 1.8 | pgvector 0.6 |
|--------|------------|------------|--------------|
| Memory (GB) | 28.5 | **18.7** | 9.2 |
| Query latency p99 (ms) | 18 | **12** | 185 |
| Index build time (min) | 45 | **32** | 210 |

pgvector's latency balloons to 185 ms (>50 ms SLA), so it fails the scale requirement.

## 5. Recommendation

**Select Qdrant 1.8.**

1. **Best performance** - lowest latency, highest throughput, fastest inserts.
2. **Efficient** - Rust implementation uses 62% of Milvus's memory.
3. **Simple ops** - single binary deployment.
4. **High recall** - HNSW index reached 0.96 on our test set.

Milvus is feature-rich but too heavy for phase 1. pgvector remains suitable for tenants under ~500k vectors and can be the lightweight fallback (see Review Comment S2).

## 6. Follow-ups

1. Wang Hao to design Qdrant cluster deployment (3-node Raft, cross-AZ DR).
2. Zhao Lei to finalize the query fusion logic (Elasticsearch + Qdrant).
3. Evaluate the cost of supporting pgvector as a swappable backend (adapter layer).

---

**Reviewer**: Wang Hao | **Approver**: Zhang Ming
