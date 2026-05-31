# Advanced RAG (Python Edition)

Python / FastAPI / React 重新實作的 RAG 知識庫平台，
參考並對齊 [TigerAI Advanced RAG Community](https://tigerai-taiwan.github.io/TigerAI-Advanced-RAG-Community/)
的架構與 [OpenGenie AI Stack](https://github.com/TigerAI-Taiwan/OpenGenie-AI-Stack)
的底層思路，但**移除 n8n 與多 container 編排**，全部用 Python service 取代。

> 對話與設計分析來源：`/home/ubuntu/opengenie-rag-analysis/session-conversation.md`

## 設計文件
- [docs/SDD.md](docs/SDD.md) — Software Design Document
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — 階段計畫

## 架構速覽

```
React (Vite, :5173 dev / :80 prod)
   │ REST + SSE
   ▼
FastAPI (:8000)
   ├─ pipelines/step1_pdf_to_md.py    ← n8n #01 Docling
   ├─ pipelines/step2_md_to_json.py   ← n8n #02 MD2JSON
   ├─ pipelines/step3_md_to_qa.py     ← n8n #03 Validation
   ├─ pipelines/step3b_quality_gate.py← n8n #03b AIQualityGate
   ├─ pipelines/step4_ingestion.py    ← n8n #04 DataIngestion
   ├─ pipelines/step5_query.py        ← n8n #05 RAG Core (Agent1+Agent2, stream/non-stream)
   └─ providers/{llm,embedding,vector_store,docling}.py  ← swappable
   │
   ├─ SQLAlchemy (SQLite default / Postgres prod)
   └─ Qdrant (default in-memory for tests)
```

### 與原系統的對應
| n8n workflow | 本實作模組 |
|--------------|-----------|
| `#01-DataTransferring-Docling` | `pipelines/step1_pdf_to_md.py` |
| `#02-DataProcessing-MD2JSON` | `pipelines/step2_md_to_json.py` |
| `#03-DataValidation-MD2QA` | `pipelines/step3_md_to_qa.py` |
| `#03b-AIQualityGate` | `pipelines/step3b_quality_gate.py` |
| `#04-DataIngestion` | `pipelines/step4_ingestion.py` |
| `#05-RAG-Core` / `#05b-Cloud-Streaming` | `pipelines/step5_query.py` |
| `Sub-Chat-Cloud` | (合併到 `step5_query.answer / answer_stream`) |

## 本機快速開始

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # 預設 mock provider，不需要任何 API key
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
打開 http://localhost:8000/docs 看 OpenAPI 介面。

### Frontend
```bash
cd frontend
npm install
npm run dev
```
打開 http://localhost:5173

### 跑測試
```bash
cd backend && source .venv/bin/activate
pytest -q          # 15 tests (unit + integration + full smoke flow)
```

### 用 Docker 一次起來

> 需要 Docker 19.03+ 及 docker-compose v1.27+（或 `docker compose` plugin）。

```bash
# 1. 複製環境變數範本並填入設定
cp .env.example .env
# 編輯 .env，填入 LLM/Embedding/Rerank 的 base_url、api_key、model 名稱

# 2. 第一次啟動（自動 build images + 啟動三個服務）
sudo docker-compose up -d --build

# 之後更新只需
sudo docker-compose up -d
```

**Port 對照表**

| 服務 | Port | 說明 |
|------|------|------|
| 前端 UI | :80 | http://localhost |
| 後端 API | :8000 | http://localhost:8000/docs |
| Qdrant | :6333 | http://localhost:6333/dashboard |

**資料持久化（bind mount）**

| 容器路徑 | 主機路徑 | 內容 |
|----------|----------|------|
| `/qdrant/storage` | `./data/qdrant/` | 向量索引 |
| `/app/data` | `./backend/data/` | SQLite DB、上傳檔、chunks、markdown |
| `/root/.cache/fastembed` | Docker volume `fastembed_cache` | BM25 模型快取 |

**健康檢查**

```bash
# 確認三個 container 都是 (healthy)
sudo docker-compose ps

# API 健康
curl http://localhost/api/health

# Qdrant collections
curl http://localhost:6333/collections

# 停止所有服務
sudo docker-compose down
```

**注意**：BM25 sparse embedding 模型（`Qdrant/bm25`）已在 build 時預先下載進 image，首次啟動不需要等待下載。

## 切換到真實 LLM
編輯 `backend/.env`：
```
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
DOCLING_PROVIDER=local   # 需 pip install docling
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
```

## 主要 API
| Method | Path | 說明 |
|--------|------|------|
| POST | `/api/projects` | 建立專案 |
| GET | `/api/projects` | 列表 |
| POST | `/api/projects/{id}/documents` | 上傳檔案 (multipart) |
| POST | `/api/projects/{id}/rule/build` | Rule Builder：產生 3 個候選 prompt |
| POST | `/api/projects/{id}/rule` | 啟用 rule |
| POST | `/api/projects/{id}/training/start` | 啟動 ingestion |
| GET | `/api/projects/{id}/training/status` | 任務狀態 |
| POST | `/api/chat` | RAG 問答 |
| POST | `/api/chat/stream` | RAG 問答 (SSE 串流) |
| GET | `/api/projects/{id}/qc/coverage` | 覆蓋率 |
| POST | `/api/projects/{id}/qc/audit` | 答案稽核 |
| GET | `/api/files/{doc_id}/download` | 下載原始檔 |

## 測試覆蓋
```
tests/
├── unit/
│   ├── test_health.py         ← API 健康檢查
│   ├── test_embedding.py      ← MockEmbedding 確定性、L2 normalise
│   ├── test_vector_store.py   ← In-memory upsert/search/filter/count
│   ├── test_step4_uuid.py     ← Point ID UUIDv5 一致性
│   ├── test_pipelines.py      ← step1 + step2
│   └── test_step5_query.py    ← Agent1+retrieve+Agent2 完整流程
└── integration/
    ├── test_projects_api.py   ← CRUD + 409 + 404
    ├── test_settings_api.py
    └── test_full_flow.py      ← upload→train→chat→QC 全鏈路
```

15/15 通過。

## 安全注意
- 預設 admin token 為 `change-me`，務必在 `.env` 改掉。
- CORS 由 `CORS_ORIGINS` 控制；正式環境請勿用 `*`。
- 上傳檔名以 `pathlib.Path(...).name` 去除路徑成分，避免路徑穿越。
- LLM prompt 使用 system / user 分離，並要求 JSON 物件回應，降低注入風險。

---

## 變更記錄

### 2026-05-31 — Async 阻塞修正（K8S liveness probe timeout）（commit `6e6b462`）

**問題**：`LocalDoclingProvider._convert()` 及多處同步 I/O 直接在 `async` 函式中呼叫，導致整個 asyncio event loop 被阻塞，FastAPI 無法於 CPU-密集作業（Docling 轉換、HybridChunker）期間回應 K8s liveness/readiness probe，造成 pod 被強制重啟。

**修正方式**：所有同步 CPU 密集 / 大型 I/O 呼叫以 `asyncio.to_thread(...)` 卸載至 thread pool worker（Python 3.9+ 標準庫，無新增套件）。

**涵蓋檔案**：

| 檔案 | 修正內容 |
|------|----------|
| `app/providers/docling.py` | `LocalDocling._convert`（torch/docling ML pipeline）、`export_to_markdown`、`export_to_dict`、圖片抽取/描述注入、`HttpDocling` 檔案讀取 |
| `app/pipelines/step1_pdf_to_md.py` | markdown / doc.json / images.json 檔案寫入 |
| `app/pipelines/step2_md_to_json.py` | **`HybridChunker.chunk`**（CPU-heavy）、`DoclingDocument.model_validate`、markdown 讀取、chunks.json 寫入 |
| `app/pipelines/step3_md_to_qa.py` | markdown 讀取 |
| `app/api/documents.py` | 上傳 `write_bytes`、chunk location `read_text` |
| `app/services/rules.py` | 範例 markdown 讀取 |

單元測試 10/10 通過，本地容器重啟後 `/api/health` 回應正常。

---

### 2026-05-31 — Dockerfile 強制安裝 opencv-python-headless（commit `ade6f65`）

**問題**：`docling` 依賴 `rapidocr`，transitively 拉入 `opencv-python`（GUI 版），在 `python:3.12-slim` 中缺少 `libxcb.so.1`，導致 import 時 `ImportError`：

```
libxcb.so.1: cannot open shared object file: No such file or directory
```

**修正**：在 `pip install -r requirements.txt` 之後加入強制替換步驟：

```dockerfile
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    pip install --no-cache-dir opencv-python-headless
```

確保最終 image 中只存在 headless 版本，不依賴 X11/libxcb。

---

### 2026-05-31 — 新增 opencv-python-headless 至 requirements.txt（commit `9cb3304`）

`requirements.txt` 新增 `opencv-python-headless>=4.8`，明確聲明 headless 依賴，避免 `docling` → `rapidocr` 自動拉入 GUI 版 opencv。

---

### 2026-05 — 資料庫切換至 PostgreSQL 17

- `docker-compose.yml` 新增 `postgres` service，backend 設定 `DATABASE_URL` 指向 PostgreSQL 17。
- SQLAlchemy async engine 改用 `asyncpg` driver。
- `requirements.txt` 新增 `asyncpg`、`psycopg2-binary`。

---

### 2026-05 — 新增 Docling 本地 PDF 轉換支援

- `requirements.txt` 新增 `docling>=2.0`。
- `DOCLING_PROVIDER=local` 時使用 `LocalDoclingProvider`，於容器內執行 torch-based PDF → Markdown 轉換，無需外部 API。
- Build 時預先下載 BM25 sparse embedding 模型（`Qdrant/bm25`），首次啟動不需等待。

---

### 2026-05 — Frontend nginx .mjs MIME type 修正

- `frontend/nginx.conf` 新增 `.mjs` → `application/javascript` MIME 對應，修正瀏覽器載入 ES module 時 MIME type 錯誤的問題。
