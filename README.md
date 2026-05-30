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
```bash
docker compose up --build
# UI:      http://localhost
# API:     http://localhost:8000/docs
# Qdrant:  http://localhost:6333/dashboard
```

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
