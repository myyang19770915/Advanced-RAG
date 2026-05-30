# Implementation Plan — Advanced RAG (Python)

> 對應 SDD v0.1.0。本計畫採「**先骨架可跑、再逐步加實作、每階段必有測試**」的 walking-skeleton 策略。

---

## 階段 0 — Repo 與工具鏈
- [x] 建立 monorepo 目錄結構
- [ ] `backend/pyproject.toml`（用 uv 或 pip-tools；本案先用 `requirements.txt` 加 `pyproject.toml`）
- [ ] `frontend/package.json` (Vite + React + TS + Tailwind)
- [ ] `Makefile` / `scripts/dev.sh`：一鍵跑後端、前端
- [ ] `.env.example`
- [ ] 根目錄 `README.md` 快速啟動

## 階段 1 — Backend Walking Skeleton
- [ ] `app/core/config.py`：Pydantic Settings
- [ ] `app/main.py`：FastAPI app + CORS + 健康檢查 `/api/health`
- [ ] `app/db/`：async SQLAlchemy engine（SQLite 預設）
- [ ] Alembic 不導入（MVP 用 `Base.metadata.create_all` 啟動時建表）
- [ ] **Test:** `tests/unit/test_health.py` 確認 `/api/health` 回 200

## 階段 2 — Models + Schemas + 基礎 CRUD
- [ ] ORM: Project / Document / Chunk / Rule / QaTest / ChatSession / ChatMessage
- [ ] Pydantic schemas
- [ ] Router: `api/projects.py`（CRUD）
- [ ] **Test:** `test_projects_api.py`（建立、列表、取單筆、刪除）

## 階段 3 — Providers（抽象 + Mock + 真實）
- [ ] `providers/llm.py`：Protocol + `OpenAICompatibleLLM` + `MockLLM`
- [ ] `providers/embedding.py`：同上，`MockEmbedding` 用 hash → vector
- [ ] `providers/vector_store.py`：`InMemoryVectorStore` + `QdrantVectorStore`
- [ ] `providers/docling.py`：`MockDocling`（依檔名回固定 markdown）+ `LocalDocling`（呼叫 `docling` SDK，optional dep）
- [ ] DI factory：依 `settings.LLM_PROVIDER` 等選擇實作
- [ ] **Test:** 每個 provider 對應 mock 測試

## 階段 4 — Pipeline Steps（核心邏輯）
- [ ] `pipelines/step1_pdf_to_md.py`
- [ ] `pipelines/step2_md_to_json.py`（含 keyword L1/L2/L3 抽取 prompt）
- [ ] `pipelines/step3_md_to_qa.py`
- [ ] `pipelines/step3b_quality_gate.py`
- [ ] `pipelines/step4_ingestion.py`（含 UUIDv5 point id）
- [ ] `pipelines/step5_query.py`（Agent1 + 檢索 + Agent2）
- [ ] **Test:**
  - `test_step2_chunking.py`：用 MockLLM 回固定 JSON，驗證 parse + metadata
  - `test_step4_uuid.py`：相同 `(file, idx)` 應產同一 ID
  - `test_step5_query.py`：mock 環境跑完整查詢

## 階段 5 — Services + Orchestration
- [ ] `services/training.py`：把 step1→4 串成一個 job，更新狀態
- [ ] `services/chat.py`：包 step5
- [ ] `services/qc.py`：coverage + audit
- [ ] `workers/runner.py`：簡單 in-process async queue
- [ ] **Test:** `test_training_orchestration.py`（用 mock pipeline）

## 階段 6 — Routers（其餘 API）
- [ ] `api/documents.py`：上傳、列表、下載
- [ ] `api/rules.py`：Rule Builder + active rule
- [ ] `api/training.py`：start / status
- [ ] `api/chat.py`：non-stream + SSE
- [ ] `api/qc.py`：coverage / audit
- [ ] `api/settings.py`
- [ ] **Test:** 每個 router 至少一個 happy-path 測試

## 階段 7 — Frontend Walking Skeleton
- [ ] Vite + React + TS + Tailwind 初始化
- [ ] axios api client + base URL from env
- [ ] Layout（左 sidebar + topbar）
- [ ] 路由：`/projects`、`/projects/:id`、`/projects/:id/chat`、`/projects/:id/qc`、`/settings`
- [ ] Projects 列表 + 建立
- [ ] Project 詳情：文件上傳、Rule Builder、Start Training、狀態
- [ ] Chat 頁（含 streaming 顯示）
- [ ] QC 頁（覆蓋率、Audit 結果）
- [ ] Settings 頁

## 階段 8 — Integration / E2E
- [ ] `scripts/e2e_smoke.py`：純 Python 跑 backend → 上傳 mock PDF → train → chat → assert
- [ ] `docker-compose.yml`：qdrant + postgres + backend + frontend
- [ ] `docs/RUNBOOK.md`：本機與 Docker 模式

## 階段 9 — Polishing
- [ ] OpenAPI tags 整理
- [ ] 結構化 logging + request id
- [ ] CI 設定 (GitHub Actions) — 留 stub
- [ ] 文件補圖、補使用情境

---

## 開發順序（這次實作）
本回合會一次完成階段 0 → 7（walking skeleton 全部 + 主要邏輯 + 單元測試）。
階段 8/9 提供 stub 與 docker-compose，使用者可在本機驗證。
