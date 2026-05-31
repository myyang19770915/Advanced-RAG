# Advanced RAG — Kubernetes 部署指南

## 檔案結構

```
k8s/
├── namespace.yaml          # Namespace: advanced-rag
├── configmap-secret.yaml   # ConfigMap（非敏感設定）+ Secret（API keys）
├── pvc.yaml                # PersistentVolumeClaim x3
├── qdrant.yaml             # Qdrant Deployment + Service
├── backend.yaml            # Backend Deployment + Service
├── frontend.yaml           # Frontend Deployment + Service
└── ingress.yaml            # Ingress（選用）
```

---

## Step 1：推送 Image 到 Registry

### 方案 A：GitHub Container Registry（推薦）

```bash
# 登入 ghcr.io（需要 GitHub Personal Access Token，scope: write:packages）
echo $GITHUB_TOKEN | docker login ghcr.io -u myyang19770915 --password-stdin

# 打 tag
docker tag advanced-rag-py_backend:latest  ghcr.io/myyang19770915/advanced-rag-backend:latest
docker tag advanced-rag-py_frontend:latest ghcr.io/myyang19770915/advanced-rag-frontend:latest

# Push
docker push ghcr.io/myyang19770915/advanced-rag-backend:latest
docker push ghcr.io/myyang19770915/advanced-rag-frontend:latest
```

接著更新 `backend.yaml` 和 `frontend.yaml` 的 `image:` 欄位：
```yaml
image: ghcr.io/myyang19770915/advanced-rag-backend:latest
imagePullPolicy: Always
```

如果 ghcr.io repo 是 private，需要建立 imagePullSecret：
```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=myyang19770915 \
  --docker-password=$GITHUB_TOKEN \
  -n advanced-rag
```
然後在 Deployment spec 加上：
```yaml
spec:
  imagePullSecrets:
    - name: ghcr-secret
```

### 方案 B：Docker Hub

```bash
docker tag advanced-rag-py_backend:latest  myyang19770915/advanced-rag-backend:latest
docker push myyang19770915/advanced-rag-backend:latest
```

---

## Step 2：設定 Secret（API keys）

```bash
# 刪除 configmap-secret.yaml 中的 stringData 明文，改用 kubectl 直接建立
kubectl create secret generic rag-secrets -n advanced-rag \
  --from-literal=LLM_API_KEY=your-llm-key \
  --from-literal=RERANK_API_KEY=your-cohere-key \
  --from-literal=EMBEDDING_API_KEY=your-embedding-key
```

---

## Step 3：部署

```bash
# 依序套用
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/qdrant.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml   # 選用

# 一次套用整個資料夾
kubectl apply -f k8s/
```

## Step 4：確認狀態

```bash
kubectl get all -n advanced-rag
kubectl get pvc -n advanced-rag
kubectl logs -n advanced-rag deployment/backend
```

---

## 注意事項

| 項目 | 說明 |
|------|------|
| **SQLite 限制** | backend `replicas` 必須為 1，若需水平擴展請改用 PostgreSQL |
| **StorageClass** | `pvc.yaml` 預設使用叢集預設 StorageClass，裸機需手動建立 PV 或使用 local-path-provisioner |
| **SSE 串流** | Ingress 已設定 `proxy-buffering: off`，確保 SSE 正常傳輸 |
| **Qdrant 版本** | 使用 `latest`，正式環境建議釘死版本號如 `qdrant/qdrant:v1.18.1` |
