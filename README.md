# RAG Nexus：垂直领域文档知识库

本项目将本地文档检索、引用可追溯的生成式问答组合为可在个人电脑运行的全栈应用。当前支持 PDF/TXT，使用本地 BGE embedding 与 reranker、Qdrant 持久化、BM25 关键词检索和 SSE 流式回答。

## 架构与亮点

```text
PDF/TXT → PyPDF → RecursiveCharacterTextSplitter (500 / 50)
        → SentenceTransformer embedding → Qdrant (Cosine)

问题 → Qdrant Top-10 ─┐
       BM25 Top-10 ────┴→ 去重 + RRF → BGE CrossEncoder → Top-3
                                                       ↓
                       AsyncOpenAI ← 限定上下文的提示词
                           ↓ SSE: citations / token / done / error
                       Next.js 对话 UI + 来源片段展开
```

- 上传大小受限、文件类型校验、解析错误可读；chunk ID 基于文档内容确定，便于稳定追溯。
- 稠密向量与 BM25 双路召回，经 Reciprocal Rank Fusion 合并，再以 BGE reranker 精排。
- 回答只使用检索上下文；无依据时要求明确拒答。SSE 先发来源元数据，再发 token。
- Qdrant 数据存于 Docker named volume；模型运行在本机，LLM 使用 OpenAI-compatible API（OpenAI 或 Ollama）。

## 文件结构

```text
outputs/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── requirements-eval.txt
│   ├── main.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── evaluate.py
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── next.config.ts
    ├── postcss.config.mjs
    └── app/
        ├── page.tsx
        ├── layout.tsx
        └── globals.css
```

## 本地启动

需要 Python 3.10+、Node.js 20+、Docker Compose。以下命令从本项目 `outputs` 目录执行。

1. 启动向量数据库：

   ```bash
   docker compose up -d qdrant
   ```

2. 配置环境变量：复制 `.env.example` 到 `backend/.env`，设置 `LLM_API_KEY`。使用 Ollama 时设置 `LLM_BASE_URL=http://localhost:11434/v1`、`LLM_MODEL=<本地模型名>`；本地 Ollama 可将 key 设置为 `ollama`。

3. 安装并启动后端：

   ```bash
   cd backend
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. 新开终端安装并启动前端：

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   打开 `http://localhost:3000`。API 文档位于 `http://localhost:8000/docs`。首次启动会下载约数百 MB 的 embedding/reranker 模型。可通过 `NEXT_PUBLIC_API_URL` 指定后端地址。

## API 与 SSE 协议

- `POST /upload`：multipart 表单字段 `file`，允许 `.pdf` / `.txt`，返回文档 ID、来源名和 chunk 数。
- `GET /documents`：查看知识库文件清单；`DELETE /documents/{document_id}`：删除文档及其索引块。
- `POST /chat`：JSON `{ "question": "..." }`。SSE 的 `citations` 事件携带 `source`、`chunk_id`、页码和原文；默认 `message` 事件中每条 `data` 为 `{ "token": "..." }`；结束时发 `done`，生成期间失败则发 `error`。
- `GET /health`：服务状态。

## 生产化说明

BM25 索引在服务启动时加载到内存，并在文档上传或删除后刷新；大规模语料应使用持久化、可增量更新的关键词索引。Reranker 分数用于排序，不应被当作校准置信概率。此本地示例未启用用户认证或租户隔离；部署到公共网络前必须放在受保护的网关后，并设置 TLS、认证、限流和数据隔离。

## 文档管理与评估

- `GET /documents` 列出已索引文档；`DELETE /documents/{document_id}` 删除该文档及其所有块。
- 评估集使用 UTF-8 JSONL，每行形如 `{"question":"根据你的资料填写问题","reference":"由领域人员编写的参考答案"}`。请使用真实文档和人工审核的参考答案，不要将占位内容作为评估数据。
- 在后端虚拟环境安装评估依赖：`pip install -r requirements-eval.txt`。确保服务运行且环境变量可用后执行：

  ```bash
  python evaluate.py --input evaluation.jsonl --output evaluation-results.json
  ```

  脚本通过实际 `/chat` 接口评估 Ragas Faithfulness、Answer Relevancy、Context Precision 与 Context Recall。评估会调用配置的 LLM 和 embedding API，产生额外耗时及费用；不同评审模型的分数可能有波动。仓库未提供虚构的评估对比数据。
