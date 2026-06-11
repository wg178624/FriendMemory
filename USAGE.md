# 使用指导

本文档说明如何在本地运行 Friend Memory Project。项目使用 Python + `uv` 管理，不需要安装成可导入包。

## 环境要求

- Python 3.11 或更高版本
- `uv`

安装依赖和运行命令时建议使用项目内缓存：

```bash
uv --cache-dir .uv-cache run python app/main.py commands --json
```

## 快速开始

先用临时状态文件运行，避免改写默认示例状态：

```bash
uv --cache-dir .uv-cache run python app/main.py --state /tmp/friend-memory-demo.json ingest "这是一次虚构的关系记忆测试。"
uv --cache-dir .uv-cache run python app/main.py --state /tmp/friend-memory-demo.json retrieve "刚才测试了什么？" --json
uv --cache-dir .uv-cache run python app/main.py --state /tmp/friend-memory-demo.json browser --json
```

默认状态文件位于 `data/friend_memory_state.json`。上传包里的默认状态是空初始化状态，不包含任何用户记忆。

## 常用 CLI

```bash
uv --cache-dir .uv-cache run python app/main.py commands --json
uv --cache-dir .uv-cache run python app/main.py doctor --json
uv --cache-dir .uv-cache run python app/main.py ai-status --json
uv --cache-dir .uv-cache run python app/main.py ai-probe "第一次一起完成样例项目，太开心了！" --json
```

记录完整的用户/助手交换：

```bash
uv --cache-dir .uv-cache run python app/main.py --state /tmp/friend-memory-demo.json ingest-exchange \
  "明天一起继续样例项目吗？" \
  "可以，明天继续。" \
  --json
```

## 启动 HTTP API

```bash
uv --cache-dir .uv-cache run python app/server.py --state /tmp/friend-memory-demo.json --host 127.0.0.1 --port 8765
```

示例请求：

```bash
curl 'http://127.0.0.1:8765/healthz'
curl -X POST http://127.0.0.1:8765/ingest \
  -H 'content-type: application/json' \
  -d '{"user":"u","ai":"a","text":"这是一次虚构的关系记忆测试。"}'
curl 'http://127.0.0.1:8765/retrieve?user=u&ai=a&q=刚才测试了什么'
```

## 接入外部 AI

默认不配置外部模型时，系统使用本地启发式实现，便于直接运行。要接外部模型，可以复制 `.env.example` 为 `.env` 后填写真实配置：

```bash
cp .env.example .env
```

`.env` 会被 `.gitignore` 忽略，不应提交到仓库。

验证外部 AI 是否真的参与：

```bash
uv --cache-dir .uv-cache run python app/main.py ai-probe "第一次一起完成样例项目，太开心了！" --require-external-ai --json
```

## 运行示例

```bash
uv --cache-dir .uv-cache run python examples/demo.py
uv --cache-dir .uv-cache run python examples/ai_participation_demo.py
uv --cache-dir .uv-cache run python examples/audit_coverage_demo.py
uv --cache-dir .uv-cache run python examples/decision_evidence_demo.py
```

## 隐私建议

- 不要提交 `.env`、真实 API key、真实用户对话或生产状态文件。
- 本地实验优先使用 `--state /tmp/...`。
- 如果需要保留长期测试状态，建议放在仓库外部路径。
- 上传 GitHub 前可再次检查是否存在 `.env`、`.venv/`、`.uv-cache/`、`__pycache__/` 和真实状态数据。
