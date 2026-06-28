# 知进拍学 (pxj)

知进拍学是面向 iPhone/iPad 的拍照学习助手，支持拍题答疑、整页批改、题目提取、语音提问、错题复习、长期记忆和 Web 可观测后台。

生产入口：<https://pxj.evowit.com>

## 目录

- `backend/`: FastAPI 后端、Web dashboard、LLM 调用、SQLite 数据层。
- `ios/`: SwiftUI iPhone/iPad App，bundle id `com.linyibin8.pxj`。
- `deploy/nginx/`: `pxj.evowit.com` 的 VPS 反代配置。
- `scripts/`: 清理、iOS 冒烟等辅助脚本。
- `docs/`: 需求、架构、开发、部署、测试和流程图。

## 后端本地运行

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8038 --reload
```

访问：

- Dashboard: <http://127.0.0.1:8038/>
- Health: <http://127.0.0.1:8038/health>

## 关键配置

- `PXJ_PUBLIC_BASE_URL`: 生产为 `https://pxj.evowit.com`
- `PXJ_LLM_BASE_URL`: 默认 `http://100.64.0.5:39000/v1`
- `PXJ_LLM_API_KEY`: 默认 `ollama`
- `PXJ_LLM_MODEL`: 默认 `evowit-agent27b`
- `PXJ_AUTH_REQUIRED`: 生产为 `true`

## 文档

从 [docs/README.md](docs/README.md) 开始阅读。常用入口：

- [需求文档](docs/REQUIREMENTS.md)
- [架构文档](docs/ARCHITECTURE.md)
- [开发文档](docs/DEVELOPMENT.md)
- [部署文档](docs/DEPLOYMENT.md)
- [测试文档](docs/TESTING.md)
- [流程图](docs/FLOW_DIAGRAMS.md)
