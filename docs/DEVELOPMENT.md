# 知进拍学开发文档

## 后端本地启动

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8038 --reload
```

常用入口：

- Dashboard: `http://127.0.0.1:8038/`
- Health: `http://127.0.0.1:8038/health`
- Auth config: `http://127.0.0.1:8038/api/auth/config`
- LLM health: `http://127.0.0.1:8038/health/llm`

## 关键环境变量

- `PXJ_PUBLIC_BASE_URL`: 公网地址，生产为 `https://pxj.evowit.com`。
- `PXJ_DATA_DIR`: 数据目录，Docker 中为 `/app/data`。
- `PXJ_LLM_BASE_URL`: 默认 `http://100.64.0.5:39000/v1`。
- `PXJ_LLM_API_KEY`: 默认 `ollama`。
- `PXJ_LLM_MODEL`: 默认 `evowit-agent27b`。
- `PXJ_AUTH_REQUIRED`: 生产应为 `true`。
- `PXJ_REGISTRATION_ENABLED`: 是否允许新用户注册。
- `PXJ_OBSERVE_DEMO_PUBLIC`: 默认 `false`，观察 demo 接口需要登录。

## iOS 开发

Xcode 工程由 XcodeGen 生成：

```bash
cd ios
xcodegen generate
open PXJ.xcodeproj
```

主 target：`PXJ`

UI test target：`PXJUITests`

测试账号不再写在仓库里。需要自动登录时设置：

```bash
export PXJ_UI_TEST_EMAIL="..."
export PXJ_UI_TEST_PASSWORD="..."
```

## 编码约定

- 运行时代码统一使用 `pxj` / `PXJ` / `知进拍学`。
- 环境变量统一使用 `PXJ_` 前缀。
- 不提交任何 `.p8`、`.p12`、`.mobileprovision`、keychain 密码或云密钥。
- 后端新增接口必须考虑账号隔离，默认通过 `principal_from_request()` 获取 account context。
- 需要公开免登录接口时，必须有显式配置项，默认关闭。
- iOS 后端地址统一从 `ContentView.serverBaseURL` 与 `Auth.swift` 维护。

## 已修复的迁移风险

- 观察 demo 上传接口默认不再公开。
- iOS UI test 删除硬编码邮箱/密码。
- 可观测快照按当前账号数据库统计，不再读固定单库。
- 本地 Xcode 工程已重新生成，包含完整 Swift 源文件。
- Swift 并发检查中 `Task.detached` 的主线程访问警告已修复。
