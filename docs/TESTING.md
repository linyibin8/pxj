# 知进拍学测试文档

## 后端检查

```powershell
python -m compileall -q backend\app backend\scripts scripts
```

本地 FastAPI TestClient 冒烟：

- `GET /health`
- `GET /`
- `GET /observe`
- `GET /prompts`
- `GET /assets`
- `GET /api/auth/config`

生产冒烟：

- `GET https://pxj.evowit.com/health`
- `GET https://pxj.evowit.com/api/auth/config`
- `POST /api/auth/register`
- `GET /api/auth/me`
- `POST /api/sessions`
- `GET /health/llm`

## iOS 检查

Mac generic build：

```bash
cd /Users/macstar/Code/PXJ
xcodebuild -project PXJ.xcodeproj -scheme PXJ -configuration Debug -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

当前 Mac 的 CoreSimulator framework 版本与 Xcode 不匹配，模拟器 UI test 会受影响；generic iOS build 和 TestFlight archive 不受此问题阻断。

## TestFlight 检查

```bash
/Users/macstar/testflight-auto/fastlane-ios-oneclick/bin/oneclick-ios status --project-dir /Users/macstar/Code/PXJ
```

期望：

- build state 为 `VALID`。
- encryption/export compliance 为 `False`。
- `PXJ Internal` 组存在。
- 目标测试员尽可能全部在组内；若邮箱不是 App Store Connect internal tester，会显示 missing。

## 当前已验证

- 后端 Python compileall 通过。
- 本地 FastAPI 页面和健康接口通过。
- 生产 `https://pxj.evowit.com/health` 返回 `service=pxj`。
- 生产注册、鉴权、创建会话、LLM health 通过。
- iOS generic build 通过。
- TestFlight build `202606282258` 已上传并为 `VALID`。
