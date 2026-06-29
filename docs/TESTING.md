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

## 智能观察提题专项

历史照片/真机回归：

- 同一页连续观察 10-20 秒：只出现少量关键帧，实时题框显示后不持续刷屏，重复题显示为弱化样式。
- 从第一页移动到下一页：1-3 秒内出现新的题框与 `fp` 短码，`observationQuestionUniqueCount` 增加。
- 停止观察后点“题目”：`POST /api/sessions/{id}/extract-all-questions` 返回 task id，任务浮层显示“提取观察题目”，完成后 `restore-page` 能渲染题集。
- 停止观察后点“报告”：`/finish` 排入 final report，任务浮层显示报告生成中。
- 停止观察后点“错题”：触发报告/错题候选整理，不直接把候选导入正式错题本。
- 低质量、无关、重复画面：不进入整轮题集，日志里能看到跳过或低质量原因。

后端专项命令：

```powershell
python -m compileall -q backend\app backend\scripts scripts
```

接口冒烟：

- `POST /api/sessions`
- `POST /api/sessions/{id}/batches`
- `POST /api/sessions/{id}/extract-all-questions`
- `GET /api/tasks`
- `GET /api/sessions/{id}/restore-page?view=restore`

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
