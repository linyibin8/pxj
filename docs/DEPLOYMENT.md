# 知进拍学部署文档

## 当前生产拓扑

```text
用户浏览器 / iPhone / iPad
  -> https://pxj.evowit.com
  -> 腾讯云广州 VPS Nginx: ubuntu@100.64.0.8 / 159.75.178.237
  -> Tailscale 内网 HTTP
  -> ydz@100.64.0.13:/home/ydz/services/pxj
  -> Docker Compose pxj container, host port 8038, container port 8028
```

## 后端部署

在服务端目录：

```bash
cd /home/ydz/services/pxj
docker compose up -d --build
docker compose ps
curl -fsS http://127.0.0.1:8038/health
```

生产 compose 关键配置：

- `PXJ_PUBLIC_BASE_URL=https://pxj.evowit.com`
- `PXJ_LLM_BASE_URL=http://100.64.0.5:39000/v1`
- `PXJ_LLM_MODEL=evowit-agent27b`
- `PXJ_AUTH_REQUIRED=true`
- `PXJ_REGISTRATION_ENABLED=true`

## 智能观察提题灰度

发布步骤：

1. 先部署后端，确认 `/api/sessions/{id}/extract-all-questions` 可排入 `question_extraction_session` 任务。
2. 再发布 iOS TestFlight，验证观察中实时题框、停止后按钮和任务浮层。
3. 灰度期优先给内部测试员开放，观察 `/api/tasks`、`llm_usage_events` 和日志中的 `source=extract`。
4. 若后台队列堆积或实时问答变慢，回滚 iOS 入口或临时关闭用户引导，后端任务仍可安全保留。

线上观测重点：

- `question_extraction_session` 的 queued/running/failed 数量。
- `final_report` 与 `vision_analysis` 的平均等待时间。
- `题目提取完成` 日志中的处理张数、新增题数、low_quality 数。
- 用户点击“错题”后错题候选数量，不应自动大量导入正式错题本。

## 域名与 Nginx

DNS：

- 域名：`pxj.evowit.com`
- 记录类型：A
- 记录值：`159.75.178.237`

Nginx 配置文件：

```text
/etc/nginx/sites-available/pxj.evowit.com
/etc/nginx/sites-enabled/pxj.evowit.com
```

检查命令：

```bash
sudo nginx -t
sudo systemctl reload nginx
curl -fsS https://pxj.evowit.com/health
```

证书由 Certbot 管理：

```bash
sudo certbot --nginx -d pxj.evowit.com --non-interactive --agree-tos --register-unsafely-without-email --redirect
```

## iOS/TestFlight

发布主机：`macstar@100.64.0.6`

项目目录：`/Users/macstar/Code/PXJ`

发布前必须先读远端发布手册，不打印任何 secret、env、p8、p12、mobileprovision 或 keychain 密码。

```bash
/Users/macstar/testflight-auto/fastlane-ios-oneclick/bin/oneclick-ios preflight --project-dir /Users/macstar/Code/PXJ
/Users/macstar/testflight-auto/fastlane-ios-oneclick/bin/oneclick-ios publish --project-dir /Users/macstar/Code/PXJ
/Users/macstar/testflight-auto/fastlane-ios-oneclick/bin/oneclick-ios status --project-dir /Users/macstar/Code/PXJ
```

当前 App Store Connect 信息：

- Bundle ID: `com.linyibin8.pxj`
- App Store name: `知进拍学 EvoWit`
- iOS display name: `知进拍学`
- TestFlight group: `PXJ Internal`

## GitHub

新仓库应使用 `git@github.com:linyibin8/pxj.git` 作为 `origin`。原项目 remote 只作为历史来源，不应继续作为 push remote。
