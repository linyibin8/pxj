# 知进拍学需求文档

## 产品定位

知进拍学是面向学生的拍照学习助手。iPhone/iPad 客户端负责拍题、整页批改、题目提取、语音提问和复习；后端负责账号、多学生档案、图片与题目存储、大模型解析、长期记忆、错题与可观测日志。

线上入口：`https://pxj.evowit.com`

## 目标用户

- 学生：拍照提问、听讲解、复习错题。
- 家长或老师：查看学习过程、错题、报告和长期画像。
- 运维/开发：在 Web dashboard 中观察日志、任务队列、模型健康和数据状态。

## MVP 范围

- 原生 iPhone/iPad App，显示名为“知进拍学”，bundle id 为 `com.linyibin8.pxj`。
- 全新后端服务，域名 `pxj.evowit.com`，数据目录独立于原项目。
- OpenAI-compatible 大模型接口，默认模型 `evowit-agent27b`。
- 账号注册/登录、JWT 鉴权、多学生档案。
- 拍照上传、整页分析、题目分割、单题问答、批改和重排版。
- 错题候选、复习队列、长期记忆、学习资产浏览。
- 智能观察实时题目候选框：观察中用端上分题结果框选题目并显示题目指纹短码，停止后可异步生成报告、提取所有题目、提取错题候选。
- Web dashboard、提示词管理、日志与任务状态。
- TestFlight 发布，支持 iPhone 和 iPad。

## 非目标

- 不复用旧域名、旧后端容器、旧 iOS bundle id 或旧 GitHub 仓库。
- 不在仓库中提交云 API 密钥、App Store Connect key、p12、p8、mobileprovision、keychain 密码。
- 不引入多 worker/Redis 架构，当前以单实例后端和 SQLite 账户分库为主。

## 验收标准

- `https://pxj.evowit.com/health` 返回 `{"ok": true, "service": "pxj"}`。
- 公网 HTTPS 通过广州 VPS 反代到 `100.64.0.13:8038`。
- 新后端可注册账号、登录、创建会话，`/health/llm` 可连通默认大模型。
- iOS App 使用新域名、新 bundle id、新显示名，iPhone/iPad generic build 通过。
- TestFlight 中存在 `com.linyibin8.pxj` 的新 App 和有效 build。
- GitHub 中存在新的 pxj 仓库，remote 不再指向原项目。
- 文档覆盖需求、架构、开发、部署、测试与流程。
- 智能观察专项验收：观察时题框与 `fp` 短码能实时出现；同题重复帧被合并；停止后“报告 / 题目 / 错题”按钮都进入后台队列并有可见状态；题集还原页可从同一会话的 `question_set` 渲染。

## 观察提题成功指标

- 同一题跨多帧重复进入题集的比例低于 5%。
- 有效关键图整轮提题成功率不低于 85%，低质量/无关画面必须有跳过原因。
- 停止观察后点击“题目”的任务排队反馈小于 1 秒，后台任务不影响实时问答优先级。
- 错题候选只进入候选区，学生确认导入后才进入正式错题本和复习队列。
