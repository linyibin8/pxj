import Foundation
import SwiftUI

// ContentView.NetworkRequestError 是 file-private，跨文件不可见，故本文件自带等价错误类型。
enum IntentNetworkError: Error {
    case missingHTTPResponse
    case badStatus(Int, String)
}

// 二期·自然语言配置管家 — AppState 端编排逻辑（隔离在独立文件，最小污染 ContentView）。
//
// 流程：submitTypedQuestion → maybeInterceptAsIntent →（本地预筛命中才调 route）
//   route 命中 config → 弹卡片；否则降级为普通 QA。confirm → applyEchoState 落唯一真源。
//
// 真源不破：preference 经 coachPreferenceTextDidChange、contextToggle 经 updateContextInclusion。
//
// 网络：ContentView 的 postJSON/serverBaseURL 是 file-private，跨文件不可见，
// 故此处用 Auth.swift 暴露的 authServerBaseURL + AuthSession.shared.authHeader 复刻同一请求形态
// （Bearer 鉴权、JSON、状态码校验），不改动 ContentView 的网络层。

extension AppState {

    // MARK: - 拦截入口（只在文本路径调用；语音/复习/双击/quick-followup 不调用）

    /// 返回 true 表示已作为配置意图处理（弹卡片），调用方应 return、不进 QA。
    /// 任何失败/超时/非配置一律返回 false → 调用方照常走 QA。
    @MainActor
    func maybeInterceptAsIntent(_ question: String) async -> Bool {
        let text = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return false }

        // 本地预筛：纯提问直接放行不走 route（省一次 LLM 往返的延迟税）。
        // P4：配置类 或 批改/分题类 任一命中才付往返。
        guard localPrefilterLooksLikeConfig(text) || localPrefilterLooksLikeModule(text) else { return false }

        // 仅疑似配置才付往返；短 timeout，失败/超时降级为 QA。
        intentRouteInFlight = true
        defer { intentRouteInFlight = false }

        let payload: [String: Any] = [
            "text": text,
            "prefilter_hit": true,
            "app_state": [
                "coach_preference_text": coachPreferenceText,
                "context_inclusion": contextInclusionSnapshot(),
            ],
        ]

        do {
            let data = try await intentPostJSON(path: "/api/intent/route", payload: payload, timeout: 12)
            let resp = try JSONDecoder().decode(IntentRouteResponse.self, from: data)
            // P4：自动进功能模块（作业批改 / 拍照分题答题），并记下原话供「其实只想问」一键纠偏。
            if resp.isModule, let module = resp.module {
                autoRouteRevertText = text
                appendUserChatMessage(text)
                log(module == "grade" ? "识别为作业批改，进入取景批改" : "识别为拍照分题，进入取景答题")
                beginQuestionSegmentation(grading: module == "grade")
                return true
            }
            guard resp.isConfig, let proposal = resp.proposal else {
                // intent_kind == "qa" 或 needs_clarification（无卡片）→ 照常走 QA。
                if resp.needsClarification == true, let msg = resp.clarification, !msg.isEmpty {
                    log("配置意图需澄清：\(msg)")
                }
                return false
            }
            pendingIntentProposal = proposal
            intentPhase = .proposed
            pendingIntentOriginalQuestion = text
            return true
        } catch {
            // 网络/超时/解析失败：不拦截，降级为普通 QA。
            return false
        }
    }

    /// 轻量本地关键词/句式预筛：不命中 → 完全不调 route。
    /// 命中只是“疑似配置”，最终仍由后端 LLM + 0.6 阈值 + 白名单裁决。
    func localPrefilterLooksLikeConfig(_ text: String) -> Bool {
        let t = text
        // 偏好类信号
        let prefSignals = ["以后", "从现在起", "讲题", "讲解", "辅导", "口气", "风格", "啰嗦",
                           "简洁", "提示", "别老", "别再", "不要再", "记住我喜欢", "偏好"]
        // 上下文开关类信号（配合白名单词）
        let toggleVerbs = ["打开", "开启", "关掉", "关闭", "停用", "启用", "别用", "用上", "带上"]
        let toggleNouns = ["上下文", "记忆", "长期记忆", "错题", "历史", "知识点", "画面", "观察", "策略", "调试"]

        for s in prefSignals where t.contains(s) { return true }
        let hasVerb = toggleVerbs.contains { t.contains($0) }
        let hasNoun = toggleNouns.contains { t.contains($0) }
        if hasVerb && hasNoun { return true }
        return false
    }

    /// P4 预筛：疑似「作业批改 / 拍照分题答题」。命中才付 route 往返（最终由后端 LLM 裁决）。
    func localPrefilterLooksLikeModule(_ text: String) -> Bool {
        let gradeCues = ["批改", "检查", "对不对", "对吗", "对了吗", "错没错", "改完", "判一下", "判对错", "看看对错", "有没有错", "做错没"]
        let segmentCues = ["分一下", "分题", "这几道题", "这些题", "好几道题", "几道题", "整页", "这一页", "这页题", "拍照解"]
        if gradeCues.contains(where: { text.contains($0) }) { return true }
        if segmentCues.contains(where: { text.contains($0) }) { return true }
        return false
    }

    /// P4 一键纠偏：从「自动进模块」退回——取消取景，把原话当普通提问发出（跳过再次拦截，防环）。
    @MainActor
    func revertAutoRouteToQA() {
        let original = autoRouteRevertText ?? ""
        autoRouteRevertText = nil
        cancelAiming()
        dismissSegmentation()
        let text = original.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        proceedTypedQuestion(text)   // 逃生口：跳过意图拦截，直接当普通提问
    }

    // MARK: - 确认 / 撤销 / 收尾

    @MainActor
    func confirmProposal() async {
        guard let proposal = pendingIntentProposal, intentPhase == .proposed else { return }
        intentPhase = .applying
        do {
            let data = try await intentPostJSON(path: "/api/intent/confirm",
                                                payload: ["proposal_id": proposal.id],
                                                timeout: 12)
            let resp = try JSONDecoder().decode(IntentConfirmResponse.self, from: data)
            guard resp.ok else { intentPhase = .proposed; return }
            // 先落本地真源（成功才认为应用完成）。
            applyEchoState(resp.echoState)
            lastAppliedProposal = proposal
            intentPhase = .applied
        } catch {
            log("配置确认失败，请重试")
            intentPhase = .proposed
        }
    }

    @MainActor
    func undoLastIntent() async {
        // 严格只认当前卡片上的 pendingIntentProposal（applied 态下它仍在）；
        // 不回退到 lastAppliedProposal，避免对已收尾/已 dismiss 的旧提案误发 undo。
        guard let proposal = pendingIntentProposal,
              intentPhase == .applied else { return }
        intentPhase = .undoing
        let payload: [String: Any] = [
            "proposal_id": proposal.id,
            "current_app_state": [
                "coach_preference_text": coachPreferenceText,
                "context_inclusion": contextInclusionSnapshot(),
            ],
        ]
        do {
            let data = try await intentPostJSON(path: "/api/intent/undo", payload: payload, timeout: 12)
            let resp = try JSONDecoder().decode(IntentUndoResponse.self, from: data)
            guard resp.ok else { intentPhase = .applied; return }
            // 写回旧值成功后才置终态（写回失败=未撤销，本地即真源，无永久漂移）。
            applyEchoState(resp.echoState)
            intentPhase = .undone
        } catch let IntentNetworkError.badStatus(code, _) where code == 409 {
            // state_drifted：用户在别处改过，拒绝盲目回写。
            log("设置已变化，未执行撤销")
            intentPhase = .applied
        } catch {
            log("撤销失败，请重试")
            intentPhase = .applied
        }
    }

    /// 把 echo_state 写到唯一真源（不新建并行状态）。
    @MainActor
    func applyEchoState(_ echo: EchoState) {
        switch echo.category {
        case .preference:
            // 真源：coachPreferenceText + UserDefaults，经唯一写入口。
            coachPreferenceTextDidChange(echo.coachPreferenceText ?? "")
        case .contextToggle:
            // 真源：必须经 updateContextInclusion，禁止直接赋 contextInclusionSettings。
            guard let ci = echo.contextInclusion else { return }
            for (key, value) in ci {
                if let kp = Self.contextKeyPath(for: key) {
                    updateContextInclusion(kp, to: value)
                }
            }
        }
    }

    @MainActor
    func dismissProposal() {
        pendingIntentProposal = nil
        lastAppliedProposal = nil          // 一并清，避免残留对已收尾提案误发 undo
        intentPhase = .proposed
        pendingIntentOriginalQuestion = nil
    }

    /// 误分类逃生：关掉卡片并把原文当普通提问处理。
    @MainActor
    func askInsteadOfConfig() {
        let original = pendingIntentOriginalQuestion
        dismissProposal()
        if let q = original, !q.isEmpty {
            proceedTypedQuestion(q)
        }
    }

    // MARK: - 辅助

    /// 当前上下文开关快照（白名单 8 项）。
    func contextInclusionSnapshot() -> [String: Bool] {
        let s = contextInclusionSettings
        return [
            "visual": s.visual,
            "observation": s.observation,
            "history": s.history,
            "mistakes": s.mistakes,
            "knowledge": s.knowledge,
            "memory": s.memory,
            "strategy": s.strategy,
            "debug": s.debug,
        ]
    }

    static func contextKeyPath(for key: String) -> WritableKeyPath<ContextInclusionSettings, Bool>? {
        switch key {
        case "visual": return \.visual
        case "observation": return \.observation
        case "history": return \.history
        case "mistakes": return \.mistakes
        case "knowledge": return \.knowledge
        case "memory": return \.memory
        case "strategy": return \.strategy
        case "debug": return \.debug
        default: return nil
        }
    }

    /// 复刻 ContentView.postJSON 的请求形态（其本体 file-private 不可跨文件复用）。
    private func intentPostJSON(path: String, payload: [String: Any], timeout: TimeInterval) async throws -> Data {
        var request = URLRequest(url: authServerBaseURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        if let auth = AuthSession.shared.authHeader {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw IntentNetworkError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw IntentNetworkError.badStatus(http.statusCode, body)
        }
        return data
    }
}
