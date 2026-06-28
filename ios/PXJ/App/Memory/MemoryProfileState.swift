import Foundation
import SwiftUI

// 三期·滚动记忆数字人 — AppState 端编排（隔离在独立文件，最小污染 ContentView）。
//
// 网络：ContentView 的 getData/postForm 是 file-private，跨文件不可见，故此处自带
// memoryRequest helper，复刻同一请求形态（Bearer 鉴权、JSON、状态码校验、401 处理），
// 但额外支持 GET/PATCH/POST（IntentRouting 的 helper 仅 POST）。

// ContentView.NetworkRequestError / IntentNetworkError 均 file-private/同名占用，
// 故本文件自带等价错误类型。
enum MemoryNetworkError: Error {
    case missingHTTPResponse
    case badStatus(Int, String)
}

// 状态属性（agentMemories / lastTurnMemoryDelta / isLoadingAgentMemories /
// memoryUndoToastVisible 为 @Published；lastMutatedMemory / currentMemoryUndoToastToken
// 为纯内部 var）均声明在 ContentView.swift 的 AppState 类体上，赋值即自动触发 @Published
// 通知与 SwiftUI 精确依赖追踪。本扩展只持有编排方法。
extension AppState {

    // MARK: - 拉取

    /// 档案页打开时拉持久记忆（后端只返回 active）。
    @MainActor
    func loadAgentMemories() async {
        guard !isLoadingAgentMemories else { return }
        isLoadingAgentMemories = true
        defer { isLoadingAgentMemories = false }
        do {
            let data = try await memoryRequest(path: "/api/memory/agent?limit=300", method: "GET")
            let resp = try JSONDecoder().decode(AgentMemoryListResponse.self, from: data)
            agentMemories = resp.memories
        } catch {
            log("学习档案读取失败：\(memoryErrorText(error))", level: "error")
        }
    }

    /// 被动拉取本轮记忆增量：无延迟、无定时器、无重试（C4）。命中即填 chip 并标记 seen。
    @MainActor
    func pullMemoryDeltas() async {
        do {
            let data = try await memoryRequest(path: "/api/memory/deltas?unseen_only=1&limit=50", method: "GET")
            let resp = try JSONDecoder().decode(MemoryDeltaListResponse.self, from: data)
            guard !resp.deltas.isEmpty else { return }   // 本轮无增量 → 静默
            lastTurnMemoryDelta = MemoryDeltaBatch(deltas: resp.deltas)
            // best-effort 去抖（M8）：标记 seen，失败不影响展示。
            let ids = resp.deltas.map { $0.id }
            Task { await markDeltasSeen(ids) }
        } catch {
            // 静默失败：下一轮 / 重开会话再被动补显。
        }
    }

    @MainActor
    private func markDeltasSeen(_ ids: [String]) async {
        guard !ids.isEmpty else { return }
        do {
            _ = try await memoryRequest(path: "/api/memory/deltas/seen", method: "POST",
                                        payload: ["ids": ids])
        } catch {
            // 去抖非幂等保证；失败下次拉取可能重复展示，UI 容忍（M8）。
        }
    }

    /// 用户主动关闭 chip / 跳转后清空，避免重复展示。
    @MainActor
    func dismissMemoryDelta() {
        lastTurnMemoryDelta = nil
    }

    // MARK: - 纠正 / 软删 / 撤销

    /// 纠正一条记忆的文本（后端同事务 reembed，失败返回 500）。
    @MainActor
    func correctMemory(id: String, text: String) async {
        let newText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !newText.isEmpty, let old = agentMemories.first(where: { $0.id == id }) else { return }
        do {
            let data = try await memoryRequest(path: "/api/memory/agent/\(id)", method: "PATCH",
                                               payload: ["text": newText])
            applyPatched(id: id, data: data, fallback: old.withText(newText))
            lastMutatedMemory = .text(id: id, previousText: old.text)
            showUndoToast()
        } catch let MemoryNetworkError.badStatus(code, _) where code == 500 {
            log("记忆纠正失败（重嵌入未完成），请重试", level: "error")
        } catch {
            log("记忆纠正失败：\(memoryErrorText(error))", level: "error")
        }
    }

    /// 软删一条记忆（status=superseded）。下一轮检索 WHERE active 自动排除。
    @MainActor
    func deleteMemory(id: String) async {
        guard let old = agentMemories.first(where: { $0.id == id }) else { return }
        do {
            let data = try await memoryRequest(path: "/api/memory/agent/\(id)", method: "PATCH",
                                               payload: ["status": "superseded"])
            applyPatched(id: id, data: data, fallback: old.withStatus("superseded"))
            lastMutatedMemory = .status(id: id, previousStatus: old.status)
            showUndoToast()
        } catch {
            log("记忆删除失败：\(memoryErrorText(error))", level: "error")
        }
    }

    /// 5s 吐司窗口内撤销最近一次变更（复用 PATCH，无独立 /restore；C5）。
    @MainActor
    func undoMemoryMutation() async {
        guard let snapshot = lastMutatedMemory else { return }
        memoryUndoToastVisible = false
        lastMutatedMemory = nil
        do {
            switch snapshot {
            case let .status(id, previousStatus):
                // 删除的撤销 = 恢复为 active；active 前后端会先跑 restore_guard（409=容量/重复）。
                let data = try await memoryRequest(path: "/api/memory/agent/\(id)", method: "PATCH",
                                                   payload: ["status": previousStatus])
                applyPatched(id: id, data: data, fallback: nil)
            case let .text(id, previousText):
                let data = try await memoryRequest(path: "/api/memory/agent/\(id)", method: "PATCH",
                                                   payload: ["text": previousText])
                applyPatched(id: id, data: data, fallback: nil)
            }
        } catch let MemoryNetworkError.badStatus(code, body) where code == 409 {
            let reason = body.contains("duplicate") ? "已存在类似记忆" : "记忆已满"
            log("无法撤销：\(reason)", level: "error")
        } catch {
            log("撤销失败：\(memoryErrorText(error))", level: "error")
        }
    }

    // MARK: - 内部

    @MainActor
    private func applyPatched(id: String, data: Data, fallback: AgentMemory?) {
        var updated = fallback
        if let resp = try? JSONDecoder().decode(AgentMemoryPatchResponse.self, from: data),
           let mem = resp.memory {
            updated = mem
        }
        guard let memory = updated else {
            // 无法解析也无 fallback：保险起见重拉。
            Task { await loadAgentMemories() }
            return
        }
        if let idx = agentMemories.firstIndex(where: { $0.id == id }) {
            var list = agentMemories
            list[idx] = memory
            agentMemories = list
        } else {
            agentMemories = agentMemories + [memory]
        }
    }

    @MainActor
    private func showUndoToast() {
        memoryUndoToastVisible = true
        let token = UUID()
        currentMemoryUndoToastToken = token
        Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            await MainActor.run {
                if self.currentMemoryUndoToastToken == token {
                    self.memoryUndoToastVisible = false
                }
            }
        }
    }

    private func memoryErrorText(_ error: Error) -> String {
        switch error {
        case let MemoryNetworkError.badStatus(code, _): return "HTTP \(code)"
        case MemoryNetworkError.missingHTTPResponse: return "无响应"
        default: return (error as NSError).localizedDescription
        }
    }

    // MARK: - 网络 helper（支持 GET / PATCH / POST；复刻 ContentView 网络层形态）

    @discardableResult
    func memoryRequest(path: String, method: String, payload: [String: Any]? = nil,
                       timeout: TimeInterval = 15) async throws -> Data {
        var request = URLRequest(url: authServerBaseURL.appending(path: path))
        request.httpMethod = method
        request.timeoutInterval = timeout
        if let auth = AuthSession.shared.authHeader {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        if let payload {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MemoryNetworkError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw MemoryNetworkError.badStatus(http.statusCode, body)
        }
        return data
    }
}

/// 可撤销变更的快照（删除恢复 status / 纠正恢复 text）。
enum MemoryMutationSnapshot {
    case status(id: String, previousStatus: String)
    case text(id: String, previousText: String)
}
