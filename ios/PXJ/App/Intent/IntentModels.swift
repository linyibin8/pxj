import Foundation

// 二期·自然语言配置管家 — 数据模型（与后端 intent_router 的 route/confirm/undo 契约一一对应）。
// 真源仍在 iOS 本地（coachPreferenceText / contextInclusionSettings）；后端只做编排，不持业务真源。

// MARK: - 分类 / 阶段

enum IntentCategory: String, Decodable {
    case preference        // 改辅导偏好（真源 coachPreferenceText）
    case contextToggle     // 调上下文开关（真源 contextInclusionSettings）

    // 后端用下划线风格：preference / context_toggle
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        switch raw {
        case "preference": self = .preference
        case "context_toggle", "contextToggle": self = .contextToggle
        default:
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                                                    debugDescription: "unknown category \(raw)"))
        }
    }
}

enum IntentPhase {
    case proposed   // 卡片刚弹出，待确认
    case applying   // 确认中（confirm 请求在途）
    case applied    // 已落本地真源，露“撤销”
    case undoing    // 撤销中
    case undone     // 已撤销（终态）
}

// MARK: - 预览 diff 行

struct IntentDiffRow: Decodable, Identifiable, Equatable {
    let label: String
    let before: String
    let after: String

    var id: String { label + before + after }
}

// MARK: - 提案（对应 route 响应里的 proposal）

struct IntentProposal: Identifiable, Decodable, Equatable {
    let id: String
    let category: IntentCategory
    let title: String
    let summary: String
    let diff: [IntentDiffRow]
    let confirmLabel: String
    let reversible: Bool
    let expiresAt: String?

    enum CodingKeys: String, CodingKey {
        case id, category, title, summary, diff, reversible
        case confirmLabel = "confirm_label"
        case expiresAt = "expires_at"
    }

    static func == (lhs: IntentProposal, rhs: IntentProposal) -> Bool { lhs.id == rhs.id }
}

// MARK: - route 响应

struct IntentRouteResponse: Decodable {
    let intentKind: String              // "qa" | "config" | "module"
    let proposal: IntentProposal?
    let needsClarification: Bool?
    let clarification: String?
    let module: String?                 // P4：intentKind=="module" 时 "grade" | "answer"

    enum CodingKeys: String, CodingKey {
        case proposal, clarification, module
        case intentKind = "intent_kind"
        case needsClarification = "needs_clarification"
    }

    var isConfig: Bool { intentKind == "config" }
    var isModule: Bool { intentKind == "module" }
}

// MARK: - confirm / undo 回吐的真源目标值（echo_state）

struct EchoState: Decodable {
    let category: IntentCategory
    let coachPreferenceText: String?
    let contextInclusion: [String: Bool]?

    enum CodingKeys: String, CodingKey {
        case category
        case coachPreferenceText = "coach_preference_text"
        case contextInclusion = "context_inclusion"
    }
}

struct IntentConfirmResponse: Decodable {
    let ok: Bool
    let proposalId: String
    let undoable: Bool
    let echoState: EchoState

    enum CodingKeys: String, CodingKey {
        case ok, undoable
        case proposalId = "proposal_id"
        case echoState = "echo_state"
    }
}

struct IntentUndoResponse: Decodable {
    let ok: Bool
    let proposalId: String
    let status: String
    let echoState: EchoState

    enum CodingKeys: String, CodingKey {
        case ok, status
        case proposalId = "proposal_id"
        case echoState = "echo_state"
    }
}
