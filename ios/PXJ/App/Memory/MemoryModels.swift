import Foundation

// 三期·滚动记忆数字人 — 解码模型（隔离在独立文件，仿二期 Intent/ 范式）。
//
// 后端契约（docs/PHASE3_SPEC.md §4）：
//   GET  /api/memory/agent          -> { memories:[AgentMemory], stats:{...} }
//   GET  /api/memory/deltas         -> { deltas:[MemoryDelta] }
//   POST /api/memory/deltas/seen    -> { updated:Int }
//   PATCH /api/memory/agent/{id}    -> { memory: AgentMemory } | 409 capacity|duplicate | 500
//
// 中文展示文案（"这次更了解你了：新增 N · 更新 M"）全部在前端拼，不来自后端（C6）。

// MARK: - 持久记忆（agent_memories 的一行）

struct AgentMemory: Identifiable, Decodable, Equatable {
    let id: String
    let kind: String
    let text: String
    let importance: Double
    let status: String
    let useCount: Int
    let createdAt: String
    let updatedAt: String
    let lastUsedAt: String

    enum CodingKeys: String, CodingKey {
        case id, kind, text, importance, status
        case useCount = "use_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case lastUsedAt = "last_used_at"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = (try? c.decode(String.self, forKey: .kind)) ?? "fact"
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        importance = (try? c.decode(Double.self, forKey: .importance)) ?? 0.5
        status = (try? c.decode(String.self, forKey: .status)) ?? "active"
        useCount = (try? c.decode(Int.self, forKey: .useCount)) ?? 0
        createdAt = (try? c.decode(String.self, forKey: .createdAt)) ?? ""
        updatedAt = (try? c.decode(String.self, forKey: .updatedAt)) ?? ""
        lastUsedAt = (try? c.decode(String.self, forKey: .lastUsedAt)) ?? ""
    }

    /// 内存内构造（PATCH 返回后局部更新用，避免整列重拉）。
    init(id: String, kind: String, text: String, importance: Double, status: String,
         useCount: Int, createdAt: String, updatedAt: String, lastUsedAt: String) {
        self.id = id; self.kind = kind; self.text = text; self.importance = importance
        self.status = status; self.useCount = useCount; self.createdAt = createdAt
        self.updatedAt = updatedAt; self.lastUsedAt = lastUsedAt
    }

    var isActive: Bool { status == "active" }
    var isMistake: Bool { kind == "mistake" }

    /// 与 ContentView.RetrievedMemory.kindLabel 同义（两文件不互相可见，故各持一份）。
    var kindLabel: String {
        switch kind {
        case "preference": return "偏好"
        case "mistake": return "易错"
        case "goal": return "目标"
        case "habit": return "习惯"
        default: return "事实"
        }
    }

    /// 复制并替换 text（纠正后乐观更新列表用）。
    func withText(_ newText: String) -> AgentMemory {
        AgentMemory(id: id, kind: kind, text: newText, importance: importance, status: status,
                    useCount: useCount, createdAt: createdAt, updatedAt: updatedAt, lastUsedAt: lastUsedAt)
    }

    func withStatus(_ newStatus: String) -> AgentMemory {
        AgentMemory(id: id, kind: kind, text: text, importance: importance, status: newStatus,
                    useCount: useCount, createdAt: createdAt, updatedAt: updatedAt, lastUsedAt: lastUsedAt)
    }
}

struct AgentMemoryListResponse: Decodable {
    let memories: [AgentMemory]
}

struct AgentMemoryPatchResponse: Decodable {
    let memory: AgentMemory?
}

// MARK: - 每轮记忆增量（memory_deltas 的一行）

enum MemoryOp: String {
    case add
    case update

    static func parse(_ raw: String) -> MemoryOp { MemoryOp(rawValue: raw) ?? .add }
}

struct MemoryDelta: Identifiable, Decodable, Equatable {
    let id: String
    let memoryId: String
    let op: String
    let kind: String
    let text: String
    let qaEventId: String
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, op, kind, text
        case memoryId = "memory_id"
        case qaEventId = "qa_event_id"
        case createdAt = "created_at"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        memoryId = (try? c.decode(String.self, forKey: .memoryId)) ?? ""
        op = (try? c.decode(String.self, forKey: .op)) ?? "add"
        kind = (try? c.decode(String.self, forKey: .kind)) ?? "fact"
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        qaEventId = (try? c.decode(String.self, forKey: .qaEventId)) ?? ""
        createdAt = (try? c.decode(String.self, forKey: .createdAt)) ?? ""
    }

    var parsedOp: MemoryOp { MemoryOp.parse(op) }
}

struct MemoryDeltaListResponse: Decodable {
    let deltas: [MemoryDelta]
}

// MARK: - 一批增量（驱动对话内 chip；中文展示文案前端现拼）

struct MemoryDeltaBatch: Equatable {
    let deltas: [MemoryDelta]

    var ids: [String] { deltas.map { $0.id } }
    var memoryIds: [String] { deltas.map { $0.memoryId }.filter { !$0.isEmpty } }

    var adds: Int { deltas.filter { $0.parsedOp == .add }.count }
    var updates: Int { deltas.count - adds }

    /// chip 主标题：永远是"这次更了解你了"。
    var headline: String { "这次更了解你了" }

    /// 副标题：新增/更新计数（C6：前端现拼，可本地化）。
    var summaryLine: String {
        var parts: [String] = []
        if adds > 0 { parts.append("新增 \(adds) 条") }
        if updates > 0 { parts.append("更新 \(updates) 条") }
        return parts.isEmpty ? "已更新你的学习档案" : parts.joined(separator: " · ")
    }

    /// 副标题第二行：取最重要（取第一条，后端按 created_at DESC 已排序）一条的 text。
    var detailLine: String {
        guard let first = deltas.first else { return "" }
        let label = AgentMemory(id: "", kind: first.kind, text: "", importance: 0, status: "active",
                                useCount: 0, createdAt: "", updatedAt: "", lastUsedAt: "").kindLabel
        return "「\(label)」\(first.text)"
    }
}
