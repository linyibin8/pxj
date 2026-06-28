import SwiftUI

// 三期·滚动记忆数字人 — 两端共享「我的学习档案」页。
//
// 画像卡（复用 memoryProfileText/UpdatedAt，与 iPadMemoryView 同源）
// + 持久记忆按 kind 分组（偏好/目标/习惯/事实直出，易错 mistake 默认折叠/二级 section，M7）
// + 每条可纠正/删除 + 5s 撤销吐司
// + highlightIds 命中条目高亮闪烁（从对话内增量 chip 跳入定位）。

struct LearningProfileView: View {
    @ObservedObject var state: AppState
    var highlightIds: [String] = []

    @State private var editingMemory: AgentMemory? = nil
    @State private var editingText: String = ""
    @State private var showMistakes: Bool = false
    @State private var flashOn: Bool = false

    private var activeMemories: [AgentMemory] { state.agentMemories.filter { $0.isActive } }

    /// 分组顺序：偏好 → 目标 → 习惯 → 事实（易错单独二级 section）。
    private let visibleKinds: [(kind: String, label: String, icon: String)] = [
        ("preference", "学习偏好", "slider.horizontal.3"),
        ("goal", "学习目标", "target"),
        ("habit", "学习习惯", "repeat"),
        ("fact", "其它事实", "info.circle"),
    ]

    private func memories(of kind: String) -> [AgentMemory] {
        activeMemories.filter { $0.kind == kind }
            .sorted { $0.importance > $1.importance }
    }

    private var mistakeMemories: [AgentMemory] {
        activeMemories.filter { $0.isMistake }.sorted { $0.importance > $1.importance }
    }

    var body: some View {
        ZStack(alignment: .top) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    profileCard
                    if state.isLoadingAgentMemories && activeMemories.isEmpty {
                        ProgressView().frame(maxWidth: .infinity).padding(.top, 40)
                    } else if activeMemories.isEmpty {
                        emptyState
                    } else {
                        ForEach(visibleKinds, id: \.kind) { group in
                            let items = memories(of: group.kind)
                            if !items.isEmpty {
                                section(title: group.label, icon: group.icon, items: items)
                            }
                        }
                        if !mistakeMemories.isEmpty {
                            mistakeSection
                        }
                    }
                }
                .padding(20)
                .frame(maxWidth: 880, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(.systemGroupedBackground))

            if state.memoryUndoToastVisible {
                undoToast.transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .navigationTitle("我的学习档案")
        .navigationBarTitleDisplayMode(.inline)
        .task { await state.loadAgentMemories() }
        .task { await flashHighlight() }
        .sheet(item: $editingMemory) { mem in
            correctSheet(mem)
        }
    }

    // MARK: - 画像卡（复用真源）

    private var profileCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("学习画像", systemImage: "person.text.rectangle")
                .font(.headline)
            if state.memoryProfileText.isEmpty {
                Text("尚未形成画像。多问几道题，AI 会在这里沉淀对你的了解。")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                Text(state.memoryProfileText)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !state.memoryProfileUpdatedAt.isEmpty {
                Text("更新于 \(state.memoryProfileUpdatedAt)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 16))
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "brain")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("还没有记住关于你的事")
                .font(.headline)
            Text("每轮问答后，AI 会自动记住你的偏好、目标和易错点，并在这里展示，你随时可纠正或删除。")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - 分组 section

    private func section(title: String, icon: String, items: [AgentMemory]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("\(title)（\(items.count)）", systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(items) { mem in
                memoryRow(mem)
            }
        }
    }

    // 易错（mistake）：隐私敏感，默认折叠/单独二级 section（M7）。
    private var mistakeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation { showMistakes.toggle() }
            } label: {
                HStack {
                    Label("易错点（\(mistakeMemories.count)）", systemImage: "exclamationmark.triangle")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: showMistakes ? "chevron.up" : "chevron.down")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tertiary)
                }
            }
            .buttonStyle(.plain)
            if showMistakes {
                Text("以下是 AI 观察到的薄弱点，仅你可见。")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                ForEach(mistakeMemories) { mem in
                    memoryRow(mem)
                }
            }
        }
    }

    // MARK: - 单条

    private func memoryRow(_ mem: AgentMemory) -> some View {
        let highlighted = highlightIds.contains(mem.id) && flashOn
        return HStack(alignment: .top, spacing: 10) {
            Image(systemName: "smallcircle.filled.circle")
                .font(.caption)
                .foregroundStyle(.tint)
                .padding(.top, 3)
            Text(mem.text)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 4)
            Menu {
                Button {
                    editingMemory = mem
                    editingText = mem.text
                } label: { Label("纠正", systemImage: "pencil") }
                Button(role: .destructive) {
                    Task { await state.deleteMemory(id: mem.id) }
                } label: { Label("删除", systemImage: "trash") }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            (highlighted ? Color.accentColor.opacity(0.18) : Color(.systemBackground)),
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(highlighted ? Color.accentColor : .clear, lineWidth: 1.5)
        )
        .animation(.easeInOut(duration: 0.35), value: highlighted)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            Button(role: .destructive) {
                Task { await state.deleteMemory(id: mem.id) }
            } label: { Label("删除", systemImage: "trash") }
            Button {
                editingMemory = mem
                editingText = mem.text
            } label: { Label("纠正", systemImage: "pencil") }
            .tint(.blue)
        }
    }

    // MARK: - 纠正弹层

    private func correctSheet(_ mem: AgentMemory) -> some View {
        NavigationStack {
            Form {
                Section("纠正这条记忆") {
                    TextField("记忆内容", text: $editingText, axis: .vertical)
                        .lineLimit(3...6)
                }
                Section {
                    Text("纠正后，AI 在下一轮会按新内容理解你（同时重建语义索引）。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle(mem.kindLabel)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { editingMemory = nil }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") {
                        let id = mem.id
                        let text = editingText
                        editingMemory = nil
                        Task { await state.correctMemory(id: id, text: text) }
                    }
                    .disabled(editingText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }

    // MARK: - 撤销吐司

    private var undoToast: some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.uturn.backward.circle.fill")
                .foregroundStyle(.white)
            Text("已更新记忆")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
            Spacer()
            Button("撤销") {
                Task { await state.undoMemoryMutation() }
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.85), in: Capsule())
        .padding(.horizontal, 24)
        .padding(.top, 12)
        .shadow(radius: 8, y: 4)
    }

    // 高亮闪烁两下后落定为高亮态。
    @MainActor
    private func flashHighlight() async {
        guard !highlightIds.isEmpty else { return }
        for _ in 0..<3 {
            withAnimation { flashOn = true }
            try? await Task.sleep(nanoseconds: 450_000_000)
            withAnimation { flashOn = false }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        withAnimation { flashOn = true }
    }
}
