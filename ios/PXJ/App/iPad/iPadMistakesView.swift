import SwiftUI

// iPad 错题本：左栏错题列表（精简：题面 + 学科 + 待复习），右栏错题卡片（清晰分区，
// 只显示非空字段，作答↔正确答案对比）。数据源 = 复习队列（无需新增后端接口）。

struct iPadMistakesView: View {
    @ObservedObject var state: AppState
    var onReview: () -> Void = {}
    @State private var selectedId: String?
    @State private var working = false

    // 单一数据源：全部未掌握错题（loadMistakeBook 拉取，due_only=false），按是否到期在前端分组，避免重复。
    private var allItems: [ReviewMistakeItem] { state.recentReviewItems }
    private var dueItems: [ReviewMistakeItem] { allItems.filter { $0.isDue } }
    private var laterItems: [ReviewMistakeItem] { allItems.filter { !$0.isDue } }
    private var selected: ReviewMistakeItem? { allItems.first { $0.id == selectedId } }

    var body: some View {
        NavigationStack {
            HStack(spacing: 0) {
                list
                    .frame(width: 360)
                Divider()
                detailPane
                    .frame(maxWidth: .infinity)
            }
            .navigationTitle("错题本")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        Task { await state.loadMistakeBook() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .task { await state.loadMistakeBook() }
        .onChange(of: selectedId) { _ in state.mistakeActionMessage = "" }
    }

    // MARK: 列表（精简）

    private var list: some View {
        Group {
            if state.isLoadingMistakeBook && allItems.isEmpty {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if allItems.isEmpty {
                ContentUnavailableCompat(title: "暂无错题", systemImage: "checkmark.seal",
                                         message: "学习中点「加入错题本」后会出现在这里。")
            } else {
                List(selection: $selectedId) {
                    if !dueItems.isEmpty {
                        Section("待复习 (\(dueItems.count))") {
                            ForEach(dueItems, id: \.id) { row($0) }
                        }
                    }
                    if !laterItems.isEmpty {
                        Section("全部错题 (\(laterItems.count))") {
                            ForEach(laterItems, id: \.id) { row($0) }
                        }
                    }
                }
                .listStyle(.sidebar)
            }
        }
        .background(Color(.secondarySystemBackground))
    }

    private func rowTitle(_ item: ReviewMistakeItem) -> String {
        let t = item.title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { return t }
        let q = item.questionText.trimmingCharacters(in: .whitespacesAndNewlines)
        return q.isEmpty ? "未命名错题" : q
    }

    private func row(_ item: ReviewMistakeItem) -> some View {
        HStack(spacing: 8) {
            if item.isDue {
                Circle().fill(Color.orange).frame(width: 7, height: 7)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(rowTitle(item))
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
                if !item.subject.isEmpty {
                    Text(item.subject).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
        .tag(item.id)
    }

    // MARK: 详情卡片（清晰分区）

    private var detailPane: some View {
        Group {
            if let item = selected {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        header(item)
                        actionBar(item)
                        provenanceCard(item)
                        if !item.questionText.trimmedIsEmpty {
                            labeledCard("题目", systemImage: "doc.text", text: item.questionText)
                        }
                        answerComparison(item)
                        if !item.errorReason.trimmedIsEmpty {
                            labeledCard("错误原因", systemImage: "exclamationmark.triangle",
                                        text: item.errorReason, tint: .orange)
                        }
                        if !item.correction.trimmedIsEmpty {
                            labeledCard("订正建议", systemImage: "checkmark.circle",
                                        text: item.correction, tint: .green)
                        }
                        if !item.nextAction.trimmedIsEmpty {
                            labeledCard("复习建议", systemImage: "lightbulb",
                                        text: item.nextAction, tint: .blue)
                        }
                        if !item.knowledgePoints.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("知识点").font(.headline)
                                FlowTags(tags: item.knowledgePoints)
                            }
                        }
                    }
                    .padding(24)
                    .frame(maxWidth: 820, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                ContentUnavailableCompat(title: "选择一道错题", systemImage: "sidebar.right",
                                         message: "在左侧选择一道错题查看详情。")
            }
        }
        .background(Color(.systemGroupedBackground))
    }

    private func header(_ item: ReviewMistakeItem) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(rowTitle(item)).font(.title2.weight(.bold))
            HStack(spacing: 8) {
                if !item.subject.isEmpty { chip(item.subject, .blue) }
                if !item.locationRef.isEmpty { chip(item.locationRef, .gray) }
                if let conf = item.provenance?.confidenceLabel, !conf.isEmpty { chip(conf, .purple) }
                if item.isDue { chip("待复习", .orange) }
                chip("已复习 \(item.reviewCount) 次", .green)
            }
        }
    }

    // 复习 / 消灭 / 保留：怎么用、怎么消灭都从这里发起；掌握后由后端移出复习队列。
    private func actionBar(_ item: ReviewMistakeItem) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if !state.mistakeActionMessage.isEmpty {
                Label(state.mistakeActionMessage, systemImage: "checkmark.seal.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.green)
            }
            HStack(spacing: 10) {
                Button {
                    Task { await state.startReviewForItem(item); onReview() }
                } label: {
                    Label("复习这道题", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                Button {
                    guard !working else { return }
                    working = true
                    Task { await state.markMistakeMastered(item.id); working = false }
                } label: {
                    Label("我已掌握·消灭", systemImage: "checkmark.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.green)

                Button {
                    guard !working else { return }
                    working = true
                    Task { await state.keepMistakeForReview(item.id); working = false }
                } label: {
                    Label("还会错·保留", systemImage: "clock.arrow.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.orange)
            }
            .disabled(working || state.isPreparingReview)
        }
    }

    // 这道错题"怎么来的"：何时 / 怎么采集 / 为什么 / 判定依据。让来源透明、不再像是凭空冒出来。
    @ViewBuilder
    private func provenanceCard(_ item: ReviewMistakeItem) -> some View {
        if let p = item.provenance {
            VStack(alignment: .leading, spacing: 8) {
                Label("这道错题怎么来的", systemImage: "info.circle").font(.headline).foregroundStyle(.secondary)
                if !p.whenDisplay.isEmpty { provRow("生成时间", p.whenDisplay, "clock") }
                if !p.methodLabel.isEmpty { provRow("采集方式", p.methodLabel, "camera.viewfinder") }
                if !p.confidenceLabel.isEmpty { provRow("当前状态", p.confidenceLabel, "flag") }
                if !p.why.isEmpty { provRow("判定理由", p.why, "questionmark.circle") }
                if !p.basis.isEmpty { provRow("判定依据", p.basis, "magnifyingglass") }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
        }
    }

    private func provRow(_ label: String, _ value: String, _ icon: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon).font(.caption).foregroundStyle(.tint).frame(width: 18)
            Text(label).font(.caption.weight(.semibold)).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
            Text(value).font(.caption).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private func answerComparison(_ item: ReviewMistakeItem) -> some View {
        let hasStudent = !item.studentAnswer.trimmedIsEmpty
        let hasExpected = !item.expectedAnswer.trimmedIsEmpty
        return Group {
            if hasStudent || hasExpected {
                HStack(alignment: .top, spacing: 12) {
                    if hasStudent {
                        answerBox("我的作答", item.studentAnswer, tint: .red)
                    }
                    if hasExpected {
                        answerBox("正确答案", item.expectedAnswer, tint: .green)
                    }
                }
            }
        }
    }

    private func answerBox(_ title: String, _ text: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(tint)
            Text(text).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }

    private func labeledCard(_ title: String, systemImage: String, text: String, tint: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage).font(.headline).foregroundStyle(tint)
            Text(text).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private func chip(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

private extension String {
    var trimmedIsEmpty: Bool { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

// 自适应标签流（知识点）
struct FlowTags: View {
    let tags: [String]
    var body: some View {
        let columns = [GridItem(.adaptive(minimum: 80), spacing: 8)]
        LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
            ForEach(tags, id: \.self) { t in
                Text(t)
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(Color.accentColor.opacity(0.12), in: Capsule())
                    .foregroundStyle(Color.accentColor)
            }
        }
    }
}
