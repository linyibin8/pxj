import SwiftUI

// iPad 历史：中栏会话列表 + 右栏报告详情（三栏观感：侧栏 | 列表 | 报告）。
// 复用 AppState.refreshHistorySessions / historyReport(for:) / startConversationFromHistory。

struct iPadHistoryView: View {
    @ObservedObject var state: AppState
    var onContinue: () -> Void = {}
    @State private var selectedId: String?
    @State private var report: HistoryReportDetail?
    @State private var loadingReport = false

    private var selectedSession: HistorySessionSummary? {
        state.historySessions.first { $0.id == selectedId }
    }

    var body: some View {
        NavigationStack {
            HStack(spacing: 0) {
                sessionList
                    .frame(width: 360)
                Divider()
                reportPane
                    .frame(maxWidth: .infinity)
            }
            .navigationTitle("学习历史")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        Task { await state.refreshHistorySessions() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .task { await state.refreshHistorySessions() }
        .onChange(of: selectedId) { _ in loadReport() }
    }

    private var sessionList: some View {
        Group {
            if state.isLoadingHistory && state.historySessions.isEmpty {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if state.historySessions.isEmpty {
                ContentUnavailableCompat(title: "暂无历史", systemImage: "clock", message: "完成一次学习后会出现在这里。")
            } else {
                List(selection: $selectedId) {
                    ForEach(state.historySessions) { session in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                Text(session.modeBadge)
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(
                                        (session.modeBadge == "实时对话" ? Color.accentColor : Color.orange).opacity(0.15),
                                        in: Capsule()
                                    )
                                    .foregroundStyle(session.modeBadge == "实时对话" ? Color.accentColor : Color.orange)
                                Text(session.displayTitle)
                                    .font(.subheadline.weight(.semibold))
                                    .lineLimit(2)
                            }
                            Text(session.countSummary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(session.displayTime)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                        .padding(.vertical, 2)
                        .accessibilityIdentifier("history-session-row-\(session.id)")
                        .tag(session.id)
                    }
                }
                .accessibilityIdentifier("ipad-history-list")
                .listStyle(.plain)
            }
        }
        .background(Color(.secondarySystemBackground))
    }

    private var reportPane: some View {
        Group {
            if let session = selectedSession {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(report?.title ?? session.displayTitle)
                                .font(.title2.weight(.bold))
                            if let subtitle = report?.subtitle, !subtitle.isEmpty {
                                Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
                            }
                            if !session.studentGoal.isEmpty {
                                Label(session.studentGoal, systemImage: "target")
                                    .font(.footnote).foregroundStyle(.secondary)
                            }
                        }

                        if loadingReport {
                            ProgressView("正在生成报告…").padding(.vertical, 20)
                        } else if let report {
                            if !report.content.isEmpty {
                                sectionCard(title: "学习报告", systemImage: report.systemImage, body: report.content)
                            }
                            if !report.qaRounds.isEmpty {
                                // #5 每个回合一张卡，标题=问题第一句，便于快速回看。
                                VStack(alignment: .leading, spacing: 10) {
                                    Label("问答回顾", systemImage: "bubble.left.and.bubble.right")
                                        .font(.headline)
                                    ForEach(Array(report.qaRounds.enumerated()), id: \.element.id) { idx, round in
                                        roundCard(index: idx + 1, round: round)
                                    }
                                }
                            } else if !report.qaPreview.isEmpty {
                                sectionCard(title: "问答回顾", systemImage: "bubble.left.and.bubble.right", body: report.qaPreview)
                            }
                        } else if !session.summaryPreview.isEmpty {
                            sectionCard(title: "概要", systemImage: "doc.text", body: session.summaryPreview)
                        }

                        Button {
                            Task {
                                await state.startConversationFromHistory(session)
                                await MainActor.run { onContinue() }
                            }
                        } label: {
                            Label("进入对话 · 接着这次学习继续", systemImage: "arrow.uturn.forward")
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .accessibilityIdentifier("history-continue-button")
                    }
                    .padding(24)
                    .frame(maxWidth: 820, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                ContentUnavailableCompat(title: "选择一条历史", systemImage: "sidebar.right", message: "在左侧选择一次学习记录查看报告。")
            }
        }
        .accessibilityIdentifier("history-report-pane")
        .background(Color(.systemGroupedBackground))
    }

    // #5 单个问答回合卡：序号 + 第一句标题，展开问/答全文。
    private func roundCard(index: Int, round: HistoryQARound) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("\(index)")
                    .font(.caption.weight(.bold).monospacedDigit())
                    .foregroundStyle(.white)
                    .frame(width: 22, height: 22)
                    .background(Color.accentColor, in: Circle())
                Text(round.title.isEmpty ? "第 \(index) 回合" : round.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
            }
            if !round.question.isEmpty {
                Text("问：\(round.question)")
                    .font(.footnote).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !round.answer.isEmpty {
                Text("答：\(round.answer)")
                    .font(.footnote).foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private func sectionCard(title: String, systemImage: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            Text(body)
                .font(.callout)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private func loadReport() {
        report = nil
        guard let session = selectedSession else { return }
        loadingReport = true
        Task {
            let detail = await state.historyReport(for: session)
            await MainActor.run {
                self.report = detail
                self.loadingReport = false
            }
        }
    }
}

// MARK: - 占位（兼容 iOS 16，无 ContentUnavailableView）

struct ContentUnavailableCompat: View {
    let title: String
    let systemImage: String
    let message: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(title)
                .font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
