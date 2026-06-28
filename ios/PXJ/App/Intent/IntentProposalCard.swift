import SwiftUI

// 二期·自然语言配置管家 — 确认卡片（纯展示组件，两端共用，无布局假设）。
// iPhone 以 .sheet 呈现、iPad 以 .overlay 呈现，本视图只负责内容与回调。

struct IntentProposalCard: View {
    let proposal: IntentProposal
    let phase: IntentPhase
    var onConfirm: () -> Void = {}
    var onCancel: () -> Void = {}
    var onUndo: () -> Void = {}
    var onAskInstead: () -> Void = {}   // “我是在提问，不是配置 → 继续提问”逃生口

    private var categoryLabel: String {
        switch proposal.category {
        case .preference: return "辅导偏好"
        case .contextToggle: return "上下文开关"
        }
    }

    private var categoryColor: Color {
        switch proposal.category {
        case .preference: return .blue
        case .contextToggle: return .purple
        }
    }

    private var busy: Bool { phase == .applying || phase == .undoing }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            diffSection
            footer
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(categoryLabel)
                    .font(.caption).bold()
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(categoryColor.opacity(0.16))
                    .foregroundStyle(categoryColor)
                    .clipShape(Capsule())
                Spacer()
                if phase == .applied {
                    Label("已应用", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                } else if phase == .undone {
                    Label("已撤销", systemImage: "arrow.uturn.backward.circle.fill")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Text(proposal.title).font(.headline)
            if !proposal.summary.isEmpty {
                Text(proposal.summary).font(.subheadline).foregroundStyle(.secondary)
            }
        }
    }

    private var diffSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(proposal.diff) { row in
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.label).font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .top, spacing: 8) {
                        Text(row.before)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .strikethrough(true, color: .secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "arrow.right")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(row.after)
                            .font(.callout).bold()
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var footer: some View {
        switch phase {
        case .proposed, .applying:
            VStack(spacing: 10) {
                HStack(spacing: 12) {
                    Button(role: .cancel, action: onCancel) {
                        Text("取消").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(busy)

                    Button(action: onConfirm) {
                        if phase == .applying {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text(proposal.confirmLabel).frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(busy)
                }
                Button(action: onAskInstead) {
                    Text("我是在提问，不是配置 → 继续提问")
                        .font(.footnote)
                }
                .buttonStyle(.borderless)
                .disabled(busy)
            }
        case .applied, .undoing:
            HStack(spacing: 12) {
                Button(action: onUndo) {
                    if phase == .undoing {
                        ProgressView().frame(maxWidth: .infinity)
                    } else {
                        Label("撤销", systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(phase == .undoing || !proposal.reversible)

                Button(action: onCancel) {
                    Text("完成").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(phase == .undoing)
            }
        case .undone:
            Button(action: onCancel) {
                Text("关闭").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
        }
    }
}
