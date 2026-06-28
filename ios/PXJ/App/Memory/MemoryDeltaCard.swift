import SwiftUI

// 三期·滚动记忆数字人 — 对话内「这次更了解你了」增量 chip。
//
// 非模态、非弹窗（不打断语音播报）：监听 state.lastTurnMemoryDelta，淡入一个可点 chip；
// 点击 → present LearningProfileView(highlightIds: 本批 memoryId) 定位刚学到的条目。
// lastTurnMemoryDelta == nil（本轮无增量）→ 完全不渲染。可容忍重复展示（M8 幂等 UI）。

struct MemoryDeltaCard: View {
    @ObservedObject var state: AppState

    @State private var showProfile: Bool = false
    @State private var appeared: Bool = false

    var body: some View {
        Group {
            if let batch = state.lastTurnMemoryDelta {
                chip(batch)
                    .opacity(appeared ? 1 : 0)
                    .offset(y: appeared ? 0 : 6)
                    .onAppear {
                        withAnimation(.easeOut(duration: 0.35)) { appeared = true }
                    }
                    .sheet(isPresented: $showProfile) {
                        NavigationStack {
                            LearningProfileView(state: state, highlightIds: batch.memoryIds)
                        }
                    }
            }
        }
    }

    private func chip(_ batch: MemoryDeltaBatch) -> some View {
        Button {
            showProfile = true
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkles")
                    .font(.subheadline)
                    .foregroundStyle(.tint)
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(batch.headline)
                            .font(.subheadline.weight(.semibold))
                        Text(batch.summaryLine)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if !batch.detailLine.isEmpty {
                        Text(batch.detailLine)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 4)
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.accentColor.opacity(0.10), in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color.accentColor.opacity(0.25), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(batch.headline)，\(batch.summaryLine)，点击查看学习档案")
    }
}
