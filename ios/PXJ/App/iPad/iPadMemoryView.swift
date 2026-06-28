import SwiftUI

// iPad 记忆：长期记忆画像 + 记忆事件流。复用 AppState.refreshMemoryDigest /
// memoryProfileText / memoryEvents（来自 /api/memory）。

struct iPadMemoryView: View {
    @ObservedObject var state: AppState

    // 三期：「记忆」tab 升级为可编辑的「我的学习档案」（两端共享 LearningProfileView）。
    // 顶部 refresh 仍可刷新服务端画像摘要；档案页打开时自动拉持久记忆。
    var body: some View {
        NavigationStack {
            LearningProfileView(state: state)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button {
                            Task { await state.refreshMemoryDigest(force: true) }
                        } label: {
                            if state.isLoadingMemoryDigest {
                                ProgressView()
                            } else {
                                Image(systemName: "arrow.clockwise")
                            }
                        }
                    }
                }
        }
        .task { await state.refreshMemoryDigest() }
    }
}
