import SwiftUI

enum iPadInputMode { case text, voice }

// iPad 学习工作台：左栏相机舞台 + 智能观察；右栏对话流 + 底部统一输入条
// （拍题 / 文字 / 语音 三合一，右下角切换语音⇄文字）。复用 iPhone 已验证的
// 相机管线、语音按钮、回答气泡，仅重排为 iPad 双栏。

struct iPadLearnView: View {
    @ObservedObject var state: AppState
    @Environment(\.openURL) private var openURL

    @State private var draft = ""
    @State private var inputMode: iPadInputMode = .voice   // #3 默认语音（右下角大麦克风）
    @State private var previewAttachment: ChatAttachment?
    @State private var contextDetail: ContextBadgeItem?
    @State private var showContextWorkspace = false
    @State private var showActivity = false
    @FocusState private var composerFocused: Bool

    var body: some View {
        NavigationStack {
            HStack(spacing: 0) {
                captureColumn
                    .frame(width: 320)
                    .background(Color(.secondarySystemBackground))
                Divider()
                conversationColumn
                    .frame(maxWidth: .infinity)
            }
            .navigationTitle("学习")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarContent }
        }
        .task {
            // 仅在真正空白时才新建对话；从历史「进入对话」已回放消息(chatMessages 非空)，不要清掉
            if state.sessionId == nil && state.chatMessages.isEmpty { state.startNewConversation() }
            #if DEBUG
            state.debugSeedChatIfRequested()
            #endif
        }
        .onDisappear {
            if !state.isBursting { state.hideInlineCameraPreview() }
        }
        .sheet(item: $previewAttachment) { iPadAttachmentPreview(attachment: $0) }
        .sheet(item: $contextDetail) { iPadContextDetailSheet(item: $0) }
        .sheet(isPresented: $showActivity) {
            ActivityLogSheet(state: state) { showActivity = false }
        }
        .sheet(isPresented: $showContextWorkspace) {
            // 两端共用同一上下文面板（MUST-8：iPad 不做三栏 inspector）。
            ContextWorkspaceSheet(state: state, draft: draft) {
                showContextWorkspace = false
            }
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        }
    }

    // MARK: 工具栏

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .navigationBarTrailing) {
            // 分题问答 / 批改作业：拍照→盖透明浮层→点一道题→只分析那道（共享核心，对齐 iPhone）
            Button {
                state.beginQuestionSegmentation(grading: false)
            } label: {
                Label("分题问答", systemImage: "rectangle.split.3x1")
            }
            Button {
                state.beginQuestionSegmentation(grading: true)
            } label: {
                Label("批改作业", systemImage: "checkmark.rectangle.stack")
            }
            Button {
                state.beginQuestionExtraction()
            } label: {
                Label("题目提取", systemImage: "doc.text.viewfinder")
            }
            // 动态（活动日志），对齐 iPhone 的「动态」(#2)。偏好已移到 设置·一句话辅导偏好(#3 移除旧菜单)
            Button {
                showActivity = true
            } label: {
                Label("动态", systemImage: "clock")
            }
            Button {
                state.startNewConversation()
                draft = ""
            } label: {
                Label("新对话", systemImage: "square.and.pencil")
            }
        }
    }

    // MARK: 左栏 — 相机 + 智能观察

    private var captureColumn: some View {
        VStack(spacing: 16) {
            cameraStage
            observationButton
            if let notice = state.observationStopNotice, !state.isBursting {
                iPadObservationPostActions(state: state, notice: notice)
            }
            if state.isBursting {
                observationTicker   // #7 智能观察实时滚动文字（对齐 iPhone 的「直播」动态感）
            }
            statusLine
            Spacer(minLength: 0)
        }
        .padding(20)
    }

    // #7 智能观察时的实时「弹幕」：随每一帧分析滚动刷新最近动态（数据源 state.logs，与 iPhone 同源）。
    private var observationTicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "dot.radiowaves.left.and.right")
                    .foregroundStyle(.red)
                Text("实时观察")
                    .font(.caption.weight(.semibold))
                Spacer()
                if !state.uploadState.isEmpty && state.uploadState != "待机" {
                    Text(state.uploadState)
                        .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 5) {
                        let lines = Array(state.logs.suffix(20))
                        if lines.isEmpty {
                            Text("正在观察画面…")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(lines) { line in
                                Text(line.text)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(line.id)
                            }
                        }
                        Color.clear.frame(height: 1).id("ticker-bottom")
                    }
                    .padding(.vertical, 2)
                }
                .frame(height: 132)
                .onChange(of: state.logs.count) { _ in
                    withAnimation(.easeOut(duration: 0.18)) { proxy.scrollTo("ticker-bottom", anchor: .bottom) }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private var cameraStage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(Color.black)
            if state.cameraPreviewVisible {
                CameraView(state: state)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                LiveQuestionRegionOverlay(
                    imageSize: state.observationQuestionOverlayImageSize,
                    candidates: state.observationQuestionOverlayCandidates,
                    visible: state.observationQuestionOverlayVisible
                )
                .clipShape(RoundedRectangle(cornerRadius: 18))
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 40))
                        .foregroundStyle(.white.opacity(0.7))
                    Text("点击下方相机按钮拍题")
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            if state.isBursting {
                VStack {
                    HStack {
                        Label("智能观察中", systemImage: "dot.radiowaves.left.and.right")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(.red.opacity(0.85), in: Capsule())
                            .foregroundStyle(.white)
                        Spacer()
                    }
                    Spacer()
                }
                .padding(12)
            }
        }
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
        .frame(maxWidth: .infinity)
    }

    private var observationButton: some View {
        Button {
            if state.isBursting { state.stopBurst() } else { state.startBurst() }
        } label: {
            Label(state.isBursting ? "停止观察" : "智能观察",
                  systemImage: state.isBursting ? "stop.circle.fill" : "eye")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
        }
        .buttonStyle(.bordered)
        .tint(state.isBursting ? .red : .accentColor)
    }

    private var statusLine: some View {
        HStack(spacing: 8) {
            Image(systemName: state.qaSystemImage).foregroundStyle(.tint)
            Text(state.uploadState == "待机" ? state.qaStateText : state.uploadState)
                .font(.subheadline).foregroundStyle(.secondary).lineLimit(2)
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: 右栏 — 对话流 + 底部统一输入条

    private var conversationColumn: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        if state.chatMessages.isEmpty && !state.isThinking {
                            emptyConversationHint
                        }
                        ForEach(state.chatMessages) { message in
                            chatRow(message).id(message.id)
                        }
                        if state.isThinking {
                            thinkingRow.id("thinking-indicator")
                        }
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 18)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: state.chatMessages.count) { _ in scrollToBottom(proxy) }
                .onChange(of: state.isThinking) { thinking in if thinking { scrollToBottom(proxy) } }
            }
            Divider()
            composer
        }
        .background(Color(.systemGroupedBackground))
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation {
            if state.isThinking {
                proxy.scrollTo("thinking-indicator", anchor: .bottom)
            } else if let last = state.chatMessages.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }

    private var emptyConversationHint: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("开始学习").font(.title3.weight(.semibold))
            Text("拍下题目，或在下方输入 / 按住说话提问。AI 会结合你的学习目标、错题和记忆来辅导。")
                .font(.callout).foregroundStyle(.secondary)
        }
        .padding(.vertical, 40)
    }

    // 思考等待器（#5）
    private var thinkingRow: some View {
        HStack(alignment: .center, spacing: 10) {
            ProgressView()
            Text("AI 正在思考你的问题…")
                .font(.callout).foregroundStyle(.secondary)
            Spacer(minLength: 24)
        }
        .padding(12)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 10))
    }

    private var latestAssistantId: UUID? {
        state.chatMessages.last(where: { $0.role == .assistant })?.id
    }

    // 仅在「有题目/有实质回答」时展示举一反三/加入错题本/形成记忆/可视化（#7）。
    // actionable 由后端按问题意图判定（闲聊如 hi/谢谢 = false），避免给闲聊也弹一堆学习按钮（#9）。
    private func messageHasActions(_ m: ChatMessage) -> Bool {
        m.actionable && (m.visualizationCandidate || m.text.count >= 50)
    }

    @ViewBuilder
    private func chatRow(_ message: ChatMessage) -> some View {
        switch message.role {
        case .user:
            VStack(alignment: .trailing, spacing: 6) {
                UserTextBubble(text: message.text)
                if !message.attachments.isEmpty {
                    AttachmentStrip(attachments: message.attachments) { previewAttachment = $0 }
                }
            }
        case .assistant:
            let acts = messageHasActions(message)
            AssistantAnswerBubble(
                answer: message.text,
                visualizationCandidate: message.visualizationCandidate,
                visualizationReason: message.visualizationReason,
                visualization: message.visualization,
                showFollowUpActions: message.id == latestAssistantId && acts,
                showAnswerActions: acts,
                onQuickFollowUp: { state.submitQuickFollowUp($0) },
                onAddMistake: { state.addLatestAnswerToMistakeBook() },
                onFormMemory: { state.formMemoryFromLatestAnswer() },
                onSmartCapture: { state.smartCaptureFromMessage(message) },
                onGenerateVisualization: { state.generateVisualization(for: message) },
                onOpenVisualization: { viz in if let url = viz.absoluteURL { openURL(url) } }
            )
        case .status:
            if !message.contextItems.isEmpty {
                iPadContextStatusRow(
                    message: message,
                    onTap: { contextDetail = $0 },
                    onSelectAttachment: { previewAttachment = $0 },
                    onOpenWorkspace: { showContextWorkspace = true }
                )
            } else {
                AssistantTextBubble(
                    title: message.title ?? "状态",
                    text: message.text,
                    systemImage: message.systemImage ?? "info.circle",
                    showsProgress: message.showsProgress
                )
            }
        }
    }

    // MARK: 底部统一输入条（拍题 / 文字 / 语音 三合一，右下角切换）（#4）

    private var composer: some View {
        HStack(spacing: 12) {
            if inputMode == .text {
                // 拍题
                Button {
                    if state.cameraPreviewVisible {
                        state.performCameraPrimaryAction()
                    } else {
                        state.openSingleCaptureCamera()
                    }
                } label: {
                    Image(systemName: "camera.fill").font(.title3).frame(width: 44, height: 44)
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("ipad-capture")

                TextField("输入问题，或先拍题…", text: $draft, axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.plain)
                    .accessibilityIdentifier("ipad-composer-field")
                    .focused($composerFocused)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .frame(maxWidth: .infinity)
                    .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    .contentShape(Rectangle())
                    .onSubmit(send)
                Button(action: send) {
                    Image(systemName: "paperplane.fill").font(.title3).frame(width: 44, height: 44)
                }
                .buttonStyle(.borderedProminent)
                .clipShape(Circle())
                .accessibilityIdentifier("ipad-composer-send")
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || state.isThinking)
            } else {
                // 语音模式：占满整条，按钮更大更好按
                VoiceHoldArea(state: state, allowsCollapse: false, pressHeight: 52)
                    .frame(maxWidth: .infinity)
            }

            // 右下角：语音⇄文字 切换
            Button {
                composerFocused = false
                withAnimation { inputMode = inputMode == .text ? .voice : .text }
            } label: {
                Image(systemName: inputMode == .text ? "mic.fill" : "keyboard")
                    .font(.title3)
                    .frame(width: 44, height: 52)
            }
            .buttonStyle(.bordered)
            .tint(inputMode == .voice ? .accentColor : .secondary)
            .accessibilityIdentifier("ipad-input-toggle")
        }
        .padding(16)
        .background(.bar)
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !state.isThinking else { return }
        state.submitTypedQuestion(text)
        draft = ""
        composerFocused = false
    }
}

struct iPadObservationPostActions: View {
    @ObservedObject var state: AppState
    let notice: ObservationStopNotice

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "icloud.and.arrow.up")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.tint)
                    .frame(width: 20)
                Text("本轮观察")
                    .font(.caption.weight(.semibold))
                Spacer(minLength: 0)
            }

            Text(notice.message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 7) {
                Button {
                    state.generateObservationReportNow()
                } label: {
                    Label("报告", systemImage: "doc.text.magnifyingglass")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .accessibilityIdentifier("ipad-observation-report-action")

                Button {
                    state.extractAllObservationQuestions()
                } label: {
                    Label("题目", systemImage: "rectangle.stack.badge.plus")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .accessibilityIdentifier("ipad-observation-questions-action")

                Button {
                    state.extractObservationMistakes()
                } label: {
                    Label("错题", systemImage: "book.closed")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .accessibilityIdentifier("ipad-observation-mistakes-action")
            }
            .font(.caption.weight(.semibold))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator).opacity(0.42), lineWidth: 1)
        )
        .accessibilityIdentifier("ipad-observation-stop-actions")
    }
}

// MARK: - 上下文：本次携带的上下文（可点击查看）（#6）

struct iPadContextStatusRow: View {
    let message: ChatMessage
    let onTap: (ContextBadgeItem) -> Void
    var onSelectAttachment: ((ChatAttachment) -> Void)? = nil
    var onOpenWorkspace: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: message.systemImage ?? "tray.full")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.tint)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(message.title ?? "随本次发送")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if onOpenWorkspace != nil {
                        Button {
                            onOpenWorkspace?()
                        } label: {
                            Label("查看上下文", systemImage: "tray.full")
                                .font(.caption2.weight(.semibold))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.tint)
                        .accessibilityLabel("打开上下文面板")
                    }
                }
                if !message.text.isEmpty {
                    Text(message.text)
                        .font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // #5 随本次发送的抓拍缩略图（iPad 之前漏渲染，只有 iPhone 显示）。
                if !message.attachments.isEmpty {
                    AttachmentStrip(attachments: message.attachments) { onSelectAttachment?($0) }
                }
                let columns = [GridItem(.adaptive(minimum: 110), spacing: 8)]
                LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                    ForEach(message.contextItems) { item in
                        Button { onTap(item) } label: {
                            HStack(spacing: 5) {
                                Image(systemName: item.systemImage).font(.caption2)
                                Text(item.title).font(.caption.weight(.medium)).lineLimit(1)
                                Image(systemName: "chevron.right").font(.system(size: 9))
                            }
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(item.tone.color.opacity(0.12), in: Capsule())
                            .foregroundStyle(item.tone.color)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(11)
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            Spacer(minLength: 24)
        }
    }
}

struct iPadContextDetailSheet: View {
    let item: ContextBadgeItem
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Label(item.title, systemImage: item.systemImage)
                        .font(.headline)
                        .foregroundStyle(item.tone.color)
                    Text((item.fullDetail ?? item.detail).isEmpty ? "（无更多详情）" : (item.fullDetail ?? item.detail))
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(20)
            }
            .navigationTitle("上下文详情")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}

// MARK: - 附件预览

struct iPadAttachmentPreview: View {
    let attachment: ChatAttachment
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                if let image = attachment.image {
                    Image(uiImage: image).resizable().scaledToFit()
                } else if let url = attachment.fullURL ?? attachment.thumbnailURL {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let img): img.resizable().scaledToFit()
                        case .failure: Image(systemName: "photo").foregroundStyle(.white)
                        default: ProgressView().tint(.white)
                        }
                    }
                } else {
                    Text(attachment.detail).foregroundStyle(.white)
                }
            }
            .navigationTitle(attachment.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}
