import AVFoundation
import CoreImage
import Foundation
import ImageIO
import Speech
import SwiftUI
import UIKit
import Vision

private let serverBaseURL = URL(string: "https://pxj.evowit.com")!

/// 缩略图 URL（供上下文面板的「画面」分类卡显示真实题图缩略图）。
/// 与 AppState.imageThumbnailURL 同源，但为 free function 以便 SwiftUI 视图直接调用。
func contextImageThumbnailURL(filename: String) -> URL? {
    let trimmed = filename.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
          let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return nil }
    return URL(string: "/api/images/\(encoded)/thumbnail", relativeTo: serverBaseURL)?.absoluteURL
}
private let ttsPrimarySpeechURL = URL(string: "https://ttsubuntu.evowit.com/v1/audio/speech")!
private let ttsFallbackGenerateURL = URL(string: "https://ttscc.evowit.com/api/generate")!
private let ttsFallbackDemoId = "demo-1"
private let ttsPrimaryModel = "tts-1"
private let ttsPrimaryVoice = "zf_xiaoxiao"
private let ttsPrimaryResponseFormat = "mp3"
private let ttsRequestTimeoutSeconds: TimeInterval = 6
private let ttsFallbackTimeoutSeconds: TimeInterval = 4
private let ttsDefaultSpeed: Double = 1.5
private let ttsMinimumSpeed: Double = 0.75
private let ttsMaximumSpeed: Double = 2.0
private let ttsSpeedStep: Double = 0.15
private let ttsPlaybackRateOptions: [Double] = [0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80, 1.95, 2.00]
private let ttsSegmentMaxCharacters = 72
private let physicalFrontCameraAvoidancePadding: CGFloat = 58
private let recentUserOperationPresenceWindow: TimeInterval = 120
private let voicePlaybackEnabledDefaultsKey = "pxj.voicePlaybackEnabled"
private let voicePlaybackRateDefaultsKey = "pxj.voicePlaybackRate"
private let learningModeDefaultsKey = "pxj.learningMode"
private let coachDepthDefaultsKey = "pxj.coachDepth"
private let textOnlyQuestionDefaultsKey = "pxj.textOnlyQuestion"
private let autoKeystoneCorrectionDefaultsKey = "pxj.autoKeystoneCorrection"
private let questionSegmentationEnabledDefaultsKey = "pxj.questionSegmentationEnabled"
private let coachPreferenceTextDefaultsKey = "pxj.coachPreferenceText"
private let longTermInstructionDefaultsKey = "pxj.longTermInstruction"
private let longTermMemoriesDefaultsKey = "pxj.longTermMemories"
private let userInputMemoryDefaultsKey = "pxj.userInputMemory"
private let contextIncludeVisualDefaultsKey = "pxj.context.includeVisual"
private let contextIncludeObservationDefaultsKey = "pxj.context.includeObservation"
private let contextIncludeHistoryDefaultsKey = "pxj.context.includeHistory"
private let contextIncludeMistakesDefaultsKey = "pxj.context.includeMistakes"
private let contextIncludeKnowledgeDefaultsKey = "pxj.context.includeKnowledge"
private let contextIncludeMemoryDefaultsKey = "pxj.context.includeMemory"
private let contextIncludeStrategyDefaultsKey = "pxj.context.includeStrategy"
private let contextIncludeDebugDefaultsKey = "pxj.context.includeDebug"
private let controlToken = (Bundle.main.object(forInfoDictionaryKey: "PAIControlToken") as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

private func normalizedVoicePlaybackRate(_ value: Double) -> Double {
    let raw = value > 0 ? value : ttsDefaultSpeed
    let clamped = min(max(raw, ttsMinimumSpeed), ttsMaximumSpeed)
    if ttsMaximumSpeed - clamped < ttsSpeedStep / 2 {
        return ttsMaximumSpeed
    }
    let steps = ((clamped - ttsMinimumSpeed) / ttsSpeedStep).rounded()
    let snapped = ttsMinimumSpeed + steps * ttsSpeedStep
    return min(max(snapped, ttsMinimumSpeed), ttsMaximumSpeed)
}

enum LearningModePreference: String, CaseIterable, Identifiable, Hashable {
    case singleProblem = "single_problem"
    case homework
    case answerCheck = "answer_check"
    case review

    var id: String { rawValue }

    var title: String {
        switch self {
        case .singleProblem:
            return "拍题"
        case .homework:
            return "写作业"
        case .answerCheck:
            return "检查"
        case .review:
            return "复习"
        }
    }

    var reportPhrase: String {
        switch self {
        case .singleProblem:
            return "拍一道题，识别题目、学生答案和卡点"
        case .homework:
            return "记录写作业过程，关注书写、停顿和订正"
        case .answerCheck:
            return "先核对答案和过程，再给订正方向"
        case .review:
            return "围绕错题、知识点和相似题安排复习"
        }
    }

    var focusPhrase: String {
        "学习场景=\(title)；\(reportPhrase)"
    }
}

enum CoachDepthPreference: String, CaseIterable, Identifiable, Hashable {
    case hintFirst = "hint_first"
    case stepByStep = "step_by_step"
    case checkOnly = "check_only"
    case fullExplain = "full_explain"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .hintFirst:
            return "先提示"
        case .stepByStep:
            return "分步讲"
        case .checkOnly:
            return "只检查"
        case .fullExplain:
            return "完整讲"
        }
    }

    var reportPhrase: String {
        switch self {
        case .hintFirst:
            return "先给提示和可尝试的一小步，不直接代做"
        case .stepByStep:
            return "一步步讲清推理过程"
        case .checkOnly:
            return "只判断对错和下一步订正，不展开代做"
        case .fullExplain:
            return "给完整解析、结论和巩固动作"
        }
    }

    var focusPhrase: String {
        "回答方式=\(title)；\(reportPhrase)"
    }
}

enum ContextWorkspaceTab: String, CaseIterable, Identifiable {
    case overview
    case assets
    case prompts
    case debug
    case settings

    var id: String { rawValue }

    /// 普通用户可见的页卡。#8「调试」是开发者排查用的原始 JSON，家长/学生看不懂，从分段控件里隐藏。
    static var visibleCases: [ContextWorkspaceTab] { [.overview, .assets, .prompts, .settings] }

    var title: String {
        switch self {
        case .overview:
            return "总览"
        case .assets:
            return "资产"
        case .prompts:
            return "提示"
        case .debug:
            return "调试"
        case .settings:
            return "配置"
        }
    }

    var systemImage: String {
        switch self {
        case .overview:
            return "tray.full"
        case .assets:
            return "books.vertical"
        case .prompts:
            return "text.bubble"
        case .debug:
            return "curlybraces"
        case .settings:
            return "switch.2"
        }
    }
}

struct ContextInclusionSettings: Equatable {
    var visual: Bool
    var observation: Bool
    var history: Bool
    var mistakes: Bool
    var knowledge: Bool
    var memory: Bool
    var strategy: Bool
    var debug: Bool

    static let defaults = ContextInclusionSettings(
        visual: true,
        observation: true,
        history: true,
        mistakes: true,
        knowledge: true,
        memory: true,
        strategy: true,
        debug: false
    )

    static func load() -> ContextInclusionSettings {
        let defaults = UserDefaults.standard
        return ContextInclusionSettings(
            visual: defaults.object(forKey: contextIncludeVisualDefaultsKey) as? Bool ?? Self.defaults.visual,
            observation: defaults.object(forKey: contextIncludeObservationDefaultsKey) as? Bool ?? Self.defaults.observation,
            history: defaults.object(forKey: contextIncludeHistoryDefaultsKey) as? Bool ?? Self.defaults.history,
            mistakes: defaults.object(forKey: contextIncludeMistakesDefaultsKey) as? Bool ?? Self.defaults.mistakes,
            knowledge: defaults.object(forKey: contextIncludeKnowledgeDefaultsKey) as? Bool ?? Self.defaults.knowledge,
            memory: defaults.object(forKey: contextIncludeMemoryDefaultsKey) as? Bool ?? Self.defaults.memory,
            strategy: defaults.object(forKey: contextIncludeStrategyDefaultsKey) as? Bool ?? Self.defaults.strategy,
            debug: defaults.object(forKey: contextIncludeDebugDefaultsKey) as? Bool ?? Self.defaults.debug
        )
    }

    func save() {
        let defaults = UserDefaults.standard
        defaults.set(visual, forKey: contextIncludeVisualDefaultsKey)
        defaults.set(observation, forKey: contextIncludeObservationDefaultsKey)
        defaults.set(history, forKey: contextIncludeHistoryDefaultsKey)
        defaults.set(mistakes, forKey: contextIncludeMistakesDefaultsKey)
        defaults.set(knowledge, forKey: contextIncludeKnowledgeDefaultsKey)
        defaults.set(memory, forKey: contextIncludeMemoryDefaultsKey)
        defaults.set(strategy, forKey: contextIncludeStrategyDefaultsKey)
        defaults.set(debug, forKey: contextIncludeDebugDefaultsKey)
    }

    var enabledContextLabels: [String] {
        var labels: [String] = ["问题", "偏好", "回合"]
        if visual { labels.append("画面") }
        if observation { labels.append("观察") }
        if history { labels.append("历史") }
        if mistakes { labels.append("错题") }
        if knowledge { labels.append("知识点") }
        if memory { labels.append("记忆") }
        if strategy { labels.append("策略") }
        return labels
    }
}

struct ContentView: View {
    @StateObject private var state = AppState()
    @State private var chatDraft = ""
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        ZStack {
            LandscapeLearningWorkbench(state: state, draft: $chatDraft)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if state.backgroundCameraActive && !state.cameraSheetVisible && !state.inlineCameraPreviewVisible {
                BackgroundCameraHost(state: state)
            }
        }
        .overlay(alignment: .topTrailing) {
            VStack(alignment: .trailing, spacing: 6) {
                AccountBadge()
                GenerationBanner()
            }
            .padding(.top, 6).padding(.trailing, 10)
        }
        .background(Color(.secondarySystemBackground).ignoresSafeArea())
        .task {
            state.log("App 启动，后端地址 \(serverBaseURL.absoluteString)")
            state.startDeviceControlPolling()
            await state.refreshReviewQueuePreview()
            await state.refreshMemoryDigest()
        }
        .onChange(of: state.studentGoal) { _ in
            state.studentGoalDidChange()
        }
        .onChange(of: state.coachPreferenceText) { _ in
            state.coachPreferenceDidChange()
        }
        // 二期·自然语言配置确认卡片：iPhone(compact) 走 sheet，iPad(regular) 走 overlay；卡片本体同一份。
        .sheet(isPresented: Binding(
            get: { horizontalSizeClass != .regular && state.pendingIntentProposal != nil },
            set: { if !$0 { state.dismissProposal() } }
        )) {
            if let proposal = state.pendingIntentProposal {
                intentCard(proposal)
                    .padding()
                    .presentationDetents([.medium])
                    // confirm/undo 在途时禁止下滑关闭，避免丢撤销入口、状态机与 UI 不一致（与 overlay 守护对齐）。
                    .interactiveDismissDisabled(state.intentPhase == .applying || state.intentPhase == .undoing)
            }
        }
        .overlay {
            if horizontalSizeClass == .regular, let proposal = state.pendingIntentProposal {
                ZStack {
                    Color.black.opacity(0.25).ignoresSafeArea()
                        .onTapGesture { if state.intentPhase != .applying && state.intentPhase != .undoing { state.dismissProposal() } }
                    intentCard(proposal)
                        .frame(maxWidth: 420)
                        .padding()
                }
            }
        }
        // P0：分题/批改取景层（叠在已运行的相机预览之上，复用同一会话，不另起 CameraView）
        .overlay {
            if state.captureAimingVisible {
                QuestionCaptureOverlay(state: state)
            }
        }
    }

    @ViewBuilder
    private func intentCard(_ proposal: IntentProposal) -> some View {
        IntentProposalCard(
            proposal: proposal,
            phase: state.intentPhase,
            onConfirm: { Task { await state.confirmProposal() } },
            onCancel: { state.dismissProposal() },
            onUndo: { Task { await state.undoLastIntent() } },
            onAskInstead: { state.askInsteadOfConfig() }
        )
    }
}

private struct LandscapeLearningWorkbench: View {
    @ObservedObject var state: AppState
    @Binding var draft: String
    @State private var showActivity = false
    @State private var showContext = false
    @State private var showHistory = false
    @State private var showObservationPrompt = false
    @State private var toolsExpanded = false
    @State private var selectedContextItem: ContextBadgeItem?
    @State private var selectedAttachment: ChatAttachment?
    @State private var selectedHistoryReport: HistoryReportDetail?
    @State private var composerMode = ComposerInputMode.voice
    @State private var voiceDockExpanded = false
    @State private var observationPromptClosingToTool = false
    @State private var chatIsScrolling = false
    @State private var voiceDockRevealTask: Task<Void, Never>?
    @FocusState private var composerFocused: Bool

    private var stageIsLive: Bool {
        state.inlineCameraPreviewVisible
    }

    private var voiceFocusModeActive: Bool {
        state.isListening || state.isPreparingVoiceInput
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            ImmersiveStageBackground(state: state)

            if !voiceFocusModeActive {
                messageCanvas
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .transition(.opacity)
            }

            if state.desktopIntroVisible && !voiceFocusModeActive {
                DesktopIntroPanel()
                    .padding(.horizontal, 24)
                    .padding(.bottom, 96)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    .accessibilityIdentifier("desktop-intro-panel")
            }

            if let notice = state.observationStopNotice, !voiceFocusModeActive {
                ObservationStopToast(state: state, notice: notice)
                    .padding(.horizontal, 18)
                    .padding(.bottom, 96)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                    .accessibilityIdentifier("observation-stop-toast")
            }

            bottomChatDock
                .offset(y: voiceDockHiddenForScroll ? 76 : 0)
                .opacity(voiceDockHiddenForScroll ? 0.08 : 1)
                .animation(.spring(response: 0.28, dampingFraction: 0.86), value: voiceDockHiddenForScroll)

            if toolsExpanded {
                Color.black.opacity(0.001)
                    .ignoresSafeArea()
                    .onTapGesture {
                        dismissKeyboard()
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                            toolsExpanded = false
                        }
                    }
                    .accessibilityHidden(true)
            }

            floatingToolLauncher
                .padding(.trailing, physicalFrontCameraAvoidancePadding)
                .padding(.bottom, 64)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
        }
        .background(Color(.secondarySystemBackground).ignoresSafeArea())
        .ignoresSafeArea(.container, edges: .all)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                if composerMode == .text {
                    Button("语音") {
                        toggleComposerMode()
                    }
                    .accessibilityIdentifier("keyboard-voice-toggle")
                }
                Button("收起") {
                    dismissKeyboard()
                }
                .accessibilityIdentifier("keyboard-dismiss")
            }
        }
        .sheet(isPresented: $showHistory) {
            HistorySessionSheet(
                state: state,
                onStartNew: { session in
                    showHistory = false
                    dismissKeyboard()
                    Task { await state.startConversationFromHistory(session) }
                },
                onViewReport: { session in
                    showHistory = false
                    dismissKeyboard()
                    Task {
                        if let report = await state.historyReport(for: session) {
                            selectedHistoryReport = report
                        }
                    }
                }
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showContext) {
            ContextWorkspaceSheet(state: state, draft: draft) {
                showContext = false
            }
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showActivity) {
            ActivityLogSheet(state: state) {
                showActivity = false
            }
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $state.extractResultVisible) {
            ExtractResultSheet(state: state)
        }
        .sheet(item: $selectedContextItem) { item in
            ContextDetailSheet(item: item) {
                selectedContextItem = nil
            }
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(item: $selectedHistoryReport) { report in
            HistoryReportSheet(report: report) {
                selectedHistoryReport = nil
            }
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .fullScreenCover(item: $selectedAttachment) { attachment in
            AttachmentPreviewScreen(attachment: attachment) {
                selectedAttachment = nil
            }
        }
        .fullScreenCover(isPresented: Binding(
            get: { state.segmentationVisible },
            set: { if !$0 { state.dismissSegmentation() } }
        )) {
            QuestionSegmentationSheet(state: state)
        }
        .onDisappear {
            voiceDockRevealTask?.cancel()
            voiceDockRevealTask = nil
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("immersive-workbench")
    }

    private var messageCanvas: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if state.hasChatStarted {
                        ForEach(state.chatMessages) { message in
                            ChatMessageBubble(
                                message: message,
                                isLatestAssistant: state.latestAssistantMessageId == message.id,
                                preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                                preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                                onQuickFollowUp: { state.submitQuickFollowUp($0) },
                                onAddMistake: { state.addMessageToMistakeBook($0) },
                                onFormMemory: { state.formMemoryFromMessage($0) },
                                onSmartCapture: { state.smartCaptureFromMessage($0) },
                                onGenerateVisualization: { state.generateVisualization(for: $0) },
                                onOpenVisualization: { state.openVisualization($0) },
                                onSelectContext: { selectedContextItem = $0 },
                                onSelectAttachment: { selectedAttachment = $0 },
                                onOpenWorkspace: { showContext = true }
                            )
                        }

                        // 三期：本轮记忆增量 chip（最近 assistant 气泡下方；本轮无增量则不渲染）。
                        MemoryDeltaCard(state: state)

                        if state.chatMessages.isEmpty,
                           !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            UserTextBubble(text: state.recognizedText)
                        }

                        if state.isListening,
                           !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                           state.chatMessages.last?.text != state.recognizedText {
                            UserTextBubble(text: state.recognizedText)
                        }

                        if state.questionSubmissionInFlight {
                            AssistantTextBubble(
                                title: "正在输入",
                                text: "我在整理思路，会直接把回答放到这段对话下面。",
                                systemImage: "ellipsis.bubble",
                                showsProgress: true
                            )
                        }

                        if state.chatMessages.isEmpty,
                           !state.qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            AssistantAnswerBubble(
                                answer: state.qaAnswer,
                                showFollowUpActions: !state.questionSubmissionInFlight,
                                preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                                preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                                onQuickFollowUp: { state.submitQuickFollowUp($0) },
                                onAddMistake: { state.addLatestAnswerToMistakeBook() },
                                onFormMemory: { state.formMemoryFromLatestAnswer() },
                                onSmartCapture: { state.smartCaptureFromLatestAnswer() }
                            )
                        }
                    }

                    if !state.hasChatStarted,
                       !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        UserTextBubble(text: state.recognizedText)
                    }

                    if !state.hasChatStarted,
                       !state.qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        AssistantAnswerBubble(
                            answer: state.qaAnswer,
                            showFollowUpActions: !state.questionSubmissionInFlight,
                            preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                            preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                            onQuickFollowUp: { state.submitQuickFollowUp($0) },
                            onAddMistake: { state.addLatestAnswerToMistakeBook() },
                            onFormMemory: { state.formMemoryFromLatestAnswer() },
                            onSmartCapture: { state.smartCaptureFromLatestAnswer() }
                        )
                    }

                    Color.clear
                        .frame(height: isVoiceComposerIdle ? 52 : 100)
                        .id("chat-bottom")
                }
                .padding(.horizontal, 10)
                .padding(.top, 8)
                .padding(.bottom, 4)
            }
            .background(stageIsLive ? Color.clear : Color(.secondarySystemBackground))
            .scrollDismissesKeyboard(.interactively)
            .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
            .simultaneousGesture(
                DragGesture(minimumDistance: 8)
                    .onChanged { _ in
                        handleChatScrollActivity()
                    }
                    .onEnded { _ in
                        scheduleVoiceDockReveal()
                    }
            )
            .onChange(of: state.logs.count) { _ in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .onChange(of: state.qaAnswer) { _ in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .onChange(of: state.isThinking) { _ in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var bottomChatDock: some View {
        VStack(spacing: 8) {
            if showObservationPrompt || state.observationGuideVisible {
                ObservationGestureGuidePanel(
                    state: state,
                    showsStartButton: showObservationPrompt && !state.isBursting,
                    closingToTool: observationPromptClosingToTool,
                    onStart: {
                        startObservationFromPrompt()
                    },
                    onClose: {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            showObservationPrompt = false
                            observationPromptClosingToTool = false
                            state.observationGuideVisible = false
                        }
                    }
                )
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            HStack(alignment: .bottom, spacing: 8) {
                Button {
                    toggleComposerMode()
                } label: {
                    Image(systemName: composerMode == .voice ? "keyboard" : "waveform")
                        .font(.headline)
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial.opacity(0.72), in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .contentShape(Rectangle())
                .accessibilityLabel(composerMode == .voice ? "切换文字输入" : "切换语音输入")
                .accessibilityIdentifier("composer-mode-toggle")

                if composerMode == .voice && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    VoiceHoldArea(
                        state: state,
                        allowsCollapse: false,
                        transparentSurface: true,
                        onBegin: {
                            voiceDockExpanded = true
                            collapseTransientSurfaces()
                        },
                        onCollapse: {
                            voiceDockExpanded = true
                        }
                    )
                } else {
                    TextField("追问...", text: $draft, axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...4)
                        .submitLabel(.send)
                        .focused($composerFocused)
                        .onSubmit(sendDraft)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .frame(minHeight: 38, maxHeight: 96)
                        .background(Color(.systemBackground).opacity(0.92))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color(.separator), lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .accessibilityIdentifier("chat-composer")

                    Button {
                        sendDraft()
                    } label: {
                        if state.intentRouteInFlight {
                            ProgressView()
                                .frame(width: 38, height: 38)
                        } else {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                                .frame(width: 38, height: 38)
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(state.questionSubmissionInFlight || state.intentRouteInFlight)
                    .accessibilityLabel("发送文字问题")
                    .accessibilityIdentifier("send-chat")
                }
            }
            .padding(.leading, 8)
            .padding(.trailing, 8 + physicalFrontCameraAvoidancePadding)
            .padding(.vertical, isVoiceComposerIdle ? 4 : 6)
            .frame(maxWidth: .infinity)
            .background {
                if isVoiceComposerIdle {
                    Color.clear
                } else {
                    Rectangle().fill(.regularMaterial)
                }
            }
            .overlay(alignment: .top) {
                if !isVoiceComposerIdle {
                    Rectangle()
                        .frame(height: 1)
                        .foregroundStyle(Color(.separator))
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("bottom-chat-dock")
        }
    }

    private var isVoiceComposerIdle: Bool {
        composerMode == .voice && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var voiceDockHiddenForScroll: Bool {
        chatIsScrolling && isVoiceComposerIdle && state.voiceDockCanAutoHide
    }

    private var floatingToolLauncher: some View {
        VStack(alignment: .trailing, spacing: 8) {
            if toolsExpanded {
                FloatingToolDock(
                    state: state,
                    showActivity: $showActivity,
                    showContext: $showContext,
                    showHistory: $showHistory,
                    toolsExpanded: $toolsExpanded,
                    showObservationPrompt: {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            showActivity = false
                            showContext = false
                            showHistory = false
                            observationPromptClosingToTool = false
                            showObservationPrompt = true
                            state.observationGuideVisible = false
                            toolsExpanded = false
                        }
                    },
                    dismissKeyboard: dismissKeyboard
                )
                .transition(.opacity.combined(with: .scale(scale: 0.86, anchor: .bottomTrailing)))
            }

            Button {
                dismissKeyboard()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                    toolsExpanded.toggle()
                }
            } label: {
                Image(systemName: toolsExpanded ? "xmark" : "circle.grid.3x3.fill")
                    .font(.headline.weight(.semibold))
                    .frame(width: 44, height: 44)
                    .background(Color.accentColor)
                    .foregroundStyle(.white)
                    .clipShape(Circle())
                    .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 5)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(toolsExpanded ? "收起工具" : "展开工具")
            .accessibilityIdentifier("tool-menu-toggle")
        }
    }

    private func collapseTransientSurfaces() {
        dismissKeyboard()
        withAnimation(.easeInOut(duration: 0.18)) {
            showContext = false
            showActivity = false
            showHistory = false
            toolsExpanded = false
            showObservationPrompt = false
            observationPromptClosingToTool = false
            state.observationGuideVisible = false
            selectedContextItem = nil
        }
    }

    private func startObservationFromPrompt() {
        observationPromptClosingToTool = true
        withAnimation(.spring(response: 0.22, dampingFraction: 0.9)) {
            showObservationPrompt = false
            toolsExpanded = false
        }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 160_000_000)
            observationPromptClosingToTool = false
            state.prepareForObservationStart()
            state.startBurst(showGuide: false)
        }
    }

    private func sendDraft() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // 配置探测在途也算忙：先于 draft="" 拦截，避免清空输入框却没发出去。
        guard !text.isEmpty, !state.questionSubmissionInFlight, !state.intentRouteInFlight else { return }
        draft = ""
        composerMode = .voice
        voiceDockExpanded = true
        collapseTransientSurfaces()
        dismissKeyboard()
        state.submitTypedQuestion(text)
    }

    private func toggleComposerMode() {
        if composerMode == .voice {
            withAnimation(.easeInOut(duration: 0.18)) {
                composerMode = .text
                voiceDockExpanded = true
            }
            Task { @MainActor in
                await Task.yield()
                try? await Task.sleep(nanoseconds: 80_000_000)
                composerFocused = true
            }
        } else {
            dismissKeyboard()
            draft = ""
            voiceDockExpanded = true
            Task { @MainActor in
                await Task.yield()
                withAnimation(.easeInOut(duration: 0.18)) {
                    composerMode = .voice
                }
            }
        }
    }

    private func handleChatScrollActivity() {
        if toolsExpanded {
            withAnimation(.spring(response: 0.22, dampingFraction: 0.86)) {
                toolsExpanded = false
            }
        }
        guard isVoiceComposerIdle, state.voiceDockCanAutoHide else { return }
        voiceDockRevealTask?.cancel()
        voiceDockRevealTask = nil
        if !chatIsScrolling {
            withAnimation(.easeOut(duration: 0.16)) {
                chatIsScrolling = true
            }
        }
    }

    private func scheduleVoiceDockReveal() {
        voiceDockRevealTask?.cancel()
        voiceDockRevealTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 620_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.spring(response: 0.28, dampingFraction: 0.86)) {
                chatIsScrolling = false
            }
        }
    }

    private func dismissKeyboard() {
        composerFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

private struct DesktopIntroPanel: View {
    var body: some View {
        VStack(spacing: 7) {
            Text("知进拍学 EvoWit.com")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)

            VStack(spacing: 2) {
                Text("为学习者建造通往知识的优化坡道")
                    .font(.footnote.weight(.medium))
                Text("Building optimized ramps to knowledge")
                    .font(.footnote)
            }
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)

            Text("最大化每秒顿悟次数 (Eurekas per second)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Color.accentColor)
                .multilineTextAlignment(.center)
                .padding(.top, 2)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 15)
        .frame(maxWidth: 430)
        .background(.ultraThinMaterial.opacity(0.78), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.22), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 18, x: 0, y: 8)
    }
}

private struct ObservationStopToast: View {
    @ObservedObject var state: AppState
    let notice: ObservationStopNotice

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "icloud.and.arrow.up")
                .font(.headline)
                .foregroundStyle(Color.accentColor)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 8) {
                Text("本轮观察")
                    .font(.caption.weight(.semibold))
                Text(notice.message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 7) {
                    Button {
                        state.generateObservationReportNow()
                    } label: {
                        Label("报告", systemImage: "doc.text.magnifyingglass")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)

                    Button {
                        state.extractAllObservationQuestions()
                    } label: {
                        Label("题目", systemImage: "rectangle.stack.badge.plus")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button {
                        state.extractObservationMistakes()
                    } label: {
                        Label("错题", systemImage: "book.closed")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .font(.caption.weight(.semibold))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: 520, alignment: .leading)
        .background(.ultraThinMaterial.opacity(0.82), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.24), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.12), radius: 18, x: 0, y: 7)
    }
}

private struct ImmersiveStageBackground: View {
    @ObservedObject var state: AppState

    var body: some View {
        ZStack {
            if state.inlineCameraPreviewVisible {
                CameraView(state: state)
                    .ignoresSafeArea()
                    .accessibilityIdentifier("landscape-camera-preview")

                LiveQuestionRegionOverlay(
                    imageSize: state.observationQuestionOverlayImageSize,
                    candidates: state.observationQuestionOverlayCandidates,
                    visible: state.observationQuestionOverlayVisible
                )
                .ignoresSafeArea()

                LinearGradient(
                    colors: [.clear, .black.opacity(0.12), .black.opacity(0.42)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ObservationDanmakuOverlay(state: state)
                    .allowsHitTesting(false)
            } else {
                Color(.secondarySystemBackground)
                    .ignoresSafeArea()
            }
        }
    }
}

private struct FloatingToolDock: View {
    @ObservedObject var state: AppState
    @Binding var showActivity: Bool
    @Binding var showContext: Bool
    @Binding var showHistory: Bool
    @Binding var toolsExpanded: Bool
    let showObservationPrompt: () -> Void
    let dismissKeyboard: () -> Void

    var body: some View {
        VStack(alignment: .trailing, spacing: 7) {
            FloatingToolButton(
                title: "收起",
                systemImage: "xmark",
                accessibilityLabel: "收起工具",
                identifier: "tool-menu-close"
            ) {
                dismissKeyboard()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                    toolsExpanded = false
                }
            }

            FloatingToolButton(
                title: "新对话",
                systemImage: "plus.bubble",
                accessibilityLabel: "新对话",
                identifier: "new-conversation-action"
            ) {
                dismissKeyboard()
                state.startNewConversation()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                    showActivity = false
                    showContext = false
                    showHistory = false
                    toolsExpanded = false
                }
            }

            FloatingToolButton(
                title: state.isBursting ? "停止观察" : "观察",
                systemImage: state.isBursting ? "stop.circle" : "rectangle.stack",
                accessibilityLabel: state.isBursting ? "停止观察" : "智能观察",
                identifier: "burst-action"
            ) {
                dismissKeyboard()
                if state.isBursting {
                    state.stopContinuousVoiceConversation(reason: "用户点击观察按钮停止")
                    state.stopBurst()
                    withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                        toolsExpanded = false
                    }
                } else {
                    showObservationPrompt()
                }
            }

            FloatingToolButton(
                title: "分题问答",
                systemImage: "rectangle.split.3x1",
                accessibilityLabel: "分题问答",
                identifier: "segment-qa-action"
            ) {
                dismissKeyboard()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                    showActivity = false
                    showContext = false
                    showHistory = false
                    toolsExpanded = false
                }
                state.beginQuestionSegmentation(grading: false)
            }

            FloatingToolButton(
                title: "批改作业",
                systemImage: "checkmark.rectangle.stack",
                accessibilityLabel: "批改作业",
                identifier: "segment-grade-action"
            ) {
                dismissKeyboard()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                    showActivity = false
                    showContext = false
                    showHistory = false
                    toolsExpanded = false
                }
                state.beginQuestionSegmentation(grading: true)
            }

            FloatingToolButton(
                title: "题目提取",
                systemImage: "doc.text.viewfinder",
                accessibilityLabel: "题目提取",
                identifier: "extract-action"
            ) {
                dismissKeyboard()
                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                    showActivity = false
                    showContext = false
                    showHistory = false
                    toolsExpanded = false
                }
                state.beginQuestionExtraction()
            }

            FloatingToolButton(
                title: "上下文",
                systemImage: showContext ? "tray.full.fill" : "tray.full",
                accessibilityLabel: "上下文",
                identifier: "context-action"
            ) {
                dismissKeyboard()
                withAnimation(.easeInOut(duration: 0.2)) {
                    showActivity = false
                    showContext = true
                    toolsExpanded = false
                }
            }

            FloatingToolButton(
                title: "动态",
                systemImage: "clock",
                accessibilityLabel: "动态",
                identifier: "activity-action"
            ) {
                dismissKeyboard()
                withAnimation(.easeInOut(duration: 0.2)) {
                    showContext = false
                    showActivity = true
                    toolsExpanded = false
                }
            }

            FloatingToolButton(
                title: "历史",
                systemImage: "clock.arrow.circlepath",
                accessibilityLabel: "历史对话",
                identifier: "history-action"
            ) {
                dismissKeyboard()
                showHistory = true
                toolsExpanded = false
                Task { await state.refreshHistorySessions() }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 9)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.18), radius: 14, x: 0, y: 6)
    }
}

private struct FloatingToolButton: View {
    let title: String
    let systemImage: String
    let accessibilityLabel: String
    let identifier: String
    var disabled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.82)
                Image(systemName: systemImage)
                    .font(.subheadline.weight(.semibold))
                    .frame(width: 24, height: 24)
            }
            .foregroundStyle(.primary)
            .padding(.leading, 10)
            .padding(.trailing, 8)
            .frame(width: 92, height: 40, alignment: .trailing)
            .background(Color(.systemBackground).opacity(0.92))
            .clipShape(Capsule())
            .shadow(color: .black.opacity(0.08), radius: 8, x: 0, y: 3)
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.42 : 1)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityIdentifier(identifier)
    }
}

// 题目分割选题层（两端复用）：冻结图 + 透明浮层；点一道题分析，或整图问答。含梯形校正切换。
struct QuestionSegmentationSheet: View {
    @ObservedObject var state: AppState

    private var hasSelection: Bool { state.selectedRegionID != nil }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button {
                    state.dismissSegmentation()
                } label: {
                    Image(systemName: "xmark")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(width: 34, height: 34)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .accessibilityIdentifier("segmentation-close")

                VStack(alignment: .leading, spacing: 2) {
                    Text(state.segmentationIsGrading ? "作业批改" : "拍照答题 · 选题")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                    if !state.segmentationNotice.isEmpty {
                        Text(state.segmentationNotice)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.78))
                    }
                    if state.segmentationIsGrading && !state.gradingNotice.isEmpty {
                        Text(state.gradingNotice)
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
                Spacer()
                if state.segmentationDidCorrect {
                    Button {
                        state.toggleSegmentationCorrection()
                    } label: {
                        Label(state.segmentationUseCorrected ? "用原图" : "用矫正图",
                              systemImage: "perspective")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .accessibilityIdentifier("segmentation-correction-toggle")
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)
            .padding(.bottom, 10)

            ZStack {
                if state.segmentationBusy {
                    ProgressView("正在校正并分割题目…")
                        .tint(.white)
                        .foregroundStyle(.white)
                } else if let image = state.segmentationDisplayImage {
                    if state.segmentationIsGrading {
                        QuestionGradingOverlay(
                            image: image,
                            regions: state.questionRegions,
                            verdicts: state.gradeQuestionsByIndex,
                            selectedIndex: Binding(
                                get: { state.selectedGradeIndex },
                                set: { state.selectedGradeIndex = $0 }
                            ),
                            onSelect: { state.selectedGradeIndex = $0.index }
                        )
                        .padding(.horizontal, 12)
                    } else {
                        QuestionRegionOverlay(
                            image: image,
                            regions: state.questionRegions,
                            selectedID: Binding(
                                get: { state.selectedRegionID },
                                set: { state.selectedRegionID = $0 }
                            ),
                            onSelect: { state.selectQuestionRegion($0) }
                        )
                        .padding(.horizontal, 12)
                    }
                } else {
                    ProgressView().tint(.white)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            HStack(spacing: 12) {
                Button {
                    state.submitWholeImageFromSegmentation()
                } label: {
                    Text("整图问答")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.white)
                }
                .accessibilityIdentifier("segmentation-whole-image")

                if state.segmentationIsGrading {
                    Text(state.gradingInFlight ? "逐题批改中…" : "点题目看批改详情")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.white.opacity(0.85))
                } else {
                    Button {
                        state.confirmSelectedRegion()
                    } label: {
                        Text("分析选中题")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(hasSelection ? Color.accentColor : Color.gray.opacity(0.5),
                                        in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.white)
                    }
                    .disabled(!hasSelection || state.segmentationBusy)
                    .accessibilityIdentifier("segmentation-confirm")
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 10)
            .padding(.bottom, 18)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.ignoresSafeArea())
        .sheet(isPresented: Binding(
            get: { state.selectedGradeIndex != nil },
            set: { if !$0 { state.selectedGradeIndex = nil } }
        )) {
            if let idx = state.selectedGradeIndex, let q = state.gradeQuestionsByIndex[idx] {
                GradeDetailSheet(state: state, question: q,
                                 onExplain: { state.explainGradedRegion(idx) },
                                 onClose: { state.selectedGradeIndex = nil })
                    .presentationDetents([.medium, .large])
                    .onDisappear { state.clearRelayout() }
            }
        }
    }
}

/// 单题批改详情卡：判分 + 学生作答 + 订正/思路 + 数字版题干(对照原图) + 「讲解这道题」。
struct GradeDetailSheet: View {
    @ObservedObject var state: AppState
    let question: GradedQuestion
    var onExplain: () -> Void
    var onClose: () -> Void
    @State private var showRelayout = false
    @State private var compareOriginal = false

    var body: some View {
        let mark = question.mark
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 10) {
                    Image(systemName: mark.symbol).font(.title2).foregroundStyle(mark.color)
                    Text("第\(question.index)题 · \(mark.label)").font(.headline)
                    Spacer()
                    Button { onClose() } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                }
                if !question.questionText.isEmpty { field("题目", question.questionText) }
                if !question.studentAnswer.isEmpty { field("学生作答", question.studentAnswer) }
                if question.gradable && !question.correctAnswer.isEmpty { field("参考答案", question.correctAnswer) }
                if !question.correction.isEmpty { field(question.gradable ? "订正" : "思路", question.correction) }
                if question.gradable && !question.errorReason.isEmpty { field("错因", question.errorReason) }
                if !question.knowledge.isEmpty { field("知识点", question.knowledge) }
                if !question.gradable {
                    Text("这类题需要看图/几何推理，AI 不硬判对错，点下面让它给你讲。")
                        .font(.caption).foregroundStyle(.secondary)
                }

                relayoutSection

                Button { onExplain() } label: {
                    Label("讲解这道题", systemImage: "sparkles")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity).padding(.vertical, 12)
                        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.white)
                }
                .padding(.top, 4)
            }
            .padding(18)
        }
    }

    // B：数字版题干（仅印刷题干，强制可对照原图）。
    @ViewBuilder private var relayoutSection: some View {
        Divider()
        if !showRelayout {
            Button {
                showRelayout = true
                Task { await state.fetchRelayout(for: question.index) }
            } label: {
                Label("看数字版题干（对照原图）", systemImage: "doc.plaintext")
                    .font(.subheadline.weight(.semibold)).foregroundStyle(Color.accentColor)
            }
        } else {
            HStack {
                Text("数字版题干").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if state.relayoutCropImage != nil {
                    Toggle("对照原图", isOn: $compareOriginal).font(.caption).labelsHidden()
                    Text(compareOriginal ? "原图" : "数字版").font(.caption2).foregroundStyle(.secondary)
                }
            }
            if state.relayoutInFlight {
                ProgressView().padding(.vertical, 6)
            } else if compareOriginal, let img = state.relayoutCropImage {
                Image(uiImage: img).resizable().scaledToFit()
                    .frame(maxHeight: 200).clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Text(state.relayoutText.isEmpty ? "（无）" : state.relayoutText)
                    .font(.subheadline).textSelection(.enabled)
                if !state.relayoutFigureNote.isEmpty {
                    Text(state.relayoutFigureNote).font(.caption).foregroundStyle(.orange)
                }
                Text("数字版仅整理印刷题干、不含手写/不重画图，务必对照原图。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func field(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text(value).font(.subheadline).textSelection(.enabled)
        }
    }
}

// P0：分题/批改「取景—自动快门」叠层。复用底层正在运行的相机预览（透明中心露出相机），
// 提供取景框 + 实时补光/清晰度提示 + 手电 + 自动/手动快门。点按穿透到相机做对焦。
struct QuestionCaptureOverlay: View {
    @ObservedObject var state: AppState

    var body: some View {
        ZStack {
            // 顶/底渐变暗角（让控件清晰，中间透明露出相机），不拦截点按
            VStack(spacing: 0) {
                LinearGradient(colors: [.black.opacity(0.55), .clear], startPoint: .top, endPoint: .bottom)
                    .frame(height: 140)
                Spacer()
                LinearGradient(colors: [.clear, .black.opacity(0.62)], startPoint: .top, endPoint: .bottom)
                    .frame(height: 190)
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            // 取景框
            GeometryReader { geo in
                let m: CGFloat = 26
                let topPad: CGFloat = 96
                let botPad: CGFloat = 168
                let rect = CGRect(x: m, y: topPad,
                                  width: geo.size.width - 2 * m,
                                  height: max(90, geo.size.height - topPad - botPad))
                FramingBrackets(rect: rect, color: state.cameraAimReady ? .green : .white)
            }
            .allowsHitTesting(false)

            VStack {
                HStack(alignment: .top, spacing: 12) {
                    Button {
                        if state.cameraTaskKind == .extract { state.cancelExtraction() } else { state.cancelAiming() }
                    } label: {
                        Image(systemName: "xmark").font(.headline.weight(.semibold)).foregroundStyle(.white)
                            .frame(width: 38, height: 38).background(.ultraThinMaterial, in: Circle())
                    }
                    .accessibilityIdentifier("aiming-close")
                    VStack(alignment: .leading, spacing: 2) {
                        Text(state.cameraTaskKind == .extract ? "题目提取 · 取景" : (state.segmentationIsGrading ? "作业批改 · 取景" : "拍照答题 · 取景"))
                            .font(.subheadline.weight(.semibold)).foregroundStyle(.white)
                        Label(state.cameraTaskKind == .extract && !state.extractNotice.isEmpty ? state.extractNotice : state.cameraAimHint,
                              systemImage: state.cameraAimReady ? "checkmark.circle.fill" : "viewfinder")
                            .font(.caption).foregroundStyle(state.cameraAimReady ? .green : .white.opacity(0.85))
                    }
                    Spacer()
                    Button { state.toggleTorch() } label: {
                        Image(systemName: state.torchOn ? "bolt.fill" : "bolt.slash")
                            .font(.headline).foregroundStyle(state.torchOn ? .yellow : .white)
                            .frame(width: 38, height: 38).background(.ultraThinMaterial, in: Circle())
                    }
                    .accessibilityIdentifier("aiming-torch")
                }
                .padding(.horizontal, 16).padding(.top, 16)

                // P4：自动进模块时给「其实只想问」一键纠偏（退回普通问答）。
                if state.autoRouteRevertText != nil {
                    Button { state.revertAutoRouteToQA() } label: {
                        Label("其实只想问问题 →", systemImage: "arrow.uturn.left")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .padding(.top, 8)
                    .accessibilityIdentifier("aiming-revert-qa")
                }

                Spacer()

                VStack(spacing: 14) {
                    Text(state.autoShutterArmed ? "对准清晰会自动拍，也可手动按快门" : "手动按快门拍照")
                        .font(.caption).foregroundStyle(.white.opacity(0.85))
                    HStack(spacing: 30) {
                        Button { state.autoShutterArmed.toggle() } label: {
                            VStack(spacing: 3) {
                                Image(systemName: state.autoShutterArmed ? "a.circle.fill" : "a.circle").font(.title3)
                                Text("自动").font(.caption2)
                            }
                            .foregroundStyle(state.autoShutterArmed ? .green : .white)
                            .frame(width: 48)
                        }
                        .accessibilityIdentifier("aiming-auto-toggle")
                        Button {
                            if state.cameraTaskKind == .extract { state.captureExtractionPhoto() } else { state.captureSegmentationPhoto() }
                        } label: {
                            ZStack {
                                Circle().strokeBorder(.white, lineWidth: 4).frame(width: 74, height: 74)
                                Circle().fill(.white).frame(width: 60, height: 60)
                            }
                        }
                        .accessibilityIdentifier("aiming-shutter")
                        if state.cameraTaskKind == .extract {
                            Button { state.finishExtraction() } label: {
                                VStack(spacing: 3) {
                                    Image(systemName: "checkmark.circle.fill").font(.title3)
                                    Text("完成").font(.caption2)
                                }
                                .foregroundStyle(.green)
                                .frame(width: 48)
                            }
                            .accessibilityIdentifier("extract-finish")
                        } else {
                            Color.clear.frame(width: 48, height: 48)
                        }
                    }
                }
                .padding(.bottom, 28)
            }
        }
    }
}

struct FramingBrackets: View {
    let rect: CGRect
    let color: Color
    var body: some View {
        Path { p in
            let L: CGFloat = 26
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + L)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + L, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - L, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + L))
            p.move(to: CGPoint(x: rect.minX, y: rect.maxY - L)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX + L, y: rect.maxY))
            p.move(to: CGPoint(x: rect.maxX - L, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - L))
        }
        .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
    }
}

private struct LearningStagePanel: View {
    @ObservedObject var state: AppState

    private var stageIsLive: Bool {
        state.inlineCameraPreviewVisible
    }

    private var stageStatusText: String {
        if state.continuousVoiceActive { return "持续倾听" }
        if state.isListening || state.isPreparingVoiceInput { return "正在听" }
        if state.isThinking { return "AI 处理中" }
        if state.isBursting { return "智能观察" }
        if state.cameraTaskKind == .singleCapture { return state.isCameraReady ? "相机就绪" : "准备相机" }
        if state.cameraTaskKind == .qaFrame { return state.isCameraReady ? "画面就绪" : "准备画面" }
        return state.uploadState
    }

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Label("学习现场", systemImage: state.cameraTaskSystemImage)
                    .font(.headline)
                    .lineLimit(1)
                Spacer(minLength: 8)
                ChatStatusChip(text: stageStatusText, systemImage: state.qaSystemImage)
            }

            ZStack(alignment: .bottom) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemBackground))

                if stageIsLive {
                    ZStack {
                        CameraView(state: state)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .accessibilityIdentifier("landscape-camera-preview")
                        LiveQuestionRegionOverlay(
                            imageSize: state.observationQuestionOverlayImageSize,
                            candidates: state.observationQuestionOverlayCandidates,
                            visible: state.observationQuestionOverlayVisible
                        )
                    }
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "camera.viewfinder")
                            .font(.system(size: 38, weight: .semibold))
                            .foregroundStyle(Color.accentColor)
                        Text("待机")
                            .font(.headline)
                        Text(state.reviewQueueState)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 14)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

                VStack(spacing: 8) {
                    if stageIsLive {
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 5) {
                                Label(state.inlineCameraTitle, systemImage: state.cameraTaskSystemImage)
                                    .font(.caption.weight(.semibold))
                                    .lineLimit(1)
                                Text(state.inlineCameraHint)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(.thinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            Spacer(minLength: 0)
                        }
                    }

                    Spacer(minLength: 0)

                }
                .padding(10)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(.separator), lineWidth: 1)
            )
            .layoutPriority(1)

            LandscapeStageControls(state: state)

            if state.activeTaskVisible || !state.pendingContextItems(draft: "").isEmpty {
                LandscapeStageStatusStrip(state: state)
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .accessibilityIdentifier("landscape-workbench")
    }
}

private struct LandscapeStageControls: View {
    @ObservedObject var state: AppState

    var body: some View {
        HStack(spacing: 8) {
            if !state.isBursting {
                Button {
                    state.startNewConversationAndListen()
                } label: {
                    Label("新对话", systemImage: "plus.bubble.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.voiceInputDisabled || state.continuousVoiceActive)
                .accessibilityIdentifier("stage-new-conversation")
            }

            Button {
                state.toggleBurst()
            } label: {
                Label(state.isBursting ? "停止" : "观察", systemImage: state.isBursting ? "stop.circle" : "rectangle.stack")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("stage-burst-action")

            if state.cameraPreviewVisible {
                Button {
                    state.hideInlineCameraPreview()
                } label: {
                    Image(systemName: "eye.slash")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("隐藏预览")
            }
        }
        .font(.caption.weight(.semibold))
        .lineLimit(1)
        .minimumScaleFactor(0.85)
    }
}

private struct LandscapeStageStatusStrip: View {
    @ObservedObject var state: AppState

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(state.runtimeTasks) { task in
                    HStack(spacing: 6) {
                        Image(systemName: task.systemImage)
                            .foregroundStyle(task.tone.color)
                        Text(task.title)
                            .font(.caption.weight(.semibold))
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(task.tone.color.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                ForEach(state.pendingContextItems(draft: "")) { item in
                    ContextChip(item: item, compact: true)
                }
            }
            .padding(.horizontal, 1)
        }
        .frame(height: 34)
    }
}

private struct BackgroundCameraHost: View {
    @ObservedObject var state: AppState

    var body: some View {
        CameraView(state: state)
            .frame(width: 2, height: 2)
            .opacity(0.01)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}

private struct CameraTaskSheet: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Label(state.cameraTaskTitle, systemImage: state.cameraTaskSystemImage)
                    .font(.headline)
                Spacer()
                Button {
                    state.closeCameraTask()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("关闭相机")
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 10)

            GeometryReader { geometry in
                CameraView(state: state)
                    .frame(height: max(220, min(geometry.size.height, state.isBursting ? 360 : 300)))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(.separator), lineWidth: 1)
                    )
                    .padding(.horizontal, 16)
            }
            .frame(minHeight: 230, idealHeight: state.isBursting ? 360 : 300, maxHeight: state.isBursting ? 380 : 320)

            HStack(spacing: 10) {
                Button {
                    state.performCameraPrimaryAction()
                } label: {
                    Label(state.cameraPrimaryActionTitle, systemImage: state.cameraPrimaryActionSystemImage)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.cameraPrimaryActionDisabled)

                Button {
                    state.closeCameraTask()
                } label: {
                    Label(state.isBursting ? "收起观察" : "回到对话", systemImage: "bubble.left.and.bubble.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .padding(16)
        }
        .background(Color(.systemBackground))
        .accessibilityElement(children: .contain)
    }
}

private struct InlineCameraPreviewPane: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Label(state.inlineCameraTitle, systemImage: state.cameraTaskSystemImage)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Spacer()
                if state.isCameraReady {
                    Label("预览中", systemImage: "checkmark.circle.fill")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.green)
                } else {
                    Label("准备相机", systemImage: "camera")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                Button {
                    state.openInlineCameraPreview()
                } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("放大相机预览")

                Button {
                    state.hideInlineCameraPreview()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("隐藏相机预览")
            }

            ZStack(alignment: .bottomLeading) {
                CameraView(state: state)
                    .frame(height: 168)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(.separator), lineWidth: 1)
                    )

                Text(state.inlineCameraHint)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(.black.opacity(0.46))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(8)
            }

        }
        .padding(10)
        .background(Color(.secondarySystemBackground))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityIdentifier("inline-camera-preview")
    }
}

private struct AssistantChatPanel: View {
    @ObservedObject var state: AppState
    @Binding var draft: String
    @State private var showPreferences = false
    @State private var showActivity = false
    @State private var showContext = false
    @State private var showHistory = false
    @State private var showRuntimePanel = false
    @State private var selectedContextItem: ContextBadgeItem?
    @State private var selectedAttachment: ChatAttachment?
    @State private var selectedHistoryReport: HistoryReportDetail?
    @State private var composerMode = ComposerInputMode.voice
    @FocusState private var composerFocused: Bool

    private var statusText: String {
        if state.continuousVoiceActive { return state.isListening ? "持续听" : "录音开" }
        if state.isListening || state.isPreparingVoiceInput { return "正在听" }
        if state.isThinking { return "思考中" }
        if state.ttsPlaybackPhase == .generating { return "生成语音" }
        if state.ttsPlaybackPhase == .paused { return "已暂停" }
        if state.ttsPlaybackPhase == .playing { return "正在回答" }
        return state.uploadState
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .center, spacing: 10) {
                Button {
                    dismissKeyboard()
                    showHistory = true
                    Task { await state.refreshHistorySessions() }
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.headline)
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("历史对话")
                .accessibilityIdentifier("history-action")

                Label("Pai 学习教练", systemImage: "sparkles")
                    .font(.headline)
                    .accessibilityIdentifier("app-title")
                Spacer()
                ChatStatusChip(text: statusText, systemImage: state.qaSystemImage)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.systemBackground))

            Divider()

            ZStack(alignment: .bottomTrailing) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            if state.hasChatStarted {
                                ForEach(state.chatMessages) { message in
                                    ChatMessageBubble(
                                        message: message,
                                        isLatestAssistant: state.latestAssistantMessageId == message.id,
                                        preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                                        preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                                        onQuickFollowUp: { state.submitQuickFollowUp($0) },
                                        onAddMistake: { state.addMessageToMistakeBook($0) },
                                        onFormMemory: { state.formMemoryFromMessage($0) },
                                        onSmartCapture: { state.smartCaptureFromMessage($0) },
                                        onGenerateVisualization: { state.generateVisualization(for: $0) },
                                        onOpenVisualization: { state.openVisualization($0) },
                                        onSelectContext: { selectedContextItem = $0 },
                                        onSelectAttachment: { selectedAttachment = $0 },
                                        onOpenWorkspace: { withAnimation(.easeInOut(duration: 0.2)) { showContext = true } }
                                    )
                                }

                                // 三期：本轮记忆增量 chip（本轮无增量则不渲染）。
                                MemoryDeltaCard(state: state)

                                if state.chatMessages.isEmpty,
                                   !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    UserTextBubble(text: state.recognizedText)
                                }

                                if state.isListening,
                                   !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                   state.chatMessages.last?.text != state.recognizedText {
                                    UserTextBubble(text: state.recognizedText)
                                }

                                if state.questionSubmissionInFlight {
                                    AssistantTextBubble(
                                        title: "正在输入",
                                        text: "我在整理思路，会直接把回答放到这段对话下面。",
                                        systemImage: "ellipsis.bubble",
                                        showsProgress: true
                                    )
                                }

                                if state.chatMessages.isEmpty,
                                   !state.qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    AssistantAnswerBubble(
                                        answer: state.qaAnswer,
                                        showFollowUpActions: !state.questionSubmissionInFlight,
                                        preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                                        preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                                        onQuickFollowUp: { state.submitQuickFollowUp($0) },
                                        onAddMistake: { state.addLatestAnswerToMistakeBook() },
                                        onFormMemory: { state.formMemoryFromLatestAnswer() },
                                        onSmartCapture: { state.smartCaptureFromLatestAnswer() }
                                    )
                                }
                            } else {
                                AssistantTextBubble(
                                    title: "知进拍学 EvoWit.com",
                                    text: "为学习者建造通往知识的优化坡道\nBuilding optimized ramps to knowledge\n\n最大化每秒顿悟次数 (Eurekas per second)",
                                    systemImage: "sparkles"
                                )

                                AssistantTextBubble(
                                    title: "当前状态",
                                    text: state.reviewQueueState,
                                    systemImage: "list.bullet.clipboard"
                                )
                            }

                            if !state.hasChatStarted,
                               !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                UserTextBubble(text: state.recognizedText)
                            }

                            if !state.hasChatStarted,
                               !state.qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                AssistantAnswerBubble(
                                    answer: state.qaAnswer,
                                    showFollowUpActions: !state.questionSubmissionInFlight,
                                    preferenceFollowUpTitle: state.preferenceFollowUpTitle,
                                    preferenceFollowUpPrompt: state.preferenceFollowUpPrompt,
                                    onQuickFollowUp: { state.submitQuickFollowUp($0) },
                                    onAddMistake: { state.addLatestAnswerToMistakeBook() },
                                    onFormMemory: { state.formMemoryFromLatestAnswer() },
                                    onSmartCapture: { state.smartCaptureFromLatestAnswer() }
                                )
                            }

                            Color.clear
                                .frame(height: state.activeTaskVisible ? 92 : 1)
                                .id("chat-bottom")
                        }
                        .padding(16)
                    }
                    .background(Color(.secondarySystemBackground))
                    .scrollDismissesKeyboard(.interactively)
                    .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
                    .onChange(of: state.logs.count) { _ in
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo("chat-bottom", anchor: .bottom)
                        }
                    }
                    .onChange(of: state.qaAnswer) { _ in
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo("chat-bottom", anchor: .bottom)
                        }
                    }
                    .onChange(of: state.isThinking) { _ in
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo("chat-bottom", anchor: .bottom)
                        }
                    }
                }

                if state.activeTaskVisible {
                    RuntimeTaskOverlay(state: state, expanded: $showRuntimePanel) {
                        dismissKeyboard()
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 10)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }

            Divider()

            VStack(spacing: 10) {
                ChatToolRail(
                    state: state,
                    showPreferences: $showPreferences,
                    showActivity: $showActivity,
                    showContext: $showContext,
                    contextCount: state.pendingContextItems(draft: draft).count,
                    dismissKeyboard: dismissKeyboard
                )

                if showContext {
                    SubmissionContextStrip(
                        state: state,
                        draft: draft,
                        onClose: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                showContext = false
                            }
                        },
                        onSelectContext: { selectedContextItem = $0 }
                    )
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                if showPreferences {
                    ChatSettingsPanel(state: state)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                if showActivity {
                    ActivityLogPanel(logs: Array(state.logs.suffix(6)))
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                if state.observationGuideVisible {
                    ObservationGestureGuidePanel(state: state)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                HStack(alignment: .bottom, spacing: 8) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            if composerMode == .voice {
                                composerMode = .text
                                composerFocused = true
                            } else {
                                composerMode = .voice
                                draft = ""
                                dismissKeyboard()
                            }
                        }
                } label: {
                    Image(systemName: composerMode == .voice ? "keyboard" : "waveform")
                        .font(.headline)
                        .frame(width: 36, height: 36)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(composerMode == .voice ? "切换文字输入" : "切换语音输入")
                .accessibilityIdentifier("composer-mode-toggle")

                if composerMode == .voice && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    VoiceHoldArea(state: state)
                        .frame(maxWidth: 430, alignment: .leading)
                    Spacer(minLength: 0)
                } else {
                        TextField("追问...", text: $draft, axis: .vertical)
                            .textFieldStyle(.plain)
                            .lineLimit(1...4)
                            .submitLabel(.send)
                            .focused($composerFocused)
                            .onSubmit(sendDraft)
                            .accessibilityIdentifier("chat-composer")

                        Button {
                            sendDraft()
                        } label: {
                            if state.intentRouteInFlight {
                                ProgressView()
                                    .frame(width: 38, height: 38)
                            } else {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.title2)
                                    .frame(width: 38, height: 38)
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(state.questionSubmissionInFlight || state.intentRouteInFlight)
                        .accessibilityLabel("发送文字问题")
                        .accessibilityIdentifier("send-chat")
                    }
                }
                .padding(.leading, 12)
                .padding(.trailing, 6)
                .padding(.vertical, 6)
                .background(Color(.secondarySystemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.separator), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .contentShape(RoundedRectangle(cornerRadius: 8))

                Text(state.composerHint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(12)
            .background(Color(.systemBackground))
        }
        .background(Color(.systemBackground))
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("收起") {
                    dismissKeyboard()
                }
            }
        }
        .sheet(isPresented: $showHistory) {
            HistorySessionSheet(
                state: state,
                onStartNew: { session in
                    showHistory = false
                    dismissKeyboard()
                    Task { await state.startConversationFromHistory(session) }
                },
                onViewReport: { session in
                    showHistory = false
                    dismissKeyboard()
                    Task {
                        if let report = await state.historyReport(for: session) {
                            selectedHistoryReport = report
                        }
                    }
                }
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
        .sheet(item: $selectedContextItem) { item in
            ContextDetailSheet(item: item) {
                selectedContextItem = nil
            }
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(item: $selectedHistoryReport) { report in
            HistoryReportSheet(report: report) {
                selectedHistoryReport = nil
            }
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .fullScreenCover(item: $selectedAttachment) { attachment in
            AttachmentPreviewScreen(attachment: attachment) {
                selectedAttachment = nil
            }
        }
    }

    private func sendDraft() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // 配置探测在途也算忙：先于 draft="" 拦截，避免清空输入框却没发出去。
        guard !text.isEmpty, !state.questionSubmissionInFlight, !state.intentRouteInFlight else { return }
        draft = ""
        composerMode = .voice
        state.submitTypedQuestion(text)
    }

    private func dismissKeyboard() {
        composerFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

private struct ChatToolRail: View {
    @ObservedObject var state: AppState
    @Binding var showPreferences: Bool
    @Binding var showActivity: Bool
    @Binding var showContext: Bool
    let contextCount: Int
    let dismissKeyboard: () -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                if !state.isBursting {
                    Button {
                        dismissKeyboard()
                        state.startNewConversationAndListen()
                    } label: {
                        Label("新对话", systemImage: "plus.bubble.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(state.voiceInputDisabled || state.continuousVoiceActive)
                }

                Button {
                    dismissKeyboard()
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showContext.toggle()
                        if showContext {
                            showPreferences = false
                            showActivity = false
                        }
                    }
                } label: {
                    Label(contextCount > 0 ? "上下文 \(contextCount)" : "上下文", systemImage: showContext ? "tray.full.fill" : "tray.full")
                }
                .buttonStyle(.bordered)
                .tint(showContext ? .accentColor : nil)
                .accessibilityIdentifier("context-action")

                Button {
                    dismissKeyboard()
                    state.toggleBurst()
                } label: {
                    Label(state.isBursting ? "停止观察" : "智能观察", systemImage: state.isBursting ? "stop.circle" : "rectangle.stack")
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("burst-action")

                Button {
                    dismissKeyboard()
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showPreferences.toggle()
                        if showPreferences {
                            showActivity = false
                            showContext = false
                        }
                    }
                } label: {
                    Label("偏好", systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)

                Button {
                    dismissKeyboard()
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showActivity.toggle()
                        if showActivity {
                            showPreferences = false
                            showContext = false
                        }
                    }
                } label: {
                    Label("动态", systemImage: "clock")
                }
                .buttonStyle(.bordered)
            }
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 1)
        }
    }
}

private struct ObservationGestureGuidePanel: View {
    @ObservedObject var state: AppState
    var showsStartButton = false
    var closingToTool = false
    var onStart: () -> Void = {}
    var onClose: (() -> Void)? = nil

    private let items: [(String, String, String)] = [
        ("hand.point.up.left", "指向", "打开语音"),
        ("hand.thumbsup", "OK", "打断回答"),
        ("hand.raised", "停止", "结束本轮")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                Label("观察手势", systemImage: "hand.raised")
                    .font(.caption.weight(.semibold))
                Spacer()
                if showsStartButton {
                    Button {
                        onStart()
                    } label: {
                        Label("开启观察", systemImage: "play.circle.fill")
                            .font(.caption.weight(.semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .accessibilityIdentifier("start-observation")
                }
                Button {
                    if let onClose {
                        onClose()
                    } else {
                        state.observationGuideVisible = false
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                    .accessibilityLabel("关闭手势提示")
            }

            Text(showsStartButton ? "开启后会清空当前桌面对话，让相机预览完整显示；学习相关画面会在后台整理成报告。" : "观察运行时可用手势和语音打断。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], alignment: .leading, spacing: 6) {
                ForEach(items, id: \.1) { item in
                    HStack(alignment: .top, spacing: 7) {
                        Image(systemName: item.0)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Color.accentColor)
                            .frame(width: 18)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(item.1)
                                .font(.caption.weight(.semibold))
                                .lineLimit(1)
                            Text(item.2)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .padding(7)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.tertiarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding(7)
        .frame(maxWidth: 500)
        .background(Color(.secondarySystemBackground))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .scaleEffect(closingToTool ? 0.2 : 1.0, anchor: .bottomTrailing)
        .opacity(closingToTool ? 0.05 : 1.0)
        .offset(x: closingToTool ? 220 : 0, y: closingToTool ? 44 : 0)
    }
}

private struct CollapsedVoiceButton: View {
    let action: () -> Void
    let doubleTapAction: () -> Void

    var body: some View {
        Image(systemName: "mic.fill")
            .font(.headline.weight(.semibold))
            .foregroundStyle(.white)
            .frame(width: 42, height: 42)
            .background(Color.accentColor)
            .clipShape(Circle())
            .shadow(color: .black.opacity(0.14), radius: 8, x: 0, y: 3)
            .contentShape(Circle())
            .gesture(collapsedTapGesture)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityLabel("展开语音输入")
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("voice-collapsed-action")
    }

    private var collapsedTapGesture: some Gesture {
        TapGesture(count: 2)
            .onEnded {
                doubleTapAction()
            }
            .exclusively(before: TapGesture(count: 1).onEnded {
                action()
            })
    }
}

enum VoiceHoldAction: Equatable {
    case sendWithPhoto   // 默认：捎带当前照片
    case sendTextOnly    // 上滑一点：只纯文字
    case cancel          // 上滑更多：取消
}

struct VoiceHoldArea: View {
    @ObservedObject var state: AppState
    var allowsCollapse = true
    var transparentSurface = false
    var pressHeight: CGFloat = 32
    var onBegin: () -> Void = {}
    var onCollapse: () -> Void = {}
    @State private var pressActive = false
    @State private var holdStarted = false
    @State private var holdStartTask: Task<Void, Never>?
    @State private var quickTapHintTask: Task<Void, Never>?
    @State private var pressBeganAt: Date?
    @State private var lastQuickTapAt: Date?
    @State private var quickTapHintVisible = false
    @State private var cancelOnRelease = false
    @State private var holdAction: VoiceHoldAction = .sendWithPhoto

    private var disabled: Bool {
        state.voiceInputDisabled
    }

    private var debugForceVoiceMenu: Bool {
        #if DEBUG
        return ProcessInfo.processInfo.environment["PXJ_FORCE_VOICE_MENU"] == "1"
        #else
        return false
        #endif
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            if pressActive || state.isListening || state.isPreparingVoiceInput {
                VoiceRecordingOverlay(
                    cancelOnRelease: cancelOnRelease,
                    transcript: state.recognizedText
                )
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            if (pressActive && holdStarted) || debugForceVoiceMenu {
                VoiceHoldOptionsMenu(active: debugForceVoiceMenu ? .sendTextOnly : holdAction)
                    .padding(.bottom, 52)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                    .animation(.easeOut(duration: 0.12), value: holdAction)
            }

            HStack(spacing: 8) {
            if allowsCollapse && state.voiceDockCanCollapse {
                Button {
                    onCollapse()
                } label: {
                    Image(systemName: "chevron.down.circle")
                        .font(.headline)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("收起语音条")
            }

            HStack(spacing: 8) {
                VoicePressLabel(
                    text: labelText,
                    systemImage: leadingSystemImage,
                    tint: leadingTint,
                    disabled: disabled,
                    pressed: pressActive
                )

                Divider()
                    .frame(height: 22)

                VoiceStatusTicker(
                    state: state,
                    frozen: pressActive || quickTapHintVisible,
                    cancelOnRelease: cancelOnRelease,
                    quickTapHint: quickTapHintVisible
                )
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 8)
            .frame(height: pressHeight)
            .frame(maxWidth: .infinity)
            .background(pressActive ? Color.accentColor.opacity(0.18) : Color(.systemBackground).opacity(transparentSurface ? 0.18 : 0.0))
            .clipShape(RoundedRectangle(cornerRadius: 7))
            .overlay(
                RoundedRectangle(cornerRadius: 7)
                    .stroke(pressActive ? Color.accentColor.opacity(0.55) : Color(.separator).opacity(transparentSurface ? 0.10 : 0.0), lineWidth: 1)
            )
            .scaleEffect(pressActive ? 0.985 : 1.0)
            .animation(.easeOut(duration: 0.12), value: pressActive)
            .contentShape(Rectangle())
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(labelText)
            .accessibilityAddTraits(.isButton)
            .accessibilityIdentifier("voice-action")
            .simultaneousGesture(pressGesture)

            if state.voiceCancelAvailable {
                Button {
                    pressActive = false
                    state.cancelVoiceForRetry()
                } label: {
                    if state.questionSubmissionInFlight {
                        Label("停止", systemImage: "stop.circle.fill")
                            .font(.caption.weight(.semibold))
                            .frame(width: 58, height: 44)
                            .background(Color.orange.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
                    } else {
                        Image(systemName: "arrow.counterclockwise.circle.fill")
                            .font(.headline)
                            .frame(width: 44, height: 44)
                            .background(Color.orange.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(.orange)
                .contentShape(Rectangle())
                .accessibilityLabel(state.questionSubmissionInFlight ? "停止当前回答并重说" : "取消重说")
                .accessibilityIdentifier("voice-retry-action")
            }

            Button {
                state.toggleVoicePlayback()
            } label: {
                Image(systemName: playbackSystemImage)
                    .font(.headline)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial.opacity(0.62), in: RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .foregroundStyle(playbackTint)
            .contentShape(Rectangle())
            .accessibilityLabel(playbackAccessibilityLabel)
            .accessibilityIdentifier("voice-playback-toggle")

            Menu {
                ForEach(ttsPlaybackRateOptions, id: \.self) { rate in
                    Button {
                        state.voicePlaybackRateDidChange(rate)
                    } label: {
                        Label(String(format: "%.2fx", rate), systemImage: abs(state.voicePlaybackRate - rate) < 0.01 ? "checkmark" : "speedometer")
                    }
                }
            } label: {
                Text(state.voicePlaybackRateText)
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .frame(width: 48, height: 44)
                    .background(.ultraThinMaterial.opacity(0.62), in: RoundedRectangle(cornerRadius: 8))
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .contentShape(Rectangle())
            .disabled(!state.voicePlaybackEnabled)
            .accessibilityLabel("朗读速度")
            .accessibilityIdentifier("voice-playback-rate-menu")

            if state.ttsPlaybackPhase == .playing || state.ttsPlaybackPhase == .paused {
                Button {
                    state.toggleSpeechPause()
                } label: {
                    Image(systemName: pauseSystemImage)
                        .font(.headline)
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial.opacity(0.62), in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
                .contentShape(Rectangle())
                .accessibilityLabel(pauseAccessibilityLabel)
                .accessibilityIdentifier("voice-playback-pause")
            }
        }
        .padding(.horizontal, transparentSurface ? 2 : 12)
        .frame(height: 46)
        .frame(maxWidth: .infinity)
        .background {
            if transparentSurface {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.ultraThinMaterial.opacity(pressActive ? 0.9 : 0.24))
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemBackground))
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator).opacity(transparentSurface ? 0.12 : 1), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .opacity(disabled && !state.voiceCancelAvailable ? 0.45 : 1)
        }
        .padding(.horizontal, 12)
        .frame(height: pressActive || state.isListening || state.isPreparingVoiceInput ? 174 : 46, alignment: .bottom)
        .onDisappear {
            cancelPendingHoldStart()
            cancelQuickTapHint()
            if holdStarted {
                holdStarted = false
                cancelOnRelease = false
                state.cancelHoldToTalk()
            }
        }
        .onChange(of: state.isListening) { isListening in
            if !isListening && !state.isPreparingVoiceInput {
                pressActive = false
                holdStarted = false
                cancelOnRelease = false
            }
        }
        .onChange(of: state.isPreparingVoiceInput) { isPreparing in
            if !isPreparing && !state.isListening {
                pressActive = false
                holdStarted = false
                cancelOnRelease = false
            }
        }
    }

    private var pressGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard !disabled, !state.continuousVoiceActive else { return }
                if !pressActive {
                    pressActive = true
                    quickTapHintVisible = false
                    cancelOnRelease = false
                    holdAction = .sendWithPhoto
                    pressBeganAt = Date()
                    scheduleHoldStart()
                }
                // 上滑分三档：默认捎带照片 → 只纯文字 → 取消
                let h = value.translation.height
                holdAction = h <= -150 ? .cancel : (h <= -70 ? .sendTextOnly : .sendWithPhoto)
                cancelOnRelease = (holdAction == .cancel)
            }
            .onEnded { _ in
                cancelPendingHoldStart()
                let endedAt = Date()
                let pressDuration = pressBeganAt.map { endedAt.timeIntervalSince($0) } ?? 0
                pressBeganAt = nil
                guard pressActive else { return }
                pressActive = false
                if holdStarted {
                    holdStarted = false
                    lastQuickTapAt = nil
                    switch holdAction {
                    case .cancel: state.cancelHoldToTalk()
                    case .sendTextOnly: state.endHoldToTalk(textOnly: true)
                    case .sendWithPhoto: state.endHoldToTalk()
                    }
                    cancelOnRelease = false
                    holdAction = .sendWithPhoto
                } else {
                    handleQuickTap(endedAt: endedAt, pressDuration: pressDuration)
                }
            }
    }

    private func scheduleHoldStart() {
        holdStartTask?.cancel()
        holdStartTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 180_000_000)
            guard !Task.isCancelled, pressActive, !holdStarted, !disabled, !state.continuousVoiceActive else { return }
            holdStarted = true
            onBegin()
            state.beginHoldToTalk()
        }
    }

    private func cancelPendingHoldStart() {
        holdStartTask?.cancel()
        holdStartTask = nil
    }

    private func handleQuickTap(endedAt: Date, pressDuration: TimeInterval) {
        guard pressDuration <= 0.32 else {
            lastQuickTapAt = nil
            cancelQuickTapHint()
            return
        }

        if let lastQuickTapAt, endedAt.timeIntervalSince(lastQuickTapAt) <= 0.48 {
            self.lastQuickTapAt = nil
            cancelQuickTapHint()
            onBegin()
            state.submitCurrentFrameShortcut()
        } else {
            lastQuickTapAt = endedAt
            showQuickTapHint()
        }
    }

    private func showQuickTapHint() {
        quickTapHintTask?.cancel()
        quickTapHintVisible = true
        quickTapHintTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 900_000_000)
            guard !Task.isCancelled else { return }
            quickTapHintVisible = false
            lastQuickTapAt = nil
        }
    }

    private func cancelQuickTapHint() {
        quickTapHintTask?.cancel()
        quickTapHintTask = nil
        quickTapHintVisible = false
    }

    private var playbackSystemImage: String {
        guard state.voicePlaybackEnabled else { return "speaker.slash.fill" }
        return state.ttsPlaybackPhase.isActive ? "speaker.wave.2.fill" : "speaker.wave.1"
    }

    private var playbackTint: Color {
        guard state.voicePlaybackEnabled else { return .orange }
        return state.ttsPlaybackPhase.isActive ? .accentColor : .secondary
    }

    private var playbackAccessibilityLabel: String {
        state.voicePlaybackEnabled ? "关闭声音" : "开启声音"
    }

    private var pauseSystemImage: String {
        state.ttsPlaybackPhase == .paused ? "play.circle.fill" : "pause.circle.fill"
    }

    private var pauseAccessibilityLabel: String {
        state.ttsPlaybackPhase == .paused ? "继续朗读" : "暂停朗读"
    }

    private var leadingSystemImage: String {
        (state.isListening || state.isPreparingVoiceInput || state.continuousVoiceActive || pressActive) ? "waveform.circle.fill" : "mic.fill"
    }

    private var leadingTint: Color {
        if state.isListening || state.isPreparingVoiceInput || state.continuousVoiceActive || pressActive {
            return .red
        }
        return disabled ? .secondary : .accentColor
    }

    private var labelText: String {
        if pressActive && holdStarted {
            switch holdAction {
            case .cancel: return "松手取消"
            case .sendTextOnly: return "松手只发文字"
            case .sendWithPhoto: return "上滑可选 · 松手发送"
            }
        }
        if quickTapHintVisible {
            return "再点发送画面"
        }
        if pressActive {
            return holdStarted || state.isListening || state.isPreparingVoiceInput ? "松开发送当前画面" : "继续按住"
        }
        if state.continuousVoiceActive {
            return state.isListening || state.isPreparingVoiceInput ? "持续倾听中" : "自动继续听"
        }
        if state.isWaitingForVoiceSubmit {
            return "识别中"
        }
        if state.questionSubmissionInFlight {
            return "等待回答"
        }
        return state.isListening || state.isPreparingVoiceInput ? "松开发送当前画面" : "按住说话"
    }
}

// 长按语音上滑出现的选项菜单：取消 / 只纯文字 / 捎带当前照片（高亮当前选中档）
private struct VoiceHoldOptionsMenu: View {
    let active: VoiceHoldAction

    var body: some View {
        VStack(spacing: 6) {
            row(.cancel, "取消", "xmark.circle.fill", .red)
            row(.sendTextOnly, "只发文字", "text.bubble.fill", .blue)
            row(.sendWithPhoto, "捎带当前照片", "camera.fill", .accentColor)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color(.separator).opacity(0.35), lineWidth: 1))
        .shadow(color: .black.opacity(0.18), radius: 10, y: 3)
        .frame(maxWidth: .infinity, alignment: .center)
    }

    private func row(_ a: VoiceHoldAction, _ title: String, _ icon: String, _ tint: Color) -> some View {
        let on = active == a
        return HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundStyle(on ? Color.white : tint)
                .frame(width: 22)
            Text(title)
                .font(.subheadline.weight(on ? .bold : .regular))
                .foregroundStyle(on ? Color.white : Color.primary)
            Spacer(minLength: 6)
            if on {
                Image(systemName: "hand.point.up.left.fill")
                    .font(.caption)
                    .foregroundStyle(Color.white)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(width: 200)
        .background(on ? tint : Color.clear, in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct VoicePressLabel: View {
    let text: String
    let systemImage: String
    let tint: Color
    let disabled: Bool
    let pressed: Bool

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.headline)
                .foregroundStyle(tint)
                .scaleEffect(pressed ? 1.08 : 1.0)
                .frame(width: 22)
            Text(text)
                .font(.callout.weight(.semibold))
                .foregroundStyle(disabled ? .secondary : .primary)
                .lineLimit(1)
                .minimumScaleFactor(0.82)
        }
        .frame(minWidth: 128, alignment: .leading)
    }
}

private struct VoiceRecordingOverlay: View {
    let cancelOnRelease: Bool
    let transcript: String

    var body: some View {
        VStack(spacing: 8) {
            VoiceWaveform(cancelOnRelease: cancelOnRelease)
                .frame(width: 168, height: 28)

            Text(cancelOnRelease ? "松手取消" : "松开发送当前画面 · 上滑取消")
                .font(.caption.weight(.semibold))
                .foregroundStyle(cancelOnRelease ? .orange : .primary)

            if !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(shortText(transcript, limit: 42))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 12)
        .frame(maxWidth: 430)
        .background(Color(.systemBackground).opacity(0.20), in: RoundedRectangle(cornerRadius: 14))
        .background(.ultraThinMaterial.opacity(0.18), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke((cancelOnRelease ? Color.orange : Color.accentColor).opacity(0.18), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.06), radius: 10, x: 0, y: 4)
        .padding(.bottom, 48)
        .accessibilityIdentifier("voice-recording-overlay")
    }
}

private struct VoiceWaveform: View {
    let cancelOnRelease: Bool

    private let bars: [CGFloat] = [0.28, 0.48, 0.68, 0.44, 0.82, 0.58, 0.36, 0.64, 0.72, 0.42, 0.88, 0.54, 0.34, 0.62, 0.78, 0.46, 0.3]

    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(bars.enumerated()), id: \.offset) { _, value in
                Capsule()
                    .fill(cancelOnRelease ? Color.orange : Color.green)
                    .frame(width: 5, height: max(8, 28 * value))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: cancelOnRelease)
    }
}

private struct ObservationDanmakuOverlay: View {
    @ObservedObject var state: AppState
    @State private var tick = Date()
    private let timer = Timer.publish(every: 1.0 / 24.0, on: .main, in: .common).autoconnect()

    private var visibleLogs: [LogLine] {
        Array(state.logs.suffix(6))
    }

    var body: some View {
        GeometryReader { proxy in
            if state.observationDanmakuVisible {
                VStack {
                    Spacer(minLength: 0)
                    HStack {
                        Spacer(minLength: 0)
                        ZStack(alignment: .bottomTrailing) {
                            ForEach(Array(visibleLogs.enumerated()), id: \.element.id) { index, log in
                                LiveCommentBubble(
                                    text: cleanLogText(log.text),
                                    index: index,
                                    tick: tick,
                                    containerSize: commentContainerSize(proxy.size)
                                )
                            }
                        }
                        .frame(
                            width: commentContainerSize(proxy.size).width,
                            height: commentContainerSize(proxy.size).height,
                            alignment: .bottomTrailing
                        )
                        .clipped()
                        .mask(
                            LinearGradient(
                                stops: [
                                    .init(color: .clear, location: 0),
                                    .init(color: .white, location: 0.2),
                                    .init(color: .white, location: 0.9),
                                    .init(color: .clear, location: 1)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                    .padding(.trailing, 14)
                    .padding(.bottom, 74)
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.18), value: state.observationDanmakuVisible)
        .onReceive(timer) { value in
            tick = value
        }
    }

    private func commentContainerSize(_ size: CGSize) -> CGSize {
        let safeWidth = max(size.width, 1)
        let safeHeight = max(size.height, 1)
        return CGSize(
            width: min(320, max(210, safeWidth * 0.34)),
            height: min(210, max(132, safeHeight * 0.36))
        )
    }
}

struct LiveQuestionRegionOverlay: View {
    let imageSize: CGSize
    let candidates: [ObservationQuestionCandidate]
    let visible: Bool

    var body: some View {
        GeometryReader { geo in
            if visible && !candidates.isEmpty {
                let display = QuestionRegionOverlay.imageDisplayRect(imageSize: imageSize, container: geo.size)
                ZStack(alignment: .topLeading) {
                    ForEach(candidates.indices, id: \.self) { index in
                        liveBox(candidates[index], in: display)
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private func liveBox(_ candidate: ObservationQuestionCandidate, in display: CGRect) -> some View {
        let rect = candidate.normalizedRect
        let w = max(rect.width * display.width, 1)
        let h = max(rect.height * display.height, 1)
        let x = display.minX + rect.minX * display.width
        let y = display.minY + rect.minY * display.height
        let color = candidate.isDuplicate ? Color.white : Color.green
        let label = "Q\(candidate.index) · fp \(String(candidate.fingerprint.prefix(6)))"

        RoundedRectangle(cornerRadius: 6)
            .fill(color.opacity(candidate.isDuplicate ? 0.05 : 0.12))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(
                        color.opacity(candidate.isDuplicate ? 0.62 : 0.95),
                        style: StrokeStyle(lineWidth: candidate.isDuplicate ? 1.2 : 2.0, dash: candidate.isDuplicate ? [5, 4] : [])
                    )
            )
            .frame(width: w, height: h)
            .overlay(alignment: .topLeading) {
                Text(label)
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(candidate.isDuplicate ? Color.primary : Color.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(.regularMaterial, in: Capsule())
                    .overlay(Capsule().stroke(color.opacity(0.42), lineWidth: 1))
                    .padding(4)
            }
            .opacity(candidate.isDuplicate ? 0.72 : 1)
            .position(x: x + w / 2, y: y + h / 2)
            .accessibilityHidden(true)
    }
}

private struct LiveCommentBubble: View {
    let text: String
    let index: Int
    let tick: Date
    let containerSize: CGSize

    private var rowHeight: CGFloat { 31 }

    var body: some View {
        let cycle = 8.0
        let stagger = Double(index) * 1.15
        let progress = ((tick.timeIntervalSinceReferenceDate + stagger)
            .truncatingRemainder(dividingBy: cycle)) / cycle
        let safeWidth = max(1, containerSize.width)
        let safeHeight = max(rowHeight, containerSize.height)
        let travel = safeHeight + rowHeight
        let y = safeHeight + rowHeight / 2 - CGFloat(progress) * travel
        let opacity = min(1, max(0, sin(progress * .pi) * 1.15))
        let scale = 0.92 + 0.08 * min(1, max(0, sin(progress * .pi)))

        HStack(spacing: 6) {
            Circle()
                .fill(Color.accentColor.opacity(0.9))
                .frame(width: 6, height: 6)
            Text(shortText(text, limit: 34))
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.82)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(.black.opacity(0.46), in: Capsule())
        .overlay(
            Capsule()
                .stroke(.white.opacity(0.12), lineWidth: 0.6)
        )
        .shadow(color: .black.opacity(0.22), radius: 5, x: 0, y: 2)
        .frame(maxWidth: safeWidth, alignment: .trailing)
        .scaleEffect(scale, anchor: .trailing)
        .opacity(opacity)
        .position(x: max(6, safeWidth - 6), y: y)
    }
}

private struct VoiceStatusTicker: View {
    @ObservedObject var state: AppState
    var frozen = false
    var cancelOnRelease = false
    var quickTapHint = false
    @State private var tickerIndex = 0
    private let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()

    private var items: [VoiceTickerItem] {
        var values: [VoiceTickerItem] = []
        values.append(contentsOf: state.runtimeTasks.map {
            VoiceTickerItem(id: "task-\($0.id)", title: $0.title, detail: $0.detail, systemImage: $0.systemImage, tone: $0.tone)
        })
        let contextCount = state.pendingContextItems(draft: "").filter { $0.id != "empty-question" }.count
        if contextCount > 0 {
            values.append(
                VoiceTickerItem(
                    id: "context-count",
                    title: "上下文",
                    detail: "本次会带上 \(contextCount) 项上下文",
                    systemImage: "tray.full",
                    tone: .neutral
                )
            )
        }
        values.append(contentsOf: state.logs.suffix(8).map {
            VoiceTickerItem(id: $0.id.uuidString, title: "动态", detail: cleanLogText($0.text), systemImage: "clock", tone: .neutral)
        })
        if values.isEmpty {
            values.append(
                VoiceTickerItem(
                    id: "idle",
                    title: "待机",
                    detail: state.composerHint,
                    systemImage: "mic",
                    tone: .neutral
                )
            )
        }
        return values
    }

    private var displayItems: [VoiceTickerItem] {
        guard frozen else { return items }
        if quickTapHint {
            return [
                VoiceTickerItem(
                    id: "quick-current-frame",
                    title: "快捷发送",
                    detail: "再点一次，直接发送当前画面。",
                    systemImage: "camera.viewfinder",
                    tone: .waiting
                )
            ]
        }
        let text = state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
        return [
            VoiceTickerItem(
                id: "voice-hold-feedback",
                title: cancelOnRelease ? "取消" : (text.isEmpty ? "正在听" : "已听到"),
                detail: cancelOnRelease ? "松手取消本次语音。" : (text.isEmpty ? "松开发送当前画面，上滑取消。" : "松开发送当前画面：\(shortText(text, limit: 42))"),
                systemImage: cancelOnRelease ? "xmark.circle.fill" : "waveform",
                tone: cancelOnRelease ? .warning : .waiting
            )
        ]
    }

    var body: some View {
        let currentItems = displayItems
        let current = currentItems[min(tickerIndex, max(0, currentItems.count - 1))]
        HStack(spacing: 7) {
            Image(systemName: current.systemImage)
                .font(.caption.weight(.semibold))
                .foregroundStyle(current.tone.color)
                .frame(width: 16)
            Text("\(current.title)：\(current.detail)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .id(current.id)
                .transition(.move(edge: .bottom).combined(with: .opacity))
        }
        .animation(.easeInOut(duration: 0.18), value: current.id)
        .onReceive(timer) { _ in
            guard !frozen else {
                tickerIndex = 0
                return
            }
            let count = displayItems.count
            guard count > 1 else {
                tickerIndex = 0
                return
            }
            tickerIndex = (tickerIndex + 1) % count
        }
        .onChange(of: currentItems.count) { count in
            tickerIndex = min(tickerIndex, max(0, count - 1))
        }
        .accessibilityLabel("\(current.title)：\(current.detail)")
    }
}

private struct VoiceTickerItem: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let systemImage: String
    let tone: CaptureQualityTone
}

private struct ChatMessageBubble: View {
    let message: ChatMessage
    var isLatestAssistant = false
    let preferenceFollowUpTitle: String
    let preferenceFollowUpPrompt: String
    let onQuickFollowUp: (String) -> Void
    let onAddMistake: (ChatMessage) -> Void
    let onFormMemory: (ChatMessage) -> Void
    let onSmartCapture: (ChatMessage) -> Void
    let onGenerateVisualization: (ChatMessage) -> Void
    let onOpenVisualization: (TeachingVisualization) -> Void
    let onSelectContext: (ContextBadgeItem) -> Void
    let onSelectAttachment: (ChatAttachment) -> Void
    var onOpenWorkspace: (() -> Void)? = nil

    var body: some View {
        switch message.role {
        case .user:
            UserTextBubble(text: message.text)
        case .assistant:
            // 仅在有题目/实质回答时才显示举一反三/加入错题本/形成记忆/可视化（"hi" 等闲聊不显示）。
            // actionable 由后端按问题意图判定（闲聊=false）；再叠加本地的可视化/长度信号。
            let hasActions = message.actionable && (message.visualizationCandidate || message.text.count >= 50)
            AssistantAnswerBubble(
                answer: message.text,
                visualizationCandidate: message.visualizationCandidate,
                visualizationReason: message.visualizationReason,
                visualization: message.visualization,
                showFollowUpActions: isLatestAssistant && hasActions,
                showAnswerActions: hasActions,
                preferenceFollowUpTitle: preferenceFollowUpTitle,
                preferenceFollowUpPrompt: preferenceFollowUpPrompt,
                onQuickFollowUp: onQuickFollowUp,
                onAddMistake: { onAddMistake(message) },
                onFormMemory: { onFormMemory(message) },
                onSmartCapture: { onSmartCapture(message) },
                onGenerateVisualization: { onGenerateVisualization(message) },
                onOpenVisualization: onOpenVisualization
            )
        case .status:
            if message.contextItems.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    AssistantTextBubble(
                        title: message.title ?? "状态",
                        text: message.text,
                        systemImage: message.systemImage ?? "sparkles",
                        showsProgress: message.showsProgress
                    )
                    AttachmentStrip(attachments: message.attachments, onSelect: onSelectAttachment)
                }
            } else {
                ContextStatusBubble(
                    title: message.title ?? "本次上下文",
                    text: message.text,
                    systemImage: message.systemImage ?? "tray.full",
                    items: message.contextItems,
                    attachments: message.attachments,
                    onSelectContext: onSelectContext,
                    onSelectAttachment: onSelectAttachment,
                    onOpenWorkspace: onOpenWorkspace
                )
            }
        }
    }
}

private struct ContextStatusBubble: View {
    let title: String
    let text: String
    let systemImage: String
    let items: [ContextBadgeItem]
    let attachments: [ChatAttachment]
    let onSelectContext: (ContextBadgeItem) -> Void
    let onSelectAttachment: (ChatAttachment) -> Void
    var onOpenWorkspace: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: systemImage)
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

            if !text.isEmpty {
                Text(text)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            AttachmentStrip(attachments: attachments, onSelect: onSelectAttachment)

            ContextChipFlow(items: items, compact: true, onSelect: onSelectContext)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.tertiarySystemBackground))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct ChatStatusChip: View {
    let text: String
    let systemImage: String

    var body: some View {
        Label(text, systemImage: systemImage)
            .font(.caption.weight(.semibold))
            .lineLimit(1)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Color(.secondarySystemBackground))
            .clipShape(Capsule())
    }
}

private struct RuntimeTaskOverlay: View {
    @ObservedObject var state: AppState
    @Binding var expanded: Bool
    let dismissKeyboard: () -> Void

    var body: some View {
        VStack(alignment: .trailing, spacing: 8) {
            if expanded {
                RuntimeTaskPanel(state: state, expanded: $expanded, dismissKeyboard: dismissKeyboard)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            if !expanded {
                HStack(spacing: 7) {
                    if let primaryTask = state.runtimeTasks.first, primaryTask.canClose {
                        Button {
                            state.closeRuntimeTask(primaryTask.id)
                            withAnimation(.easeInOut(duration: 0.2)) {
                                expanded = false
                            }
                        } label: {
                            Image(systemName: primaryTask.closeSystemImage)
                                .font(.caption.weight(.bold))
                                .frame(width: 28, height: 28)
                                .background(Color(.systemBackground).opacity(0.9))
                                .clipShape(Circle())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel(primaryTask.closeTitle)
                        .accessibilityIdentifier("runtime-task-close")
                    }

                    Button {
                        dismissKeyboard()
                        withAnimation(.easeInOut(duration: 0.2)) {
                            expanded.toggle()
                        }
                    } label: {
                        RuntimeTaskCapsule(state: state, expanded: expanded)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("runtime-task-overlay")
                }
            }
        }
        .onChange(of: state.activeTaskVisible) { visible in
            if !visible {
                expanded = false
            }
        }
    }
}

private struct RuntimeTaskCapsule: View {
    @ObservedObject var state: AppState
    let expanded: Bool

    private var primaryTask: RuntimeTaskItem? {
        state.runtimeTasks.first
    }

    var body: some View {
        if let primaryTask {
            HStack(spacing: 9) {
                Image(systemName: primaryTask.systemImage)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(primaryTask.tone.color)
                    .frame(width: 20, height: 20)

                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 5) {
                        Text(primaryTask.title)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if state.runtimeTasks.count > 1 {
                            Text("+\(state.runtimeTasks.count - 1)")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(Color.accentColor)
                                .clipShape(Capsule())
                        }
                    }
                    Text(primaryTask.detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                if state.runtimeTasks.contains(where: { $0.showsProgress }) {
                    ProgressView()
                        .scaleEffect(0.72)
                }

                Image(systemName: expanded ? "chevron.down" : "chevron.up")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: 330, alignment: .leading)
            .background(.regularMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(.separator), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .shadow(color: .black.opacity(0.12), radius: 12, x: 0, y: 5)
        }
    }
}

private struct RuntimeTaskPanel: View {
    @ObservedObject var state: AppState
    @Binding var expanded: Bool
    let dismissKeyboard: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Label("正在运行 \(state.runtimeTasks.count) 项", systemImage: "square.stack.3d.up")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Spacer()
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        expanded = false
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("收起任务浮层")
            }

            VStack(spacing: 8) {
                ForEach(state.runtimeTasks) { task in
                    RuntimeTaskRow(task: task, state: state, dismissKeyboard: dismissKeyboard)
                }
            }

            ContextChipFlow(items: state.pendingContextItems(draft: ""), compact: true)
        }
        .padding(12)
        .frame(maxWidth: 360, alignment: .leading)
        .background(.regularMaterial)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .shadow(color: .black.opacity(0.14), radius: 16, x: 0, y: 7)
    }
}

private struct RuntimeTaskRow: View {
    let task: RuntimeTaskItem
    @ObservedObject var state: AppState
    let dismissKeyboard: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: task.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(task.tone.color)
                .frame(width: 22, height: 22)

            VStack(alignment: .leading, spacing: 3) {
                Text(task.title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(task.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if task.id == "voice",
                   !state.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(shortText(state.recognizedText, limit: 76))
                        .font(.caption)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(7)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }

            if task.showsProgress {
                ProgressView()
                    .scaleEffect(0.66)
            }

            VStack(spacing: 4) {
                if task.canOpen {
                    Button {
                        dismissKeyboard()
                        state.openRuntimeTask(task.id)
                    } label: {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("放大\(task.title)")
                }

                if task.canClose {
                    Button {
                        state.closeRuntimeTask(task.id)
                    } label: {
                        Image(systemName: task.closeSystemImage)
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel(task.closeTitle)
                }
            }
            .font(.caption.weight(.semibold))
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground).opacity(0.74))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .onTapGesture {
            guard task.canOpen else { return }
            dismissKeyboard()
            state.openRuntimeTask(task.id)
        }
    }
}

private struct SubmissionContextStrip: View {
    @ObservedObject var state: AppState
    let draft: String
    var onClose: () -> Void = {}
    var onSelectContext: (ContextBadgeItem) -> Void = { _ in }

    @State private var showLearningProfile = false

    private var pendingItems: [ContextBadgeItem] {
        state.pendingContextItems(draft: draft)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Label("将带上的上下文", systemImage: "paperplane")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !state.lastSubmittedContextItems.isEmpty {
                    Text("上次已记录")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                Button {
                    onClose()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("关闭上下文")
            }

            ContextChipFlow(items: pendingItems, onSelect: onSelectContext)

            if !state.lastSubmittedContextItems.isEmpty {
                Divider()
                ContextChipFlow(items: state.lastSubmittedContextItems, compact: true, onSelect: onSelectContext)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Button {
                        Task { await state.refreshMemoryDigest() }
                        onSelectContext(state.memoryDigestContextItem())
                    } label: {
                        Label("记忆整理", systemImage: "brain.head.profile")
                    }
                    .buttonStyle(.plain)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        Task { await state.refreshMemoryDigest(force: true) }
                    } label: {
                        Label("整理", systemImage: "arrow.triangle.2.circlepath")
                    }
                    .font(.caption2.weight(.semibold))
                    .buttonStyle(.bordered)
                    Button {
                        state.clearLongTermMemories()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .font(.caption2.weight(.semibold))
                    .buttonStyle(.bordered)
                    .accessibilityLabel("清空本地记忆整理")
                    .disabled(state.longTermMemories.isEmpty && state.userInputMemory.isEmpty)
                }

                // 三期：进入「我的学习档案」（持久记忆，可纠正/删除/撤销）。
                Button {
                    showLearningProfile = true
                } label: {
                    Label("我的学习档案", systemImage: "person.text.rectangle")
                }
                .buttonStyle(.plain)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.tint)

                if state.memoryDigestSummary.isEmpty {
                    Text("系统会根据语音转文字和纯文字输入自动整理；服务器每小时汇总一次，相关时作为上下文候选。")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(state.memoryDigestSummary.prefix(4), id: \.self) { memory in
                        Label(shortText(memory, limit: 42), systemImage: "smallcircle.filled.circle")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    if state.memoryDigestSummary.count > 4 {
                        Text("还有 \(state.memoryDigestSummary.count - 4) 条")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !state.lastTurnMemories.isEmpty {
                Divider()
                ContextRetrievedMemorySection(state: state)
            }
        }
        .padding(10)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .sheet(isPresented: $showLearningProfile) {
            NavigationStack {
                LearningProfileView(state: state)
            }
        }
    }
}

/// Shows the durable memories the server semantically retrieved into the latest answer,
/// each with its score breakdown — the concrete answer to "which memories entered this turn".
private struct ContextRetrievedMemorySection: View {
    @ObservedObject var state: AppState

    /// 命中记忆的真相源：本轮 manifest（来自 context_trace）；无 manifest 时回退 lastTurnMemories。
    private var memories: [RetrievedMemory] {
        state.lastTurnManifest?.retrievedMemories ?? state.lastTurnMemories
    }

    /// 空态/降级态（MUST-4）：区分「已关闭」「暂无相关」「首轮未问」。
    private var emptyStateText: String? {
        if !memories.isEmpty { return nil }
        if !state.contextInclusionSettings.memory { return "你已关闭长期记忆，本轮未调用任何记忆。" }
        if state.lastTurnManifest != nil { return "暂无与本轮问题相关的记忆。" }
        return "本轮未调用长期记忆（先提问，问答后这里显示命中的记忆）。"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                Text("本轮带入的记忆")
                Spacer()
                Text("\(memories.count) 条")
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)

            Text("服务器按 语义+新近度+重要性 检索；「常驻」= 始终带入的偏好/目标/习惯。关掉某条 → 下一轮不带入（且不计入使用次数）。")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let emptyStateText {
                Text(emptyStateText)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            ForEach(memories) { memory in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(memory.kindLabel)
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color(.tertiarySystemBackground))
                            .clipShape(Capsule())
                        Text(memory.text)
                            .font(.caption2)
                            .lineLimit(2)
                        Spacer(minLength: 4)
                        Text(memory.isPersona ? "常驻" : String(format: "%.2f", memory.score ?? 0))
                            .font(.caption2.weight(.semibold).monospacedDigit())
                            .foregroundStyle(memory.isPersona ? Color.purple : Color.accentColor)
                        // 逐条开关：写 memoryOverrides，下一轮进 memory_excludes。
                        Toggle("", isOn: Binding(
                            get: { !state.memoryOverrides.contains(memory.id) },
                            set: { isOn in
                                if isOn { state.memoryOverrides.remove(memory.id) }
                                else { state.memoryOverrides.insert(memory.id) }
                            }
                        ))
                        .labelsHidden()
                        .scaleEffect(0.72)
                        .frame(width: 40)
                    }
                    if !memory.isPersona {
                        MemoryBreakdownBars(memory: memory)
                    }
                    if state.memoryOverrides.contains(memory.id) {
                        Text("已关闭 · 下一轮不带入")
                            .font(.system(size: 9).weight(.semibold))
                            .foregroundStyle(.orange)
                    }
                }
            }
        }
    }
}

/// 4 维相关度条：语义/新近/重要/用过，直观展示记忆为何被检索（含数值）。
private struct MemoryBreakdownBars: View {
    let memory: RetrievedMemory

    private var rows: [(String, Double)] {
        [("语义", memory.semantic), ("新近", memory.recency), ("重要", memory.importance), ("用过", memory.usage)]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(rows, id: \.0) { row in
                HStack(spacing: 5) {
                    Text(row.0)
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.tertiary)
                        .frame(width: 22, alignment: .leading)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color(.tertiarySystemBackground))
                            Capsule()
                                .fill(Color.accentColor.opacity(0.55))
                                .frame(width: max(0, min(1, row.1)) * geo.size.width)
                        }
                    }
                    .frame(height: 4)
                    Text(String(format: "%.2f", row.1))
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.tertiary)
                        .frame(width: 28, alignment: .trailing)
                }
            }
        }
    }
}

private struct ContextChipFlow: View {
    let items: [ContextBadgeItem]
    var compact = false
    var onSelect: ((ContextBadgeItem) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(items) { item in
                    ContextChip(item: item, compact: compact, onSelect: onSelect)
                }
            }
            .padding(.horizontal, 1)
        }
    }
}

private struct ContextChip: View {
    let item: ContextBadgeItem
    var compact = false
    var onSelect: ((ContextBadgeItem) -> Void)? = nil

    var body: some View {
        Button {
            onSelect?(item)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: item.systemImage)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(item.tone.color)
                    .frame(width: 14)

                VStack(alignment: .leading, spacing: 1) {
                    Text(item.title)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    if !compact && !item.detail.isEmpty {
                        Text(item.detail)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, compact ? 5 : 7)
            .frame(minHeight: compact ? 28 : 36)
            .background(item.tone.color.opacity(0.10))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(item.tone.color.opacity(0.22), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(item.title)：\(item.detail)")
    }
}

struct AttachmentStrip: View {
    let attachments: [ChatAttachment]
    let onSelect: (ChatAttachment) -> Void

    var body: some View {
        if !attachments.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(attachments) { attachment in
                        Button {
                            onSelect(attachment)
                        } label: {
                            AttachmentThumbnail(attachment: attachment)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 1)
            }
        }
    }
}

private struct AttachmentThumbnail: View {
    let attachment: ChatAttachment

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.secondarySystemBackground))
                if let image = attachment.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else if let url = attachment.thumbnailURL {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .empty:
                            ProgressView()
                        case .success(let image):
                            image
                                .resizable()
                                .scaledToFill()
                        case .failure:
                            Image(systemName: "photo.badge.exclamationmark")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                        @unknown default:
                            EmptyView()
                        }
                    }
                } else {
                    Image(systemName: "photo")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 92, height: 68)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(.separator), lineWidth: 1)
            )

            Text(attachment.title)
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
            if !attachment.detail.isEmpty {
                Text(attachment.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .frame(width: 96, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

private struct AttachmentPreviewScreen: View {
    let attachment: ChatAttachment
    let onClose: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 12) {
                Spacer(minLength: 40)
                if let image = attachment.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let url = attachment.fullURL ?? attachment.thumbnailURL {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .empty:
                            ProgressView()
                                .tint(.white)
                        case .success(let image):
                            image
                                .resizable()
                                .scaledToFit()
                        case .failure:
                            Label("图片加载失败", systemImage: "photo.badge.exclamationmark")
                                .foregroundStyle(.white)
                        @unknown default:
                            EmptyView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Label("没有可预览的图片", systemImage: "photo")
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(attachment.title)
                        .font(.headline)
                    if !attachment.detail.isEmpty {
                        Text(attachment.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 18)
                .padding(.bottom, 18)
            }

            Button {
                onClose()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.largeTitle)
                    .foregroundStyle(.white.opacity(0.92))
                    .frame(width: 54, height: 54)
            }
            .buttonStyle(.plain)
            .padding(.top, 12)
            .padding(.trailing, 12)
            .accessibilityLabel("关闭图片预览")
        }
    }
}

private struct ContextDetailSheet: View {
    let item: ContextBadgeItem
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Label(item.title, systemImage: item.systemImage)
                        .font(.headline)
                        .foregroundStyle(item.tone.color)

                    Text(item.fullDetail ?? item.detail)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .padding(16)
            }
            .navigationTitle("上下文详情")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("关闭上下文详情")
                }
            }
        }
    }
}

struct AssistantTextBubble: View {
    let title: String
    let text: String
    let systemImage: String
    var showsProgress = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.tint)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    if showsProgress {
                        ProgressView()
                            .scaleEffect(0.72)
                    }
                }
                Text(text)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(11)
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            Spacer(minLength: 24)
        }
    }
}

struct UserTextBubble: View {
    let text: String

    var body: some View {
        HStack(alignment: .top) {
            Spacer(minLength: 38)
            Text(text)
                .font(.callout)
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
                .padding(11)
                .background(Color.accentColor)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

struct AssistantAnswerBubble: View {
    let answer: String
    var visualizationCandidate = false
    var visualizationReason = ""
    var visualization: TeachingVisualization?
    var showFollowUpActions = false
    var showAnswerActions = true
    var preferenceFollowUpTitle = "按偏好"
    var preferenceFollowUpPrompt = "请按我的学习偏好，基于上面的回答继续安排下一步。"
    var onQuickFollowUp: (String) -> Void = { _ in }
    var onAddMistake: () -> Void = {}
    var onFormMemory: () -> Void = {}
    var onSmartCapture: () -> Void = {}
    var onGenerateVisualization: () -> Void = {}
    var onOpenVisualization: (TeachingVisualization) -> Void = { _ in }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "sparkles")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.tint)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 8) {
                Text("AI 回答")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                QARichAnswerView(answer: answer)
                if visualizationCandidate || visualization?.absoluteURL != nil {
                    VisualizationActionRow(
                        visualization: visualization,
                        reason: visualizationReason,
                        onGenerate: onGenerateVisualization,
                        onOpen: onOpenVisualization
                    )
                }
                if showFollowUpActions {
                    QuickFollowUpActions(
                        preferenceTitle: preferenceFollowUpTitle,
                        preferencePrompt: preferenceFollowUpPrompt,
                        onSubmit: onQuickFollowUp
                    )
                }
                if showAnswerActions {
                    AssistantAnswerActionRow(
                        onSmartCapture: onSmartCapture,
                        onAddMistake: onAddMistake,
                        onFormMemory: onFormMemory
                    )
                }
            }
            .padding(11)
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            Spacer(minLength: 24)
        }
    }
}

struct QuickFollowUpActions: View {
    let preferenceTitle: String
    let preferencePrompt: String
    let onSubmit: (String) -> Void

    private var actions: [(title: String, prompt: String, systemImage: String)] {
        [
            ("举一反三", "请基于上面的题目举一反三，给我一道相似变式题，并先让我尝试。", "sparkles"),
            ("总结知识点", "请总结上面这题涉及的知识点、易错点和检查方法。", "list.bullet.rectangle"),
            (preferenceTitle, preferencePrompt, "slider.horizontal.3")
        ]
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(actions, id: \.title) { action in
                    Button {
                        onSubmit(action.prompt)
                    } label: {
                        Label(action.title, systemImage: action.systemImage)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10)
                            .frame(height: 32)
                            .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .accessibilityLabel(action.title)
                }
            }
            .padding(.horizontal, 1)
        }
        .padding(.top, 2)
        .accessibilityIdentifier("assistant-follow-up-actions")
    }
}

private struct VisualizationActionRow: View {
    let visualization: TeachingVisualization?
    let reason: String
    let onGenerate: () -> Void
    let onOpen: (TeachingVisualization) -> Void

    private var isRunning: Bool {
        visualization?.status == "running"
    }

    private var readyVisualization: TeachingVisualization? {
        guard let visualization, visualization.absoluteURL != nil, visualization.status != "running" else { return nil }
        return visualization
    }

    private var buttonTitle: String {
        if readyVisualization != nil { return "打开可视化" }
        if isRunning { return "生成中" }
        return "生成可视化"
    }

    private var detailText: String {
        if readyVisualization != nil {
            return "已生成，稍后也可以回到这条 AI 回复下打开。"
        }
        if isRunning {
            return "正在生成交互 HTML 教学页，通常需要几十秒；完成后会自动打开。"
        }
        return reason.isEmpty ? "适合做图形演示，点击后生成；完成后会自动打开。" : reason
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                if let readyVisualization {
                    onOpen(readyVisualization)
                } else {
                    onGenerate()
                }
            } label: {
                HStack(spacing: 8) {
                    if isRunning {
                        ProgressView()
                            .scaleEffect(0.72)
                    } else {
                        Image(systemName: readyVisualization == nil ? "cube.transparent" : "safari")
                    }
                    Text(buttonTitle)
                        .font(.caption.weight(.semibold))
                }
                .padding(.horizontal, 11)
                .frame(height: 32)
                .background(Color.teal.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.teal)
            .disabled(isRunning)
            .accessibilityLabel(buttonTitle)

            Text(detailText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
        .accessibilityIdentifier("assistant-visualization-action")
    }
}

private struct AssistantAnswerActionRow: View {
    let onSmartCapture: () -> Void
    let onAddMistake: () -> Void
    let onFormMemory: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Button {
                onSmartCapture()
            } label: {
                Label("智能沉淀", systemImage: "wand.and.stars")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 11)
                    .frame(height: 32)
                    .background(Color.accentColor.opacity(0.13), in: RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.accentColor)
            .accessibilityLabel("智能沉淀")

            Menu {
                Button {
                    onAddMistake()
                } label: {
                    Label("仅加入错题本", systemImage: "book.closed.fill")
                }

                Button {
                    onFormMemory()
                } label: {
                    Label("仅形成记忆", systemImage: "brain.head.profile")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.headline)
                    .frame(width: 32, height: 32)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .accessibilityLabel("更多沉淀选项")
        }
        .accessibilityIdentifier("assistant-answer-actions")
    }
}

private struct ChatSettingsPanel: View {
    @ObservedObject var state: AppState

    // 编辑时仅更新 published 值（实时显示），持久化与策略同步在 onSubmit 触发。
    private var coachPreferenceBinding: Binding<String> {
        Binding(get: { state.coachPreferenceText }, set: { state.coachPreferenceText = $0 })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // #4 去掉重复的「本轮偏好」（临时 studentGoal）——本轮要求可直接在对话里说；只保留持久的「一句话辅导偏好」。
            Label("一句话辅导偏好", systemImage: "slider.horizontal.3")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            TextField("例如：先给提示别直接报答案，讲慢一点", text: coachPreferenceBinding, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
                .submitLabel(.done)
                .onSubmit { state.coachPreferenceTextDidChange(state.coachPreferenceText) }

            HStack {
                Label(state.strategySyncState, systemImage: "icloud")
                Spacer()
                Text("随上下文发送")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Toggle(isOn: Binding(get: { state.textOnlyQuestion }, set: { state.textOnlyQuestionDidChange($0) })) {
                    Label("纯文字提问（不开相机）", systemImage: "keyboard")
                }
                .font(.caption.weight(.semibold))

                Toggle(isOn: Binding(get: { state.questionSegmentationEnabled }, set: { state.questionSegmentationEnabledDidChange($0) })) {
                    Label("多题自动分割", systemImage: "rectangle.split.3x1")
                }
                .font(.caption.weight(.semibold))

                Toggle(isOn: Binding(get: { state.autoKeystoneCorrection }, set: { state.autoKeystoneCorrectionDidChange($0) })) {
                    Label("自动梯形校正", systemImage: "perspective")
                }
                .font(.caption.weight(.semibold))

                Toggle(isOn: $state.voicePlaybackEnabled) {
                    Label("朗读回答", systemImage: state.voicePlaybackEnabled ? "speaker.wave.2.fill" : "speaker.slash")
                }
                .font(.caption.weight(.semibold))
                .onChange(of: state.voicePlaybackEnabled) { enabled in
                    state.voicePlaybackEnabledDidChange(enabled)
                }
                // #6 朗读语速移出上下文面板——它属于语音播放设置，不是上下文配置；在「设置」里调。
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct ActivityLogPanel: View {
    let logs: [LogLine]

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Label("最近动态", systemImage: "clock")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            if logs.isEmpty {
                Text("暂无动态")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(logs) { log in
                    Text(log.text)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct ContextWorkspaceSheet: View {
    @ObservedObject var state: AppState
    let draft: String
    let onClose: () -> Void
    @State private var selectedTab: ContextWorkspaceTab = .overview

    private var contextItems: [ContextBadgeItem] {
        state.pendingContextItems(draft: draft)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ContextWorkspaceHeader(state: state, draft: draft, contextItems: contextItems)
                    .padding(.horizontal, 14)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                Picker("上下文页卡", selection: $selectedTab) {
                    ForEach(ContextWorkspaceTab.visibleCases) { tab in
                        Label(tab.title, systemImage: tab.systemImage).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 14)
                .padding(.bottom, 10)

                Divider()

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        switch selectedTab {
                        case .overview:
                            ContextWorkspaceOverviewTab(state: state, draft: draft)
                        case .assets:
                            ContextWorkspaceAssetsTab(state: state, draft: draft)
                        case .prompts:
                            ContextWorkspacePromptsTab(state: state, draft: draft)
                        case .debug:
                            ContextWorkspaceDebugTab(state: state, draft: draft)
                        case .settings:
                            ContextWorkspaceSettingsTab(state: state)
                        }
                    }
                    .padding(14)
                }
            }
            .navigationTitle("上下文")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("关闭上下文")
                }
            }
        }
    }
}

struct ActivityLogSheet: View {
    @ObservedObject var state: AppState
    let onClose: () -> Void

    private var logs: [LogLine] {
        Array(Array(state.logs.suffix(120)).reversed())
    }

    var body: some View {
        NavigationStack {
            List {
                if !state.runtimeTasks.isEmpty {
                    Section("当前状态") {
                        ForEach(state.runtimeTasks) { task in
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: task.systemImage)
                                    .foregroundStyle(task.tone.color)
                                    .frame(width: 22)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(task.title)
                                        .font(.subheadline.weight(.semibold))
                                    Text(task.detail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                if task.showsProgress {
                                    Spacer()
                                    ProgressView()
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }

                Section("最近动态") {
                    if logs.isEmpty {
                        Text("暂无动态")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(logs) { log in
                            Text(log.text)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .padding(.vertical, 2)
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("动态")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("关闭动态")
                }
            }
        }
    }
}

private struct ContextWorkspaceHeader: View {
    @ObservedObject var state: AppState
    let draft: String
    let contextItems: [ContextBadgeItem]

    private var enabledSummary: String {
        state.contextInclusionSettings.enabledContextLabels.joined(separator: "、")
    }

    private var assetSummary: ContextAssetSummary {
        state.contextAssetSummary(for: draft)
    }

    private var visibleContextCount: Int {
        contextItems.filter { $0.id != "empty-question" }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: "tray.full")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 28, height: 28)

                VStack(alignment: .leading, spacing: 2) {
                    Text("本次会发送 \(visibleContextCount) 类上下文")
                        .font(.subheadline.weight(.semibold))
                    Text("已启用：\(enabledSummary)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                Text("\(assetSummary.total)")
                    .font(.callout.monospacedDigit().weight(.bold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.secondarySystemBackground), in: Capsule())
                    .accessibilityLabel("候选学习资产 \(assetSummary.total) 条")
            }

            Text("先看总览；错题、记忆、完整 JSON 分页查看，打开会更快。配置里的开关会影响实际发给大模型的上下文。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct ContextWorkspaceOverviewTab: View {
    @ObservedObject var state: AppState
    let draft: String

    /// 是否已有本轮真相（来自 context_trace）；首轮/无 manifest 时仍展示各类开关与说明。
    private var usingTurnTruth: Bool {
        state.lastTurnManifest != nil
    }

    var body: some View {
        // #7 朗读/纯文字提问/一句话辅导偏好属于「系统设置」，已在「设置」页提供；
        // 上下文面板只负责「配置本轮发送什么上下文」，不再混入这些开关。
        ContextSectionCard(title: usingTurnTruth ? "这一轮发给 AI 的内容" : "下一轮会发给 AI 的内容", systemImage: "tray.full") {
            VStack(alignment: .leading, spacing: 10) {
                Text(usingTurnTruth
                     ? "下面按每一类分别说明：这一轮有没有发给 AI、发了什么。绿点=本轮带上了，灰点=本轮没用到。每一类都能关掉，关了下一轮就不发。"
                     : "先提一个问题，AI 回答后，这里会逐类显示本轮真正发给 AI 的内容。下面是各类开关与说明。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                ForEach(ContextCategory.ordered, id: \.key) { category in
                    ContextCategoryCard(state: state, category: category)
                }
            }
        }

        ContextSectionCard(title: "使用策略", systemImage: "flowchart") {
            SelectableTextBlock(text: state.contextUsePolicyPreview(draft: draft), lineLimit: 8)
        }
    }
}

/// 一个「配置开关」对应的上下文类别（与设置页顺序一致）。普通家长/学生看得懂的中文名 +
/// 一句话大白话说明 + 对应 trace key + 对应类级开关 keyPath。
struct ContextCategory {
    let key: String                 // 与 context_trace channel.key / inclusionKeyPath 对应
    let title: String               // 中文名
    let blurb: String               // 一句话「这是什么/为什么影响 AI」
    let systemImage: String
    let keyPath: WritableKeyPath<ContextInclusionSettings, Bool>

    /// 顺序与「发送配置」设置页一致（视觉画面/智能观察/历史记录/错题本/知识库/长期记忆/学习策略）。
    static let ordered: [ContextCategory] = [
        ContextCategory(key: "visual", title: "视觉画面",
                        blurb: "把当前拍到的题目画面发给 AI，让它看清题再回答。",
                        systemImage: "camera.metering.center.weighted", keyPath: \.visual),
        ContextCategory(key: "observation", title: "智能观察",
                        blurb: "把后台连续观察到的学习画面摘要发给 AI，帮它了解你正在做什么。",
                        systemImage: "rectangle.stack", keyPath: \.observation),
        ContextCategory(key: "history", title: "历史记录",
                        blurb: "把之前对话里相关的内容发给 AI，让追问能接上、不用重复说。",
                        systemImage: "clock.arrow.circlepath", keyPath: \.history),
        ContextCategory(key: "mistakes", title: "错题本",
                        blurb: "把相关的错题、错因和订正线索发给 AI，复习时更有针对性。",
                        systemImage: "book.closed", keyPath: \.mistakes),
        ContextCategory(key: "knowledge", title: "知识库",
                        blurb: "把和这道题相关的知识点发给 AI，方便讲清概念、举一反三。",
                        systemImage: "lightbulb", keyPath: \.knowledge),
        ContextCategory(key: "memory", title: "长期记忆",
                        blurb: "把 AI 记住的你的偏好、常错点、学习目标发给它，让回答更贴合你。",
                        systemImage: "brain.head.profile", keyPath: \.memory),
        ContextCategory(key: "strategy", title: "学习策略",
                        blurb: "把你的辅导偏好（怎么讲、讲多深）和本轮状态发给 AI，决定它怎么回答。",
                        systemImage: "slider.horizontal.3", keyPath: \.strategy),
    ]
}

/// 单类上下文分区卡：中文名 + 大白话说明 + 本轮是否带上（绿/灰点）+ 类级开关 + 可展开真实内容预览。
/// 两端（iPad/iPhone）共用的 ContextWorkspaceSheet 都走这里。
private struct ContextCategoryCard: View {
    @ObservedObject var state: AppState
    let category: ContextCategory

    @State private var expanded = false

    private var trace: ContextChannelTrace? {
        state.lastTurnManifest?.channel(category.key)
    }
    private var hasTurn: Bool { state.lastTurnManifest != nil }
    private var included: Bool { trace?.included ?? false }
    private var enabled: Bool { state.contextInclusionSettings[keyPath: category.keyPath] }

    /// 本轮状态文案（绿点=带上了；灰点=没用到 / 已关闭 / 还没提问）。
    private var statusText: String {
        if !enabled { return "已关闭，本轮不发" }
        if !hasTurn { return "等待提问" }
        return included ? "本轮已带上" : "本轮没用到"
    }
    private var statusColor: Color {
        if !enabled { return .secondary }
        if !hasTurn { return .orange }
        return included ? .green : .secondary
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 9) {
                Image(systemName: category.systemImage)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(included && enabled ? Color.green : Color.secondary)
                    .frame(width: 22)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                        Text(category.title)
                            .font(.caption.weight(.semibold))
                        Text(statusText)
                            .font(.caption2)
                            .foregroundStyle(statusColor)
                    }
                    Text(category.blurb)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                Toggle("", isOn: Binding(
                    get: { state.contextInclusionSettings[keyPath: category.keyPath] },
                    set: { state.updateContextInclusion(category.keyPath, to: $0) }
                ))
                .labelsHidden()
            }

            // 真实内容预览：本轮带上了才展示「实际发给 AI 的内容」。
            if enabled && included, hasPreviewContent {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
                } label: {
                    Label(expanded ? "收起实际内容" : "查看实际发给 AI 的内容",
                          systemImage: expanded ? "chevron.up" : "chevron.down")
                        .font(.caption2.weight(.semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tint)

                if expanded {
                    previewContent
                        .padding(.top, 2)
                }
            } else if enabled && !included && hasTurn {
                Text(emptyHint)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((included && enabled ? Color.green : Color.secondary).opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var emptyHint: String {
        switch category.key {
        case "visual": return "本轮没有用到画面（可能是纯文字提问，或没拍到题）。"
        case "observation": return "本轮没有用到智能观察画面。"
        case "history": return "本轮没有携带历史记录。"
        case "mistakes": return "本轮没有用到错题本。"
        case "knowledge": return "本轮没有命中相关知识点。"
        case "memory": return "本轮没有用到长期记忆。"
        case "strategy": return "本轮没有带上学习策略。"
        default: return "本轮没用到。"
        }
    }

    private var hasPreviewContent: Bool {
        guard let trace else { return false }
        switch category.key {
        case "visual": return !trace.visualFilename.isEmpty
        case "history": return !trace.historyPreview.isEmpty
        case "mistakes": return !trace.mistakeItems.isEmpty
        case "knowledge": return !trace.knowledgeHits.isEmpty
        case "memory": return !(state.lastTurnManifest?.retrievedMemories.isEmpty ?? true)
        case "observation": return trace.observationFrames > 0
        case "strategy": return !trace.strategyLearningMode.isEmpty || !trace.strategyCoachDepth.isEmpty
        default: return false
        }
    }

    @ViewBuilder
    private var previewContent: some View {
        switch category.key {
        case "visual":
            if let url = contextImageThumbnailURL(filename: trace?.visualFilename ?? "") {
                VStack(alignment: .leading, spacing: 4) {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image):
                            image.resizable().scaledToFill()
                        case .failure:
                            Image(systemName: "photo.badge.exclamationmark").foregroundStyle(.secondary)
                        default:
                            ProgressView()
                        }
                    }
                    .frame(width: 132, height: 99)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(.separator), lineWidth: 1))
                    if let mode = trace?.visualMode, !mode.isEmpty {
                        Text("画面模式：\(mode)")
                            .font(.system(size: 10)).foregroundStyle(.secondary)
                    }
                }
            }
        case "history":
            SelectableTextBlock(text: trace?.historyPreview ?? "", lineLimit: 8)
        case "mistakes":
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array((trace?.mistakeItems ?? []).enumerated()), id: \.offset) { _, item in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 5) {
                            if item.active {
                                Text("本轮复习")
                                    .font(.system(size: 9).weight(.semibold))
                                    .padding(.horizontal, 5).padding(.vertical, 1)
                                    .background(Color.orange.opacity(0.2))
                                    .clipShape(Capsule())
                            }
                            Text(item.title)
                                .font(.caption2.weight(.semibold))
                        }
                        if !item.detail.isEmpty {
                            Text(item.detail)
                                .font(.caption2).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        case "knowledge":
            VStack(alignment: .leading, spacing: 5) {
                ForEach(Array((trace?.knowledgeHits ?? []).enumerated()), id: \.offset) { _, hit in
                    HStack(alignment: .top, spacing: 5) {
                        Image(systemName: "lightbulb").font(.caption2).foregroundStyle(.secondary)
                        Text(hit.preview)
                            .font(.caption2)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 4)
                        if let score = hit.score {
                            Text(String(format: "%.2f", score))
                                .font(.system(size: 9).monospacedDigit())
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
            }
        case "memory":
            // 复用已有命中记忆视图（每条原文 + 4 维相关度条 + 逐条开关 + 进档案纠正）。
            ContextRetrievedMemorySection(state: state)
        case "observation":
            Text("本轮带上了约 \(trace?.observationFrames ?? 0) 帧后台观察画面摘要。")
                .font(.caption2).foregroundStyle(.secondary)
        case "strategy":
            VStack(alignment: .leading, spacing: 3) {
                if let mode = trace?.strategyLearningMode, !mode.isEmpty {
                    Text("场景：\(mode)").font(.caption2).foregroundStyle(.secondary)
                }
                if let depth = trace?.strategyCoachDepth, !depth.isEmpty {
                    Text("回答方式：\(depth)").font(.caption2).foregroundStyle(.secondary)
                }
                Text("可在「设置」里调整你的一句话辅导偏好。")
                    .font(.system(size: 10)).foregroundStyle(.tertiary)
            }
        default:
            EmptyView()
        }
    }
}

private struct ContextWorkspaceAssetsTab: View {
    @ObservedObject var state: AppState
    let draft: String

    private var assetGroups: [ContextAssetGroup] {
        state.contextAssetGroups(for: draft)
    }

    var body: some View {
        ContextSectionCard(title: "资产列表", systemImage: "books.vertical") {
            if assetGroups.allSatisfy({ $0.items.isEmpty }) {
                Text("暂无可用的错题、知识点或记忆。智能沉淀后会显示在这里，并作为后续上下文候选。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(assetGroups.filter { !$0.items.isEmpty }) { group in
                        ContextAssetGroupView(group: group)
                    }
                }
            }
        }
    }
}

private struct ContextWorkspacePromptsTab: View {
    @ObservedObject var state: AppState
    let draft: String

    var body: some View {
        ContextSectionCard(title: "用户提示词", systemImage: "text.bubble") {
            SelectableTextBlock(text: state.contextUserPromptPreview(draft: draft))
        }

        ContextSectionCard(title: "系统提示词", systemImage: "gearshape.2") {
            SelectableTextBlock(text: state.contextSystemPromptPreview)
        }

        ContextSectionCard(title: "使用策略", systemImage: "flowchart") {
            SelectableTextBlock(text: state.contextUsePolicyPreview(draft: draft))
        }
    }
}

private struct ContextWorkspaceDebugTab: View {
    @ObservedObject var state: AppState
    let draft: String

    var body: some View {
        ContextSectionCard(title: "调试 JSON", systemImage: "curlybraces") {
            if state.contextInclusionSettings.debug {
                SelectableTextBlock(text: state.contextPayloadPreview(draft: draft), monospaced: true)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Label("完整 JSON 默认关闭", systemImage: "eye.slash")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text("需要排查模型上下文时，可在配置页打开“调试 JSON”。关闭时不会在面板打开时生成大段 JSON。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button {
                        state.setContextDebugEnabled(true)
                    } label: {
                        Label("打开调试 JSON", systemImage: "curlybraces")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
        }
    }
}

private struct ContextWorkspaceSettingsTab: View {
    @ObservedObject var state: AppState

    var body: some View {
        ContextSectionCard(title: "发送配置", systemImage: "switch.2") {
            VStack(alignment: .leading, spacing: 10) {
                // #7「纯文字提问」是系统级提问方式，已移到「设置」；这里只保留各类上下文的发送开关。
                ContextToggleRow(
                    title: "当前画面",
                    detail: "控制是否抓拍或复用题图；关闭后本轮按文字和对话理解。",
                    systemImage: "camera.metering.center.weighted",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.visual },
                        set: { state.updateContextInclusion(\.visual, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "智能观察",
                    detail: "控制后台观察帧摘要是否进入大模型上下文。",
                    systemImage: "rectangle.stack",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.observation },
                        set: { state.updateContextInclusion(\.observation, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "历史回合",
                    detail: "控制从历史对话携带来的压缩上下文。",
                    systemImage: "clock.arrow.circlepath",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.history },
                        set: { state.updateContextInclusion(\.history, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "错题",
                    detail: "控制到期错题、错因、订正线索。",
                    systemImage: "book.closed",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.mistakes },
                        set: { state.updateContextInclusion(\.mistakes, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "知识点",
                    detail: "控制知识点汇总、易错点和举一反三素材。",
                    systemImage: "list.bullet.rectangle",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.knowledge },
                        set: { state.updateContextInclusion(\.knowledge, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "记忆整理",
                    detail: "控制用户画像、近期输入、个性化偏好。",
                    systemImage: "brain.head.profile",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.memory },
                        set: { state.updateContextInclusion(\.memory, to: $0) }
                    )
                )
                ContextToggleRow(
                    title: "动态策略",
                    detail: "控制本轮偏好、当前画面、回合状态等组合策略。",
                    systemImage: "slider.horizontal.3",
                    isOn: Binding(
                        get: { state.contextInclusionSettings.strategy },
                        set: { state.updateContextInclusion(\.strategy, to: $0) }
                    )
                )
                // #8「调试 JSON」开关已随调试页卡一起从普通用户界面隐藏。
            }
        }
    }
}

private struct ContextToggleRow: View {
    let title: String
    let detail: String
    let systemImage: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn) {
            HStack(alignment: .top, spacing: 9) {
                Image(systemName: systemImage)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(isOn ? .blue : .secondary)
                    .frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.caption.weight(.semibold))
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .toggleStyle(.switch)
    }
}

private struct ContextSectionCard<Content: View>: View {
    let title: String
    let systemImage: String
    private let content: Content

    init(title: String, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.separator), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct ContextWorkspaceRow: View {
    let item: ContextBadgeItem
    var compact = false

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: item.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(item.tone.color)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(item.title)
                    .font(.caption.weight(.semibold))
                Text(compact ? item.detail : (item.fullDetail ?? item.detail))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(compact ? 2 : nil)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(item.tone.color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct ContextAssetGroupView: View {
    let group: ContextAssetGroup

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: group.systemImage)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                Text(group.title)
                    .font(.caption.weight(.semibold))
                Text("\(group.items.count)")
                    .font(.caption2.monospacedDigit().weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(.secondarySystemBackground), in: Capsule())
            }

            VStack(spacing: 7) {
                ForEach(group.items) { item in
                    ContextAssetRow(item: item)
                }
            }
        }
    }
}

private struct ContextAssetRow: View {
    let item: ContextAssetItem

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: item.systemImage)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(item.tone.color)
                    .frame(width: 22)

                VStack(alignment: .leading, spacing: 3) {
                    Text(item.title)
                        .font(.caption.weight(.semibold))
                        .lineLimit(1)
                    Text(item.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Label(item.useRule, systemImage: "arrow.triangle.branch")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(item.tone.color.opacity(0.08))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(item.tone.color.opacity(0.18), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct SelectableTextBlock: View {
    let text: String
    var monospaced = false
    var lineLimit: Int? = nil

    var body: some View {
        Text(text.isEmpty ? "无" : text)
            .font(monospaced ? .system(size: 12, design: .monospaced) : .caption)
            .foregroundStyle(.primary)
            .textSelection(.enabled)
            .lineLimit(lineLimit)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct HistorySessionSheet: View {
    @ObservedObject var state: AppState
    let onStartNew: (HistorySessionSummary) -> Void
    let onViewReport: (HistorySessionSummary) -> Void

    var body: some View {
        NavigationStack {
            Group {
                if state.isLoadingHistory {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("正在读取历史对话")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if state.historySessions.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "clock.arrow.circlepath")
                            .font(.largeTitle)
                            .foregroundStyle(.secondary)
                        Text("暂无历史对话")
                            .font(.headline)
                        Text("完成一次问答或智能观察后，这里会显示可复用的历史回合。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        Section {
                            ForEach(state.historySessions) { session in
                                VStack(alignment: .leading, spacing: 10) {
                                    HistorySessionRow(session: session)
                                    HStack(spacing: 8) {
                                        Button {
                                            onStartNew(session)
                                        } label: {
                                            Label("新对话", systemImage: "plus.bubble")
                                                .frame(maxWidth: .infinity)
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .controlSize(.small)

                                        Button {
                                            onViewReport(session)
                                        } label: {
                                            Label("查看报告", systemImage: session.imageCount > 0 ? "doc.text.magnifyingglass" : "text.bubble")
                                                .frame(maxWidth: .infinity)
                                        }
                                        .buttonStyle(.bordered)
                                        .controlSize(.small)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        } footer: {
                            Text("新对话会携带该回合压缩上下文；查看报告会打开观察报告，纯问答回合会生成一段回合总结。")
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("历史对话")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await state.refreshHistorySessions() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("刷新历史对话")
                }
            }
        }
        .task {
            if state.historySessions.isEmpty {
                await state.refreshHistorySessions()
            }
        }
    }
}

private struct HistorySessionRow: View {
    let session: HistorySessionSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline) {
                Text(session.displayTitle)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Spacer()
                Text(session.displayTime)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Text(session.summaryPreview.isEmpty ? session.studentGoal : session.summaryPreview)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            Label(session.countSummary, systemImage: "tray.full")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

private struct HistoryReportSheet: View {
    let report: HistoryReportDetail
    var onClose: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(report.title)
                            .font(.headline)
                            .fixedSize(horizontal: false, vertical: true)
                        Label(report.subtitle, systemImage: report.systemImage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    SelectableTextBlock(text: report.content)

                    if !report.qaRounds.isEmpty {
                        // #5 每回合一张卡，标题=问题第一句。
                        VStack(alignment: .leading, spacing: 8) {
                            Label("问答回顾", systemImage: "bubble.left.and.bubble.right")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            ForEach(Array(report.qaRounds.enumerated()), id: \.element.id) { idx, round in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 6) {
                                        Text("\(idx + 1)")
                                            .font(.caption2.weight(.bold).monospacedDigit())
                                            .foregroundStyle(.white)
                                            .frame(width: 18, height: 18)
                                            .background(Color.accentColor, in: Circle())
                                        Text(round.title.isEmpty ? "第 \(idx + 1) 回合" : round.title)
                                            .font(.subheadline.weight(.semibold))
                                            .lineLimit(2)
                                    }
                                    if !round.question.isEmpty {
                                        Text("问：\(round.question)").font(.caption).foregroundStyle(.secondary)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                    if !round.answer.isEmpty {
                                        Text("答：\(round.answer)").font(.caption)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    } else if !report.qaPreview.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Label("最近问答", systemImage: "bubble.left.and.bubble.right")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            SelectableTextBlock(text: report.qaPreview)
                        }
                    }
                }
                .padding(16)
            }
            .navigationTitle("查看报告")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") {
                        onClose()
                    }
                }
            }
        }
    }
}

struct LogLine: Identifiable {
    let id = UUID()
    let text: String
}

struct ObservationStopNotice: Identifiable, Equatable {
    let id = UUID()
    let message: String
}

enum ChatMessageRole: Equatable {
    case user
    case assistant
    case status
}

enum CameraTaskKind: Equatable {
    case none
    case singleCapture
    case burst
    case qaFrame
    case extract   // 题目提取取景（连拍多张试卷，逐张提取去重）
}

enum TTSPlaybackPhase: Equatable {
    case idle
    case generating
    case playing
    case paused

    var isActive: Bool {
        self != .idle
    }
}

private enum TTSService: String, CaseIterable {
    case local
    case primary
    case fallback

    var displayName: String {
        switch self {
        case .local:
            return "本机语音"
        case .primary:
            return "网络语音"
        case .fallback:
            return "备用语音"
        }
    }
}

private enum ComposerInputMode {
    case voice
    case text
}

enum LearningCaptureSource: Equatable {
    case manualMistake
    case manualMemory
    case smartCapture

    var mistakeReason: String {
        switch self {
        case .smartCapture:
            return "智能沉淀：本轮问答可能包含易错点、关键知识点或后续复习价值"
        case .manualMistake, .manualMemory:
            return "用户手动从 AI 回答加入错题本"
        }
    }

    var mistakeSourceSummary: String {
        switch self {
        case .smartCapture:
            return "iOS 智能沉淀为错题线索"
        case .manualMistake, .manualMemory:
            return "iOS 手动加入错题本"
        }
    }

    var knowledgeTag: String {
        switch self {
        case .smartCapture:
            return "智能沉淀"
        case .manualMistake, .manualMemory:
            return "手动收藏"
        }
    }

    var memorySource: String {
        switch self {
        case .smartCapture:
            return "ios-smart-capture"
        case .manualMistake, .manualMemory:
            return "ios-manual"
        }
    }

    var memoryAction: String {
        switch self {
        case .smartCapture:
            return "assistant_answer_smart_capture"
        case .manualMistake, .manualMemory:
            return "assistant_answer_button"
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: ChatMessageRole
    var text: String
    var question: String = ""
    var qaEventId: String = ""
    var visualizationCandidate = false
    var visualizationReason: String = ""
    var visualization: TeachingVisualization?
    /// 这轮是否值得展示学习动作（举一反三/错题本/可视化等）。闲聊（hi/谢谢）由后端判为 false。
    var actionable = true
    var title: String?
    var systemImage: String?
    var showsProgress = false
    var statusKey: String? = nil   // 同 key 的进度状态可被「就地更新」（spinner→结果），避免重复气泡 + 永久转圈。
    var contextItems: [ContextBadgeItem] = []
    var attachments: [ChatAttachment] = []
}

struct TeachingVisualization: Equatable {
    var id: String
    var status: String
    var title: String
    var url: String
    var triggerReason: String
    var canOpen: Bool

    init(id: String = "", status: String = "", title: String = "", url: String = "", triggerReason: String = "", canOpen: Bool = false) {
        self.id = id
        self.status = status
        self.title = title
        self.url = url
        self.triggerReason = triggerReason
        self.canOpen = canOpen
    }

    init?(json: [String: Any]?) {
        guard let json else { return nil }
        let id = json["id"] as? String ?? ""
        let status = json["status"] as? String ?? ""
        let title = json["title"] as? String ?? ""
        let url = json["url"] as? String ?? ""
        let triggerReason = (json["trigger_reason"] as? String) ?? (json["triggerReason"] as? String) ?? ""
        let canOpen = (json["can_open"] as? Bool) ?? (json["canOpen"] as? Bool) ?? !url.isEmpty
        guard !id.isEmpty || !url.isEmpty || !status.isEmpty else { return nil }
        self.init(id: id, status: status, title: title, url: url, triggerReason: triggerReason, canOpen: canOpen)
    }

    var absoluteURL: URL? {
        guard !url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return URL(string: url, relativeTo: serverBaseURL)?.absoluteURL
    }
}

struct ChatAttachment: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    var image: UIImage?
    var thumbnailURL: URL?
    var fullURL: URL?
}

struct ContextBadgeItem: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let systemImage: String
    let tone: CaptureQualityTone
    var fullDetail: String? = nil
    /// "为何这一类上下文被本轮调用" — client-generated explanation (CUT-6: not from backend).
    var reason: String? = nil
}

/// The single source of truth for "what this turn actually carried", reconciled from the
/// server's `context_trace` after the answer returns (MUST-5: fixes the one-turn lag).
/// Deliberately holds only the current turn — no per-turn history dictionary (CUT-2).
struct TurnContextManifest {
    let id: UUID
    let turn: Int
    var sentItems: [ContextBadgeItem]
    var retrievedMemories: [RetrievedMemory]
    var usedImageContext: Bool
    var imageContextMode: String
    /// 本轮每个上下文通道的原始 trace（来自后端 context_trace.channels）。
    /// 让分类卡能直接读「本轮是否带上 + 真实内容预览」，而不再依赖摘要徽章。
    var channels: [ContextChannelTrace] = []

    func channel(_ key: String) -> ContextChannelTrace? {
        channels.first { $0.key == key }
    }
}

/// 一个上下文通道的本轮真相：是否带上 + 后端给的 detail（含真实内容预览）。
/// 直接镜像 context_trace.channels 的一项，分类卡按 key 取用。
struct ContextChannelTrace {
    let key: String
    let included: Bool
    let detail: [String: Any]

    /// 本轮历史回合携带的真实正文预览（后端 history.detail.preview）。
    var historyPreview: String { (detail["preview"] as? String) ?? "" }
    var historyChars: Int { (detail["chars"] as? Int) ?? 0 }

    /// 本轮命中的错题列表（后端 mistakes.detail.items：title/detail/active）。
    var mistakeItems: [(title: String, detail: String, active: Bool)] {
        guard let raw = detail["items"] as? [[String: Any]] else { return [] }
        return raw.map {
            (title: ($0["title"] as? String) ?? "",
             detail: ($0["detail"] as? String) ?? "",
             active: ($0["active"] as? Bool) ?? false)
        }
    }
    var mistakeCount: Int { (detail["count"] as? Int) ?? mistakeItems.count }

    /// 本轮命中的知识点条目（后端 knowledge.detail.semantic_hits：kind/score/preview）。
    var knowledgeHits: [(preview: String, score: Double?)] {
        guard let raw = detail["semantic_hits"] as? [[String: Any]] else { return [] }
        return raw.map {
            (preview: ($0["preview"] as? String) ?? "",
             score: $0["score"] as? Double)
        }
    }

    /// 本轮带入画面的文件名（后端 visual.detail.filename），用于显示缩略图。
    var visualFilename: String { (detail["filename"] as? String) ?? "" }
    var visualMode: String { (detail["mode"] as? String) ?? "" }
    var visualRejected: Bool { (detail["rejected"] as? Bool) ?? false }

    /// 本轮智能观察缓冲帧数（后端 observation.detail.frames）。
    var observationFrames: Int { (detail["frames"] as? Int) ?? 0 }

    /// 本轮动态策略带上的偏好（后端 strategy.detail）。
    var strategyLearningMode: String { (detail["learning_mode"] as? String) ?? "" }
    var strategyCoachDepth: String { (detail["coach_depth"] as? String) ?? "" }

    static func list(from trace: [String: Any]?) -> [ContextChannelTrace] {
        guard let channels = trace?["channels"] as? [[String: Any]] else { return [] }
        return channels.compactMap { channel in
            guard let key = channel["key"] as? String else { return nil }
            return ContextChannelTrace(
                key: key,
                included: (channel["included"] as? Bool) ?? false,
                detail: channel["detail"] as? [String: Any] ?? [:]
            )
        }
    }
}

/// A durable memory the server retrieved (semantically) into the most recent QA turn,
/// with the score breakdown that explains *why* it was carried into context.
/// `score == nil` marks an always-on persona memory (preference/goal/habit).
struct RetrievedMemory: Identifiable, Equatable {
    let id: String
    let kind: String
    let text: String
    let score: Double?
    let semantic: Double
    let recency: Double
    let importance: Double
    let usage: Double

    var isPersona: Bool { score == nil }

    var kindLabel: String {
        switch kind {
        case "preference": return "偏好"
        case "mistake": return "易错"
        case "goal": return "目标"
        case "habit": return "习惯"
        default: return "事实"
        }
    }

    static func list(from raw: Any?) -> [RetrievedMemory] {
        guard let array = raw as? [[String: Any]] else { return [] }
        return array.compactMap { item in
            guard let text = (item["text"] as? String), !text.isEmpty else { return nil }
            let breakdown = item["breakdown"] as? [String: Any] ?? [:]
            func num(_ value: Any?) -> Double {
                if let d = value as? Double { return d }
                if let n = value as? NSNumber { return n.doubleValue }
                return 0
            }
            return RetrievedMemory(
                id: (item["id"] as? String) ?? UUID().uuidString,
                kind: (item["kind"] as? String) ?? "fact",
                text: text,
                score: item["score"] as? Double,
                semantic: num(breakdown["semantic"]),
                recency: num(breakdown["recency"]),
                importance: num(breakdown["importance"]),
                usage: num(breakdown["usage"])
            )
        }
    }
}

struct ContextAssetGroup: Identifiable {
    let id: String
    let title: String
    let systemImage: String
    let items: [ContextAssetItem]
}

struct ContextAssetSummary {
    let mistakes: Int
    let knowledge: Int
    let memory: Int

    var total: Int {
        mistakes + knowledge + memory
    }
}

struct ContextAssetItem: Identifiable, Equatable {
    let id: String
    let kind: String
    let title: String
    let detail: String
    let useRule: String
    let systemImage: String
    let tone: CaptureQualityTone
    var fullDetail: String = ""
    var payload: [String: Any] = [:]

    static func == (lhs: ContextAssetItem, rhs: ContextAssetItem) -> Bool {
        lhs.id == rhs.id &&
        lhs.kind == rhs.kind &&
        lhs.title == rhs.title &&
        lhs.detail == rhs.detail &&
        lhs.useRule == rhs.useRule &&
        lhs.systemImage == rhs.systemImage &&
        lhs.tone == rhs.tone &&
        lhs.fullDetail == rhs.fullDetail
    }

    var contextBadge: ContextBadgeItem {
        ContextBadgeItem(
            id: "asset-\(id)",
            title: title,
            detail: detail,
            systemImage: systemImage,
            tone: tone,
            fullDetail: fullDetail.isEmpty ? "\(detail)\n\n使用规则：\(useRule)" : "\(fullDetail)\n\n使用规则：\(useRule)"
        )
    }

    var compactPayload: [String: Any] {
        var value = payload
        value["id"] = id
        value["kind"] = kind
        value["title"] = title
        value["detail"] = detail
        value["use_rule"] = useRule
        return value
    }
}

struct RuntimeTaskItem: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let systemImage: String
    let tone: CaptureQualityTone
    var showsProgress = false
    var canOpen = false
    var canClose = true
    var closeTitle = "关闭"
    var closeSystemImage = "xmark.circle"
}

// #8 后端后台生成任务（GET /api/tasks，账号隔离）。
struct ServerBackgroundTask: Identifiable, Decodable {
    let id: String
    let title: String
    let label: String
    let lane: String
    let state: String          // waiting / running / cancelling
    let sessionId: String
    let ageSeconds: Int

    enum CodingKeys: String, CodingKey {
        case id, title, label, lane, state
        case sessionId = "session_id"
        case ageSeconds = "age_seconds"
    }

    var stateText: String {
        switch state {
        case "running": return "生成中"
        case "cancelling": return "正在取消"
        default: return "排队中"
        }
    }

    var laneText: String {
        lane == "realtime" ? "实时优先" : "后台队列"
    }

    var systemImage: String {
        let text = (label + " " + title).lowercased()
        if text.contains("question") || text.contains("extract") || title.contains("题目") {
            return "rectangle.stack.badge.plus"
        }
        if text.contains("report") || title.contains("报告") {
            return "doc.text.magnifyingglass"
        }
        if text.contains("memory") || title.contains("记忆") {
            return "brain.head.profile"
        }
        return "sparkles"
    }
}

struct ServerTasksResponse: Decodable {
    let tasks: [ServerBackgroundTask]
}

struct TTSGenerateResponse: Decodable {
    let audioBase64: String

    enum CodingKeys: String, CodingKey {
        case audioBase64 = "audio_base64"
    }
}

struct HistorySessionsResponse: Decodable {
    let sessions: [HistorySessionSummary]
}

struct MemoryDigestResponse: Decodable {
    let profile: MemoryProfile
    let events: [MemoryEvent]
}

struct MemoryProfile: Decodable {
    let scope: String
    let profile: String
    let sourceCount: Int
    let latestEventAt: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case scope
        case profile
        case sourceCount = "source_count"
        case latestEventAt = "latest_event_at"
        case updatedAt = "updated_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        scope = (try? container.decode(String.self, forKey: .scope)) ?? "global"
        profile = (try? container.decode(String.self, forKey: .profile)) ?? ""
        sourceCount = (try? container.decode(Int.self, forKey: .sourceCount)) ?? 0
        latestEventAt = (try? container.decode(String.self, forKey: .latestEventAt)) ?? ""
        updatedAt = (try? container.decode(String.self, forKey: .updatedAt)) ?? ""
    }
}

struct MemoryEvent: Identifiable, Decodable {
    let id: String
    let sessionId: String
    let source: String
    let messageType: String
    let text: String
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case sessionId = "session_id"
        case source
        case messageType = "message_type"
        case text
        case createdAt = "created_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? UUID().uuidString
        sessionId = (try? container.decode(String.self, forKey: .sessionId)) ?? ""
        source = (try? container.decode(String.self, forKey: .source)) ?? ""
        messageType = (try? container.decode(String.self, forKey: .messageType)) ?? ""
        text = (try? container.decode(String.self, forKey: .text)) ?? ""
        createdAt = (try? container.decode(String.self, forKey: .createdAt)) ?? ""
    }
}

struct HistorySessionSummary: Identifiable, Decodable {
    let id: String
    let mode: String
    let title: String
    let status: String
    let createdAt: String
    let updatedAt: String
    let studentGoal: String
    let summaryPreview: String
    let firstQuestion: String
    let imageCount: Int
    let analysisCount: Int
    let qaCount: Int
    let mistakeCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case mode
        case title
        case status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case studentGoal = "student_goal"
        case summaryPreview = "summary_preview"
        case firstQuestion = "first_question"
        case imageCount = "image_count"
        case analysisCount = "analysis_count"
        case qaCount = "qa_count"
        case mistakeCount = "mistake_count"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? ""
        mode = (try? container.decode(String.self, forKey: .mode)) ?? ""
        title = (try? container.decode(String.self, forKey: .title)) ?? ""
        status = (try? container.decode(String.self, forKey: .status)) ?? ""
        createdAt = (try? container.decode(String.self, forKey: .createdAt)) ?? ""
        updatedAt = (try? container.decode(String.self, forKey: .updatedAt)) ?? ""
        studentGoal = (try? container.decode(String.self, forKey: .studentGoal)) ?? ""
        summaryPreview = (try? container.decode(String.self, forKey: .summaryPreview)) ?? ""
        firstQuestion = (try? container.decode(String.self, forKey: .firstQuestion)) ?? ""
        imageCount = (try? container.decode(Int.self, forKey: .imageCount)) ?? 0
        analysisCount = (try? container.decode(Int.self, forKey: .analysisCount)) ?? 0
        qaCount = (try? container.decode(Int.self, forKey: .qaCount)) ?? 0
        mistakeCount = (try? container.decode(Int.self, forKey: .mistakeCount)) ?? 0
    }

    /// 回合类型徽章：区分「实时对话」与「智能观察」（#4）。
    var modeBadge: String {
        switch mode {
        case "qa", "today_review":
            return "实时对话"
        case "burst", "observation":
            return "智能观察"
        default:
            return qaCount > 0 ? "实时对话" : "智能观察"
        }
    }

    /// 标题优先取「第一句提问」，让历史一眼能认出聊了什么；纯观察无提问时退回目标/摘要（#4）。
    var displayTitle: String {
        let q = firstQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        if !q.isEmpty {
            // 取第一句（按中英文句末标点/换行截断），并限制长度。
            let firstSentence = q.split(whereSeparator: { "。！？!?\n".contains($0) }).first.map(String.init) ?? q
            return shortText(firstSentence.trimmingCharacters(in: .whitespacesAndNewlines), limit: 30)
        }
        return firstNonEmpty(studentGoal, summaryPreview, modeBadge + "回合", "历史回合 " + String(id.prefix(8)))
    }

    var countSummary: String {
        "\(imageCount) 张图 · \(qaCount) 次问答 · \(mistakeCount) 个错题"
    }

    var displayTime: String {
        HistoryDateFormatter.displayString(from: firstNonEmpty(updatedAt, createdAt))
    }
}

// 提示词记录：对应 GET /api/prompts 返回的每条记录（只读预览 + 是否自定义）。
struct CoachPromptRecord: Identifiable, Decodable {
    let key: String
    let label: String
    let description: String
    let content: String
    let isCustom: Bool

    var id: String { key }

    enum CodingKeys: String, CodingKey {
        case key
        case label
        case description
        case content
        case isCustom = "is_custom"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        key = (try? container.decode(String.self, forKey: .key)) ?? ""
        label = (try? container.decode(String.self, forKey: .label)) ?? ""
        description = (try? container.decode(String.self, forKey: .description)) ?? ""
        content = (try? container.decode(String.self, forKey: .content)) ?? ""
        isCustom = (try? container.decode(Bool.self, forKey: .isCustom)) ?? false
    }
}

private struct CoachPromptsResponse: Decodable {
    let prompts: [CoachPromptRecord]
}

private struct CoachPromptResetResponse: Decodable {
    let prompt: CoachPromptRecord
}

private enum HistoryDateFormatter {
    private static let outputFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.dateFormat = "MM-dd HH:mm"
        return formatter
    }()

    static func displayString(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if let date = ISO8601DateFormatter.paiCodexFormatter.date(from: trimmed) ??
            ISO8601DateFormatter.paiCodexFractionalFormatter.date(from: trimmed) {
            return outputFormatter.string(from: date)
        }
        return trimmed.replacingOccurrences(of: "T", with: " ").prefix(16).description
    }
}

private extension ISO8601DateFormatter {
    static let paiCodexFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static let paiCodexFractionalFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}

private struct HistoryCarryContext {
    let sourceSessionId: String
    let title: String
    let summary: String
    let detail: String
    let payload: [String: Any]
    let previewFilename: String?

    var badge: ContextBadgeItem {
        ContextBadgeItem(
            id: "history-context",
            title: "历史回合",
            detail: summary,
            systemImage: "clock.arrow.circlepath",
            tone: .neutral,
            fullDetail: detail
        )
    }
}

struct HistoryQARound: Identifiable {
    let id = UUID()
    let title: String       // #5 回合标题：取问题第一句（空则取答案第一句）
    let question: String
    let answer: String
}

struct HistoryReportDetail: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String
    let content: String
    let qaPreview: String
    let qaRounds: [HistoryQARound]
    let systemImage: String
}

/// #5 取一段文本的「第一句」作为回合标题：在第一个句末标点（。！？.!?\n）处截断，限长。
func firstSentenceTitle(_ text: String, limit: Int = 24) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return "" }
    let terminators = CharacterSet(charactersIn: "。！？.!?\n")
    if let range = trimmed.rangeOfCharacter(from: terminators) {
        let sentence = String(trimmed[trimmed.startIndex..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
        if !sentence.isEmpty { return shortText(sentence, limit: limit) }
    }
    return shortText(trimmed, limit: limit)
}

struct QAOverlay: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: state.qaSystemImage)
                    .font(.title3)
                Text(state.qaStateText)
                    .font(.headline)
                Spacer()
                if state.isThinking {
                    ProgressView()
                }
            }

            if !state.recognizedText.isEmpty {
                Text(state.recognizedText)
                    .font(.subheadline)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if !state.qaAnswer.isEmpty {
                ScrollView {
                    QARichAnswerView(answer: state.qaAnswer)
                }
                .frame(maxHeight: 230)
            }

            HStack(spacing: 8) {
                Button {
                    state.startNewConversation()
                } label: {
                    Label("新对话", systemImage: "plus.bubble")
                }
                .buttonStyle(.bordered)
                .disabled(!state.canStartNewConversation)

                Spacer()
            }
            .font(.caption)
        }
        .padding(14)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .shadow(radius: 12)
    }
}

struct QARichAnswerView: View {
    let answer: String

    var sections: [QARichAnswerSection] {
        QARichAnswerSection.parse(answer)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(sections) { section in
                HStack(alignment: .top, spacing: 8) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(section.tint)
                        .frame(width: 3)
                    VStack(alignment: .leading, spacing: 5) {
                        HStack(spacing: 5) {
                            Text(section.icon)
                                .font(.caption)
                            Text(section.title)
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundStyle(section.tint)
                        }
                        if section.lines.count == 1 {
                            Text(section.lines[0])
                                .font(.callout)
                                .fixedSize(horizontal: false, vertical: true)
                        } else {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(Array(section.lines.enumerated()), id: \.offset) { index, line in
                                    HStack(alignment: .top, spacing: 6) {
                                        Text("\(index + 1).")
                                            .font(.caption)
                                            .monospacedDigit()
                                            .foregroundStyle(.secondary)
                                            .frame(width: 16, alignment: .trailing)
                                        Text(line)
                                            .font(.callout)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.vertical, 7)
                .padding(.horizontal, 9)
                .background(section.tint.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct QARichAnswerSection: Identifiable {
    let id = UUID()
    let title: String
    let lines: [String]
    let tint: Color
    let icon: String

    static func parse(_ answer: String) -> [QARichAnswerSection] {
        var sections: [(String, [String])] = []
        var currentTitle = "回答"
        var currentLines: [String] = []

        func flush() {
            let clean = currentLines
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if !clean.isEmpty {
                sections.append((normalizedTitle(currentTitle), clean))
            }
            currentLines.removeAll()
        }

        for rawLine in answer.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            if let split = splitLabeledLine(line) {
                flush()
                currentTitle = split.title
                if !split.body.isEmpty {
                    currentLines.append(split.body)
                }
            } else {
                currentLines.append(stripListPrefix(line))
            }
        }
        flush()
        if sections.isEmpty {
            sections = [("回答", [answer.trimmingCharacters(in: .whitespacesAndNewlines)])]
        }
        return sections.map { title, lines in
            QARichAnswerSection(title: title, lines: lines, tint: tint(for: title), icon: icon(for: title))
        }
    }

    private static func splitLabeledLine(_ line: String) -> (title: String, body: String)? {
        let separators = ["：", ":"]
        for separator in separators {
            guard let range = line.range(of: separator) else { continue }
            let title = String(line[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let body = String(line[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if (2...12).contains(title.count), title.rangeOfCharacter(from: .letters) != nil {
                return (title, stripListPrefix(body))
            }
        }
        return nil
    }

    private static func stripListPrefix(_ value: String) -> String {
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefixes = ["- ", "* ", "• "]
        for prefix in prefixes where text.hasPrefix(prefix) {
            text.removeFirst(prefix.count)
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let dot = text.firstIndex(where: { $0 == "." || $0 == "、" || $0 == ")" || $0 == "）" }) {
            let prefix = text[..<dot]
            if !prefix.isEmpty, prefix.allSatisfy({ $0.isNumber || $0 == "(" || $0 == "（" }) {
                text = String(text[text.index(after: dot)...])
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func normalizedTitle(_ title: String) -> String {
        let clean = title.trimmingCharacters(in: .whitespacesAndNewlines)
        switch clean {
        case "题号", "题目", "问题":
            return "题目"
        case "已知", "条件", "关键条件":
            return "关键条件"
        case "学生答案", "作答":
            return "学生答案"
        case "检查结果", "结果":
            return "检查结果"
        case "思路", "解题思路":
            return "解题思路"
        case "步骤", "过程", "正确计算", "计算", "解法":
            return "解题步骤"
        case "错因", "错误原因":
            return "错因提醒"
        case "订正", "改正":
            return "订正建议"
        case "答案", "结论":
            return "结论"
        case "知识点", "知识点候选", "知识板块", "考点":
            return "知识点"
        case "追问建议", "可以追问", "下一步":
            return "追问建议"
        default:
            return clean.isEmpty ? "回答" : clean
        }
    }

    private static func icon(for title: String) -> String {
        switch title {
        case "题目": return "📖"
        case "关键条件": return "🔑"
        case "先想一想", "解题思路": return "💡"
        case "学生答案": return "✏️"
        case "检查结果": return "✅"
        case "解题步骤": return "🪜"
        case "错因提醒": return "⚠️"
        case "订正建议": return "🛠️"
        case "结论": return "🎯"
        case "知识点": return "📚"
        case "下一步小任务": return "🚀"
        case "追问建议": return "💬"
        default: return "📌"
        }
    }

    private static func tint(for title: String) -> Color {
        switch title {
        case "题目":
            return .blue
        case "关键条件", "解题思路":
            return .teal
        case "学生答案":
            return .purple
        case "检查结果", "错因提醒":
            return .red
        case "解题步骤":
            return .orange
        case "订正建议", "结论":
            return .green
        case "知识点":
            return .indigo
        case "追问建议":
            return .cyan
        default:
            return .secondary
        }
    }
}

struct CaptureQualityFeedback: Equatable {
    let title: String
    let detail: String
    let systemImage: String
    let tone: CaptureQualityTone

    static let idle = CaptureQualityFeedback(
        title: "对准学习材料",
        detail: "让课本、试卷或屏幕尽量完整入镜",
        systemImage: "camera.viewfinder",
        tone: .neutral
    )

    static let preparing = CaptureQualityFeedback(
        title: "创建学习回合",
        detail: "马上开始观察学习材料和画面变化",
        systemImage: "rectangle.stack",
        tone: .neutral
    )

    static let waitingForMaterial = CaptureQualityFeedback(
        title: "未检测到学习材料",
        detail: "建议调整相机或补全入镜，把纸张/屏幕放到画面中央",
        systemImage: "exclamationmark.triangle",
        tone: .warning
    )

    static let stopped = CaptureQualityFeedback(
        title: "已停止观察",
        detail: "本轮关键帧会继续上传并生成报告",
        systemImage: "stop.circle",
        tone: .neutral
    )

    static func observationStopped(collectedCount: Int, pendingCount: Int) -> CaptureQualityFeedback {
        let countText = collectedCount == 0 ? "尚未采集到学习相关图片" : "共采集 \(collectedCount) 张与学习相关的图片"
        let pendingText = pendingCount > 0 ? "，其中 \(pendingCount) 张正在异步传输到服务器" : "，图片已交给服务器后台处理"
        return CaptureQualityFeedback(
            title: "本轮观察已停止",
            detail: "\(countText)\(pendingText)，后台会自动生成该回合报告。",
            systemImage: "icloud.and.arrow.up",
            tone: collectedCount > 0 ? .good : .neutral
        )
    }

    static func materialVisible(signals: StudyMaterialSignals, cachedCount: Int) -> CaptureQualityFeedback {
        CaptureQualityFeedback(
            title: "学习材料可见",
            detail: "\(presenceHint(for: signals)) · 已缓存 \(cachedCount) 张关键帧",
            systemImage: signals.hasStudentPresence ? "hand.raised" : "checkmark.circle",
            tone: .good
        )
    }

    static func sceneChanged(signals: StudyMaterialSignals, cachedCount: Int, distance: Double?) -> CaptureQualityFeedback {
        let changeText = distance.map { String(format: "变化 %.1f", $0) } ?? "首张关键帧"
        return CaptureQualityFeedback(
            title: "画面有变化",
            detail: "\(changeText) · \(presenceHint(for: signals)) · 已缓存 \(cachedCount) 张",
            systemImage: "arrow.triangle.2.circlepath",
            tone: .good
        )
    }

    static func waitingForChange(signals: StudyMaterialSignals, distance: Double?, similarCount: Int) -> CaptureQualityFeedback {
        let changeText = distance.map { String(format: "变化 %.1f", $0) } ?? "变化很小"
        let waitText = similarCount >= 4 ? "等待翻页、书写或补充步骤后再上传" : "当前画面相似，先不重复上传"
        return CaptureQualityFeedback(
            title: "等待画面变化",
            detail: "\(waitText) · \(changeText) · \(presenceHint(for: signals))",
            systemImage: "pause.circle",
            tone: .waiting
        )
    }

    static func lowQuality(_ assessment: CaptureQualityAssessment) -> CaptureQualityFeedback {
        CaptureQualityFeedback(
            title: "未上传低质量画面",
            detail: assessment.userMessage,
            systemImage: "exclamationmark.triangle",
            tone: .warning
        )
    }

    static func cameraError(_ message: String) -> CaptureQualityFeedback {
        CaptureQualityFeedback(
            title: "相机不可用",
            detail: message,
            systemImage: "exclamationmark.triangle",
            tone: .warning
        )
    }

    static func uploading(_ state: String) -> CaptureQualityFeedback {
        CaptureQualityFeedback(
            title: state,
            detail: "正在把已保留的有效画面发送到后端",
            systemImage: "icloud.and.arrow.up",
            tone: .neutral
        )
    }

    private static func presenceHint(for signals: StudyMaterialSignals) -> String {
        if signals.handCount > 0 {
            return "手/笔/学生在场"
        }
        if signals.faceCount > 0 || signals.bodyCount > 0 {
            return "学生在场"
        }
        return "未确认手/学生，必要时补全入镜"
    }
}

struct CaptureQualityAssessment {
    let status: String
    let reasons: [String]
    let shouldUpload: Bool
    let blurScore: Double
    let materialConfidence: Double
    let occlusionScore: Double
    let motionScore: Double
    let isLowQuality: Bool

    var userMessage: String {
        if reasons.isEmpty {
            return "画面质量可用"
        }
        return reasons.map(Self.reasonText).joined(separator: "、")
    }

    static func evaluate(signals: StudyMaterialSignals, visualDistance: Double?, textDistance: Double?, isFirstUsefulFrame: Bool) -> CaptureQualityAssessment {
        let blurScore = max(0, min(1, signals.edgeDensity / 0.06))
        var materialConfidence = 0.0
        materialConfidence += min(Double(signals.textCount), 4.0) * 0.18
        materialConfidence += min(Double(signals.rectangleCount), 2.0) * 0.16
        materialConfidence += min(signals.lightCoverage / 0.22, 1.0) * 0.18
        materialConfidence += min(signals.edgeDensity / 0.07, 1.0) * 0.16
        materialConfidence += min(signals.contrast / 22.0, 1.0) * 0.16
        materialConfidence = max(0, min(1, materialConfidence))

        let occlusionScore = max(0, min(1, Double(signals.handCount) * 0.45 + Double(signals.bodyCount) * 0.25 + Double(signals.faceCount) * 0.15))
        let visualMotionScore = max(0, min(1, (visualDistance ?? 0) / 4.5))
        let textMotionScore = max(0, min(1, (textDistance ?? 0) / 0.35))
        let motionScore = isFirstUsefulFrame ? 1.0 : max(visualMotionScore, textMotionScore)
        var reasons: [String] = []

        if signals.maybeCameraCovered {
            reasons.append("camera_covered")
        }
        if !signals.hasStudyMaterial {
            reasons.append("no_material")
        }
        if blurScore < 0.32 && signals.textCount == 0 {
            reasons.append("too_blurry")
        }
        if occlusionScore >= 0.70 && signals.textCount < 2 {
            reasons.append("heavy_occlusion")
        }
        let shouldUpload = reasons.isEmpty
        return CaptureQualityAssessment(
            status: shouldUpload ? "ok" : "low_quality",
            reasons: reasons,
            shouldUpload: shouldUpload,
            blurScore: blurScore,
            materialConfidence: materialConfidence,
            occlusionScore: occlusionScore,
            motionScore: motionScore,
            isLowQuality: !shouldUpload
        )
    }

    private static func reasonText(_ reason: String) -> String {
        switch reason {
        case "camera_covered":
            return "镜头可能被手或物体挡住，请露出相机"
        case "no_material":
            return "未检测到学习材料"
        case "too_blurry":
            return "画面太模糊"
        case "heavy_occlusion":
            return "手/笔遮挡较多"
        case "not_enough_change":
            return "画面变化不足"
        default:
            return reason
        }
    }
}

enum CaptureQualityTone: Equatable {
    case neutral
    case good
    case waiting
    case warning

    var color: Color {
        switch self {
        case .neutral:
            return .secondary
        case .good:
            return .green
        case .waiting:
            return .blue
        case .warning:
            return .orange
        }
    }
}

private struct CaptureQualityBanner: View {
    let feedback: CaptureQualityFeedback

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: feedback.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(feedback.tone.color)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(feedback.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(feedback.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: 360, alignment: .leading)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .combine)
    }
}

private struct FrameFingerprint {
    let values: [UInt8]

    var hash: String {
        guard !values.isEmpty else { return "" }
        let mean = values.map(Double.init).reduce(0, +) / Double(values.count)
        var bytes = [UInt8]()
        var current: UInt8 = 0
        for (index, value) in values.enumerated() {
            if Double(value) >= mean {
                current |= UInt8(1 << (7 - (index % 8)))
            }
            if index % 8 == 7 {
                bytes.append(current)
                current = 0
            }
        }
        if values.count % 8 != 0 {
            bytes.append(current)
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    func distance(to other: FrameFingerprint) -> Double {
        guard values.count == other.values.count, !values.isEmpty else { return .greatestFiniteMagnitude }
        let mean = values.map(Double.init).reduce(0, +) / Double(values.count)
        let otherMean = other.values.map(Double.init).reduce(0, +) / Double(other.values.count)
        let total = zip(values, other.values).reduce(0.0) { partial, pair in
            let lhs = Double(pair.0) - mean
            let rhs = Double(pair.1) - otherMean
            return partial + abs(lhs - rhs)
        }
        return total / Double(values.count)
    }
}

struct StudyMaterialSignals {
    let rectangleCount: Int
    let textCount: Int
    let lightCoverage: Double
    let edgeDensity: Double
    let contrast: Double
    let blurScore: Double
    let materialConfidence: Double
    let handCount: Int
    let faceCount: Int
    let bodyCount: Int

    private static let strongMaterialConfidence = 0.45
    private static let softMaterialConfidence = 0.34

    var hasExplicitStudyEvidence: Bool {
        textCount >= 2 || rectangleCount >= 2
    }

    var isReliableQAContext: Bool {
        if textCount >= 4 {
            return blurScore >= 0.28 || contrast >= 8
        }
        if textCount >= 2 && rectangleCount >= 1 {
            return blurScore >= 0.28 || edgeDensity > 0.04 || contrast >= 10
        }
        if rectangleCount >= 2 {
            return lightCoverage > 0.14 && edgeDensity > 0.055 && contrast > 12
        }
        return false
    }

    var hasStudyMaterial: Bool {
        if materialConfidence >= Self.strongMaterialConfidence {
            return true
        }
        if textCount >= 2 { return true }
        if materialConfidence >= Self.softMaterialConfidence && (edgeDensity > 0.045 || contrast > 12 || lightCoverage > 0.14) {
            return true
        }
        if rectangleCount > 0 && (lightCoverage > 0.10 || edgeDensity > 0.035 || contrast > 13) {
            return true
        }
        return lightCoverage > 0.18 && edgeDensity > 0.055 && contrast > 10
    }

    var maybeCameraCovered: Bool {
        textCount == 0 &&
        rectangleCount == 0 &&
        handCount == 0 &&
        faceCount == 0 &&
        bodyCount == 0 &&
        lightCoverage < 0.035 &&
        edgeDensity < 0.018 &&
        contrast < 6
    }

    var hasStudentPresence: Bool {
        handCount > 0 || faceCount > 0 || bodyCount > 0
    }

    var studentPresenceStatus: String {
        hasStudentPresence ? "present" : "unknown"
    }

    var presenceSummary: String {
        if hasStudentPresence {
            var parts: [String] = []
            if handCount > 0 { parts.append("手\(handCount)") }
            if faceCount > 0 { parts.append("脸\(faceCount)") }
            if bodyCount > 0 { parts.append("身体\(bodyCount)") }
            return "学生在场：" + parts.joined(separator: "、")
        }
        return "学生在场未识别"
    }

    var activitySummary: String {
        if maybeCameraCovered {
            return "画面过暗且缺少纹理，镜头可能被手或物体挡住"
        }
        if handCount > 0 {
            return textCount >= 2 ? "手部在画面中，疑似读题/书写/指题" : "手部在画面中，动作未识别"
        }
        if faceCount > 0 || bodyCount > 0 {
            return "人体在画面中，具体动作未识别"
        }
        return "未检测到手/脸/身体，不能确认学生在场"
    }

    var summary: String {
        if maybeCameraCovered {
            return "画面过暗、纹理很低，可能遮挡镜头"
        }
        // #6 对话场景下相机多对着题目而非学生，「学生在场未识别」属于无意义噪音，会让人误解。
        // 仅在真的识别到人时才附带在场信息；否则只报画面信号。
        var parts = ["文字\(textCount) 处", "矩形\(rectangleCount) 个", "纹理\(String(format: "%.2f", edgeDensity))"]
        if hasStudentPresence {
            parts.append(presenceSummary)
        }
        return parts.joined(separator: "、")
    }
}

private struct BurstFrameAnalysis {
    let fingerprint: FrameFingerprint
    let signals: StudyMaterialSignals
    let textTokens: Set<String>
}

private struct QAFrameCandidate {
    let image: UIImage
    let analysis: BurstFrameAnalysis
    let assessment: CaptureQualityAssessment

    var shouldUseAsContext: Bool {
        assessment.shouldUpload && analysis.signals.isReliableQAContext
    }

    var userMessage: String {
        if assessment.shouldUpload && !analysis.signals.isReliableQAContext {
            return "沿用已有上下文继续回答"
        }
        return assessment.userMessage
    }
}

private struct BurstFrame {
    let image: UIImage
    let capturedAt: Date
    let sequenceIndex: Int
    let signalSummary: String
    let rectangleCount: Int
    let textCount: Int
    let lightCoverage: Double
    let edgeDensity: Double
    let contrast: Double
    let hasStudyMaterial: Bool
    let hasExplicitStudyEvidence: Bool
    let visualHash: String
    let visualSample: [UInt8]
    let textTokens: [String]
    let visualDistance: Double?
    let textDistance: Double?
    let qualityStatus: String
    let qualityReasons: [String]
    let shouldUpload: Bool
    let blurScore: Double
    let materialConfidence: Double
    let occlusionScore: Double
    let motionScore: Double
    let hasStudentPresence: Bool
    let studentPresenceStatus: String
    let presenceSummary: String
    let activitySummary: String
    let handCount: Int
    let faceCount: Int
    let bodyCount: Int
}

struct ObservationQuestionCandidate {
    let index: Int
    let key: String
    let fingerprint: String
    let preview: String
    let normalizedRect: CGRect
    let isDuplicate: Bool
}

struct QAFocus {
    var x: Double
    var y: Double
    var label: String
    var trigger: String
    var stableFrames: Int
    // 题目分割·选中题（可选；不设置=旧行为，后端不带这些键即旧逻辑）。
    var region: CGRect? = nil       // 归一化 bbox（左上原点），相对已校正上传图
    var questionIndex: Int? = nil   // 选中题在端上分割结果里的序号（1 基）
    var questionText: String? = nil // 端上 OCR 题干（可空）
    var crop: Bool = false          // 上传图是否已裁成该题
}

struct ReviewQueueResponse: Decodable {
    let items: [ReviewMistakeItem]
    let total: Int
    let dueOnly: Bool
    let pageSize: Int

    enum CodingKeys: String, CodingKey {
        case items
        case total
        case dueOnly = "due_only"
        case pageSize = "page_size"
    }
}

/// 一道错题的"来历"：何时生成、怎么采集、为什么判为错、判定依据、置信度。后端 build_mistake_provenance 输出。
struct MistakeProvenance: Decodable {
    let methodLabel: String
    let when: String
    let why: String
    let basis: String
    let confidenceLabel: String
    let sourceSummary: String

    enum CodingKeys: String, CodingKey {
        case methodLabel = "method_label"
        case when
        case why
        case basis
        case confidenceLabel = "confidence_label"
        case sourceSummary = "source_summary"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        methodLabel = (try? c.decode(String.self, forKey: .methodLabel)) ?? ""
        when = (try? c.decode(String.self, forKey: .when)) ?? ""
        why = (try? c.decode(String.self, forKey: .why)) ?? ""
        basis = (try? c.decode(String.self, forKey: .basis)) ?? ""
        confidenceLabel = (try? c.decode(String.self, forKey: .confidenceLabel)) ?? ""
        sourceSummary = (try? c.decode(String.self, forKey: .sourceSummary)) ?? ""
    }

    var whenDisplay: String { HistoryDateFormatter.displayString(from: when) }
}

struct ReviewMistakeItem: Decodable {
    let id: String
    let title: String
    let subject: String
    let pageRef: String
    let questionRef: String
    let locationRef: String
    let questionText: String
    let studentAnswer: String
    let expectedAnswer: String
    let errorType: String
    let errorReason: String
    let correction: String
    let nextAction: String
    let knowledgePoints: [String]
    let status: String
    let reviewState: String
    let nextReviewAt: String
    let reviewCount: Int
    let isDue: Bool
    let provenance: MistakeProvenance?

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case provenance
        case subject
        case pageRef = "page_ref"
        case questionRef = "question_ref"
        case locationRef = "location_ref"
        case questionText = "question_text"
        case studentAnswer = "student_answer"
        case expectedAnswer = "expected_answer"
        case errorType = "error_type"
        case errorReason = "error_reason"
        case correction
        case nextAction = "next_action"
        case knowledgePoints = "knowledge_points"
        case status
        case reviewState = "review_state"
        case nextReviewAt = "next_review_at"
        case reviewCount = "review_count"
        case isDue = "is_due"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? ""
        title = (try? container.decode(String.self, forKey: .title)) ?? ""
        subject = (try? container.decode(String.self, forKey: .subject)) ?? ""
        pageRef = (try? container.decode(String.self, forKey: .pageRef)) ?? ""
        questionRef = (try? container.decode(String.self, forKey: .questionRef)) ?? ""
        locationRef = (try? container.decode(String.self, forKey: .locationRef)) ?? ""
        questionText = (try? container.decode(String.self, forKey: .questionText)) ?? ""
        studentAnswer = (try? container.decode(String.self, forKey: .studentAnswer)) ?? ""
        expectedAnswer = (try? container.decode(String.self, forKey: .expectedAnswer)) ?? ""
        errorType = (try? container.decode(String.self, forKey: .errorType)) ?? ""
        errorReason = (try? container.decode(String.self, forKey: .errorReason)) ?? ""
        correction = (try? container.decode(String.self, forKey: .correction)) ?? ""
        nextAction = (try? container.decode(String.self, forKey: .nextAction)) ?? ""
        knowledgePoints = (try? container.decode([String].self, forKey: .knowledgePoints)) ?? []
        status = (try? container.decode(String.self, forKey: .status)) ?? ""
        reviewState = (try? container.decode(String.self, forKey: .reviewState)) ?? ""
        nextReviewAt = (try? container.decode(String.self, forKey: .nextReviewAt)) ?? ""
        reviewCount = (try? container.decode(Int.self, forKey: .reviewCount)) ?? 0
        isDue = (try? container.decode(Bool.self, forKey: .isDue)) ?? true
        provenance = try? container.decode(MistakeProvenance.self, forKey: .provenance)
    }

    var displayTitle: String {
        firstNonEmpty(title, questionRef, questionText, "这道错题")
    }

    var locationText: String {
        [subject, pageRef, questionRef, locationRef]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " · ")
    }

    var compactContext: [String: Any] {
        [
            "mistake_id": id,
            "title": title,
            "subject": subject,
            "page_ref": pageRef,
            "question_ref": questionRef,
            "location_ref": locationRef,
            "question_text": questionText,
            "student_answer": studentAnswer,
            "expected_answer": expectedAnswer,
            "error_type": errorType,
            "error_reason": errorReason,
            "correction": correction,
            "next_action": nextAction,
            "knowledge_points": knowledgePoints,
            "status": status,
            "review_state": reviewState,
            "next_review_at": nextReviewAt,
            "review_count": reviewCount,
            "is_due": isDue
        ]
    }
}

struct RemoteControlCommand: Decodable {
    let id: String
    let commandType: String
    let payload: [String: String]

    enum CodingKeys: String, CodingKey {
        case id
        case commandType = "command_type"
        case payload
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        commandType = try container.decode(String.self, forKey: .commandType)
        payload = (try? container.decode([String: String].self, forKey: .payload)) ?? [:]
    }
}

struct DeviceControlPollResponse: Decodable {
    let commands: [RemoteControlCommand]
    let pollIntervalSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case commands
        case pollIntervalSeconds = "poll_interval_seconds"
    }
}

private enum CaptureTimeFormatter {
    static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    static func string(from date: Date) -> String {
        iso8601.string(from: date)
    }
}

private enum BurstFrameAnalyzer {
    static func analyze(_ image: UIImage) -> BurstFrameAnalysis {
        let fingerprint = frameFingerprint(image)
        let studySignals = studyMaterialSignals(image)
        return BurstFrameAnalysis(
            fingerprint: fingerprint,
            signals: studySignals.signals,
            textTokens: studySignals.textTokens
        )
    }

    private static func frameFingerprint(_ image: UIImage) -> FrameFingerprint {
        guard let cg = image.cgImage else { return FrameFingerprint(values: []) }
        let width = 32
        let height = 32
        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        pixels.withUnsafeMutableBytes { ptr in
            if let ctx = CGContext(data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: 0) {
                ctx.interpolationQuality = .medium
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
        }
        return FrameFingerprint(values: pixels)
    }

    private static func studyMaterialSignals(_ image: UIImage) -> (signals: StudyMaterialSignals, textTokens: Set<String>) {
        let metrics = imageMetrics(image)
        let vision = visionStudySignals(image)
        let blurScore = max(0, min(1, metrics.edgeDensity / 0.06))
        var materialConfidence = 0.0
        materialConfidence += min(Double(vision.textTokens.count), 4.0) * 0.18
        materialConfidence += min(Double(vision.rectangleCount), 2.0) * 0.16
        materialConfidence += min(metrics.lightCoverage / 0.22, 1.0) * 0.18
        materialConfidence += min(metrics.edgeDensity / 0.07, 1.0) * 0.16
        materialConfidence += min(metrics.contrast / 22.0, 1.0) * 0.16
        materialConfidence = max(0, min(1, materialConfidence))
        return (
            StudyMaterialSignals(
                rectangleCount: vision.rectangleCount,
                textCount: vision.textTokens.count,
                lightCoverage: metrics.lightCoverage,
                edgeDensity: metrics.edgeDensity,
                contrast: metrics.contrast,
                blurScore: blurScore,
                materialConfidence: materialConfidence,
                handCount: vision.handCount,
                faceCount: vision.faceCount,
                bodyCount: vision.bodyCount
            ),
            vision.textTokens
        )
    }

    private static func imageMetrics(_ image: UIImage) -> (lightCoverage: Double, edgeDensity: Double, contrast: Double) {
        guard let cg = image.cgImage else { return (0, 0, 0) }
        let width = 48
        let height = 48
        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        pixels.withUnsafeMutableBytes { ptr in
            if let ctx = CGContext(data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: 0) {
                ctx.interpolationQuality = .medium
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
        }

        let xRange = 6..<(width - 6)
        let yRange = 6..<(height - 6)
        var lightPixels = 0
        var edgePixels = 0
        var totalPixels = 0
        var contrastTotal = 0.0

        for y in yRange {
            for x in xRange {
                let index = y * width + x
                let value = Int(pixels[index])
                if value > 168 {
                    lightPixels += 1
                }
                if x + 1 < width {
                    let diff = abs(value - Int(pixels[y * width + x + 1]))
                    contrastTotal += Double(diff)
                    if diff > 18 {
                        edgePixels += 1
                    }
                }
                if y + 1 < height {
                    let diff = abs(value - Int(pixels[(y + 1) * width + x]))
                    contrastTotal += Double(diff)
                    if diff > 18 {
                        edgePixels += 1
                    }
                }
                totalPixels += 1
            }
        }

        let edgeChecks = max(1, totalPixels * 2)
        return (
            Double(lightPixels) / Double(max(1, totalPixels)),
            Double(edgePixels) / Double(edgeChecks),
            contrastTotal / Double(edgeChecks)
        )
    }

    private static func visionStudySignals(_ image: UIImage) -> (rectangleCount: Int, textTokens: Set<String>, handCount: Int, faceCount: Int, bodyCount: Int) {
        let visionImage = image.resizedForVision(maxSide: 900)
        guard let cg = visionImage.cgImage else { return (0, [], 0, 0, 0) }
        let orientation = visionImage.cgImagePropertyOrientation
        let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])

        let rectangleRequest = VNDetectRectanglesRequest()
        rectangleRequest.maximumObservations = 6
        rectangleRequest.minimumConfidence = 0.45
        rectangleRequest.minimumAspectRatio = 0.20
        rectangleRequest.maximumAspectRatio = 1.0
        rectangleRequest.minimumSize = 0.16
        rectangleRequest.quadratureTolerance = 28

        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .fast
        textRequest.usesLanguageCorrection = false
        textRequest.minimumTextHeight = 0.015
        textRequest.recognitionLanguages = ["zh-Hans", "en-US"]

        do {
            try handler.perform([rectangleRequest, textRequest])
        } catch {
            return (0, [], 0, 0, 0)
        }

        let rectangleCount = rectangleRequest.results?.filter { $0.confidence >= 0.45 }.count ?? 0
        let textTokens = Set((textRequest.results ?? []).compactMap { observation -> String? in
            let token = observation.topCandidates(1).first?.string
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: " ", with: "")
            guard let token, token.count >= 2 else { return nil }
            return token
        })
        let presence = visionPresenceSignals(cgImage: cg, orientation: orientation)
        let handCount = presence.handCount
        let faceCount = presence.faceCount
        let bodyCount = presence.bodyCount
        return (rectangleCount, textTokens, handCount, faceCount, bodyCount)
    }

    private static func visionPresenceSignals(cgImage: CGImage, orientation: CGImagePropertyOrientation) -> (handCount: Int, faceCount: Int, bodyCount: Int) {
        var handCount = 0
        var faceCount = 0
        var bodyCount = 0

        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:]).perform([handRequest])
            handCount = handRequest.results?.count ?? 0
        } catch {
            handCount = 0
        }

        let faceRequest = VNDetectFaceRectanglesRequest()
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:]).perform([faceRequest])
            faceCount = faceRequest.results?.count ?? 0
        } catch {
            faceCount = 0
        }

        let bodyRequest = VNDetectHumanBodyPoseRequest()
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:]).perform([bodyRequest])
            bodyCount = bodyRequest.results?.count ?? 0
        } catch {
            bodyCount = 0
        }

        return (handCount, faceCount, bodyCount)
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var logs: [LogLine] = []
    @Published var uploadState = "待机"
    @Published var isBursting = false
    @Published var sessionId: String?
    @Published var studentGoal = ""
    @Published var strategySyncState = "未同步"
    @Published var qualityFeedback = CaptureQualityFeedback.idle
    @Published var qaOverlayVisible = false
    @Published var qaStateText = "AI 问答"
    @Published var qaSystemImage = "sparkles"
    @Published var recognizedText = ""
    @Published var qaAnswer = ""
    @Published var isListening = false
    @Published var isPreparingVoiceInput = false
    @Published var continuousVoiceActive = false
    @Published var observationGuideVisible = false
    @Published var isSpeaking = false
    @Published var voicePlaybackEnabled = UserDefaults.standard.object(forKey: voicePlaybackEnabledDefaultsKey) as? Bool ?? true
    @Published var voicePlaybackRate = normalizedVoicePlaybackRate(UserDefaults.standard.double(forKey: voicePlaybackRateDefaultsKey))
    @Published var ttsPlaybackPhase = TTSPlaybackPhase.idle
    @Published var ttsCurrentSegmentIndex = 0
    @Published var ttsTotalSegmentCount = 0
    @Published var ttsServiceText = TTSService.local.displayName
    @Published var observationStopNotice: ObservationStopNotice?
    @Published var observationQuestionCandidateCount = 0
    @Published var observationQuestionUniqueCount = 0
    @Published var observationQuestionDuplicateCount = 0
    @Published var observationQuestionLastAction = ""
    @Published var observationQuestionEvents: [String] = []
    @Published var observationQuestionOverlayCandidates: [ObservationQuestionCandidate] = []
    @Published var observationQuestionOverlayImageSize: CGSize = .zero
    @Published var observationQuestionOverlayUpdatedAt: Date?
    @Published var isThinking = false
    @Published var cameraSheetVisible = false
    @Published var cameraPreviewVisible = false
    @Published var isCameraReady = false
    @Published var chatMessages: [ChatMessage] = []
    @Published var serverTasks: [ServerBackgroundTask] = []   // #8 后端在跑的后台生成任务（账号隔离，可取消）
    @Published var cameraTaskKind = CameraTaskKind.none
    @Published var backgroundCameraEnabled = true
    // 题目分割 + 浮层选题（拍照答题/作业批改）。空状态=旧拍题问答链路完全不受影响。
    @Published var segmentationVisible = false             // 是否在展示「冻结图 + 透明浮层」选题层
    @Published var segmentationBusy = false                // 校正/分割进行中
    @Published var segmentationDisplayImage: UIImage?       // 当前展示用图（原图或校正图）
    @Published var questionRegions: [QuestionRegion] = []   // 端上分割出的题目区域
    @Published var selectedRegionID: UUID?                  // 当前选中的浮层
    @Published var segmentationDidCorrect = false           // 本次是否做了梯形校正
    @Published var segmentationUseCorrected = true          // 「用矫正图 ⇄ 用原图」切换
    // MARK: 题目提取（观察模式·融入 iPhone/iPad）。拍清试卷→后端提取结构化题目→服务端去重累积成题集→还原页/空白卷。
    @Published var extractSessionId: String?                // 本回合题目提取会话（mode=extract，创建即入历史）
    @Published var extractShotCount = 0                     // 已拍并成功提取的张数
    @Published var extractUniqueCount = 0                   // 服务端去重后的题集题数（question_set_count）
    @Published var extractInflight = 0                      // 进行中的上传数（串行 maxInflight=1）
    @Published var extractNotice = ""                       // 取景/提取实时提示文案
    @Published var extractResultVisible = false            // 弹出还原/空白卷结果 sheet
    @Published var extractRestoreHTML: String?             // 还原页 HTML（服务端渲染，WKWebView 显示）
    @Published var extractAutoScan = false                 // 自动扫描：开相机时端上跟踪、发现新题才异步识别
    fileprivate var extractScanTimer: Timer?               // 自动扫描定时器（~1.2s 一帧）
    fileprivate var extractSeenKeys: [String] = []         // 已捕获题的端上指纹（题文归一化），用于跟踪去重
    fileprivate var extractScanBusy = false                // 端上扫描计算中（避免叠帧）
    @Published var segmentationIsGrading = false            // true=作业批改, false=拍照答题
    @Published var segmentationNotice = ""                  // 顶部提示（未框到题等兜底文案）
    // P2 整页批改：端上准 bbox(questionRegions) + 后端 grade-page 逐题判分，按 index 对齐叠 ✓/✗/?。
    @Published var gradeQuestionsByIndex: [Int: GradedQuestion] = [:]  // region.index → 判分
    @Published var gradingInFlight = false                  // 批改请求在途（题上转圈）
    @Published var gradingNotice = ""                       // 批改兜底文案（未识别/太糊）
    @Published var selectedGradeIndex: Int?                 // 点开的题（详情卡）
    @Published var autoRouteRevertText: String?             // P4：自动进模块时记下原话，供「其实只想问」一键纠偏
    // B：印刷题干数字化重排版（仅印刷题干，强制对照原图；学生作答/几何不重生成）。
    @Published var relayoutText = ""                        // 数字版题干纯文本
    @Published var relayoutFigureNote = ""                  // 「此题含图，见原图」
    @Published var relayoutInFlight = false
    @Published var relayoutForIndex: Int?                   // 已取数字版的题序
    var relayoutCropImage: UIImage?                        // 对照原图用的该题裁剪图（非 @Published）
    var segmentationRawImage: UIImage?                      // 原图（orientation 归一），非 @Published
    var segmentationCorrectedImage: UIImage?               // 校正图，非 @Published
    fileprivate var pendingRegionQAFrame: QAFrameCandidate?  // 选中题裁剪帧，喂给下一次提交（类型 private，故 fileprivate）
    // P0：分题/批改「取景—自动快门」阶段（先对准拍清楚，再分割）
    @Published var captureAimingVisible = false             // 取景拍照层是否在展示
    @Published var cameraAimBrightness = 0.5                // 实时平均亮度 0..1
    @Published var cameraAimSharpness = 0.0                 // 实时锐度（清晰度代理）0..1
    @Published var cameraAimMaterial = 0.0                  // 是否有题目印刷体（书本/试卷/屏幕）0..1
    @Published var torchOn = false                          // 补光手电状态
    @Published var autoShutterArmed = true                  // 自动快门开关（达标自动拍）
    private var aimReadyStreak = 0                          // 连续达标帧计数（自动快门去抖）
    private var aimCapturing = false                        // 抓拍在途，防重复
    /// 取景是否达标（有题目印刷体 + 够亮 + 够清晰）。先判断「有没有题目」是关键——
    /// 没对准书本/试卷/屏幕上的题目就不自动拍、也不去识别。
    var cameraAimReady: Bool { cameraAimMaterial >= 0.5 && cameraAimBrightness >= 0.18 && cameraAimSharpness >= 0.34 }
    var cameraAimHint: String {
        if cameraAimMaterial < 0.5 { return "请把镜头对准书本/试卷/屏幕上的题目" }
        if cameraAimBrightness < 0.18 { return "光线偏暗，点亮手电或挪到亮处" }
        if cameraAimSharpness < 0.34 { return "对焦中，把题目放平、点击对焦" }
        return "已识别到题目，清晰，可以拍了"
    }
    // 设置开关：自动梯形校正 / 多题自动分割（默认开；关=回到老整图链路）
    @Published var autoKeystoneCorrection: Bool = {
        if UserDefaults.standard.object(forKey: autoKeystoneCorrectionDefaultsKey) == nil { return true }
        return UserDefaults.standard.bool(forKey: autoKeystoneCorrectionDefaultsKey)
    }()
    @Published var questionSegmentationEnabled: Bool = {
        if UserDefaults.standard.object(forKey: questionSegmentationEnabledDefaultsKey) == nil { return true }
        return UserDefaults.standard.bool(forKey: questionSegmentationEnabledDefaultsKey)
    }()
    @Published var lastSubmittedContextItems: [ContextBadgeItem] = []
    /// Memories the server semantically retrieved into the most recent answer (with score breakdown).
    @Published var lastTurnMemories: [RetrievedMemory] = []
    /// Reconciled truth for the current turn (from server `context_trace`); the panel reads this.
    @Published var lastTurnManifest: TurnContextManifest?
    /// Memory ids the user turned off for the *next* turn; sent as `memory_excludes`.
    @Published var memoryOverrides: Set<String> = []
    @Published var textOnlyQuestion: Bool = {
        #if DEBUG
        if ProcessInfo.processInfo.environment["PXJ_TEXT_ONLY"] == "1" { return true }
        #endif
        return UserDefaults.standard.bool(forKey: textOnlyQuestionDefaultsKey)
    }()
    @Published var learningMode = LearningModePreference(rawValue: UserDefaults.standard.string(forKey: learningModeDefaultsKey) ?? "") ?? .singleProblem
    @Published var coachDepth = CoachDepthPreference(rawValue: UserDefaults.standard.string(forKey: coachDepthDefaultsKey) ?? "") ?? .hintFirst
    /// 一句话辅导偏好（自然语言）。非空时直接作为教练偏好主体；为空时回退中性默认。独立持久化，不复用 studentGoal。
    @Published var coachPreferenceText = UserDefaults.standard.string(forKey: coachPreferenceTextDefaultsKey) ?? ""
    /// 提示词列表（GET /api/prompts，只读预览 + 一键恢复默认）。
    @Published var coachPrompts: [CoachPromptRecord] = []
    @Published var isLoadingPrompts = false
    @Published var contextInclusionSettings = ContextInclusionSettings.load()
    // 二期·自然语言配置管家（逻辑在 Intent/IntentRouting.swift；UI 在 Intent/IntentProposalCard.swift）。
    @Published var pendingIntentProposal: IntentProposal?
    @Published var lastAppliedProposal: IntentProposal?          // 持 echo_state.before 供就地撤销
    @Published var intentRouteInFlight = false
    @Published var intentPhase: IntentPhase = .proposed
    var pendingIntentOriginalQuestion: String?                   // 误判逃生时回填原文
    @Published var longTermInstruction = UserDefaults.standard.string(forKey: longTermInstructionDefaultsKey) ?? ""
    @Published var longTermMemories = UserDefaults.standard.stringArray(forKey: longTermMemoriesDefaultsKey) ?? []
    @Published var userInputMemory = UserDefaults.standard.stringArray(forKey: userInputMemoryDefaultsKey) ?? []
    @Published var memoryProfileText = ""
    @Published var memoryProfileUpdatedAt = ""
    @Published var memoryEvents: [MemoryEvent] = []
    @Published var isLoadingMemoryDigest = false
    // 三期·滚动记忆数字人 — UI 驱动状态（编排逻辑在 Memory/MemoryProfileState.swift 的 extension AppState）。
    @Published var agentMemories: [AgentMemory] = []           // 档案页持久记忆（仅 active）
    @Published var lastTurnMemoryDelta: MemoryDeltaBatch? = nil // 驱动对话内增量 chip；nil=不渲染
    @Published var isLoadingAgentMemories = false
    @Published var memoryUndoToastVisible = false              // 5s 撤销吐司可见
    // 纯内部、非 UI 驱动：撤销快照 + 吐司去抖 token（不需 @Published）。
    var lastMutatedMemory: MemoryMutationSnapshot? = nil
    var currentMemoryUndoToastToken: UUID? = nil
    @Published var reviewQueueState = "复习队列待刷新"
    @Published var isPreparingReview = false
    @Published var historySessions: [HistorySessionSummary] = []
    @Published var isLoadingHistory = false
    @Published var dueReviewItems: [ReviewMistakeItem] = []
    @Published var recentReviewItems: [ReviewMistakeItem] = []
    @Published private var reviewDueCount: Int?

    var captureSingle: (() -> Void)?
    var captureBurstFrame: (() -> Bool)?
    var captureQAFrame: (() -> Bool)?
    var captureSegment: (() -> Bool)?          // P0：分题/批改取景后全分辨率抓拍
    var captureExtract: (() -> Bool)?          // 题目提取取景后全分辨率抓拍（连拍）
    var latestPreviewFrame: (() -> UIImage?)?  // 题目提取「自动扫描」端上跟踪用：取当前预览帧
    var setTorch: ((Bool) -> Void)?            // P0：补光手电

    private var burstBuffer: [BurstFrame] = []
    private var burstTimer: Timer?
    private var secondTimer: Timer?
    private var lastAcceptedFingerprint: FrameFingerprint?
    private var lastAcceptedTextTokens = Set<String>()
    private var lastAcceptedAt: Date?
    private var lastSceneActivityAt: Date?
    private var lastUserOperationAt = Date.distantPast
    private var lastUserOperationKind = "app_launch"
    private var lastFrameHadStudyMaterial: Bool?
    private var emptySceneCount = 0
    private var similarSceneCount = 0
    private var isFlushingBurst = false
    private var isAnalyzingBurstFrame = false
    private var shouldFinishAfterFlush = false
    private var nextBurstSequenceIndex = 0
    private var observationAcceptedFrameCount = 0
    private var observationQuestionSeenKeys: [String] = []
    private var observationQuestionScanBusy = false
    private var observationQuestionScanSerial = 0
    private var burstGeneration = 0
    private var strategySyncTask: Task<Void, Never>?
    private var observationGuideHideTask: Task<Void, Never>?
    private var observationStopNoticeTask: Task<Void, Never>?
    private var danmakuSuppressedUntil = Date.distantPast
    private var qaSubmissionGeneration = 0
    private var lastSyncedStrategySignature = ""
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private let audioEngine = AVAudioEngine()
    private var ttsPlayer: AVAudioPlayer?
    private let localSpeechSynthesizer = AVSpeechSynthesizer()
    private var ttsPlaybackDelegate = AudioPlaybackFinishDelegate()
    private var ttsPlaybackTask: Task<Void, Never>?
    private var ttsPrefetchTask: Task<(data: Data, service: TTSService), Error>?
    private var ttsPlaybackGeneration = 0
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var silenceTimer: Timer?
    private var recognitionRefreshTimer: Timer?
    private var voiceSubmitGraceTask: Task<Void, Never>?
    private var qaHideTimer: Timer?
    private var pendingQATrigger = "voice"
    private var pendingQAFocus: QAFocus?
    private var pendingQAQuestion = ""
    private var shouldSubmitWhenVoiceReady = false
    private var isWaitingForVoiceSubmitGrace = false
    private var voiceInputIsStarting = false
    private var holdToTalkActive = false
    private var anchoredQAFrame: QAFrameCandidate?
    private var activeReviewItem: ReviewMistakeItem?
    private var carriedHistoryContext: HistoryCarryContext?
    private var qaTurnIndex = 0
    private var hasSubmittedCurrentListen = false
    private var speechAuthorizationRequested = false
    private var pendingQAFrameContinuation: CheckedContinuation<QAFrameCandidate?, Never>?
    private var pendingQAFrameTimeoutTask: Task<Void, Never>?
    @Published private var isSubmittingQuestion = false
    private var pendingSingleCapture = false
    private var currentCameraHostId: UUID?
    private var backgroundCameraResumeTask: Task<Void, Never>?
    private let qaFrameTimeoutSeconds: TimeInterval = 4.0
    private var deviceControlTask: Task<Void, Never>?
    private var handledRemoteCommandIds = Set<String>()
    private var remotePollIntervalSeconds: TimeInterval = 1.0
    private let remoteDeviceId = UIDevice.current.identifierForVendor?.uuidString ?? "iphone"
    private let burstAnalysisQueue = DispatchQueue(label: "com.pxj.burst-analysis", qos: .userInitiated)
    private let burstBatchSize = 5
    private let activeCaptureInterval: TimeInterval = 2.0
    private let similarCaptureInterval: TimeInterval = 4.0
    private let longSimilarCaptureInterval: TimeInterval = 8.0
    private let similarSceneProbeInterval: TimeInterval = 45.0
    private let emptyCaptureIntervals: [TimeInterval] = [4.0, 8.0, 12.0, 20.0]
    private let stillSceneAutoStopInterval: TimeInterval = 30 * 60
    private var idleTimerDisabledBeforeBurst: Bool?
    var modeTitle: String {
        isBursting ? "智能观察学习回合" : "单张拍题解析"
    }

    var sessionText: String {
        let goal = trimmedStudentGoal
        if let sessionId {
            return goal.isEmpty ? "回合 " + String(sessionId.prefix(8)) : "回合 " + String(sessionId.prefix(8)) + " · " + goal
        }
        return goal.isEmpty ? "尚未创建学习回合" : "待开始 · " + goal
    }

    var reviewButtonTitle: String {
        if isPreparingReview {
            return "准备复习"
        }
        if let reviewDueCount, reviewDueCount > 0 {
            return "复习任务 \(reviewDueCount)"
        }
        return "复习任务"
    }

    private var trimmedStudentGoal: String {
        studentGoal.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var trimmedCoachPreferenceText: String {
        coachPreferenceText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // 用户未填偏好时的默认辅导风格 = 原"先给提示"默认（保证默认问答质量不退化）
    private static let defaultCoachStyle = "默认辅导风格：先给提示、循序渐进地引导学生自己想，必要时再完整讲解"

    private var coachReportStyle: String {
        let preference = trimmedCoachPreferenceText
        let base = preference.isEmpty ? AppState.defaultCoachStyle : "家长/学生偏好：\(preference)"
        return "\(base)；输出要保留错题、知识点、过程线索、家长三句话和下一步复习建议"
    }

    private var coachAssistantFocus: String {
        let preference = trimmedCoachPreferenceText
        let base = preference.isEmpty ? AppState.defaultCoachStyle : "家长/学生偏好：\(preference)"
        return "\(base)；每次回答尽量给一个下一步小任务，帮助学生继续自己做"
    }

    private var strategySignature: String {
        [trimmedStudentGoal, trimmedCoachPreferenceText, coachReportStyle, coachAssistantFocus].joined(separator: "|")
    }

    var hasChatStarted: Bool {
        !chatMessages.isEmpty || isThinking || isListening || ttsPlaybackPhase.isActive || !qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var latestAssistantMessageId: UUID? {
        chatMessages.last(where: { $0.role == .assistant })?.id
    }

    var canStartNewConversation: Bool {
        hasChatStarted || sessionId != nil || !lastSubmittedContextItems.isEmpty || cameraTaskKind == .qaFrame
    }

    var desktopIntroVisible: Bool {
        !inlineCameraPreviewVisible &&
        !hasChatStarted &&
        !isBursting &&
        !isThinking &&
        !isListening &&
        !questionSubmissionInFlight &&
        recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var voiceInputDisabled: Bool {
        isThinking || isSubmittingQuestion || isPreparingReview
    }

    var voiceDockCanCollapse: Bool {
        !isListening &&
        !isPreparingVoiceInput &&
        !isWaitingForVoiceSubmitGrace &&
        !continuousVoiceActive &&
        !questionSubmissionInFlight &&
        !ttsPlaybackPhase.isActive
    }

    var voiceDockCanAutoHide: Bool {
        voiceDockCanCollapse && !voiceCancelAvailable && !isSpeaking && !isPreparingReview
    }

    var voiceCancelAvailable: Bool {
        continuousVoiceActive || isListening || isPreparingVoiceInput || isWaitingForVoiceSubmitGrace || questionSubmissionInFlight
    }

    var isWaitingForVoiceSubmit: Bool {
        isWaitingForVoiceSubmitGrace
    }

    var voicePlaybackRateText: String {
        String(format: "%.2fx", voicePlaybackRate)
    }

    var questionSubmissionInFlight: Bool {
        isThinking || isSubmittingQuestion
    }

    var preferenceFollowUpTitle: String {
        "按偏好"
    }

    var preferenceFollowUpPrompt: String {
        "请按我的学习偏好（\(learningMode.title) · \(coachDepth.title)），基于上面的回答继续安排下一步。"
    }

    var observationDanmakuVisible: Bool {
        isBursting &&
        inlineCameraPreviewVisible &&
        !observationGuideVisible &&
        !isListening &&
        !isPreparingVoiceInput &&
        !isWaitingForVoiceSubmitGrace &&
        !questionSubmissionInFlight &&
        !ttsPlaybackPhase.isActive &&
        Date() >= danmakuSuppressedUntil &&
        !logs.isEmpty
    }

    var observationQuestionOverlayVisible: Bool {
        guard isBursting, inlineCameraPreviewVisible, let updatedAt = observationQuestionOverlayUpdatedAt else { return false }
        return Date().timeIntervalSince(updatedAt) < 4.5
    }

    var inlineCameraPreviewVisible: Bool {
        cameraPreviewVisible && !cameraSheetVisible && (cameraTaskKind == .qaFrame || cameraTaskKind == .extract || isBursting || cameraTaskKind == .burst || cameraTaskKind == .singleCapture)
    }

    var activeTaskVisible: Bool {
        !runtimeTasks.isEmpty
    }

    var backgroundCameraActive: Bool {
        backgroundCameraEnabled && (isBursting || cameraTaskKind == .qaFrame) && !inlineCameraPreviewVisible
    }

    var runtimeTasks: [RuntimeTaskItem] {
        var tasks: [RuntimeTaskItem] = []

        if isBursting {
            let bufferedText = burstBuffer.isEmpty ? "等待有效画面" : "已保留 \(burstBuffer.count) 张关键帧"
            let questionText: String
            if observationQuestionUniqueCount > 0 {
                questionText = "；题目指纹 \(observationQuestionUniqueCount) 新 / \(observationQuestionDuplicateCount) 重"
            } else if observationQuestionScanBusy {
                questionText = "；题目指纹扫描中"
            } else {
                questionText = ""
            }
            tasks.append(RuntimeTaskItem(
                id: "observation",
                title: "智能观察运行中",
                detail: "\(bufferedText)\(questionText)；作为问答背景和后续总结。",
                systemImage: "rectangle.stack.fill",
                tone: .good,
                showsProgress: true,
                canOpen: true,
                canClose: true,
                closeTitle: "停止观察",
                closeSystemImage: "stop.circle"
            ))
        }

        if isListening || isPreparingVoiceInput {
            let text = recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
            tasks.append(RuntimeTaskItem(
                id: "voice",
                title: continuousVoiceActive ? "持续倾听中" : (isListening ? "语音转文字中" : "准备语音"),
                detail: text.isEmpty ? (continuousVoiceActive ? "停顿后自动提交；回答完会继续听。" : "松开发送；会抓拍当前预览画面一起提交。") : "已听到：\(shortText(text, limit: 46))",
                systemImage: "waveform.circle.fill",
                tone: .waiting,
                showsProgress: true,
                canOpen: false,
                canClose: true,
                closeTitle: continuousVoiceActive ? "关闭录音" : "取消语音",
                closeSystemImage: "mic.slash"
            ))
        }

        if isThinking || isSubmittingQuestion {
            let detail: String
            switch qaStateText {
            case "创建问答回合":
                detail = "正在连接服务，准备把问题发给 AI。"
            case "截取当前画面":
                detail = "正在截取当前预览画面，随后提交问题。"
            case "已截取画面，思考中", "复用首轮画面，思考中":
                detail = "画面和问题已准备好，正在等待 AI 回答。"
            case "问答失败", "创建问答回合失败", "问题为空":
                detail = qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "本轮没有成功提交，请重新问一次。" : qaAnswer
            default:
                detail = "正在提交本轮问题；回答会回到聊天流里。"
            }
            tasks.append(RuntimeTaskItem(
                id: "thinking",
                title: qaStateText == "截取当前画面" ? "正在截取画面" : "AI 正在处理",
                detail: detail,
                systemImage: "brain.head.profile",
                tone: .neutral,
                showsProgress: true,
                canOpen: false,
                canClose: false
            ))
        }

        if cameraTaskKind == .qaFrame && !isListening && !isPreparingVoiceInput && !questionSubmissionInFlight {
            tasks.append(RuntimeTaskItem(
                id: "qa-frame",
                title: "当前画面待用",
                detail: isCameraReady ? "相机已准备好，按住说话会截取当前画面。" : "正在准备相机，用于下一次提问。",
                systemImage: "camera.metering.center.weighted",
                tone: isCameraReady ? .good : .waiting,
                showsProgress: !isCameraReady,
                canOpen: true,
                canClose: true,
                closeTitle: "隐藏预览",
                closeSystemImage: "xmark.circle"
            ))
        }

        if isSpeaking || ttsPlaybackPhase.isActive {
            let segmentSuffix = ttsTotalSegmentCount > 1 ? "（\(max(1, ttsCurrentSegmentIndex))/\(ttsTotalSegmentCount)）" : ""
            switch ttsPlaybackPhase {
            case .generating:
                tasks.append(RuntimeTaskItem(
                    id: "tts-generating",
                    title: "\(ttsServiceText)合成中\(segmentSuffix)",
                    detail: ttsServiceText == TTSService.local.displayName ? "文字答案已经显示，正在准备本机语音播放。" : "文字答案已经显示，正在合成语音并等待播放。",
                    systemImage: "waveform.path.ecg",
                    tone: .waiting,
                    showsProgress: true,
                    canOpen: false,
                    canClose: true,
                    closeTitle: "取消朗读",
                    closeSystemImage: "speaker.slash"
                ))
            case .playing, .paused:
                tasks.append(RuntimeTaskItem(
                    id: "speaking",
                    title: ttsPlaybackPhase == .paused ? "语音已暂停\(segmentSuffix)" : "\(ttsServiceText)播放中\(segmentSuffix)",
                    detail: "\(ttsServiceText)播放中；文字答案已经在聊天里，当前速度 \(voicePlaybackRateText)。",
                    systemImage: ttsPlaybackPhase == .paused ? "pause.circle.fill" : "speaker.wave.2.fill",
                    tone: .neutral,
                    showsProgress: false,
                    canOpen: false,
                    canClose: true,
                    closeTitle: "停止朗读",
                    closeSystemImage: "speaker.slash"
                ))
            case .idle:
                break
            }
        }

        if uploadState == "上传单张" || uploadState == "后台解析中" || uploadState == "上传批次" {
            tasks.append(RuntimeTaskItem(
                id: "upload",
                title: uploadState == "后台解析中" ? "拍题后台解析中" : "上传画面中",
                detail: runtimeTaskDetail(from: qualityFeedback.detail),
                systemImage: "arrow.up.doc",
                tone: uploadState.contains("失败") ? .warning : .waiting,
                showsProgress: true,
                canOpen: false,
                canClose: false
            ))
        }

        if uploadState == "生成报告" || uploadState == "报告生成中" || uploadState == "报告触发失败" {
            tasks.append(RuntimeTaskItem(
                id: "report",
                title: uploadState == "报告触发失败" ? "报告触发失败" : "学习报告处理中",
                detail: runtimeTaskDetail(from: qualityFeedback.detail),
                systemImage: "doc.text.magnifyingglass",
                tone: uploadState == "报告触发失败" ? .warning : .waiting,
                showsProgress: uploadState != "报告触发失败",
                canOpen: false,
                canClose: uploadState == "报告触发失败",
                closeTitle: "关闭",
                closeSystemImage: "xmark.circle"
            ))
        }

        if uploadState == "批次失败" || uploadState == "上传失败" {
            tasks.append(RuntimeTaskItem(
                id: "upload-error",
                title: uploadState,
                detail: runtimeTaskDetail(from: qualityFeedback.detail, fallback: "网络连接失败，暂时没有拿到 AI 回复。"),
                systemImage: "exclamationmark.triangle",
                tone: .warning,
                showsProgress: false,
                canOpen: false,
                canClose: true,
                closeTitle: "关闭",
                closeSystemImage: "xmark.circle"
            ))
        }

        if uploadState == "相机错误" {
            tasks.append(RuntimeTaskItem(
                id: "camera-error",
                title: "相机不可用",
                detail: runtimeTaskDetail(from: qualityFeedback.detail),
                systemImage: "camera.fill.badge.ellipsis",
                tone: .warning,
                showsProgress: false,
                canOpen: false,
                canClose: true,
                closeTitle: "关闭",
                closeSystemImage: "xmark.circle"
            ))
        }

        for task in serverTasks.prefix(3) {
            tasks.append(RuntimeTaskItem(
                id: "server-task:\(task.id)",
                title: task.title,
                detail: "\(task.stateText) · \(task.laneText)",
                systemImage: task.systemImage,
                tone: .waiting,
                showsProgress: true,
                canOpen: false,
                canClose: task.state != "cancelling",
                closeTitle: "取消后台任务",
                closeSystemImage: "xmark.circle"
            ))
        }

        return tasks
    }

    var cameraTaskTitle: String {
        switch cameraTaskKind {
        case .burst:
            return "智能观察"
        case .qaFrame:
            return "语音画面上下文"
        case .extract:
            return "题目提取"
        case .singleCapture, .none:
            return "拍题解析"
        }
    }

    var cameraTaskSystemImage: String {
        switch cameraTaskKind {
        case .burst:
            return "rectangle.stack.fill"
        case .qaFrame:
            return "camera.metering.center.weighted"
        case .extract:
            return "doc.text.viewfinder"
        case .singleCapture, .none:
            return "camera.viewfinder"
        }
    }

    var cameraPrimaryActionTitle: String {
        switch cameraTaskKind {
        case .burst:
            return "停止并生成总结"
        case .qaFrame:
            return isListening ? "结束收听并提交" : "回到对话"
        case .extract:
            return "完成提取"
        case .singleCapture, .none:
            return "拍照答题"
        }
    }

    var cameraPrimaryActionSystemImage: String {
        switch cameraTaskKind {
        case .burst:
            return "stop.circle"
        case .qaFrame:
            return isListening ? "paperplane.circle.fill" : "bubble.left.and.bubble.right"
        case .extract:
            return "checkmark.circle.fill"
        case .singleCapture, .none:
            return "camera.circle.fill"
        }
    }

    var cameraPrimaryActionDisabled: Bool {
        switch cameraTaskKind {
        case .burst:
            return false
        case .qaFrame:
            return isThinking
        case .extract:
            return false
        case .singleCapture, .none:
            return !isCameraReady
        }
    }

    var inlineCameraTitle: String {
        if isBursting {
            return "智能观察预览"
        }
        if cameraTaskKind == .qaFrame {
            return qaTurnIndex == 0 ? "新对话画面" : "追问画面"
        }
        return "相机预览"
    }

    var inlineCameraHint: String {
        if continuousVoiceActive {
            return "持续倾听中，停顿后自动带当前画面提交"
        }
        if isBursting {
            return "观察在后台运行，按底部语音条可继续追问"
        }
        if isListening || isPreparingVoiceInput {
            return "松开发送当前画面，上滑取消"
        }
        return "按住下方语音条说话，松手会连同当前画面发送"
    }

    var composerHint: String {
        if continuousVoiceActive {
            return "持续录音中：停顿后自动提交，回答完继续听；可说“停止录音”或用停止手势关闭。"
        }
        if isBursting {
            return "智能观察正在后台收集学习过程；用底部语音条提问，会带上观察上下文。"
        }
        if isThinking {
            return "正在整理回答，完成后会显示在对话里。"
        }
        if isListening {
            return "松开发送当前画面；上滑取消本次语音。"
        }
        return "按住说话，松开发送当前画面；连续点两下可直接发送当前画面。"
    }

    var contextSystemPromptPreview: String {
        [
            "学习报告策略：\(coachReportStyle)",
            "回答策略：\(coachAssistantFocus)",
            "动态策略：\(dynamicStrategyPreview())",
            "资产规则：错题用于复习与纠错，知识点用于解释和举一反三，记忆用于个性化表达；相关性不足时只作弱参考。",
            "线程规则：追问默认沿用本轮首张题图；除非用户明确换题或点击新对话。",
            "视觉规则：当前画面和智能观察只作为辅助上下文，不覆盖用户当前问题。"
        ].joined(separator: "\n")
    }

    func updateContextInclusion(_ keyPath: WritableKeyPath<ContextInclusionSettings, Bool>, to value: Bool) {
        var settings = contextInclusionSettings
        settings[keyPath: keyPath] = value
        contextInclusionSettings = settings
        settings.save()
        log("上下文配置已更新：\(settings.enabledContextLabels.joined(separator: "、"))")
    }

    func setContextDebugEnabled(_ enabled: Bool) {
        updateContextInclusion(\.debug, to: enabled)
    }

    /// Maps a trace-derived badge (id == "trace-<channel>") to its class-level inclusion
    /// keypath, so the panel can offer a per-row toggle that flows through the single
    /// write entry point `updateContextInclusion` (二期 NL 配置同源)。Returns nil for
    /// non-toggleable rows (e.g. the question itself).
    func inclusionKeyPath(forBadgeId id: String) -> WritableKeyPath<ContextInclusionSettings, Bool>? {
        guard id.hasPrefix("trace-") else { return nil }
        switch String(id.dropFirst("trace-".count)) {
        case "visual": return \.visual
        case "observation": return \.observation
        case "history": return \.history
        case "mistakes": return \.mistakes
        case "knowledge": return \.knowledge
        case "memory": return \.memory
        case "strategy": return \.strategy
        default: return nil
        }
    }

    private func contextAssetKindEnabled(_ kind: String) -> Bool {
        switch kind {
        case "mistake":
            return contextInclusionSettings.mistakes
        case "knowledge":
            return contextInclusionSettings.knowledge
        case "memory_profile", "local_memory", "memory_event":
            return contextInclusionSettings.memory
        default:
            return true
        }
    }

    func contextUsePolicyPreview(draft: String) -> String {
        [
            "1. 先判断本轮意图：检查答案、讲解新题、复习错题、总结知识点、继续追问。",
            "2. 当前画面/首轮题图优先级最高；没有可靠画面时，才更多依赖当前对话和结构化资产。",
            "3. 错题资产只在题目、知识点、错因或复习意图相关时使用；复习时先让学生回忆，再给提示。",
            "4. 知识点资产用于补齐概念、易错点和举一反三，不要替代当前题目判断。",
            "5. 记忆资产用于个性化：偏好、常犯错、表达方式和最近目标；不得编造成事实答案。",
            "6. 若使用了资产，回答里用一句自然话体现依据，例如“结合你之前常错的单位换算”。",
            "本次启用：\(contextInclusionSettings.enabledContextLabels.joined(separator: "、"))。",
            "本次候选资产：\(contextAssetCount(for: draft)) 条。"
        ].joined(separator: "\n")
    }

    func contextUserPromptPreview(draft: String) -> String {
        let question = draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? qaQuestionTextForSubmission() : draft
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "等待用户输入；按住说话后会把语音文本冻结为本轮问题。"
        }
        return question.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func contextAssetGroups(for draft: String) -> [ContextAssetGroup] {
        let question = contextUserPromptPreview(draft: draft)
        return [
            ContextAssetGroup(
                id: "mistakes",
                title: "错题",
                systemImage: "book.closed",
                items: contextInclusionSettings.mistakes ? mistakeContextAssets(for: question) : []
            ),
            ContextAssetGroup(
                id: "knowledge",
                title: "知识点",
                systemImage: "list.bullet.rectangle",
                items: contextInclusionSettings.knowledge ? knowledgeContextAssets(for: question) : []
            ),
            ContextAssetGroup(
                id: "memories",
                title: "记忆",
                systemImage: "brain.head.profile",
                items: contextInclusionSettings.memory ? memoryContextAssets(for: question) : []
            )
        ]
    }

    func contextAssets(for question: String) -> [ContextAssetItem] {
        var assets: [ContextAssetItem] = []
        if contextInclusionSettings.mistakes {
            assets += mistakeContextAssets(for: question)
        }
        if contextInclusionSettings.knowledge {
            assets += knowledgeContextAssets(for: question)
        }
        if contextInclusionSettings.memory {
            assets += memoryContextAssets(for: question)
        }
        return Array(assets.prefix(12))
    }

    func contextAssetCount(for draft: String) -> Int {
        contextAssetSummary(for: draft).total
    }

    func contextAssetSummary(for draft: String) -> ContextAssetSummary {
        _ = draft
        let reviewItems = uniqueReviewItems((activeReviewItem.map { [$0] } ?? []) + dueReviewItems + recentReviewItems)
        let mistakeCount = contextInclusionSettings.mistakes ? min(reviewItems.count, 6) : 0
        let knowledgeCount = contextInclusionSettings.knowledge ? min(Set(reviewItems.flatMap { $0.knowledgePoints }.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { $0.count >= 2 }).count, 6) : 0
        let memoryRawCount = (memoryProfileText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0 : 1) + min((longTermMemories + userInputMemory).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.count, 8) + min(memoryEvents.count, 8)
        let memoryCount = contextInclusionSettings.memory ? min(memoryRawCount, 8) : 0
        return ContextAssetSummary(
            mistakes: mistakeCount,
            knowledge: knowledgeCount,
            memory: memoryCount
        )
    }

    private func mistakeContextAssets(for question: String) -> [ContextAssetItem] {
        let source = uniqueReviewItems((activeReviewItem.map { [$0] } ?? []) + dueReviewItems + recentReviewItems)
        return Array(source.prefix(6)).map { item in
            let title = shortText(item.displayTitle, limit: 38)
            let location = item.locationText
            let reason = firstNonEmpty(item.errorReason, item.errorType, item.nextAction, "复习时先核对错因和订正步骤")
            let fullDetail = [
                "题目：\(item.displayTitle)",
                location.isEmpty ? "" : "位置：\(location)",
                item.questionText.isEmpty ? "" : "题干：\(shortText(item.questionText, limit: 260))",
                item.studentAnswer.isEmpty ? "" : "学生答案：\(item.studentAnswer)",
                item.expectedAnswer.isEmpty ? "" : "参考答案：\(item.expectedAnswer)",
                reason.isEmpty ? "" : "错因/重点：\(reason)",
                item.nextAction.isEmpty ? "" : "下一步：\(item.nextAction)",
                item.knowledgePoints.isEmpty ? "" : "知识点：\(item.knowledgePoints.prefix(6).joined(separator: "、"))",
                "状态：\(item.status) · \(item.reviewState) · 复习 \(item.reviewCount) 次"
            ].filter { !$0.isEmpty }.joined(separator: "\n")
            return ContextAssetItem(
                id: "mistake-\(item.id)",
                kind: "mistake",
                title: title.isEmpty ? "错题" : title,
                detail: shortText(firstNonEmpty(location, reason, item.questionText), limit: 72),
                useRule: item.isDue ? "到期错题：复习意图或相似知识点时优先使用。" : "历史错题：相关时用于提醒易错点，不强行套用。",
                systemImage: item.isDue ? "calendar.badge.clock" : "book.closed",
                tone: item.isDue ? .waiting : .neutral,
                fullDetail: fullDetail,
                payload: item.compactContext
            )
        }
    }

    private func knowledgeContextAssets(for question: String) -> [ContextAssetItem] {
        var counts: [String: Int] = [:]
        var examples: [String: String] = [:]
        for item in uniqueReviewItems(dueReviewItems + recentReviewItems) {
            for point in item.knowledgePoints {
                let clean = point.trimmingCharacters(in: .whitespacesAndNewlines)
                guard clean.count >= 2 else { continue }
                counts[clean, default: 0] += 1
                if examples[clean] == nil {
                    examples[clean] = item.displayTitle
                }
            }
        }
        let ranked = counts.sorted { lhs, rhs in
            if lhs.value == rhs.value {
                return relevanceScore(text: lhs.key, question: question) > relevanceScore(text: rhs.key, question: question)
            }
            return lhs.value > rhs.value
        }
        return Array(ranked.prefix(6)).map { point, count in
            let example = examples[point] ?? ""
            return ContextAssetItem(
                id: "knowledge-\(point)",
                kind: "knowledge",
                title: point,
                detail: example.isEmpty ? "出现 \(count) 次" : "出现 \(count) 次 · \(shortText(example, limit: 42))",
                useRule: "讲解、总结知识点、举一反三时使用；检查答案时只作为辅助依据。",
                systemImage: "list.bullet.rectangle",
                tone: .neutral,
                fullDetail: [
                    "知识点：\(point)",
                    "关联错题数：\(count)",
                    example.isEmpty ? "" : "例子：\(example)",
                    "使用方式：解释概念、列易错点、生成相似题。"
                ].filter { !$0.isEmpty }.joined(separator: "\n"),
                payload: [
                    "knowledge_point": point,
                    "count": count,
                    "example": example
                ]
            )
        }
    }

    private func memoryContextAssets(for question: String) -> [ContextAssetItem] {
        var items: [ContextAssetItem] = []
        let profile = memoryProfileText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !profile.isEmpty {
            items.append(ContextAssetItem(
                id: "memory-profile",
                kind: "memory_profile",
                title: "用户画像",
                detail: shortText(profile, limit: 72),
                useRule: "用于选择解释粒度、提醒常犯错和匹配偏好；不能代替题目事实。",
                systemImage: "person.text.rectangle",
                tone: .neutral,
                fullDetail: profile,
                payload: [
                    "profile": profile,
                    "updated_at": memoryProfileUpdatedAt
                ]
            ))
        }
        let localMemories = Array((longTermMemories + userInputMemory).prefix(8))
        for (index, memory) in localMemories.enumerated() where !memory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            items.append(ContextAssetItem(
                id: "local-memory-\(index)",
                kind: "local_memory",
                title: "本机记忆",
                detail: shortText(memory, limit: 72),
                useRule: "当前问题相关时作为个性化偏好和近期目标。",
                systemImage: "brain.head.profile",
                tone: .neutral,
                fullDetail: memory,
                payload: ["text": memory]
            ))
        }
        for event in memoryEvents.prefix(8) {
            let title = event.messageType == "formed_memory" ? "形成记忆" : "最近输入"
            items.append(ContextAssetItem(
                id: "memory-event-\(event.id)",
                kind: "memory_event",
                title: title,
                detail: shortText(event.text, limit: 72),
                useRule: "相关时引用最近表达、偏好或学习问题；不相关时忽略。",
                systemImage: event.messageType == "formed_memory" ? "wand.and.stars" : "text.quote",
                tone: event.messageType == "formed_memory" ? .good : .neutral,
                fullDetail: [
                    "时间：\(HistoryDateFormatter.displayString(from: event.createdAt))",
                    "来源：\(event.source)",
                    "类型：\(event.messageType)",
                    event.text
                ].joined(separator: "\n"),
                payload: [
                    "event_id": event.id,
                    "message_type": event.messageType,
                    "source": event.source,
                    "text": event.text,
                    "created_at": event.createdAt
                ]
            ))
        }
        return Array(items.sorted { lhs, rhs in
            relevanceScore(text: lhs.detail + lhs.fullDetail, question: question) > relevanceScore(text: rhs.detail + rhs.fullDetail, question: question)
        }.prefix(8))
    }

    private func uniqueReviewItems(_ items: [ReviewMistakeItem]) -> [ReviewMistakeItem] {
        var seen = Set<String>()
        var result: [ReviewMistakeItem] = []
        for item in items where !item.id.isEmpty {
            guard seen.insert(item.id).inserted else { continue }
            result.append(item)
        }
        return result
    }

    private func relevanceScore(text: String, question: String) -> Int {
        let compactQuestion = question.lowercased()
        let compactText = text.lowercased()
        guard !compactQuestion.isEmpty else { return 0 }
        let tokens = compactQuestion
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
        return tokens.reduce(0) { partial, token in
            partial + (compactText.contains(token) ? 1 : 0)
        }
    }

    func contextPayloadPreview(draft: String) -> String {
        let question = contextUserPromptPreview(draft: draft)
        let previewFrame: QAFrameCandidate? = anchoredQAFrame
        var payload: [String: Any] = [
            "question": question,
            "trigger_type": pendingQATrigger,
            "session_id": sessionId ?? "",
            "qa_context": qaContextPayload(
                qaFrame: previewFrame,
                turn: max(qaTurnIndex + 1, 1),
                submittedCurrentFrame: false,
                reusedAnchorFrame: previewFrame != nil,
                transcriptOverride: question
            ),
            "pending_context_items": pendingContextItems(draft: draft).map { item in
                [
                    "id": item.id,
                    "title": item.title,
                    "detail": item.detail
                ]
            },
            "structured_context_assets": contextAssets(for: question).map { $0.compactPayload },
            "context_use_policy": contextUsePolicyPreview(draft: draft)
        ]
        if !lastSubmittedContextItems.isEmpty {
            payload["last_submitted_context_items"] = lastSubmittedContextItems.map { item in
                [
                    "id": item.id,
                    "title": item.title,
                    "detail": item.detail
                ]
            }
        }
        return prettyJSONString(payload)
    }

    private var cameraCaptureAvailable: Bool {
        (cameraSheetVisible || inlineCameraPreviewVisible || backgroundCameraActive) && isCameraReady && captureQAFrame != nil
    }

    private var cameraCanBecomeAvailableForQA: Bool {
        cameraCaptureAvailable || cameraTaskKind == .qaFrame || cameraTaskKind == .extract || isBursting || cameraSheetVisible || cameraPreviewVisible
    }

    private var trimmedLongTermInstruction: String {
        longTermInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var memoryDigestSummary: [String] {
        let profileLines = memoryProfileText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let eventLines = memoryEvents.prefix(8).map { event in
            "\(HistoryDateFormatter.displayString(from: event.createdAt)) \(event.text)"
        }
        return Array((profileLines + longTermMemories + userInputMemory + eventLines).prefix(20))
    }

    func pendingContextItems(draft: String) -> [ContextBadgeItem] {
        let draftText = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let pendingQuestion = draftText.isEmpty ? qaQuestionTextForSubmission() : draftText
        var items: [ContextBadgeItem] = []

        if !pendingQuestion.isEmpty {
            let title = isListening ? "语音转文字" : (draftText.isEmpty && pendingQATrigger != "typed_chat" ? "语音问题" : "文字问题")
            items.append(ContextBadgeItem(
                id: "question",
                title: title,
                detail: shortText(pendingQuestion, limit: 34),
                systemImage: isListening ? "waveform" : "text.bubble",
                tone: .neutral
            ))
        } else if isListening {
            items.append(ContextBadgeItem(
                id: "listening",
                title: "语音转文字",
                detail: "正在听，停顿后提交",
                systemImage: "waveform",
                tone: .waiting
            ))
        } else {
            items.append(ContextBadgeItem(
                id: "empty-question",
                title: "等待输入",
                detail: "可纯文字、语音或带图提问",
                systemImage: "keyboard",
                tone: .neutral
            ))
        }

        if contextInclusionSettings.observation && isBursting {
            items.append(ContextBadgeItem(
                id: "burst",
                title: "智能观察",
                detail: observationContextSummary(),
                systemImage: "rectangle.stack",
                tone: .good,
                fullDetail: observationContextDetail()
            ))
        }

        if contextInclusionSettings.visual && (cameraTaskKind == .qaFrame || isListening) {
            items.append(ContextBadgeItem(
                id: "qa-frame",
                title: "当前画面",
                detail: isCameraReady ? "检查/首问时截取" : "正在准备",
                systemImage: "camera.metering.center.weighted",
                tone: isCameraReady ? .good : .waiting
            ))
        }

        if sessionId != nil {
            items.append(ContextBadgeItem(
                id: "session",
                title: "当前回合",
                detail: sessionText,
                systemImage: "tray.full",
                tone: .neutral
            ))
        } else {
            items.append(ContextBadgeItem(
                id: "new-session",
                title: "新回合",
                detail: "提交时自动创建",
                systemImage: "plus.bubble",
                tone: .neutral
            ))
        }

        if contextInclusionSettings.history, let carriedHistoryContext {
            items.append(carriedHistoryContext.badge)
        }

        if contextInclusionSettings.mistakes, let activeReviewItem {
            items.append(ContextBadgeItem(
                id: "review",
                title: "复习错题",
                detail: shortText(reviewQueueSummary(for: activeReviewItem), limit: 34),
                systemImage: "calendar.badge.clock",
                tone: .waiting
            ))
        }

        if !trimmedStudentGoal.isEmpty {
            items.append(ContextBadgeItem(
                id: "goal",
                title: "目标",
                detail: shortText(trimmedStudentGoal, limit: 34),
                systemImage: "scope",
                tone: .neutral
            ))
        }

        let memorySummary = contextInclusionSettings.memory ? memoryDigestSummary : []
        if !memorySummary.isEmpty {
            items.append(memoryDigestContextItem())
        }

        let assetSummary = contextAssetSummary(for: pendingQuestion)
        if assetSummary.total > 0 {
            items.append(ContextBadgeItem(
                id: "structured-assets",
                title: "学习资产",
                detail: "错题 \(assetSummary.mistakes) · 知识点 \(assetSummary.knowledge) · 记忆 \(assetSummary.memory)",
                systemImage: "books.vertical",
                tone: .neutral
            ))
        }

        if contextInclusionSettings.strategy {
            items.append(ContextBadgeItem(
                id: "strategy",
                title: "动态策略",
                detail: dynamicStrategyPreview(),
                systemImage: "slider.horizontal.3",
                tone: .neutral,
                fullDetail: dynamicStrategyDetail()
            ))
        }

        return items
    }

    private func observationContextSummary() -> String {
        guard isBursting else { return "未运行" }
        let questionSuffix = observationQuestionUniqueCount > 0 ? " · 题目指纹 \(observationQuestionUniqueCount) 新/\(observationQuestionDuplicateCount) 重" : ""
        if let latest = burstBuffer.last {
            return "已保留 \(burstBuffer.count) 张观察帧 · \(shortText(latest.signalSummary, limit: 24))\(questionSuffix)"
        }
        if uploadState == "智能观察中" || uploadState == "等待学习材料" || uploadState == "创建学习回合" {
            return "后台运行中，等待有效画面\(questionSuffix)"
        }
        return uploadState
    }

    private func observationContextDetail() -> String {
        guard isBursting || !burstBuffer.isEmpty else {
            return "智能观察未运行。"
        }
        var lines = [
            "状态：\(isBursting ? "后台运行中" : "已停止")",
            "用途：作为即时问答的背景信息，也用于后续学习总结；不阻塞新对话和按住说话。",
            "待上传观察帧：\(burstBuffer.count) 张",
            "题目指纹：新题 \(observationQuestionUniqueCount) 道，重复合并 \(observationQuestionDuplicateCount) 道，候选 \(observationQuestionCandidateCount) 条"
        ]
        if !observationQuestionLastAction.isEmpty {
            lines.append("最近题目动态：\(observationQuestionLastAction)")
        }
        for event in observationQuestionEvents.suffix(4) {
            lines.append("题目动态：\(event)")
        }
        if let lastAcceptedAt {
            lines.append("最近有效观察：\(CaptureTimeFormatter.string(from: lastAcceptedAt))")
        }
        for frame in burstBuffer.suffix(4) {
            let capturedAt = CaptureTimeFormatter.string(from: frame.capturedAt)
            lines.append("帧 \(frame.sequenceIndex)：\(capturedAt) · \(frame.signalSummary) · \(frame.presenceSummary) · \(frame.activitySummary)")
        }
        if burstBuffer.isEmpty {
            lines.append("还没有保留关键帧；相机会继续等待学习材料或画面变化。")
        }
        return lines.joined(separator: "\n")
    }

    private func observationContextPayload() -> [String: Any] {
        var payload: [String: Any] = [
            "is_active": isBursting,
            "role": "background_observation",
            "policy": "Use this as supporting process context for the current answer and later summary. Do not let old observation override the user's current question or the current QA frame.",
            "buffered_frame_count": burstBuffer.count,
            "upload_state": uploadState,
            "question_fingerprint": [
                "candidate_count": observationQuestionCandidateCount,
                "unique_count": observationQuestionUniqueCount,
                "duplicate_count": observationQuestionDuplicateCount,
                "last_action": observationQuestionLastAction,
                "recent_events": Array(observationQuestionEvents.suffix(6))
            ]
        ]
        if let lastAcceptedAt {
            payload["last_accepted_at"] = CaptureTimeFormatter.string(from: lastAcceptedAt)
        }
        payload["recent_frames"] = burstBuffer.suffix(4).map { frame -> [String: Any] in
            [
                "sequence_index": frame.sequenceIndex,
                "captured_at": CaptureTimeFormatter.string(from: frame.capturedAt),
                "signal_summary": frame.signalSummary,
                "presence_summary": frame.presenceSummary,
                "activity_summary": frame.activitySummary,
                "text_count": frame.textCount,
                "rectangle_count": frame.rectangleCount,
                "has_study_material": frame.hasStudyMaterial,
                "has_explicit_study_evidence": frame.hasExplicitStudyEvidence,
                "has_student_presence": frame.hasStudentPresence,
                "text_tokens": Array(frame.textTokens.prefix(12))
            ]
        }
        return payload
    }

    func memoryDigestContextItem() -> ContextBadgeItem {
        let summary = memoryDigestSummary
        let detail = summary.isEmpty ? "服务器会每小时整理一次用户输入" : "\(summary.count) 条输入/画像候选"
        let profile = memoryProfileText.trimmingCharacters(in: .whitespacesAndNewlines)
        var lines: [String] = []
        lines.append("服务器整理：\(memoryProfileUpdatedAt.isEmpty ? "等待整理" : HistoryDateFormatter.displayString(from: memoryProfileUpdatedAt))")
        lines.append("画像：")
        lines.append(profile.isEmpty ? "暂未形成稳定画像。" : profile)
        let recent = memoryEvents.prefix(30)
        if !recent.isEmpty {
            lines.append("")
            lines.append("最近用户输入：")
            for event in recent {
                lines.append("- \(HistoryDateFormatter.displayString(from: event.createdAt)) \(event.text)")
            }
        } else if !userInputMemory.isEmpty {
            lines.append("")
            lines.append("本机最近输入：")
            for text in userInputMemory.prefix(30) {
                lines.append("- \(text)")
            }
        }
        return ContextBadgeItem(
            id: "memory-digest",
            title: "记忆整理",
            detail: detail,
            systemImage: "brain.head.profile",
            tone: isLoadingMemoryDigest ? .waiting : .neutral,
            fullDetail: lines.joined(separator: "\n")
        )
    }

    private func dynamicStrategyPreview() -> String {
        var parts = [learningMode.title, coachDepth.title]
        if isBursting {
            parts.append("观察 \(observationAcceptedFrameCount) 张")
        }
        if !memoryDigestSummary.isEmpty {
            parts.append("记忆 \(memoryDigestSummary.count) 条")
        }
        if carriedHistoryContext != nil {
            parts.append("带历史")
        }
        return parts.joined(separator: " · ")
    }

    private func dynamicStrategyDetail() -> String {
        let payload = dynamicStrategyPayload(
            transcript: pendingQAQuestion.isEmpty ? recognizedText : pendingQAQuestion,
            turn: max(qaTurnIndex + 1, 1),
            qaFrame: anchoredQAFrame
        )
        return prettyJSONString(payload)
    }

    private func dynamicStrategyPayload(transcript: String, turn: Int, qaFrame: QAFrameCandidate?) -> [String: Any] {
        var payload: [String: Any] = [
            "current_turn": turn,
            "current_preference": [
                "learning_mode": learningMode.rawValue,
                "learning_mode_title": learningMode.title,
                "coach_depth": coachDepth.rawValue,
                "coach_depth_title": coachDepth.title,
                "assistant_focus": coachAssistantFocus
            ],
            "current_session": [
                "session_id": sessionId ?? "",
                "session_text": sessionText,
                "student_goal": trimmedStudentGoal,
                "qa_turn_index": qaTurnIndex,
                "conversation_mode": continuousVoiceActive ? "continuous_observation_voice" : "turn_based_chat",
                "device_interaction_presence": userOperationPresencePayload()
            ],
            "current_question": shortText(transcript, limit: 260),
            "current_frame": [
                "available": contextInclusionSettings.visual && (qaFrame != nil || anchoredQAFrame != nil),
                "camera_ready": isCameraReady,
                "policy": contextInclusionSettings.visual ? "use_current_or_anchor_frame_only_when relevant to the user question" : "disabled_by_context_settings"
            ],
            "instruction": "Combine current preference, current frame, current session, current turn, memory digest, active review item and carried history. Prioritize the user's current question and do not let old context override it."
        ]
        if contextInclusionSettings.observation {
            payload["observation"] = [
                "is_active": isBursting,
                "accepted_frame_count": observationAcceptedFrameCount,
                "buffered_frame_count": burstBuffer.count,
                "summary": observationContextSummary()
            ]
        }
        if contextInclusionSettings.memory {
            payload["memory"] = [
                "profile_updated_at": memoryProfileUpdatedAt,
                "summary": Array(memoryDigestSummary.prefix(8))
            ]
        }
        if contextInclusionSettings.history, let carriedHistoryContext {
            payload["carried_history"] = carriedHistoryContext.payload
        }
        if contextInclusionSettings.mistakes, let activeReviewItem {
            payload["active_review"] = activeReviewItem.compactContext
        }
        return payload
    }

    init() {
        normalizeCoachPreferenceForMode()
        UserDefaults.standard.set(learningMode.rawValue, forKey: learningModeDefaultsKey)
        UserDefaults.standard.set(coachDepth.rawValue, forKey: coachDepthDefaultsKey)
    }

    private func normalizeCoachPreferenceForMode() {
        if learningMode == .answerCheck && coachDepth == .hintFirst {
            coachDepth = .checkOnly
        }
    }

    func refreshReviewQueuePreview() async {
        guard !isPreparingReview else { return }
        do {
            let queue = try await fetchReviewQueue(dueOnly: true, pageSize: 6)
            reviewDueCount = queue.total
            dueReviewItems = queue.items
            if let item = queue.items.first {
                reviewQueueState = "待复习：\(reviewQueueSummary(for: item))"
            } else {
                let fallback = try await fetchReviewQueue(dueOnly: false, pageSize: 8)
                recentReviewItems = fallback.items
                if let item = fallback.items.first {
                    reviewQueueState = "暂无到期错题，可提前复习：\(reviewQueueSummary(for: item))"
                } else {
                    reviewQueueState = "暂无错题队列；拍题或智能观察后会自动沉淀复习内容"
                }
            }
            if !queue.items.isEmpty {
                let fallback = try? await fetchReviewQueue(dueOnly: false, pageSize: 8)
                recentReviewItems = fallback?.items ?? queue.items
            }
        } catch {
            reviewQueueState = "复习队列读取失败"
            log("复习队列读取失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    /// 错题本页专用：一次性拉取全部未掌握错题（含未到期），按是否到期在前端分组展示。
    /// 复用 /api/review-queue（due_only=false），无需新增后端接口。
    @Published var isLoadingMistakeBook = false
    func loadMistakeBook() async {
        guard !isLoadingMistakeBook else { return }
        isLoadingMistakeBook = true
        defer { isLoadingMistakeBook = false }
        do {
            let all = try await fetchReviewQueue(dueOnly: false, pageSize: 100)
            recentReviewItems = all.items
            dueReviewItems = all.items.filter { $0.isDue }
            reviewDueCount = dueReviewItems.count
        } catch {
            log("错题本读取失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    /// 错题本操作的轻提示（错题本页顶部短暂横幅）。
    @Published var mistakeActionMessage: String = ""

    private func submitReviewResult(mistakeId: String, result: String, note: String = "") async -> Bool {
        let id = mistakeId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else { return false }
        var payload: [String: Any] = ["result": result]
        if !note.isEmpty { payload["review_note"] = note }
        do {
            _ = try await postJSON(path: "/api/mistakes/\(id)/review-events", payload: payload)
            await loadMistakeBook()
            return true
        } catch {
            log("错题状态更新失败（\(result)）：\(networkErrorDescription(error))", level: "error")
            mistakeActionMessage = "操作失败，请检查网络后重试"
            return false
        }
    }

    /// 消灭这道错题：标记已掌握（status=mastered），移出复习队列。
    func markMistakeMastered(_ mistakeId: String) async {
        if await submitReviewResult(mistakeId: mistakeId, result: "mastered", note: "学生确认已掌握，不再错") {
            mistakeActionMessage = "已消灭这道错题 · 不再进入复习"
            log("错题已标记掌握并移出队列")
        }
    }

    /// 还会错：保留并改约下次复习（postpone），让它继续留在错题本里。
    func keepMistakeForReview(_ mistakeId: String) async {
        if await submitReviewResult(mistakeId: mistakeId, result: "postpone") {
            mistakeActionMessage = "已保留，安排稍后再复习"
        }
    }

    /// 复习指定错题：切到复习模式，带着这道题进入对话，由 AI 先回忆再检查是否掌握。
    func startReviewForItem(_ item: ReviewMistakeItem) async {
        guard !isPreparingReview, !isSubmittingQuestion else { return }
        isPreparingReview = true
        defer { isPreparingReview = false }
        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        qaHideTimer?.invalidate()
        learningMode = .review
        coachDepth = .hintFirst
        UserDefaults.standard.set(learningMode.rawValue, forKey: learningModeDefaultsKey)
        UserDefaults.standard.set(coachDepth.rawValue, forKey: coachDepthDefaultsKey)
        sessionId = nil
        qaTurnIndex = 0
        lastSyncedStrategySignature = ""
        strategySyncState = "待同步"
        activeReviewItem = item
        studentGoal = reviewGoal(for: item, isDue: item.isDue)
        let question = reviewQuestion(for: item)
        reviewQueueState = "正在复习：\(reviewQueueSummary(for: item))"
        qaOverlayVisible = true
        qaAnswer = ""
        qaStateText = "准备复习这道错题"
        qaSystemImage = "arrow.triangle.2.circlepath"
        uploadState = "准备复习"
        studentGoalDidChange()
        pendingQATrigger = "review_today"
        pendingQAFocus = nil
        pendingQAQuestion = question
        recognizedText = question
        appendUserChatMessage(question)
        isThinking = true
        qaStateText = "思考中"
        qaSystemImage = "brain.head.profile"
        let generation = beginQuestionSubmissionGeneration()
        await submitRecognizedQuestion(generation: generation)
    }

    func refreshHistorySessions() async {
        guard !isLoadingHistory else { return }
        isLoadingHistory = true
        defer { isLoadingHistory = false }
        do {
            let data = try await getData(path: "/api/sessions")
            let response = try JSONDecoder().decode(HistorySessionsResponse.self, from: data)
            historySessions = response.sessions.filter { !$0.id.isEmpty }
            log("历史对话已刷新：\(historySessions.count) 个回合")
        } catch {
            log("历史对话读取失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    func refreshMemoryDigest(force: Bool = false) async {
        guard !isLoadingMemoryDigest else { return }
        isLoadingMemoryDigest = true
        defer { isLoadingMemoryDigest = false }
        do {
            if force {
                _ = try await postForm(path: "/api/memory/consolidate", fields: [:], files: [])
            }
            let data = try await getData(path: "/api/memory")
            let response = try JSONDecoder().decode(MemoryDigestResponse.self, from: data)
            memoryProfileText = response.profile.profile
            memoryProfileUpdatedAt = response.profile.updatedAt
            memoryEvents = response.events
            log("记忆整理已刷新：最近输入 \(response.events.count) 条，画像来源 \(response.profile.sourceCount) 条")
        } catch {
            log("记忆整理读取失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    func historyReport(for session: HistorySessionSummary) async -> HistoryReportDetail? {
        do {
            if session.imageCount == 0 && session.qaCount > 0 {
                _ = try? await postForm(
                    path: "/api/sessions/\(session.id)/finish",
                    fields: ["device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone"],
                    files: []
                )
            }
            let components = URLComponents(url: serverBaseURL.appending(path: "/api/sessions/\(session.id)/overview"), resolvingAgainstBaseURL: false)
            let data = try await getData(url: components?.url ?? serverBaseURL.appending(path: "/api/sessions/\(session.id)/overview"))
            let object = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            let finalAnalysis = object["final_analysis"] as? [String: Any]
            let sessionObject = object["session"] as? [String: Any] ?? [:]
            let qaEvents = object["qa_events"] as? [[String: Any]] ?? []
            let finalContent = firstNonEmpty(
                finalAnalysis?["content"] as? String ?? "",
                sessionObject["summary"] as? String ?? "",
                session.summaryPreview,
                "报告正在生成中，请稍后刷新历史对话后再查看。"
            )
            let qaRounds = qaEvents.prefix(12).compactMap { event -> HistoryQARound? in
                let question = (event["question"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let answer = (event["answer"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                guard !question.isEmpty || !answer.isEmpty else { return nil }
                let title = firstSentenceTitle(question.isEmpty ? answer : question)
                return HistoryQARound(title: title, question: question, answer: answer)
            }
            let qaPreview = qaRounds.prefix(8).map {
                "问：\(shortText($0.question, limit: 70))\n答：\(shortText($0.answer, limit: 110))"
            }.joined(separator: "\n\n")
            let isObservation = session.imageCount > 0
            return HistoryReportDetail(
                title: session.displayTitle,
                subtitle: isObservation ? "观察报告 · \(session.countSummary)" : "问答总结 · \(session.countSummary)",
                content: finalContent,
                qaPreview: qaPreview,
                qaRounds: qaRounds,
                systemImage: isObservation ? "doc.text.magnifyingglass" : "text.bubble"
            )
        } catch {
            log("历史报告读取失败：\(networkErrorDescription(error))", level: "error")
            return HistoryReportDetail(
                title: session.displayTitle,
                subtitle: "读取失败",
                content: networkErrorUserMessage(error),
                qaPreview: "",
                qaRounds: [],
                systemImage: "exclamationmark.triangle"
            )
        }
    }

    func startConversationFromHistory(_ session: HistorySessionSummary) async {
        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        var detail: [String: Any] = [:]
        do {
            let data = try await getData(path: "/api/sessions/\(session.id)")
            detail = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        } catch {
            log("历史回合详情读取失败，使用列表摘要：\(networkErrorDescription(error))", level: "error")
        }
        let context = makeHistoryCarryContext(summary: session, detail: detail)
        startNewConversation()
        carriedHistoryContext = context
        studentGoal = "接着历史回合：\(context.title)"
        lastSubmittedContextItems = [context.badge]
        // 进入历史是「回看+接着聊」，不主动开相机
        if cameraTaskKind != .burst {
            cameraTaskKind = .none
            cameraPreviewVisible = false
        }
        appendStatusChatMessage(
            title: "已接入历史回合",
            text: "下面回看「\(context.title)」的历史内容，你可以直接接着提问。",
            systemImage: "clock.arrow.circlepath",
            contextItems: [context.badge]
        )
        replayHistoryMessages(detail: detail, session: session)
        log("已从历史回合新开对话并回放历史：\(session.id)")
    }

    // 把历史回合的问答/观察内容回放到对话流，让用户看得见（#4）
    private func replayHistoryMessages(detail: [String: Any], session: HistorySessionSummary) {
        var replay: [ChatMessage] = []
        if let qaEvents = detail["qa_events"] as? [[String: Any]], !qaEvents.isEmpty {
            let sorted = qaEvents.sorted { ($0["created_at"] as? String ?? "") < ($1["created_at"] as? String ?? "") }
            for ev in sorted {
                let q = (ev["question"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let a = (ev["answer"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if !q.isEmpty { replay.append(ChatMessage(role: .user, text: q)) }
                if !a.isEmpty {
                    replay.append(ChatMessage(role: .assistant, text: a, question: q,
                                              qaEventId: ev["id"] as? String ?? ""))
                }
            }
        }
        // 观察回合（或无问答）：回放报告/分析摘要
        if replay.isEmpty {
            let reports = (detail["report_events"] as? [[String: Any]]) ?? []
            let analyses = (detail["analyses"] as? [[String: Any]]) ?? []
            let texts: [String] = (reports + analyses).compactMap { row in
                for key in ["summary", "content", "text", "report", "analysis"] {
                    if let s = row[key] as? String, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return s }
                }
                return nil
            }
            for t in texts.prefix(6) {
                replay.append(ChatMessage(role: .assistant, text: t, title: "历史观察", systemImage: "eye"))
            }
            if replay.isEmpty && !session.summaryPreview.isEmpty {
                replay.append(ChatMessage(role: .assistant, text: session.summaryPreview, title: "历史摘要", systemImage: "doc.text"))
            }
        }
        guard !replay.isEmpty else { return }
        chatMessages.append(contentsOf: replay)
    }

    private func makeHistoryCarryContext(summary: HistorySessionSummary, detail: [String: Any]) -> HistoryCarryContext {
        let session = detail["session"] as? [String: Any] ?? [:]
        let images = detail["images"] as? [[String: Any]] ?? []
        let qaEvents = detail["qa_events"] as? [[String: Any]] ?? []
        let learningItems = detail["learning_items"] as? [[String: Any]] ?? []
        let mistakeItems = detail["mistake_items"] as? [[String: Any]] ?? []
        let analyses = detail["analyses"] as? [[String: Any]] ?? []
        let finalAnalysis = analyses.last(where: { ($0["scope"] as? String) == "final" })
        let recentQA = Array(qaEvents.prefix(4))
        let recentMistakes = Array(mistakeItems.prefix(4))
        let recentLearning = Array(learningItems.prefix(4))
        let previewFilename = (recentQA.compactMap { firstNonEmpty($0["selected_image_filename"] as? String ?? "", $0["image_filename"] as? String ?? "") }.first)
            ?? (images.last?["filename"] as? String)
        let title = firstNonEmpty(
            summary.title,
            session["title"] as? String ?? "",
            summary.studentGoal,
            "历史回合 " + String(summary.id.prefix(8))
        )
        let finalSummary = firstNonEmpty(
            finalAnalysis?["content"] as? String ?? "",
            session["summary"] as? String ?? "",
            summary.summaryPreview
        )
        let questionSnippets = recentQA.compactMap { event -> String? in
            let question = firstNonEmpty(event["question"] as? String ?? "", (event["context"] as? [String: Any])?["transcript"] as? String ?? "")
            guard !question.isEmpty else { return nil }
            return "问：\(shortText(question, limit: 42))"
        }
        let answerSnippets = recentQA.compactMap { event -> String? in
            let answer = event["answer"] as? String ?? ""
            guard !answer.isEmpty else { return nil }
            return "答：\(shortText(answer, limit: 70))"
        }
        let mistakeSnippets = recentMistakes.compactMap { item -> String? in
            let title = firstNonEmpty(item["title"] as? String ?? "", item["question_text"] as? String ?? "")
            guard !title.isEmpty else { return nil }
            return "错题：\(shortText(title, limit: 58))"
        }
        let learningSnippets = recentLearning.compactMap { item -> String? in
            let title = firstNonEmpty(item["title"] as? String ?? "", item["knowledge_point"] as? String ?? "", item["content"] as? String ?? "")
            guard !title.isEmpty else { return nil }
            return "知识点：\(shortText(title, limit: 58))"
        }
        let lines = [
            "来源：\(title)",
            "统计：\(summary.countSummary)",
            finalSummary.isEmpty ? "" : "摘要：\(shortText(finalSummary, limit: 220))"
        ] + questionSnippets + answerSnippets + mistakeSnippets + learningSnippets
        let detailText = lines.filter { !$0.isEmpty }.joined(separator: "\n")
        let payload: [String: Any] = [
            "source_session_id": summary.id,
            "title": title,
            "stats": [
                "image_count": summary.imageCount,
                "qa_count": summary.qaCount,
                "mistake_count": summary.mistakeCount,
                "analysis_count": summary.analysisCount
            ],
            "summary": finalSummary,
            "recent_questions": questionSnippets,
            "recent_answers": answerSnippets,
            "mistakes": mistakeSnippets,
            "learning_items": learningSnippets,
            "instruction": "Use this compressed historical session only when it is relevant to the user's new question; do not assume the user is still on the old problem unless they ask to continue it."
        ]
        return HistoryCarryContext(
            sourceSessionId: summary.id,
            title: title,
            summary: shortText(firstNonEmpty(finalSummary, title, summary.countSummary), limit: 48),
            detail: detailText,
            payload: payload,
            previewFilename: previewFilename
        )
    }

    func startTodayReview() async {
        guard !isPreparingReview, !isSubmittingQuestion else { return }
        isPreparingReview = true
        defer { isPreparingReview = false }

        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        qaHideTimer?.invalidate()
        activeReviewItem = nil
        learningMode = .review
        coachDepth = .hintFirst
        UserDefaults.standard.set(learningMode.rawValue, forKey: learningModeDefaultsKey)
        UserDefaults.standard.set(coachDepth.rawValue, forKey: coachDepthDefaultsKey)
        sessionId = nil
        qaTurnIndex = 0
        lastSyncedStrategySignature = ""
        strategySyncState = "待同步"

        qaOverlayVisible = true
        qaAnswer = ""
        qaStateText = "准备今日复习"
        qaSystemImage = "calendar.badge.clock"
        uploadState = "准备复习"
        reviewQueueState = "正在读取复习队列"

        let question: String
        do {
            let dueQueue = try await fetchReviewQueue(dueOnly: true, pageSize: 1)
            reviewDueCount = dueQueue.total
            let fallbackItem: ReviewMistakeItem?
            if dueQueue.items.isEmpty {
                fallbackItem = try await fetchReviewQueue(dueOnly: false, pageSize: 1).items.first
            } else {
                fallbackItem = nil
            }

            if let item = dueQueue.items.first ?? fallbackItem {
                activeReviewItem = item
                studentGoal = reviewGoal(for: item, isDue: item.isDue)
                question = reviewQuestion(for: item)
                reviewQueueState = "正在复习：\(reviewQueueSummary(for: item))"
                log("今日复习已选中错题：\(reviewQueueSummary(for: item))")
            } else {
                studentGoal = "今日复习：暂无到期错题，请根据最近学习记录安排 5 分钟复习"
                question = "今天没有到期错题。请根据我最近的学习记录，安排一个 5 分钟复习计划，并告诉我先从哪里开始。"
                reviewQueueState = "暂无待复习错题"
                log("今日复习队列为空，改为生成 5 分钟复习计划")
            }
        } catch {
            studentGoal = "今日复习：复习队列暂时不可用，请先安排一个 5 分钟复习计划"
            question = "复习队列暂时读取失败。请先帮我安排一个 5 分钟复习计划，要求先回忆、再练一道、最后检查掌握。"
            reviewQueueState = "复习队列读取失败"
            log("今日复习读取失败，降级为复习计划：\(networkErrorDescription(error))", level: "error")
        }

        studentGoalDidChange()
        pendingQATrigger = "review_today"
        pendingQAFocus = nil
        pendingQAQuestion = question
        recognizedText = question
        appendUserChatMessage(question)
        isThinking = true
        qaStateText = "思考中"
        qaSystemImage = "brain.head.profile"
        let generation = beginQuestionSubmissionGeneration()
        await submitRecognizedQuestion(generation: generation)
    }

    func startVoiceQuestion(trigger: String, focus: QAFocus? = nil) {
        suppressObservationDanmakuBriefly()
        qaHideTimer?.invalidate()
        shouldSubmitWhenVoiceReady = false
        pendingQATrigger = trigger
        pendingQAFocus = focus
        pendingQAQuestion = ""
        if cameraTaskKind != .burst {
            cameraTaskKind = .qaFrame
        }
        cameraPreviewVisible = true
        qaOverlayVisible = true
        recognizedText = ""
        qaAnswer = ""
        uploadState = "准备语音"
        stopSpeaking()
        isPreparingVoiceInput = true
        ensureSpeechAuthorization { [weak self] granted in
            guard let self else { return }
            self.isPreparingVoiceInput = false
            if granted {
                self.speak("请说") { [weak self] in
                    Task { @MainActor in
                        self?.beginListening()
                    }
                }
            } else {
                self.qaStateText = "语音权限不可用"
                self.qaSystemImage = "mic.slash"
                self.cameraTaskKind = .none
                self.cameraPreviewVisible = false
                self.log("语音识别或麦克风权限不可用", level: "error")
            }
        }
    }

    func startContinuousVoiceConversation() {
        guard !voiceInputDisabled else { return }
        recordUserOperation("start_continuous_voice")
        if !isBursting {
            startBurst()
        }
        continuousVoiceActive = true
        observationGuideVisible = false
        shouldSubmitWhenVoiceReady = false
        qaHideTimer?.invalidate()
        pendingQATrigger = qaTurnIndex == 0 ? "continuous_voice_first_turn" : "continuous_voice_followup"
        pendingQAFocus = nil
        pendingQAQuestion = ""
        cameraTaskKind = .burst
        cameraPreviewVisible = true
        qaOverlayVisible = true
        qaAnswer = ""
        uploadState = "持续倾听"
        appendStatusChatMessage(
            title: "连续语音已开启",
            text: "现在会自动听你说话；停顿后提交，AI 回答完会继续听。说“停止录音”可以关闭。",
            systemImage: "waveform.circle.fill",
            showsProgress: true
        )
        stopSpeaking()
        if isListening || isPreparingVoiceInput || voiceInputIsStarting {
            return
        }
        isPreparingVoiceInput = true
        ensureSpeechAuthorization { [weak self] granted in
            guard let self else { return }
            self.isPreparingVoiceInput = false
            if granted {
                self.beginContinuousListeningTurn()
            } else {
                self.continuousVoiceActive = false
                self.qaStateText = "语音权限不可用"
                self.qaSystemImage = "mic.slash"
                self.log("语音识别或麦克风权限不可用", level: "error")
            }
        }
    }

    func stopContinuousVoiceConversation(reason: String) {
        suppressObservationDanmakuBriefly()
        continuousVoiceActive = false
        voiceInputIsStarting = false
        shouldSubmitWhenVoiceReady = false
        stopListening(submit: false)
        stopSpeaking()
        qaStateText = "录音已关闭"
        qaSystemImage = "mic.slash"
        uploadState = isBursting ? "智能观察中" : "待机"
        log("持续录音已关闭：\(reason)")
    }

    private func beginContinuousListeningTurn() {
        guard continuousVoiceActive, !voiceInputDisabled else { return }
        guard !isListening, !isPreparingVoiceInput, !voiceInputIsStarting, !isThinking, !ttsPlaybackPhase.isActive else { return }
        suppressObservationDanmakuBriefly()
        pendingQATrigger = qaTurnIndex == 0 ? "continuous_voice_first_turn" : "continuous_voice_followup"
        pendingQAFocus = nil
        pendingQAQuestion = ""
        recognizedText = ""
        qaOverlayVisible = true
        qaStateText = "持续倾听"
        qaSystemImage = "waveform"
        uploadState = "持续倾听"
        if cameraTaskKind != .burst {
            cameraTaskKind = .qaFrame
        }
        cameraPreviewVisible = true
        beginListening()
    }

    private var pendingVoiceTextOnly = false

    func beginHoldToTalk() {
        guard !voiceInputDisabled else { return }
        pendingVoiceTextOnly = false
        suppressObservationDanmakuBriefly()
        recordUserOperation("hold_to_talk")
        if continuousVoiceActive {
            stopContinuousVoiceConversation(reason: "用户长按切换关闭")
            return
        }
        if isListening || isPreparingVoiceInput || voiceInputIsStarting { return }
        voiceInputIsStarting = true
        holdToTalkActive = true
        shouldSubmitWhenVoiceReady = false
        qaHideTimer?.invalidate()
        pendingQATrigger = qaTurnIndex == 0 ? "hold_to_talk_first_turn" : "hold_to_talk_followup"
        pendingQAFocus = nil
        pendingQAQuestion = ""
        if cameraTaskKind != .burst {
            cameraTaskKind = .qaFrame
        }
        cameraPreviewVisible = true
        qaOverlayVisible = true
        recognizedText = ""
        qaAnswer = ""
        uploadState = "按住说话"
        stopSpeaking()
        isPreparingVoiceInput = true
        ensureSpeechAuthorization { [weak self] granted in
            guard let self else { return }
            self.voiceInputIsStarting = false
            self.isPreparingVoiceInput = false
            guard self.holdToTalkActive else {
                self.qaStateText = "语音已取消"
                self.qaSystemImage = "mic.slash"
                return
            }
            if granted {
                self.beginListening()
            } else {
                self.holdToTalkActive = false
                self.qaStateText = "语音权限不可用"
                self.qaSystemImage = "mic.slash"
                self.cameraTaskKind = .none
                self.cameraPreviewVisible = false
                self.log("语音识别或麦克风权限不可用", level: "error")
            }
        }
    }

    func endHoldToTalk(textOnly: Bool = false) {
        suppressObservationDanmakuBriefly(seconds: 4.0)
        if isPreparingVoiceInput {
            cancelHoldToTalk()
            return
        }
        guard isListening else { return }
        // 上滑选了「只纯文字」：本次语音提交不带当前画面
        pendingVoiceTextOnly = textOnly
        if textOnly {
            cameraTaskKind = .none
            cameraPreviewVisible = false
        }
        holdToTalkActive = false
        scheduleVoiceSubmitAfterRecognitionGrace(maxWait: 3.4)
    }

    func cancelHoldToTalk() {
        suppressObservationDanmakuBriefly()
        voiceInputIsStarting = false
        holdToTalkActive = false
        shouldSubmitWhenVoiceReady = false
        isPreparingVoiceInput = false
        if continuousVoiceActive {
            continuousVoiceActive = false
        }
        qaStateText = "语音已取消"
        qaSystemImage = "mic.slash"
        guard isListening else { return }
        stopListening(submit: false)
        if cameraTaskKind == .qaFrame {
            cameraTaskKind = .none
            cameraSheetVisible = false
            cameraPreviewVisible = false
        }
    }

    func cancelVoiceForRetry() {
        qaSubmissionGeneration += 1
        suppressObservationDanmakuBriefly()
        continuousVoiceActive = false
        voiceSubmitGraceTask?.cancel()
        voiceSubmitGraceTask = nil
        stopListening(submit: false)
        stopSpeaking()
        completePendingQAFrame(nil)
        isWaitingForVoiceSubmitGrace = false
        voiceInputIsStarting = false
        holdToTalkActive = false
        shouldSubmitWhenVoiceReady = false
        isPreparingVoiceInput = false
        isThinking = false
        isSubmittingQuestion = false
        pendingQAQuestion = ""
        recognizedText = ""
        qaAnswer = ""
        qaStateText = "已取消，可重说"
        qaSystemImage = "arrow.counterclockwise"
        uploadState = isBursting ? "智能观察中" : "待机"
        log("本轮语音已取消，可重新按住说话")
    }

    func toggleVoicePlayback() {
        voicePlaybackEnabledDidChange(!voicePlaybackEnabled)
    }

    // 自动梯形校正：拍照分题前先把拍歪的纸张扶正（检测不到/形变很小则用原图）
    func autoKeystoneCorrectionDidChange(_ on: Bool) {
        autoKeystoneCorrection = on
        UserDefaults.standard.set(on, forKey: autoKeystoneCorrectionDefaultsKey)
        log(on ? "自动梯形校正已开启" : "自动梯形校正已关闭：用相机原图")
    }

    // 多题自动分割：拍照后把多道题各盖一个浮层供点选（关=直接整图问答）
    func questionSegmentationEnabledDidChange(_ on: Bool) {
        questionSegmentationEnabled = on
        UserDefaults.standard.set(on, forKey: questionSegmentationEnabledDefaultsKey)
        log(on ? "多题自动分割已开启" : "多题自动分割已关闭：拍照按整张图分析")
    }

    // 纯文字提问（不开相机）：开启后打字提问不再打开相机/抓取画面，按纯文字理解
    func textOnlyQuestionDidChange(_ on: Bool) {
        textOnlyQuestion = on
        UserDefaults.standard.set(on, forKey: textOnlyQuestionDefaultsKey)
        log(on ? "纯文字提问已开启：打字提问不再开相机" : "纯文字提问已关闭：打字提问会结合当前画面")
    }

    // 一句话辅导偏好：失焦/提交时持久化并触发策略同步（不复用 studentGoal）
    func coachPreferenceTextDidChange(_ text: String) {
        coachPreferenceText = text
        UserDefaults.standard.set(text, forKey: coachPreferenceTextDefaultsKey)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        log(trimmed.isEmpty ? "辅导偏好已清空，回到默认策略" : "辅导偏好已更新：\(trimmed)")
        coachPreferenceDidChange()
    }

    // 提示词：加载列表（GET /api/prompts）。已加载且非空时不重复拉取。
    func loadCoachPrompts() async {
        guard !isLoadingPrompts, coachPrompts.isEmpty else { return }
        isLoadingPrompts = true
        defer { isLoadingPrompts = false }
        do {
            let data = try await getData(path: "/api/prompts")
            let decoded = try JSONDecoder().decode(CoachPromptsResponse.self, from: data)
            coachPrompts = decoded.prompts
        } catch {
            log("加载提示词失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    // 提示词：恢复某条默认（POST /api/prompts/{key}/reset），成功后刷新该条。
    func resetCoachPrompt(key: String) async {
        do {
            let data = try await postJSON(path: "/api/prompts/\(key)/reset", payload: [String: Any]())
            let decoded = try JSONDecoder().decode(CoachPromptResetResponse.self, from: data)
            if let index = coachPrompts.firstIndex(where: { $0.key == key }) {
                coachPrompts[index] = decoded.prompt
            }
            log("已恢复默认提示词：\(decoded.prompt.label)")
        } catch {
            log("恢复默认提示词失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    #if DEBUG
    // 仅 DEBUG：注入示例对话，便于布局/截图回归（无网络、无相机）
    func debugSeedChatIfRequested() {
        guard ProcessInfo.processInfo.environment["PXJ_SEED_CHAT"] == "1", chatMessages.isEmpty else { return }
        chatMessages = [
            ChatMessage(role: .user, text: "请讲解：一个长方形长8宽5，面积和周长各是多少？另外正方形和长方形有什么区别？"),
            ChatMessage(role: .assistant,
                        text: "好的，我们一步一步来看这道题。\n\n【面积】长方形面积 = 长 × 宽 = 8 × 5 = 40（平方单位）。\n【周长】长方形周长 = (长 + 宽) × 2 = (8 + 5) × 2 = 26（长度单位）。\n\n正方形是四条边都相等的特殊长方形；长方形对边相等、相邻边不一定相等。要点：面积是“铺满平面”的大小，周长是“绕一圈”的长度，两者单位不同，别混淆。\n\n再来一道变式题练练手：如果一个长方形的周长是 30，宽是 6，那么它的长是多少？试着先自己算一算。",
                        question: "请讲解长方形面积和周长",
                        visualizationCandidate: true,
                        visualizationReason: "适合用图形演示长与宽")
        ]
        qaTurnIndex = 1
    }
    #endif

    func voicePlaybackEnabledDidChange(_ enabled: Bool) {
        let savedEnabled = UserDefaults.standard.object(forKey: voicePlaybackEnabledDefaultsKey) as? Bool
        guard voicePlaybackEnabled != enabled || savedEnabled != enabled || (!enabled && ttsPlaybackPhase.isActive) else { return }
        voicePlaybackEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: voicePlaybackEnabledDefaultsKey)
        if enabled {
            qaStateText = isSpeaking ? (ttsPlaybackPhase == .generating ? "生成语音" : "正在回答") : "声音已开启"
            qaSystemImage = "speaker.wave.2"
            log("语音播放已开启")
        } else {
            stopSpeaking()
            qaStateText = "声音已关闭"
            qaSystemImage = "speaker.slash"
            log("语音播放已关闭")
        }
    }

    func voicePlaybackRateDidChange(_ rate: Double) {
        let clamped = normalizedVoicePlaybackRate(rate)
        if clamped != voicePlaybackRate {
            voicePlaybackRate = clamped
        }
        UserDefaults.standard.set(clamped, forKey: voicePlaybackRateDefaultsKey)
        ttsPlayer?.rate = Float(clamped)
        if ttsPlaybackPhase == .playing || ttsPlaybackPhase == .paused {
            qaStateText = "\(ttsServiceText)播放中 \(voicePlaybackRateText)"
        }
    }

    func toggleSpeechPause() {
        guard voicePlaybackEnabled else { return }
        switch ttsPlaybackPhase {
        case .playing:
            if let player = ttsPlayer {
                player.pause()
            } else if localSpeechSynthesizer.isSpeaking {
                localSpeechSynthesizer.pauseSpeaking(at: .immediate)
            }
            ttsPlaybackPhase = .paused
            qaStateText = "朗读已暂停"
            qaSystemImage = "pause.circle"
            log("朗读已暂停")
        case .paused:
            let resumed: Bool
            if let player = ttsPlayer {
                resumed = player.play()
            } else if localSpeechSynthesizer.isPaused {
                resumed = localSpeechSynthesizer.continueSpeaking()
            } else {
                resumed = false
            }
            if resumed {
                ttsPlaybackPhase = .playing
                isSpeaking = true
                qaStateText = "\(ttsServiceText)播放中 \(voicePlaybackRateText)"
                qaSystemImage = "speaker.wave.2"
                log("朗读继续")
            }
        default:
            break
        }
    }

    private func suppressObservationDanmakuBriefly(seconds: TimeInterval = 3.0) {
        danmakuSuppressedUntil = Date().addingTimeInterval(seconds)
    }

    private func recordUserOperation(_ kind: String) {
        lastUserOperationAt = Date()
        lastUserOperationKind = kind
    }

    private func userOperationPresencePayload(referenceDate: Date = Date()) -> [String: Any] {
        let secondsAgo = referenceDate.timeIntervalSince(lastUserOperationAt)
        let isRecent = secondsAgo >= 0 && secondsAgo <= recentUserOperationPresenceWindow
        return [
            "present": isRecent,
            "source": "device_user_operation",
            "operation": lastUserOperationKind,
            "last_operation_at": lastUserOperationAt == Date.distantPast ? "" : CaptureTimeFormatter.string(from: lastUserOperationAt),
            "seconds_since_operation": lastUserOperationAt == Date.distantPast ? -1 : max(0, secondsAgo),
            "summary": isRecent ? "用户刚操作过手机，证明用户在场；视觉未识别到手/脸时不可直接判定离开。" : "近期没有手机操作，只能结合视觉证据判断。"
        ]
    }

    private func beginQuestionSubmissionGeneration() -> Int {
        qaSubmissionGeneration += 1
        return qaSubmissionGeneration
    }

    private func submissionIsCurrent(_ generation: Int) -> Bool {
        generation == qaSubmissionGeneration
    }

    func startNewConversation() {
        stopSpeaking()
        continuousVoiceActive = false
        stopListening(submit: false)
        completePendingQAFrame(nil)
        qaHideTimer?.invalidate()
        if cameraTaskKind == .qaFrame {
            cameraTaskKind = .none
            cameraSheetVisible = false
        }
        cameraPreviewVisible = false
        let shouldResetReviewContext = activeReviewItem != nil ||
            pendingQATrigger == "review_today" ||
            learningMode == .review ||
            trimmedStudentGoal.hasPrefix("今日复习")
        sessionId = nil
        qaTurnIndex = 0
        pendingQATrigger = "typed_chat"
        pendingQAFocus = nil
        pendingQAQuestion = ""
        anchoredQAFrame = nil
        carriedHistoryContext = nil
        recognizedText = ""
        qaAnswer = ""
        chatMessages.removeAll()
        lastSubmittedContextItems.removeAll()
        lastTurnManifest = nil
        activeReviewItem = nil
        if shouldResetReviewContext {
            studentGoal = ""
            reviewQueueState = "复习任务未开始"
            if learningMode == .review {
                learningMode = .singleProblem
                UserDefaults.standard.set(learningMode.rawValue, forKey: learningModeDefaultsKey)
            }
        }
        isThinking = false
        qaOverlayVisible = false
        qaStateText = "新对话"
        qaSystemImage = "plus.bubble"
        if isBursting {
            resetObservationForNewConversationBoundary()
            cameraTaskKind = .burst
            cameraPreviewVisible = true
            uploadState = "智能观察中"
            log("已开启新对话；智能观察继续作为背景上下文，首问会创建新的学习回合")
        } else {
            cameraTaskKind = .none
            cameraPreviewVisible = false
            qualityFeedback = .idle
            uploadState = "待机"
            log("已开启新对话")
        }
    }

    func startNewConversationAndListen() {
        guard !voiceInputDisabled else { return }
        startNewConversation()
        appendStatusChatMessage(
            title: "新对话",
            text: "相机预览已准备；按住底部输入栏说话时，会抓拍当前画面和上下文一起发送。",
            systemImage: "plus.bubble"
        )
    }

    func submitTypedQuestion(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // intentRouteInFlight：配置探测窗口（route timeout 内）也算"忙"，防 12s 内重复 send → 双卡片/重复 LLM 往返。
        guard !question.isEmpty, !isSubmittingQuestion, !intentRouteInFlight else { return }
        // 二期：仅文本路径做自然语言配置探测。命中 → 弹确认卡片，不进 QA；
        // 失败/超时/非配置一律降级为普通 QA（proceedTypedQuestion 即原逻辑）。
        Task { @MainActor in
            if await maybeInterceptAsIntent(question) { return }
            proceedTypedQuestion(question)
        }
    }

    // internal（非 private）：供同文件 submitTypedQuestion 与 Intent/IntentRouting.swift 的逃生口调用。
    @MainActor
    func proceedTypedQuestion(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !isSubmittingQuestion else { return }
        suppressObservationDanmakuBriefly(seconds: 4.0)
        recordUserOperation("typed_chat")
        qaHideTimer?.invalidate()
        stopSpeaking()
        stopListening(submit: false)
        pendingQATrigger = "typed_chat"
        pendingQAFocus = nil
        pendingQAQuestion = question
        recognizedText = question
        qaAnswer = ""
        if !textOnlyQuestion {
            if cameraTaskKind != .burst {
                cameraTaskKind = .qaFrame
            }
            cameraPreviewVisible = true
        }
        qaOverlayVisible = true
        lastSubmittedContextItems = pendingContextItems(draft: question)
        appendUserChatMessage(question)
        isThinking = true
        qaStateText = "思考中"
        qaSystemImage = "brain.head.profile"
        let generation = beginQuestionSubmissionGeneration()
        Task { await submitRecognizedQuestion(generation: generation) }
    }

    func submitQuickFollowUp(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !questionSubmissionInFlight else { return }
        suppressObservationDanmakuBriefly(seconds: 4.0)
        recordUserOperation("quick_followup")
        qaHideTimer?.invalidate()
        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        pendingQATrigger = "quick_followup_text_only"
        pendingQAFocus = nil
        pendingQAQuestion = question
        recognizedText = question
        qaAnswer = ""
        if isBursting {
            cameraPreviewVisible = true
        } else {
            cameraTaskKind = .none
            cameraPreviewVisible = false
        }
        qaOverlayVisible = true
        lastSubmittedContextItems = pendingContextItems(draft: question)
        appendUserChatMessage(question)
        isThinking = true
        qaStateText = "思考中"
        qaSystemImage = "brain.head.profile"
        log("快捷追问：仅携带对话和已有上下文，不截取当前画面")
        let generation = beginQuestionSubmissionGeneration()
        Task { await submitRecognizedQuestion(generation: generation) }
    }

    func submitCurrentFrameShortcut() {
        guard !voiceInputDisabled else { return }
        suppressObservationDanmakuBriefly(seconds: 4.0)
        recordUserOperation("double_tap_current_frame")
        qaHideTimer?.invalidate()
        voiceSubmitGraceTask?.cancel()
        voiceSubmitGraceTask = nil
        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        isWaitingForVoiceSubmitGrace = false
        voiceInputIsStarting = false
        holdToTalkActive = false
        shouldSubmitWhenVoiceReady = false
        isPreparingVoiceInput = false
        continuousVoiceActive = false
        pendingQATrigger = qaTurnIndex == 0 ? "double_tap_current_frame_first_turn" : "double_tap_current_frame_followup"
        pendingQAFocus = nil
        pendingQAQuestion = "帮我看一下当前画面。"
        recognizedText = pendingQAQuestion
        qaAnswer = ""
        if cameraTaskKind != .burst {
            cameraTaskKind = .qaFrame
        }
        cameraPreviewVisible = true
        qaOverlayVisible = true
        lastSubmittedContextItems = pendingContextItems(draft: pendingQAQuestion)
        appendUserChatMessage(pendingQAQuestion)
        isThinking = true
        qaStateText = "截取当前画面"
        qaSystemImage = "camera.viewfinder"
        log("双击按住说话：直接发送当前画面给 AI")
        let generation = beginQuestionSubmissionGeneration()
        Task { await submitRecognizedQuestion(generation: generation) }
    }

    func interruptForFollowUp() {
        log("追问入口已合并到底部长按语音，旧 OK 追问触发已忽略")
    }

    func endQARound() {
        stopSpeaking()
        stopListening(submit: false)
        completePendingQAFrame(nil)
        if cameraTaskKind == .qaFrame {
            cameraTaskKind = .none
            cameraSheetVisible = false
            cameraPreviewVisible = false
        }
        isThinking = false
        qaStateText = "本轮已结束"
        qaSystemImage = "checkmark.circle"
        scheduleQAOverlayHide(after: 1.5)
        log("AI 问答本轮结束")
    }

    func didCaptureQAFrame(_ image: UIImage) {
        guard pendingQAFrameContinuation != nil else {
            log("收到问答画面，但当前没有等待中的语音问题，已忽略")
            return
        }
        let analysis = BurstFrameAnalyzer.analyze(image)
        let assessment = CaptureQualityAssessment.evaluate(
            signals: analysis.signals,
            visualDistance: nil,
            textDistance: nil,
            isFirstUsefulFrame: true
        )
        completePendingQAFrame(QAFrameCandidate(image: image, analysis: analysis, assessment: assessment))
    }

    // MARK: - 题目分割 + 浮层选题（拍照答题 / 作业批改）

    /// 入口：先进「取景—自动快门」阶段对准拍清楚 → 抓全分辨率真照片 → 梯形校正 → 题目分割 → 浮层选题。
    func beginQuestionSegmentation(grading: Bool) {
        guard !segmentationBusy, !isSubmittingQuestion, !captureAimingVisible else { return }
        recordUserOperation(grading ? "segment_grading" : "segment_qa")
        segmentationIsGrading = grading
        selectedRegionID = nil
        questionRegions = []
        segmentationDisplayImage = nil
        pendingRegionQAFrame = nil
        // 进取景阶段：打开相机预览 + 取景框 + 自动快门
        aimReadyStreak = 0
        aimCapturing = false
        cameraAimSharpness = 0
        cameraAimBrightness = 0.5
        cameraAimMaterial = 0
        cameraTaskKind = .qaFrame
        cameraPreviewVisible = true
        cameraSheetVisible = false
        captureAimingVisible = true
        qaStateText = "对准取景"
        qaSystemImage = "viewfinder"
        log(grading ? "作业批改：进入取景拍照" : "拍照答题：进入取景拍照")
    }

    /// 取景阶段实时质量回调（控制器每帧调用）。达标连续若干帧 → 自动快门。
    func updateCameraAim(brightness: Double, sharpness: Double, material: Double) {
        guard captureAimingVisible else { return }
        cameraAimBrightness = brightness
        cameraAimSharpness = sharpness
        cameraAimMaterial = material
        guard autoShutterArmed, !aimCapturing else { return }
        if cameraAimReady {
            aimReadyStreak += 1
            if aimReadyStreak >= 3 {   // ~连续 3 帧达标(约 1 秒)才自动拍，去抖
                if cameraTaskKind == .extract {
                    captureExtractionPhoto()
                } else {
                    captureSegmentationPhoto()
                }
            }
        } else {
            aimReadyStreak = 0
        }
    }

    /// 手动/自动快门：抓一张全分辨率真照片用于分题。
    func captureSegmentationPhoto() {
        guard captureAimingVisible, !aimCapturing else { return }
        aimCapturing = true
        aimReadyStreak = 0
        qaStateText = "拍照中"
        qaSystemImage = "camera"
        if captureSegment?() != true {
            aimCapturing = false
            log("分题取景：抓拍入口不可用（模拟器无相机？）", level: "warning")
        }
    }

    /// 取景阶段抓到全分辨率照片 → 关取景层 → 校正 + 分割 → 展示浮层选题层。
    func didCaptureSegmentationPhoto(_ image: UIImage) {
        guard captureAimingVisible else { return }
        autoRouteRevertText = nil   // 已拍照committed，纠偏入口收起
        if torchOn { setTorch?(false); torchOn = false }
        captureAimingVisible = false
        aimCapturing = false
        segmentationVisible = true
        segmentationBusy = true
        segmentationNotice = ""
        qaStateText = "校正并分题"
        qaSystemImage = "rectangle.dashed"
        // 重活放到下一个 runloop，先让取景层切到选题层的 loading
        DispatchQueue.main.async { [weak self] in
            self?.processSegmentation(raw: image)
        }
    }

    func toggleTorch() {
        torchOn.toggle()
        setTorch?(torchOn)
    }

    func cancelAiming() {
        if torchOn { setTorch?(false); torchOn = false }
        captureAimingVisible = false
        aimCapturing = false
        aimReadyStreak = 0
        autoRouteRevertText = nil
        if cameraTaskKind == .qaFrame { cameraTaskKind = .none }
        cameraPreviewVisible = false
    }

    // MARK: - 题目提取（观察模式）：拍清试卷 → 逐张提取 → 服务端去重累积 → 还原页/空白卷。
    /// 进入题目提取取景：连拍多张试卷，逐张提取去重。强调取景质量（试卷铺满+清晰）。
    func beginQuestionExtraction() {
        guard !captureAimingVisible, !segmentationBusy, !isSubmittingQuestion else { return }
        recordUserOperation("question_extract")
        extractShotCount = 0
        extractUniqueCount = 0
        extractInflight = 0
        extractRestoreHTML = nil
        extractNotice = "把试卷铺满取景框、放平、对焦，清晰后会自动拍；也可手动按快门"
        aimReadyStreak = 0
        aimCapturing = false
        cameraAimSharpness = 0
        cameraAimBrightness = 0.5
        cameraAimMaterial = 0
        cameraTaskKind = .extract
        cameraPreviewVisible = true
        cameraSheetVisible = false
        captureAimingVisible = true
        qaStateText = "对准试卷"
        qaSystemImage = "doc.text.viewfinder"
        speak("把试卷对准取景框、拍清楚，我来提取题目", completion: nil)
        log("题目提取：进入取景拍照")
        extractSeenKeys = []
        Task { _ = await ensureExtractSession() }
        if extractAutoScan { startExtractScan() }
    }

    /// 手动/自动快门：抓一张全分辨率真照片用于提取。串行（一次只在跑一张）。
    func captureExtractionPhoto() {
        guard captureAimingVisible, cameraTaskKind == .extract, !aimCapturing, extractInflight == 0 else { return }
        aimCapturing = true
        aimReadyStreak = 0
        qaStateText = "拍照中"
        if captureExtract?() != true {
            aimCapturing = false
            extractNotice = "抓拍入口不可用（模拟器无相机？请用真机测试）"
            log("题目提取取景：抓拍入口不可用", level: "warning")
        }
    }

    /// 抓到全分辨率照片 → 不关取景层（连拍）→ 异步上传提取，完成后允许下一张。
    func didCaptureExtractionPhoto(_ image: UIImage) {
        guard captureAimingVisible, cameraTaskKind == .extract else { return }
        qaStateText = "提取中"
        Task { await uploadExtractionPhoto(image) }
    }

    private func uploadExtractionPhoto(_ image: UIImage) async {
        guard await ensureExtractSession(), let sid = extractSessionId else {
            extractNotice = "未能创建题目提取会话，请重试"
            aimCapturing = false
            return
        }
        extractInflight += 1
        defer { extractInflight -= 1; aimCapturing = false }
        do {
            let files = [MultipartFile(field: "image", name: "extract.jpg", mime: "image/jpeg", data: try jpegData(image))]
            let data = try await postForm(
                path: "/api/sessions/\(sid)/extract-questions",
                fields: ["source": UIDevice.current.identifierForVendor?.uuidString ?? "ios"],
                files: files
            )
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let isMaterial = (json?["is_study_material"] as? Bool) ?? false
            let lowQuality = (json?["low_quality"] as? Bool) ?? false
            let thisCount = (json?["questions"] as? [[String: Any]])?.count ?? 0
            let setCount = (json?["question_set_count"] as? Int) ?? extractUniqueCount
            extractUniqueCount = setCount
            // 语音 + 文字提醒：拍不到题就引导重拍（用户要求“没有的话需要前端语音提醒”）。
            if !isMaterial {
                extractNotice = "这张没看到题目（像是桌面/无关画面），对准试卷再拍一张"
                speak("这张没看到题目，请把试卷对准取景框", completion: nil)
            } else if lowQuality || thisCount == 0 {
                extractNotice = "这张题目没看清，请靠近一点、放平、对焦后重拍"
                speak("这张没看清，请靠近一点重拍", completion: nil)
            } else {
                extractShotCount += 1
                extractNotice = "已拍 \(extractShotCount) 张 · 去重后 \(setCount) 道题"
            }
        } catch {
            extractNotice = "提取失败（网络或服务繁忙），可再拍一张"
            log("题目提取上传失败：\(error.localizedDescription)", level: "error")
        }
        if captureAimingVisible { qaStateText = "对准试卷" }
    }

    private func ensureExtractSession() async -> Bool {
        if extractSessionId != nil { return true }
        do {
            var fields = [
                "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                "mode": "extract",
                "title": "题目提取"
            ]
            if !AuthSession.shared.activeStudentId.isEmpty {
                fields["student_profile_id"] = AuthSession.shared.activeStudentId
            }
            let data = try await postForm(path: "/api/sessions", fields: fields, files: [])
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let id = json["session_id"] as? String {
                extractSessionId = id
                log("题目提取会话创建成功：\(id)")
                return true
            }
        } catch {
            log("题目提取会话创建失败：\(error.localizedDescription)", level: "error")
        }
        return false
    }

    /// 完成取景：关相机，拉还原页 HTML 弹出结果。
    func finishExtraction() {
        if torchOn { setTorch?(false); torchOn = false }
        captureAimingVisible = false
        aimCapturing = false
        aimReadyStreak = 0
        cameraTaskKind = .none
        cameraPreviewVisible = false
        if extractUniqueCount > 0 {
            Task {
                if let html = await fetchRestoreHTML(view: "restore") { extractRestoreHTML = html }
                extractResultVisible = true
            }
        } else {
            extractNotice = "还没提取到题目，把试卷拍清楚、铺满取景框再试"
            speak("还没提取到题目，请把试卷拍清楚再试", completion: nil)
        }
    }

    func cancelExtraction() {
        if torchOn { setTorch?(false); torchOn = false }
        captureAimingVisible = false
        aimCapturing = false
        aimReadyStreak = 0
        if cameraTaskKind == .extract { cameraTaskKind = .none }
        cameraPreviewVisible = false
    }

    /// 取还原页/空白卷 HTML（服务端渲染，供 WKWebView 显示与打印）。view = restore | blank。
    func fetchRestoreHTML(view: String) async -> String? {
        guard let sid = extractSessionId else { return nil }
        // 用 queryItems 拼 view，避免 URL.appending(path:) 把 "?" 百分号编码导致 query 失效。
        let url = serverBaseURL
            .appending(path: "/api/sessions/\(sid)/restore-page")
            .appending(queryItems: [URLQueryItem(name: "view", value: view)])
        do {
            let data = try await getData(url: url)
            return String(data: data, encoding: .utf8)
        } catch {
            log("还原页获取失败：\(error.localizedDescription)", level: "error")
            return nil
        }
    }

    // MARK: - 题目提取「自动扫描」（实验）：开相机时端上每秒扫一次→按题文指纹跟踪→发现新题才异步识别入库。
    // 思路：端上 Vision 切题+题文指纹（几乎零成本）做「这题见过没」的跟踪去重，只有新题才发后端大模型，
    // 所以被动扫描也不会卡（绝大多数帧不触发任何 VLM 调用）。文字相同仅图不同的题会被指纹合并（已知边界）。
    func toggleExtractAutoScan() {
        extractAutoScan.toggle()
        if extractAutoScan && cameraTaskKind == .extract {
            extractNotice = "自动扫描中：对准试卷慢慢移动，发现新题会自动识别"
            startExtractScan()
        } else {
            stopExtractScan()
            extractNotice = "已停自动扫描，可手动按快门"
        }
    }

    private func startExtractScan() {
        extractScanTimer?.invalidate()
        extractScanTimer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.extractScanTick() }
        }
    }

    private func stopExtractScan() {
        extractScanTimer?.invalidate()
        extractScanTimer = nil
    }

    private func extractScanTick() {
        guard extractAutoScan, cameraTaskKind == .extract, captureAimingVisible,
              !extractScanBusy, extractInflight == 0,
              let frame = latestPreviewFrame?() else { return }
        extractScanBusy = true
        Task.detached(priority: .userInitiated) {
            // 端上 fast 分题（轻量），取每题题文做指纹。
            let regions = QuestionSegmenter.segment(frame, fast: true)
            let keys = regions.compactMap { Self.extractScanKey($0.ocrText ?? "") }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.extractScanBusy = false
                guard self.extractAutoScan, self.cameraTaskKind == .extract, self.extractInflight == 0 else { return }
                let newKeys = keys.filter { k in !self.extractSeenKeys.contains { Self.extractKeySimilar($0, k) } }
                guard !newKeys.isEmpty else { return }
                self.extractSeenKeys.append(contentsOf: newKeys)
                self.extractNotice = "发现新题，正在识别…（已捕获 \(self.extractUniqueCount) 题）"
                // 整帧异步发后端精提取（服务端再切题+去重），更新题集计数。
                Task { await self.uploadExtractionPhoto(frame) }
            }
        }
    }

    /// 题文指纹：去空白标点+小写，截前 40 字。太短(<8)的不作数（噪声）。
    nonisolated private static func extractScanKey(_ text: String) -> String? {
        let norm = text.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
        return norm.count >= 8 ? String(norm.prefix(40)) : nil
    }

    /// 两个指纹是否同题（容忍 OCR 抖动）：相等/包含/前 12 字相同 即视为同题。
    nonisolated private static func extractKeySimilar(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        let shorter = a.count <= b.count ? a : b
        let longer = a.count <= b.count ? b : a
        if longer.contains(shorter) { return true }
        let p = min(shorter.count, 12)
        return p >= 8 && shorter.prefix(p) == longer.prefix(p)
    }

    /// 校正 + 分割（一次性，发生在显式点击「分题」之后，短暂主线程计算可接受）。
    private func processSegmentation(raw: UIImage) {
        let rect: (image: UIImage, didCorrect: Bool) = autoKeystoneCorrection
            ? QuestionSegmenter.rectify(raw)
            : (image: raw, didCorrect: false)
        segmentationRawImage = raw
        segmentationCorrectedImage = rect.image
        segmentationDidCorrect = rect.didCorrect
        segmentationUseCorrected = true
        let display = rect.didCorrect ? rect.image : raw
        segmentationDisplayImage = display
        let regions = questionSegmentationEnabled ? QuestionSegmenter.segment(display) : []
        questionRegions = regions
        selectedRegionID = regions.count == 1 ? regions.first?.id : nil
        segmentationBusy = false
        if regions.isEmpty {
            // P0：分不出题多半是拍糊/太暗——给出可执行的重拍引导，而不是笼统“没框出题目”。
            let signals = BurstFrameAnalyzer.analyze(display).signals
            if signals.textCount < 1 {
                segmentationNotice = "没识别到题目，请对准书本/试卷/屏幕上的题目重拍；也可直接按整张图问答"
            } else if signals.blurScore < 0.35 {
                segmentationNotice = "图像偏糊，建议把题目放平、点击对焦后重拍；也可直接按整张图问答"
            } else if signals.lightCoverage < 0.12 {
                segmentationNotice = "光线偏暗，建议开灯/补光后重拍；也可直接按整张图问答"
            } else {
                segmentationNotice = "没自动框出整题，可直接按整张图问答"
            }
        } else {
            segmentationNotice = segmentationIsGrading ? "正在逐题批改…" : "点一道题，我来讲"
        }
        log("题目分割完成：\(regions.count) 道题，校正=\(rect.didCorrect)")
        // P2 作业批改：分割出题后，对整页发起后端逐题判分，按 index 把 ✓/✗/? 叠到端上准 bbox。
        if segmentationIsGrading && !regions.isEmpty {
            Task { await gradePageForSegmentation() }
        }
    }

    /// 整页批改：把（校正后）展示图发后端 grade-page，按 region.index 对齐逐题判分。
    func gradePageForSegmentation() async {
        guard let image = segmentationDisplayImage else { return }
        gradingInFlight = true
        gradeQuestionsByIndex = [:]
        gradingNotice = ""
        defer { gradingInFlight = false }
        guard await ensureQASession(), let sessionId else {
            gradingNotice = "未能创建批改会话，可整图问答"
            return
        }
        do {
            let files = [MultipartFile(field: "image", name: "grade.jpg", mime: "image/jpeg", data: try jpegData(image))]
            let data = try await postForm(
                path: "/api/sessions/\(sessionId)/grade-page",
                fields: ["source": UIDevice.current.identifierForVendor?.uuidString ?? "ios"],
                files: files
            )
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let isMaterial = (json?["is_study_material"] as? Bool) ?? false
            let lowQuality = (json?["low_quality"] as? Bool) ?? false
            let rawQuestions = (json?["questions"] as? [[String: Any]]) ?? []
            var map: [Int: GradedQuestion] = [:]
            for q in rawQuestions {
                let g = GradedQuestion.from(q)
                if g.index > 0 { map[g.index] = g }
            }
            // 端上分出的题但后端没判到的，补一个「待讲解」占位，避免题上一直转圈。
            for region in questionRegions where map[region.index] == nil {
                map[region.index] = GradedQuestion(
                    index: region.index, questionText: region.ocrText ?? "", studentAnswer: "",
                    verdict: "未识别", gradable: false, correctAnswer: "", correction: "",
                    errorReason: "", knowledge: ""
                )
            }
            gradeQuestionsByIndex = map
            if !isMaterial || rawQuestions.isEmpty {
                gradingNotice = lowQuality ? "图像不够清晰，建议重拍；也可整图问答" : "未识别到可批改的题目，可整图问答"
            } else {
                let correct = map.values.filter { $0.mark == .correct }.count
                let wrong = map.values.filter { $0.mark == .wrong || $0.mark == .partial }.count
                let pending = map.values.filter { $0.mark == .uncertain }.count
                segmentationNotice = "批改完成：对 \(correct)・错 \(wrong)・待讲解 \(pending)（点题看详情）"
                log("整页批改完成：对\(correct) 错\(wrong) 待讲解\(pending)")
            }
        } catch {
            gradingNotice = "批改失败，可整图问答或重拍"
            log("整页批改失败：\(error.localizedDescription)", level: "error")
        }
    }

    /// B：取该题的「数字版印刷题干」（裁剪该题区域发后端 relayout，强制对照原图）。
    func fetchRelayout(for index: Int) async {
        guard let region = questionRegions.first(where: { $0.index == index }),
              let base = segmentationDisplayImage else { return }
        relayoutInFlight = true
        relayoutText = ""
        relayoutFigureNote = ""
        defer { relayoutInFlight = false }
        let padded = paddedSegmentationRect(region.normalizedRect)
        let cropped = QuestionSegmenter.crop(base, to: padded)
        relayoutCropImage = cropped
        guard await ensureQASession(), let sessionId else { relayoutText = "暂时取不到数字版，可看原图"; return }
        do {
            let files = [MultipartFile(field: "image", name: "relayout.jpg", mime: "image/jpeg", data: try jpegData(cropped))]
            let data = try await postForm(
                path: "/api/sessions/\(sessionId)/relayout-question",
                fields: ["source": UIDevice.current.identifierForVendor?.uuidString ?? "ios"],
                files: files
            )
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let isPrinted = (json?["is_printed_question"] as? Bool) ?? false
            let plain = (json?["plain_text"] as? String) ?? ""
            relayoutFigureNote = (json?["figure_note"] as? String) ?? ""
            relayoutText = (isPrinted && !plain.isEmpty) ? plain : "这道题没读到清晰的印刷题干，请看原图"
            relayoutForIndex = index
        } catch {
            relayoutText = "数字版获取失败，请看原图"
            log("relayout 失败：\(error.localizedDescription)", level: "error")
        }
    }

    func clearRelayout() {
        relayoutText = ""
        relayoutFigureNote = ""
        relayoutForIndex = nil
        relayoutCropImage = nil
    }

    /// 在批改详情里「讲解这道题」：裁剪该题 → 走拍照答题链路讲解（不再判分）。
    func explainGradedRegion(_ index: Int) {
        guard let region = questionRegions.first(where: { $0.index == index }),
              let base = segmentationDisplayImage else { return }
        selectedGradeIndex = nil
        let padded = paddedSegmentationRect(region.normalizedRect)
        let cropped = QuestionSegmenter.crop(base, to: padded)
        let focus = QAFocus(
            x: Double(region.normalizedRect.midX), y: Double(region.normalizedRect.midY),
            label: "explain_question_\(region.index)", trigger: "region_select",
            stableFrames: 0, region: region.normalizedRect, questionIndex: region.index,
            questionText: region.ocrText, crop: true
        )
        segmentationIsGrading = false   // 讲解走答题链路
        dispatchSegmentationQuestion(image: cropped, focus: focus, questionLabel: "第\(region.index)题")
    }

    /// 「用原图 ⇄ 用矫正图」切换：重画并按新图重算分割（保证浮层坐标与展示图一致）。
    func toggleSegmentationCorrection() {
        guard let raw = segmentationRawImage, let corrected = segmentationCorrectedImage else { return }
        segmentationUseCorrected.toggle()
        let display = segmentationUseCorrected ? corrected : raw
        segmentationDisplayImage = display
        let regions = questionSegmentationEnabled ? QuestionSegmenter.segment(display) : []
        questionRegions = regions
        selectedRegionID = regions.count == 1 ? regions.first?.id : nil
    }

    func selectQuestionRegion(_ region: QuestionRegion) {
        selectedRegionID = region.id
    }

    /// 确认选中的一题：裁剪子图 → 设为下一次提交的题图 → 拍照答题进语音 / 作业批改直接发。
    func confirmSelectedRegion() {
        guard let id = selectedRegionID,
              let region = questionRegions.first(where: { $0.id == id }),
              let base = segmentationDisplayImage else {
            submitWholeImageFromSegmentation()
            return
        }
        let padded = paddedSegmentationRect(region.normalizedRect)
        let cropped = QuestionSegmenter.crop(base, to: padded)
        let focus = QAFocus(
            x: Double(region.normalizedRect.midX),
            y: Double(region.normalizedRect.midY),
            label: "selected_question_\(region.index)",
            trigger: segmentationIsGrading ? "correction_check" : "region_select",
            stableFrames: 0,
            region: region.normalizedRect,
            questionIndex: region.index,
            questionText: region.ocrText,
            crop: true
        )
        dispatchSegmentationQuestion(image: cropped, focus: focus, questionLabel: "第\(region.index)题")
    }

    /// 回退：不选题，直接用（校正后的）整图走一次拍照问答 / 整页批改。
    func submitWholeImageFromSegmentation() {
        guard let base = segmentationDisplayImage else { dismissSegmentation(); return }
        let focus = QAFocus(
            x: 0.5, y: 0.5,
            label: "whole_image",
            trigger: segmentationIsGrading ? "correction_check" : "region_select",
            stableFrames: 0
        )
        dispatchSegmentationQuestion(image: base, focus: focus, questionLabel: nil)
    }

    private func dispatchSegmentationQuestion(image: UIImage, focus: QAFocus, questionLabel: String?) {
        let analysis = BurstFrameAnalyzer.analyze(image)
        let assessment = CaptureQualityAssessment.evaluate(
            signals: analysis.signals,
            visualDistance: nil,
            textDistance: nil,
            isFirstUsefulFrame: true
        )
        pendingRegionQAFrame = QAFrameCandidate(image: image, analysis: analysis, assessment: assessment)
        let grading = segmentationIsGrading
        dismissSegmentation(keepPendingFrame: true)
        if grading {
            stopSpeaking()
            stopListening(submit: false)
            pendingQATrigger = "correction_check"
            pendingQAFocus = focus
            let prompt = questionLabel.map { "请批改\($0)，指出对错并给订正建议。" }
                ?? "请批改这页作业，逐题指出对错与订正。"
            pendingQAQuestion = prompt
            recognizedText = prompt
            qaAnswer = ""
            qaOverlayVisible = true
            lastSubmittedContextItems = pendingContextItems(draft: prompt)
            appendUserChatMessage(prompt)
            isThinking = true
            qaStateText = "批改中"
            qaSystemImage = "checkmark.rectangle.stack"
            let generation = beginQuestionSubmissionGeneration()
            Task { await submitRecognizedQuestion(generation: generation) }
        } else {
            // 拍照答题：选中题→讲解那一道；整图问答(questionLabel==nil)→解答整页所有题，
            // 不能再用“这道题”单数话术(否则整页多题会上下文错配)。均不再等语音，直接出答案。
            stopSpeaking()
            stopListening(submit: false)
            pendingQATrigger = "region_select"
            pendingQAFocus = focus
            let prompt = questionLabel.map { "请讲解\($0)：给出解题思路和最终答案。" }
                ?? "请解答这张图里的题目：如果有多道题，请逐题（按题号）给出解题思路和最终答案；图中看不清的地方如实说明，不要编造。"
            pendingQAQuestion = prompt
            recognizedText = prompt
            qaAnswer = ""
            qaOverlayVisible = true
            lastSubmittedContextItems = pendingContextItems(draft: prompt)
            appendUserChatMessage(prompt)
            isThinking = true
            qaStateText = "讲解中"
            qaSystemImage = "text.bubble"
            let generation = beginQuestionSubmissionGeneration()
            Task { await submitRecognizedQuestion(generation: generation) }
        }
    }

    /// 选中题裁剪留白：上下各 +10% 题高、左右各 +2% 题宽，clamp 到图内。
    private func paddedSegmentationRect(_ r: CGRect) -> CGRect {
        let padX = r.width * 0.02
        let padY = r.height * 0.10
        let minX = max(0, r.minX - padX)
        let minY = max(0, r.minY - padY)
        let maxX = min(1, r.maxX + padX)
        let maxY = min(1, r.maxY + padY)
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    func dismissSegmentation(keepPendingFrame: Bool = false) {
        segmentationVisible = false
        segmentationBusy = false
        questionRegions = []
        selectedRegionID = nil
        segmentationDisplayImage = nil
        segmentationRawImage = nil
        segmentationCorrectedImage = nil
        segmentationDidCorrect = false
        segmentationNotice = ""
        gradeQuestionsByIndex = [:]
        gradingInFlight = false
        gradingNotice = ""
        selectedGradeIndex = nil
        if !keepPendingFrame {
            pendingRegionQAFrame = nil
            if cameraTaskKind == .qaFrame { cameraTaskKind = .none }
            cameraPreviewVisible = false
            cameraSheetVisible = false
        }
    }

    func cameraDidRecognizeGesture(_ gesture: CameraGesture, stableFrames: Int, point: CGPoint?) {
        suppressObservationDanmakuBriefly()
        switch gesture {
        case .point:
            guard !isListening, !isThinking else { return }
            if isBursting && !continuousVoiceActive {
                log("检测到指向手势，开启智能观察连续语音")
                startContinuousVoiceConversation()
                return
            }
            let normalized = point ?? CGPoint(x: 0.5, y: 0.5)
            let focus = QAFocus(
                x: max(0, min(1, normalized.x)),
                y: max(0, min(1, normalized.y)),
                label: "pointed_region",
                trigger: "point",
                stableFrames: stableFrames
            )
            pendingQAFocus = focus
            log("检测到稳定指向手势，准备听取问题")
            startVoiceQuestion(trigger: "point", focus: focus)
        case .ok:
            if continuousVoiceActive {
                stopSpeaking()
                qaStateText = "继续听"
                qaSystemImage = "waveform"
                scheduleContinuousListeningRestart(after: 0.2)
                log("检测到 OK 手势，打断并继续倾听")
            } else {
                interruptForFollowUp()
            }
        case .victory:
            if continuousVoiceActive {
                stopContinuousVoiceConversation(reason: "停止手势")
            } else {
                endQARound()
            }
        }
    }

    private func ensureSpeechAuthorization(_ completion: @escaping (Bool) -> Void) {
        let microphoneStatus = AVAudioSession.sharedInstance().recordPermission
        if !speechAuthorizationRequested {
            speechAuthorizationRequested = true
            SFSpeechRecognizer.requestAuthorization { speechStatus in
                AVAudioSession.sharedInstance().requestRecordPermission { micGranted in
                    DispatchQueue.main.async {
                        completion(speechStatus == .authorized && micGranted)
                    }
                }
            }
            return
        }
        completion(SFSpeechRecognizer.authorizationStatus() == .authorized && microphoneStatus == .granted)
    }

    private func beginListening() {
        guard !audioEngine.isRunning else { return }
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            continuousVoiceActive = false
            isPreparingVoiceInput = false
            qaStateText = "语音识别暂不可用"
            qaSystemImage = "mic.slash"
            uploadState = isBursting ? "智能观察中" : "待机"
            log("语音识别器暂不可用，无法开启录音", level: "error")
            return
        }
        stopSpeaking()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true
        recognizedText = ""
        hasSubmittedCurrentListen = false
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            holdToTalkActive = false
            voiceInputIsStarting = false
            isPreparingVoiceInput = false
            qaStateText = "麦克风启动失败"
            log("麦克风启动失败：\(networkErrorDescription(error))", level: "error")
            return
        }
        let inputNode = audioEngine.inputNode
        inputNode.removeTap(onBus: 0)
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak recognitionRequest] buffer, _ in
            recognitionRequest?.append(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            holdToTalkActive = false
            voiceInputIsStarting = false
            isPreparingVoiceInput = false
            qaStateText = "语音识别启动失败"
            log("语音识别启动失败：\(networkErrorDescription(error))", level: "error")
            return
        }
        isListening = true
        isPreparingVoiceInput = false
        qaStateText = continuousVoiceActive ? "持续倾听" : "正在听"
        qaSystemImage = "waveform"
        if continuousVoiceActive {
            scheduleContinuousRecognitionRefresh()
        } else if !holdToTalkActive {
            resetSilenceTimer()
        }
        if shouldSubmitWhenVoiceReady {
            shouldSubmitWhenVoiceReady = false
            scheduleVoiceSubmitAfterRecognitionGrace(maxWait: 3.4)
        }
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    self.recognizedText = result.bestTranscription.formattedString
                    if !self.holdToTalkActive,
                       !self.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.resetSilenceTimer()
                    }
                    if result.isFinal && !self.holdToTalkActive {
                        self.stopListening(submit: true)
                    }
                }
                if error != nil {
                    if self.isWaitingForVoiceSubmitGrace {
                        return
                    }
                    let hasText = !self.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    if self.holdToTalkActive {
                        self.qaStateText = hasText ? "继续说" : "正在听"
                        self.qaSystemImage = "waveform"
                        return
                    }
                    self.stopListening(submit: hasText)
                    if self.continuousVoiceActive && !hasText {
                        self.qaStateText = "继续听"
                        self.qaSystemImage = "waveform"
                        self.scheduleContinuousListeningRestart(after: 0.5)
                    }
                }
            }
        }
    }

    private func resetSilenceTimer() {
        silenceTimer?.invalidate()
        let interval: TimeInterval = continuousVoiceActive ? 2.0 : 3.0
        silenceTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.stopListening(submit: true)
            }
        }
    }

    private func scheduleVoiceSubmitAfterRecognitionGrace(maxWait: TimeInterval) {
        voiceSubmitGraceTask?.cancel()
        isWaitingForVoiceSubmitGrace = true
        silenceTimer?.invalidate()
        silenceTimer = nil
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest?.endAudio()
        qaStateText = "识别中"
        qaSystemImage = "waveform"
        voiceSubmitGraceTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let deadline = Date().addingTimeInterval(maxWait)
            var lastText = self.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
            var stableSince = Date()
            while !Task.isCancelled,
                  Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
                let currentText = self.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
                if currentText != lastText {
                    lastText = currentText
                    stableSince = Date()
                    continue
                }
                if !currentText.isEmpty && Date().timeIntervalSince(stableSince) >= 1.25 {
                    break
                }
            }
            guard !Task.isCancelled else { return }
            self.isWaitingForVoiceSubmitGrace = false
            self.stopListening(submit: true)
        }
    }

    private func scheduleContinuousRecognitionRefresh() {
        recognitionRefreshTimer?.invalidate()
        recognitionRefreshTimer = Timer.scheduledTimer(withTimeInterval: 45.0, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.continuousVoiceActive, self.isListening else { return }
                let hasText = !self.recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                self.stopListening(submit: hasText)
                if !hasText {
                    self.qaStateText = "继续听"
                    self.qaSystemImage = "waveform"
                    self.scheduleContinuousListeningRestart(after: 0.25)
                }
            }
        }
    }

    private func stopListening(submit: Bool) {
        voiceSubmitGraceTask?.cancel()
        voiceSubmitGraceTask = nil
        isWaitingForVoiceSubmitGrace = false
        silenceTimer?.invalidate()
        silenceTimer = nil
        recognitionRefreshTimer?.invalidate()
        recognitionRefreshTimer = nil
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isListening = false
        isPreparingVoiceInput = false
        shouldSubmitWhenVoiceReady = false
        guard submit else { return }
        guard !hasSubmittedCurrentListen else { return }
        hasSubmittedCurrentListen = true
        let rawText = recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawText.isEmpty else {
            if continuousVoiceActive {
                qaStateText = "继续听"
                qaSystemImage = "waveform"
                scheduleContinuousListeningRestart(after: 0.35)
            } else {
                qaStateText = "没有听清"
                qaSystemImage = "mic.slash"
            }
            return
        }
        let text = rawText
        guard !text.isEmpty else {
            qaStateText = "没有听清"
            qaSystemImage = "mic.slash"
            if continuousVoiceActive {
                scheduleContinuousListeningRestart(after: 0.35)
            }
            return
        }
        if handleVoiceCommandIfNeeded(text) {
            return
        }
        pendingQAQuestion = text
        lastSubmittedContextItems = pendingContextItems(draft: "")
        appendUserChatMessage(text)
        qaStateText = "思考中"
        qaSystemImage = "brain.head.profile"
        isThinking = true
        let generation = beginQuestionSubmissionGeneration()
        Task { await submitRecognizedQuestion(generation: generation) }
    }

    private func handleVoiceCommandIfNeeded(_ text: String) -> Bool {
        let compact = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        guard !compact.isEmpty else { return false }
        let stopCommands = ["停止录音", "关闭录音", "结束录音", "停止倾听", "关闭麦克风", "结束对话", "先停一下", "不用听了"]
        if stopCommands.contains(where: { compact.contains($0) }) {
            stopContinuousVoiceConversation(reason: "语音指令：\(text)")
            return true
        }
        let interruptCommands = ["打断", "停一下", "别说了", "先别说", "继续问", "我追问"]
        if continuousVoiceActive && interruptCommands.contains(where: { compact.contains($0) }) {
            stopSpeaking()
            qaStateText = "继续听"
            qaSystemImage = "waveform"
            scheduleContinuousListeningRestart(after: 0.2)
            log("收到语音打断指令：\(text)")
            return true
        }
        return false
    }

    private func scheduleContinuousListeningRestart(after seconds: TimeInterval) {
        guard continuousVoiceActive else { return }
        Task { [weak self] in
            let nanoseconds = UInt64(seconds * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)
            await MainActor.run {
                guard let self,
                      self.continuousVoiceActive,
                      !self.isListening,
                      !self.isPreparingVoiceInput,
                      !self.isThinking,
                      !self.ttsPlaybackPhase.isActive else { return }
                self.beginContinuousListeningTurn()
            }
        }
    }

    private func submitRecognizedQuestion(generation: Int) async {
        guard !isSubmittingQuestion else { return }
        isSubmittingQuestion = true
        defer {
            if submissionIsCurrent(generation) {
                isSubmittingQuestion = false
            }
        }

        guard await ensureQASession() else {
            guard submissionIsCurrent(generation) else { return }
            finishQuestionSubmissionFailure(
                qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "创建问答回合失败，问题还没有提交到 AI。请稍后重试。" : qaAnswer,
                stateText: "创建问答回合失败",
                systemImage: "exclamationmark.triangle",
                shouldSpeak: false
            )
            return
        }
        guard submissionIsCurrent(generation) else { return }
        await syncStudentGoalIfNeeded()
        guard submissionIsCurrent(generation) else { return }

        let currentTurn = qaTurnIndex + 1
        let intentHint = qaIntentHint(for: qaQuestionTextForSubmission())
        let shouldCaptureFrame = shouldCaptureVisualContext(intentHint: intentHint)
        let frame: QAFrameCandidate?
        // 题目分割·选中题：已有裁剪好的单题图，直接使用、跳过重新抓帧（强制采用）。
        let forceSubmitRegionFrame = pendingRegionQAFrame != nil
        if let regionFrame = pendingRegionQAFrame {
            frame = regionFrame
            pendingRegionQAFrame = nil
            qaStateText = "已锁定选中题，思考中"
            qaSystemImage = "viewfinder.rectangular"
            log("题目分割：使用选中题裁剪图作为上下文")
        } else if shouldCaptureFrame {
            qaStateText = "截取当前画面"
            qaSystemImage = "camera.viewfinder"
            log("AI 问答准备截取当前画面作为上下文")
            frame = await captureCurrentQAFrame()
            guard submissionIsCurrent(generation) else { return }
        } else {
            qaStateText = "思考中"
            qaSystemImage = "brain.head.profile"
            frame = nil
            if pendingQATrigger == "review_today" {
                log("今日复习将使用错题队列上下文，不额外抓取当前画面")
            } else {
                log("本轮使用文字和已有回合上下文，不额外抓取当前画面")
            }
        }
        let shouldSubmitCurrentFrame = forceSubmitRegionFrame || (frame.map { shouldSubmitQAFrame($0, intentHint: intentHint, turn: currentTurn) } ?? false)
        if currentTurn == 1, let frame, shouldSubmitCurrentFrame {
            anchoredQAFrame = frame
        } else if let frame, shouldSubmitCurrentFrame {
            anchoredQAFrame = frame
        }
        let shouldReuseAnchorFrame = !shouldSubmitCurrentFrame && currentTurn > 1 && anchoredQAFrame != nil && pendingQATrigger != "review_today"
        let submittedFrame = shouldSubmitCurrentFrame ? frame : (shouldReuseAnchorFrame ? anchoredQAFrame : nil)
        let image = submittedFrame?.image
        if let frame, shouldSubmitCurrentFrame {
            qaStateText = "已截取画面，思考中"
            qaSystemImage = "photo"
            log("已截取当前有效画面，将和语音问题一起发送给 AI（\(frame.analysis.signals.summary)）")
        } else if shouldReuseAnchorFrame, let anchoredQAFrame {
            qaStateText = "复用首轮画面，思考中"
            qaSystemImage = "photo.on.rectangle"
            log("本次追问复用新对话首轮题图作为上下文（\(anchoredQAFrame.analysis.signals.summary)）")
        } else if frame != nil {
            qaStateText = "思考中"
            qaSystemImage = "brain.head.profile"
            log("当前抓拍已忽略，将沿用已有上下文回答语音问题", level: "warning")
        } else {
            qaStateText = "思考中"
            qaSystemImage = "brain.head.profile"
            log("未采用当前抓拍，将沿用本回合已有上下文回答", level: "warning")
        }
        lastSubmittedContextItems = submittedContextItems(
            question: qaQuestionTextForSubmission(),
            intentHint: intentHint,
            qaFrame: submittedFrame ?? frame,
            submittedCurrentFrame: image != nil,
            reusedAnchorFrame: shouldReuseAnchorFrame
        )
        appendSubmittedContextMessage(
            items: lastSubmittedContextItems,
            capturedFrame: frame,
            submittedFrame: submittedFrame,
            reusedAnchorFrame: shouldReuseAnchorFrame,
            submittedCurrentFrame: image != nil
        )
        await submitQuestion(image: image, qaFrame: submittedFrame ?? frame, currentTurn: currentTurn, reusedAnchorFrame: shouldReuseAnchorFrame, generation: generation)
    }

    private func shouldCaptureVisualContext(intentHint: String) -> Bool {
        if !contextInclusionSettings.visual {
            return false
        }
        // 纯文字提问：打字/快捷追问不抓画面（语音、拍题、观察不受影响）
        if textOnlyQuestion && (pendingQATrigger == "typed_chat" || pendingQATrigger.hasPrefix("quick_followup")) {
            return false
        }
        // 长按语音上滑选了「只纯文字」：本次不抓画面
        if pendingVoiceTextOnly && pendingQATrigger.hasPrefix("hold_to_talk") {
            return false
        }
        if pendingQATrigger == "review_today" {
            return false
        }
        if pendingQATrigger == "quick_followup_text_only" {
            return false
        }
        return cameraCanBecomeAvailableForQA
    }

    private func shouldSubmitQAFrame(_ frame: QAFrameCandidate, intentHint: String, turn: Int) -> Bool {
        // #1 用户主动拍题/提问（新问题 / 批改检查 / 双击当前画面 / 指向某题）：人已在现场、
        // 画面就是要问的东西 → 只要画面质量过关就随问发出，不再要求「识别到学习材料/学生在场」
        // （手/脸/身体的在场检测只服务于智能观察，不卡实时问答）。纯追问仍走「复用首轮题图」。
        let qualityOK = frame.assessment.shouldUpload
        if pendingQATrigger.hasPrefix("double_tap_current_frame") {
            return qualityOK
        }
        if pendingQATrigger != "review_today" && qualityOK && frame.analysis.signals.hasStudyMaterial {
            return true
        }
        if ["correction_check", "answer_check", "visual_check"].contains(intentHint) {
            return qualityOK
        }
        if intentHint == "new_question" && turn <= 1 {
            return qualityOK
        }
        if intentHint == "new_question" && qaQuestionHasNewProblemReference(qaQuestionTextForSubmission()) {
            return qualityOK
        }
        if pendingQATrigger.lowercased().contains("point") && intentHint == "new_question" {
            return qualityOK
        }
        return false
    }

    private func captureCurrentQAFrame() async -> QAFrameCandidate? {
        completePendingQAFrame(nil)
        if captureQAFrame == nil {
            cameraTaskKind = .qaFrame
            cameraPreviewVisible = true
            cameraSheetVisible = false
            let ready = await waitForCameraReadyForQA()
            guard ready else {
                log("相机问答抓帧入口不可用", level: "warning")
                qaStateText = "思考中"
                qaSystemImage = "brain.head.profile"
                return nil
            }
        }
        guard let captureQAFrame else {
            log("相机问答抓帧入口不可用", level: "warning")
            return nil
        }
        return await withCheckedContinuation { continuation in
            pendingQAFrameContinuation = continuation
            pendingQAFrameTimeoutTask = Task { [weak self] in
                let seconds = self?.qaFrameTimeoutSeconds ?? 2.5
                let nanoseconds = UInt64(seconds * 1_000_000_000)
                try? await Task.sleep(nanoseconds: nanoseconds)
                await MainActor.run {
                    self?.log("问答抓拍等待超时，将沿用已有上下文", level: "warning")
                    self?.completePendingQAFrame(nil)
                }
            }
            if captureQAFrame() != true {
                completePendingQAFrame(nil)
            }
        }
    }

    private func waitForCameraReadyForQA() async -> Bool {
        let deadline = Date().addingTimeInterval(2.2)
        while Date() < deadline {
            if isCameraReady, captureQAFrame != nil {
                return true
            }
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
        return isCameraReady && captureQAFrame != nil
    }

    private func submittedContextItems(question: String, intentHint: String, qaFrame: QAFrameCandidate?, submittedCurrentFrame: Bool, reusedAnchorFrame: Bool = false) -> [ContextBadgeItem] {
        let isTextOnlyFollowUp = pendingQATrigger == "quick_followup_text_only"
        var items: [ContextBadgeItem] = [
            ContextBadgeItem(
                id: "submitted-question",
                title: isTextOnlyFollowUp ? "连续追问" : (pendingQATrigger == "typed_chat" ? "文字问题" : "语音转写"),
                detail: shortText(question, limit: 36),
                systemImage: isTextOnlyFollowUp || pendingQATrigger == "typed_chat" ? "text.bubble" : "waveform",
                tone: .neutral
            )
        ]

        if contextInclusionSettings.visual && submittedCurrentFrame {
            let detail = qaFrame.map {
                submittedVisualSummary(frame: $0, submittedCurrentFrame: submittedCurrentFrame, reusedAnchorFrame: reusedAnchorFrame)
            } ?? (reusedAnchorFrame ? "沿用首轮题图" : "已带上当前画面")
            items.append(ContextBadgeItem(
                id: "submitted-current-frame",
                title: reusedAnchorFrame ? "首轮题图" : "抓拍画面",
                detail: shortText(detail, limit: 36),
                systemImage: reusedAnchorFrame ? "photo.on.rectangle" : "photo",
                tone: .good,
                fullDetail: qaFrame.map {
                    submittedVisualDetail(frame: $0, submittedCurrentFrame: submittedCurrentFrame, reusedAnchorFrame: reusedAnchorFrame)
                }
            ))
        } else if contextInclusionSettings.visual && qaFrame != nil {
            items.append(ContextBadgeItem(
                id: "submitted-reused-frame",
                title: "抓拍未采用",
                detail: qaFrame.map { shortText(visualRejectionReason(for: $0), limit: 36) } ?? "画面质量不足，沿用文字/记忆",
                systemImage: "photo.on.rectangle",
                tone: .warning,
                fullDetail: qaFrame.map {
                    submittedVisualDetail(frame: $0, submittedCurrentFrame: false, reusedAnchorFrame: false)
                }
            ))
        } else if pendingQATrigger == "review_today" {
            items.append(ContextBadgeItem(
                id: "submitted-review",
                title: "复习错题",
                detail: activeReviewItem.map { shortText(reviewQueueSummary(for: $0), limit: 34) } ?? "复习计划",
                systemImage: "calendar.badge.clock",
                tone: .waiting
            ))
        } else if isTextOnlyFollowUp {
            items.append(ContextBadgeItem(
                id: "submitted-conversation-context",
                title: "对话上下文",
                detail: "不截屏，沿用本轮上下文",
                systemImage: "bubble.left.and.bubble.right",
                tone: .waiting
            ))
        } else if !contextInclusionSettings.visual {
            items.append(ContextBadgeItem(
                id: "submitted-visual-disabled",
                title: "画面已关闭",
                detail: "按配置不抓拍/不复用题图",
                systemImage: "camera.slash",
                tone: .neutral
            ))
        } else {
            items.append(ContextBadgeItem(
                id: "submitted-text-context",
                title: "未抓到画面",
                detail: "使用文字和历史上下文",
                systemImage: "tray.full",
                tone: .waiting
            ))
        }

        if sessionId != nil {
            items.append(ContextBadgeItem(
                id: "submitted-session",
                title: "本轮对话",
                detail: qaTurnIndex == 0 ? "新对话首问" : "追问上下文",
                systemImage: "bubble.left.and.bubble.right",
                tone: .neutral
            ))
        }

        if contextInclusionSettings.observation && (isBursting || !burstBuffer.isEmpty) {
            items.append(ContextBadgeItem(
                id: "submitted-observation",
                title: "智能观察",
                detail: observationContextSummary(),
                systemImage: "rectangle.stack",
                tone: isBursting ? .good : .neutral,
                fullDetail: observationContextDetail()
            ))
        }

        if contextInclusionSettings.history, let carriedHistoryContext {
            items.append(carriedHistoryContext.badge)
        }

        if contextInclusionSettings.mistakes && (activeReviewItem != nil || reviewDueCount != nil) {
            items.append(ContextBadgeItem(
                id: "submitted-memory",
                title: "错题复习",
                detail: activeReviewItem.map { shortText(reviewQueueSummary(for: $0), limit: 34) } ?? "历史错题/复习队列",
                systemImage: "book.closed",
                tone: .neutral
            ))
        }

        if !trimmedStudentGoal.isEmpty {
            items.append(ContextBadgeItem(
                id: "submitted-goal",
                title: "目标",
                detail: shortText(trimmedStudentGoal, limit: 34),
                systemImage: "scope",
                tone: .neutral
            ))
        }

        if contextInclusionSettings.memory && !lastTurnMemories.isEmpty {
            let detailLines = lastTurnMemories.map { memory -> String in
                let scoreText = memory.isPersona ? "常驻" : String(format: "%.2f", memory.score ?? 0)
                return "[\(memory.kindLabel) \(scoreText)] \(memory.text)"
            }
            items.append(ContextBadgeItem(
                id: "submitted-long-term-memory",
                title: "本轮记忆",
                detail: "\(lastTurnMemories.count) 条（语义检索）",
                systemImage: "brain.head.profile",
                tone: .neutral,
                fullDetail: detailLines.joined(separator: "\n")
            ))
        }

        let assets = contextAssets(for: question)
        if !assets.isEmpty {
            let mistakeCount = assets.filter { $0.kind == "mistake" }.count
            let knowledgeCount = assets.filter { $0.kind == "knowledge" }.count
            let memoryCount = assets.count - mistakeCount - knowledgeCount
            items.append(ContextBadgeItem(
                id: "submitted-structured-assets",
                title: "学习资产",
                detail: "错题 \(mistakeCount) · 知识点 \(knowledgeCount) · 记忆 \(memoryCount)",
                systemImage: "books.vertical",
                tone: .neutral,
                fullDetail: assets.map { asset in
                    "【\(asset.title)】\(asset.detail)\n使用：\(asset.useRule)"
                }.joined(separator: "\n\n")
            ))
        }

        if contextInclusionSettings.strategy {
            items.append(ContextBadgeItem(
                id: "submitted-coach",
                title: "动态策略",
                detail: dynamicStrategyPreview(),
                systemImage: "slider.horizontal.3",
                tone: .neutral,
                fullDetail: dynamicStrategyDetail()
            ))
        }

        return items
    }

    private func submittedVisualSummary(frame: QAFrameCandidate, submittedCurrentFrame: Bool, reusedAnchorFrame: Bool) -> String {
        let policy = reusedAnchorFrame ? "沿用首轮题图" : (submittedCurrentFrame ? "采用当前抓拍" : "未采用当前抓拍")
        let reason = submittedCurrentFrame || reusedAnchorFrame ? visualSubmissionReason(for: frame, reusedAnchorFrame: reusedAnchorFrame) : visualRejectionReason(for: frame)
        return "\(policy)：\(frame.analysis.signals.summary)；\(reason)"
    }

    private func submittedVisualDetail(frame: QAFrameCandidate, submittedCurrentFrame: Bool, reusedAnchorFrame: Bool) -> String {
        // #1 实时问答的明细不再展示「学生在场/手脸身体」这类在场识别（那是智能观察的概念，
        // 用户主动提问时人显然在场，写出来只会让人误以为在卡在场检测）。
        [
            submittedVisualSummary(frame: frame, submittedCurrentFrame: submittedCurrentFrame, reusedAnchorFrame: reusedAnchorFrame),
            "质量判断：\(frame.assessment.userMessage)",
            "是否建议上传：\(frame.assessment.shouldUpload ? "是" : "否")",
            "学习材料：\(frame.analysis.signals.hasStudyMaterial ? "已识别" : "未充分识别")"
        ].joined(separator: "\n")
    }

    private func visualSubmissionReason(for frame: QAFrameCandidate, reusedAnchorFrame: Bool) -> String {
        if reusedAnchorFrame {
            return "原因：追问沿用首轮题图，避免重复截屏。"
        }
        if frame.assessment.shouldUpload && frame.analysis.signals.hasStudyMaterial {
            return "原因：画面达到上传阈值，且识别到学习材料。"
        }
        if frame.analysis.signals.hasStudyMaterial {
            return "原因：识别到学习材料，作为本轮视觉上下文。"
        }
        return "原因：本轮需要视觉上下文，已随问题一起提交。"
    }

    private func visualRejectionReason(for frame: QAFrameCandidate) -> String {
        if !frame.analysis.signals.hasStudyMaterial {
            return "原因：未充分识别到学习材料，避免把低价值画面发给大模型。"
        }
        if !frame.assessment.shouldUpload {
            return "原因：\(frame.assessment.userMessage)"
        }
        return "原因：本轮策略选择沿用已有上下文。"
    }

    private func appendSubmittedContextMessage(
        items: [ContextBadgeItem],
        capturedFrame: QAFrameCandidate?,
        submittedFrame: QAFrameCandidate?,
        reusedAnchorFrame: Bool,
        submittedCurrentFrame: Bool
    ) {
        let text: String
        if reusedAnchorFrame {
            let detail = submittedFrame.map {
                submittedVisualSummary(frame: $0, submittedCurrentFrame: true, reusedAnchorFrame: true)
            } ?? "沿用首轮题图。"
            text = "本次追问复用首轮题图：\(shortText(detail, limit: 96)) 同时携带本轮对话和记忆整理。"
        } else if submittedCurrentFrame {
            let detail = (submittedFrame ?? capturedFrame).map {
                submittedVisualSummary(frame: $0, submittedCurrentFrame: true, reusedAnchorFrame: false)
            } ?? "采用当前抓拍。"
            text = "已随本次发送当前画面：\(shortText(detail, limit: 96)) 同时携带本轮对话和记忆整理。"
        } else if let capturedFrame {
            text = "当前抓拍未随本次发送：\(shortText(visualRejectionReason(for: capturedFrame), limit: 80)) 画面信号：\(shortText(capturedFrame.analysis.signals.summary, limit: 72))。本次改用文字、本轮对话和记忆上下文。"
        } else {
            text = "未抓到当前画面，本次使用文字、本轮对话和记忆上下文。"
        }
        let observationSuffix = isBursting ? " 智能观察正在后台运行，也会作为背景信息一起参考。" : ""
        appendStatusChatMessage(
            title: "随本次发送",
            text: text + observationSuffix,
            systemImage: submittedCurrentFrame ? (reusedAnchorFrame ? "photo.on.rectangle" : "photo.fill") : "tray.full",
            contextItems: items,
            attachments: submittedContextAttachments(submittedFrame: submittedFrame, reusedAnchorFrame: reusedAnchorFrame)
        )
    }

    private func submittedContextAttachments(submittedFrame: QAFrameCandidate?, reusedAnchorFrame: Bool) -> [ChatAttachment] {
        var attachments: [ChatAttachment] = []
        if let submittedFrame {
            attachments.append(
                ChatAttachment(
                    title: reusedAnchorFrame ? "首轮题图" : "抓拍画面",
                    detail: shortText(submittedVisualSummary(frame: submittedFrame, submittedCurrentFrame: true, reusedAnchorFrame: reusedAnchorFrame), limit: 72),
                    image: submittedFrame.image.thumbnail(maxSide: 520),
                    thumbnailURL: nil,
                    fullURL: nil
                )
            )
        }
        if let carriedHistoryContext, let filename = carriedHistoryContext.previewFilename {
            attachments.append(
                ChatAttachment(
                    title: "历史画面",
                    detail: carriedHistoryContext.title,
                    image: nil,
                    thumbnailURL: imageThumbnailURL(filename: filename),
                    fullURL: imageURL(filename: filename)
                )
            )
        }
        return attachments
    }

    // MARK: - 上下文真相源 finalize（I-3）

    /// 客户端为每个通道生成的「为何被调用」说明（CUT-6：不依赖后端 why）。
    private func contextChannelReason(_ key: String) -> String {
        switch key {
        case "visual": return "本轮带上了当前画面/题图，用于看清题目"
        case "history": return "携带了本轮对话历史，保持追问连贯"
        case "mistakes": return "结合了相关错题/复习内容"
        case "knowledge": return "命中了相关知识点"
        case "memory": return "按语义+新近度+重要性检索到的长期记忆"
        case "observation": return "结合了后台智能观察的画面"
        case "strategy": return "带上了你的辅导偏好与动态策略"
        default: return ""
        }
    }

    private func contextChannelMeta(_ key: String) -> (title: String, systemImage: String) {
        switch key {
        case "visual": return ("画面", "photo")
        case "history": return ("对话历史", "bubble.left.and.bubble.right")
        case "mistakes": return ("错题复习", "book.closed")
        case "knowledge": return ("知识点", "lightbulb")
        case "memory": return ("长期记忆", "brain.head.profile")
        case "observation": return ("智能观察", "rectangle.stack")
        case "strategy": return ("动态策略", "slider.horizontal.3")
        default: return (key, "tray.full")
        }
    }

    /// 把后端 `context_trace.channels` 重建为本轮真实徽章（仅 included 的通道）。
    private func badges(fromTrace trace: [String: Any]?) -> [ContextBadgeItem] {
        guard let channels = trace?["channels"] as? [[String: Any]] else { return [] }
        var items: [ContextBadgeItem] = []
        for channel in channels {
            guard let key = channel["key"] as? String,
                  (channel["included"] as? Bool) == true else { continue }
            let meta = contextChannelMeta(key)
            let detail = channel["detail"] as? [String: Any] ?? [:]
            var detailText = ""
            switch key {
            case "visual":
                let mode = (detail["mode"] as? String) ?? ""
                detailText = mode.isEmpty ? "已带上画面" : "模式：\(mode)"
            case "history":
                if let chars = detail["chars"] as? Int { detailText = "约 \(chars) 字" }
            case "mistakes":
                if let count = detail["count"] as? Int { detailText = "\(count) 条" }
            case "knowledge":
                if let hits = detail["semantic_hits"] as? [[String: Any]] { detailText = "命中 \(hits.count) 条" }
            case "memory":
                let mems = detail["memories"] as? [[String: Any]] ?? []
                detailText = "\(mems.count) 条（语义检索）"
            case "strategy":
                detailText = "偏好/策略已带上"
            default:
                detailText = ""
            }
            items.append(ContextBadgeItem(
                id: "trace-\(key)",
                title: meta.title,
                detail: detailText,
                systemImage: meta.systemImage,
                tone: .neutral,
                reason: contextChannelReason(key)
            ))
        }
        return items
    }

    /// 收到 QA 响应后用 `context_trace` finalize 本轮真相（修「滞后一轮」，MUST-5）。
    /// 重建本轮徽章与命中记忆，写 `lastTurnManifest`，并回挂到对应 status 消息的 contextItems。
    private func finalizeTurnContext(
        trace: [String: Any]?,
        fallbackMemories: [RetrievedMemory],
        turn: Int,
        usedImageContext: Bool,
        imageContextMode: String
    ) {
        // 命中记忆：优先用 trace 的 memory 通道（含 breakdown），否则回退 agent_memories。
        var memories = fallbackMemories
        if let channels = trace?["channels"] as? [[String: Any]],
           let memoryChannel = channels.first(where: { ($0["key"] as? String) == "memory" }),
           let detail = memoryChannel["detail"] as? [String: Any] {
            let parsed = RetrievedMemory.list(from: detail["memories"])
            if (memoryChannel["included"] as? Bool) == true || !parsed.isEmpty {
                memories = parsed
            } else {
                memories = []
            }
        }
        lastTurnMemories = memories

        let traceBadges = badges(fromTrace: trace)
        // trace 为空（旧后端/降级）时回退到提交前已记录的徽章，保证不空白。
        let sentItems = traceBadges.isEmpty ? lastSubmittedContextItems : traceBadges
        lastSubmittedContextItems = sentItems

        let manifest = TurnContextManifest(
            id: UUID(),
            turn: turn,
            sentItems: sentItems,
            retrievedMemories: memories,
            usedImageContext: usedImageContext,
            imageContextMode: imageContextMode,
            channels: ContextChannelTrace.list(from: trace)
        )
        lastTurnManifest = manifest

        // 回挂到最近一条「随本次发送」status 消息，让那条徽章显示真实通道（而非提交前猜测）。
        if !sentItems.isEmpty,
           let index = chatMessages.lastIndex(where: { $0.role == .status && !$0.contextItems.isEmpty && ($0.title ?? "") == "随本次发送" }) {
            chatMessages[index].contextItems = sentItems
        }
    }

    func longTermInstructionDidChange() {
        UserDefaults.standard.set(trimmedLongTermInstruction, forKey: longTermInstructionDefaultsKey)
    }

    func clearLongTermMemories() {
        longTermMemories.removeAll()
        userInputMemory.removeAll()
        UserDefaults.standard.set(longTermMemories, forKey: longTermMemoriesDefaultsKey)
        UserDefaults.standard.set(userInputMemory, forKey: userInputMemoryDefaultsKey)
        log("记忆整理已清空本机缓存")
    }

    private func updateLongTermMemories(question: String, answer: String) {
        let question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard question.count >= 4 else { return }
        let intent = qaIntentHint(for: question)
        let shouldRemember = ["new_question", "answer_check", "correction_check", "visual_check"].contains(intent) ||
            question.contains("我") ||
            question.contains("以后") ||
            question.contains("总是") ||
            question.contains("容易") ||
            question.contains("记住")
        guard shouldRemember else { return }
        let mode = learningMode.title
        let depth = coachDepth.title
        let summary = "偏好/学习记忆：\(mode) · \(depth)；最近问题：\(shortText(question, limit: 42))"
        guard !longTermMemories.contains(summary) else { return }
        longTermMemories.insert(summary, at: 0)
        if longTermMemories.count > 20 {
            longTermMemories = Array(longTermMemories.prefix(20))
        }
        UserDefaults.standard.set(longTermMemories, forKey: longTermMemoriesDefaultsKey)
        log("记忆整理已更新：\(summary)")
    }

    private func recordUserInputMemory(_ text: String) {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard cleanText.count >= 2 else { return }
        if userInputMemory.first == cleanText { return }
        userInputMemory.removeAll { $0 == cleanText }
        userInputMemory.insert(cleanText, at: 0)
        if userInputMemory.count > 80 {
            userInputMemory = Array(userInputMemory.prefix(80))
        }
        UserDefaults.standard.set(userInputMemory, forKey: userInputMemoryDefaultsKey)
    }

    func appendUserChatMessage(_ text: String) {   // internal：供 Intent/IntentRouting.swift P4 自动分流复用
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else { return }
        recordUserInputMemory(cleanText)
        if chatMessages.last?.role == .user && chatMessages.last?.text == cleanText {
            return
        }
        chatMessages.append(ChatMessage(role: .user, text: cleanText))
    }

    private func appendAssistantChatMessage(
        _ text: String,
        question: String = "",
        qaEventId: String = "",
        visualizationCandidate: Bool = false,
        visualizationReason: String = "",
        visualization: TeachingVisualization? = nil,
        actionable: Bool = true
    ) {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else { return }
        chatMessages.append(
            ChatMessage(
                role: .assistant,
                text: cleanText,
                question: question.trimmingCharacters(in: .whitespacesAndNewlines),
                qaEventId: qaEventId.trimmingCharacters(in: .whitespacesAndNewlines),
                visualizationCandidate: visualizationCandidate,
                visualizationReason: visualizationReason.trimmingCharacters(in: .whitespacesAndNewlines),
                visualization: visualization,
                actionable: actionable
            )
        )
    }

    private func appendStatusChatMessage(
        title: String,
        text: String,
        systemImage: String,
        showsProgress: Bool = false,
        contextItems: [ContextBadgeItem] = [],
        attachments: [ChatAttachment] = []
    ) {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty || !contextItems.isEmpty || !attachments.isEmpty else { return }
        chatMessages.append(
            ChatMessage(
                role: .status,
                text: cleanText,
                title: title,
                systemImage: systemImage,
                showsProgress: showsProgress,
                contextItems: contextItems,
                attachments: attachments
            )
        )
    }

    /// 进度类状态气泡的「就地更新」：相同 statusKey 已存在则替换其文案/图标/进度（spinner→结果），
    /// 否则新增一条。修复「正在生成/加入错题本」转圈不消失、还另起一条结果气泡的重复问题（#2）。
    @discardableResult
    private func upsertStatusChatMessage(
        key: String,
        title: String,
        text: String,
        systemImage: String,
        showsProgress: Bool = false
    ) -> Bool {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else { return false }
        if let index = chatMessages.lastIndex(where: { $0.statusKey == key }) {
            var msg = chatMessages[index]
            msg.text = cleanText
            msg.title = title
            msg.systemImage = systemImage
            msg.showsProgress = showsProgress
            chatMessages[index] = msg
            return true
        }
        chatMessages.append(
            ChatMessage(
                role: .status,
                text: cleanText,
                title: title,
                systemImage: systemImage,
                showsProgress: showsProgress,
                statusKey: key
            )
        )
        return true
    }

    func addLatestAnswerToMistakeBook() {
        guard let message = chatMessages.last(where: { $0.role == .assistant }) else {
            appendStatusChatMessage(title: "加入错题本", text: "还没有可加入错题本的 AI 回答。", systemImage: "book.closed")
            return
        }
        addMessageToMistakeBook(message)
    }

    func formMemoryFromLatestAnswer() {
        guard let message = chatMessages.last(where: { $0.role == .assistant }) else {
            appendStatusChatMessage(title: "形成记忆", text: "还没有可整理成记忆的 AI 回答。", systemImage: "brain.head.profile")
            return
        }
        formMemoryFromMessage(message)
    }

    func smartCaptureFromLatestAnswer() {
        guard let message = chatMessages.last(where: { $0.role == .assistant }) else {
            appendStatusChatMessage(title: "智能沉淀", text: "还没有可沉淀的 AI 回答。", systemImage: "wand.and.stars")
            return
        }
        smartCaptureFromMessage(message)
    }

    func smartCaptureFromMessage(_ message: ChatMessage) {
        guard message.role == .assistant else { return }
        guard sessionId != nil else {
            appendStatusChatMessage(title: "智能沉淀", text: "当前还没有学习回合，无法保存沉淀内容。", systemImage: "exclamationmark.triangle")
            return
        }
        // 不带 spinner：下面两个子步骤各自会追加「已沉淀/已形成」结果气泡作为完成信号（避免父气泡永久转圈）。
        appendStatusChatMessage(
            title: "智能沉淀",
            text: "我会把这轮问答同时整理为错题线索和个性化记忆：错题用于复习排队，记忆用于后续上下文优先参考。你不用判断该放哪一类。",
            systemImage: "wand.and.stars"
        )
        addMessageToMistakeBook(message, source: .smartCapture)
        formMemoryFromMessage(message, source: .smartCapture)
    }

    func generateVisualization(for message: ChatMessage) {
        guard message.role == .assistant else { return }
        guard let sessionId else {
            appendStatusChatMessage(title: "生成可视化", text: "当前还没有学习回合，无法生成可视化页面。", systemImage: "exclamationmark.triangle")
            return
        }
        let sourceId = message.qaEventId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sourceId.isEmpty else {
            appendStatusChatMessage(title: "生成可视化", text: "这条回答还没有可追踪的问答记录，暂时不能生成可视化。", systemImage: "exclamationmark.triangle")
            return
        }
        if let visualization = message.visualization, visualization.absoluteURL != nil, visualization.status != "running" {
            openVisualization(visualization)
            return
        }
        if let index = chatMessages.firstIndex(where: { $0.id == message.id }) {
            chatMessages[index].visualizationCandidate = true
            chatMessages[index].visualization = TeachingVisualization(status: "running", title: "可视化讲解", triggerReason: message.visualizationReason)
        }
        let vizStatusKey = "viz-\(message.id.uuidString)"
        upsertStatusChatMessage(
            key: vizStatusKey,
            title: "正在生成可视化",
            text: "我正在生成交互 HTML 教学页，通常需要几十秒。完成后会自动打开；以后也可以回到这条 AI 回复下点“打开可视化”。",
            systemImage: "cube.transparent",
            showsProgress: true
        )
        // 可视化是后端「空闲时异步生成」：POST 立即返回 status=running（无 url），
        // 真正生成完才有可打开页面。这里轮询直到就绪再自动打开；只有真异常才算失败。
        let vizSessionId = sessionId
        Task {
            let maxAttempts = 8
            for _ in 0..<maxAttempts {
                do {
                    let data = try await postJSON(
                        path: "/api/visualizations",
                        payload: [
                            "source_type": "qa_event",
                            "source_id": sourceId,
                            "session_id": vizSessionId
                        ],
                        timeout: 60
                    )
                    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    let visualizationJSON = json?["visualization"] as? [String: Any]
                    if let visualization = TeachingVisualization(json: visualizationJSON),
                       visualization.absoluteURL != nil,
                       visualization.status != "running" {
                        if let index = chatMessages.firstIndex(where: { $0.id == message.id }) {
                            chatMessages[index].visualization = visualization
                            chatMessages[index].visualizationCandidate = true
                            if chatMessages[index].visualizationReason.isEmpty {
                                chatMessages[index].visualizationReason = visualization.triggerReason
                            }
                        }
                        upsertStatusChatMessage(
                            key: vizStatusKey,
                            title: "可视化已生成",
                            text: "页面已打开。稍后也可以回到这条 AI 回复下点“打开可视化”。",
                            systemImage: "checkmark.circle"
                        )
                        openVisualization(visualization)
                        return
                    }
                    // 仍在排队/生成中：后台空闲时才生成，继续等待。
                } catch {
                    if let index = chatMessages.firstIndex(where: { $0.id == message.id }) {
                        chatMessages[index].visualizationCandidate = true
                    }
                    upsertStatusChatMessage(key: vizStatusKey, title: "可视化生成失败", text: networkErrorUserMessage(error), systemImage: "exclamationmark.triangle")
                    log("可视化生成失败：\(networkErrorDescription(error))", level: "error")
                    return
                }
                try? await Task.sleep(nanoseconds: 8_000_000_000)
            }
            // 轮询结束仍未就绪：把 spinner 收尾为「仍在生成」（就地更新，不再永久转圈）。
            upsertStatusChatMessage(
                key: vizStatusKey,
                title: "可视化仍在生成",
                text: "已加入空闲生成队列，可能需要稍等。稍后回到这条 AI 回复点“打开可视化”即可查看。",
                systemImage: "clock"
            )
        }
    }

    func openVisualization(_ visualization: TeachingVisualization) {
        guard let url = visualization.absoluteURL else {
            appendStatusChatMessage(title: "打开可视化失败", text: "这页可视化没有可打开的地址。", systemImage: "exclamationmark.triangle")
            return
        }
        UIApplication.shared.open(url)
    }

    func addMessageToMistakeBook(_ message: ChatMessage, source: LearningCaptureSource = .manualMistake) {
        guard message.role == .assistant else { return }
        guard let sessionId else {
            appendStatusChatMessage(title: "加入错题本", text: "当前还没有学习回合，无法保存错题。", systemImage: "exclamationmark.triangle")
            return
        }
        let answer = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let question = resolvedQuestion(for: message)
        guard !answer.isEmpty || !question.isEmpty else { return }
        let mistakeStatusKey = "mistake-\(message.id.uuidString)"
        if source == .manualMistake {
            upsertStatusChatMessage(key: mistakeStatusKey, title: "加入错题本", text: "正在把本轮问答沉淀为错题本条目。", systemImage: "book.closed", showsProgress: true)
        }
        Task {
            do {
                let payload: [String: Any] = [
                    "title": shortText(question.isEmpty ? answer : question, limit: 80),
                    "question_text": question,
                    "answer": answer,
                    "error_reason": source.mistakeReason,
                    "correction": shortText(answer, limit: 600),
                    "next_action": "复习时先说出错因，再做一道举一反三题。",
                    "status": "confirmed",
                    "source_summary": source.mistakeSourceSummary,
                    "qa_event_id": message.qaEventId,
                    "knowledge_points": [source.knowledgeTag, learningMode.title, coachDepth.title]
                ]
                _ = try await postJSON(path: "/api/sessions/\(sessionId)/mistakes", payload: payload)
                if source == .manualMistake {
                    upsertStatusChatMessage(
                        key: mistakeStatusKey,
                        title: "已加入错题本",
                        text: "这轮问答已保存，后续复习和上下文会优先参考。",
                        systemImage: "checkmark.circle"
                    )
                } else {
                    appendStatusChatMessage(
                        title: "错题线索已沉淀",
                        text: "已保存为复习线索，后续会进入错题队列和上下文参考。",
                        systemImage: "checkmark.circle"
                    )
                }
                await refreshReviewQueuePreview()
            } catch {
                if source == .manualMistake {
                    upsertStatusChatMessage(key: mistakeStatusKey, title: "加入错题本失败", text: networkErrorUserMessage(error), systemImage: "exclamationmark.triangle")
                } else {
                    appendStatusChatMessage(title: "加入错题本失败", text: networkErrorUserMessage(error), systemImage: "exclamationmark.triangle")
                }
                log("加入错题本失败：\(networkErrorDescription(error))", level: "error")
            }
        }
    }

    func formMemoryFromMessage(_ message: ChatMessage, source: LearningCaptureSource = .manualMemory) {
        guard message.role == .assistant else { return }
        guard let sessionId else {
            appendStatusChatMessage(title: "形成记忆", text: "当前还没有学习回合，无法保存记忆。", systemImage: "exclamationmark.triangle")
            return
        }
        let question = resolvedQuestion(for: message)
        let answer = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let memoryText = [
            question.isEmpty ? "" : "用户问题：\(shortText(question, limit: 220))",
            "应重点记住：\(shortText(answer, limit: 520))"
        ].filter { !$0.isEmpty }.joined(separator: "\n")
        guard !memoryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let memoryStatusKey = "memory-\(message.id.uuidString)"
        if source == .manualMemory {
            upsertStatusChatMessage(key: memoryStatusKey, title: "形成记忆", text: "正在把这轮问答整理成高优先级记忆。", systemImage: "brain.head.profile", showsProgress: true)
        }
        Task {
            do {
                let payload: [String: Any] = [
                    "text": memoryText,
                    "qa_event_id": message.qaEventId,
                    "source": source.memorySource,
                    "message_type": "formed_memory",
                    "payload": [
                        "learning_mode": learningMode.rawValue,
                        "coach_depth": coachDepth.rawValue,
                        "action": source.memoryAction
                    ]
                ]
                _ = try await postJSON(path: "/api/sessions/\(sessionId)/memory", payload: payload)
                updateLocalFormedMemory(memoryText)
                if source == .manualMemory {
                    upsertStatusChatMessage(
                        key: memoryStatusKey,
                        title: "记忆已形成",
                        text: "后续回答会把这条记忆放进重点上下文。",
                        systemImage: "checkmark.circle"
                    )
                } else {
                    appendStatusChatMessage(
                        title: "个性化记忆已形成",
                        text: "已提炼成后续上下文的重点记忆，回答会优先参考。",
                        systemImage: "checkmark.circle"
                    )
                }
                await refreshMemoryDigest()
            } catch {
                if source == .manualMemory {
                    upsertStatusChatMessage(key: memoryStatusKey, title: "形成记忆失败", text: networkErrorUserMessage(error), systemImage: "exclamationmark.triangle")
                } else {
                    appendStatusChatMessage(title: "形成记忆失败", text: networkErrorUserMessage(error), systemImage: "exclamationmark.triangle")
                }
                log("形成记忆失败：\(networkErrorDescription(error))", level: "error")
            }
        }
    }

    private func resolvedQuestion(for message: ChatMessage) -> String {
        let stored = message.question.trimmingCharacters(in: .whitespacesAndNewlines)
        if !stored.isEmpty {
            return stored
        }
        if let index = chatMessages.firstIndex(where: { $0.id == message.id }) {
            for candidate in chatMessages[..<index].reversed() where candidate.role == .user {
                return candidate.text
            }
        }
        return pendingQAQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func updateLocalFormedMemory(_ text: String) {
        let summary = "重点记忆：\(shortText(text, limit: 90))"
        longTermMemories.removeAll { $0 == summary }
        longTermMemories.insert(summary, at: 0)
        if longTermMemories.count > 20 {
            longTermMemories = Array(longTermMemories.prefix(20))
        }
        UserDefaults.standard.set(longTermMemories, forKey: longTermMemoriesDefaultsKey)
    }

    private func completePendingQAFrame(_ frame: QAFrameCandidate?) {
        guard let continuation = pendingQAFrameContinuation else { return }
        pendingQAFrameContinuation = nil
        pendingQAFrameTimeoutTask?.cancel()
        pendingQAFrameTimeoutTask = nil
        continuation.resume(returning: frame)
    }

    private func ensureQASession() async -> Bool {
        if sessionId != nil {
            return true
        }
        qaStateText = "创建问答回合"
        qaSystemImage = "bubble.left.and.bubble.right"
        uploadState = "创建问答回合"
        do {
            var fields = [
                "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                "mode": "qa",
                "title": "实时语音问答学习回合",
                "report_style": coachReportStyle,
                "assistant_focus": coachAssistantFocus
            ]
            let goal = trimmedStudentGoal
            if !goal.isEmpty {
                fields["student_goal"] = goal
            }
            if !AuthSession.shared.activeStudentId.isEmpty {
                fields["student_profile_id"] = AuthSession.shared.activeStudentId
            }
            let data = try await postForm(path: "/api/sessions", fields: fields, files: [])
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let id = json["session_id"] as? String {
                sessionId = id
                lastSyncedStrategySignature = strategySignature
                strategySyncState = "已同步"
                uploadState = "问答回合已创建"
                log("实时语音问答回合创建成功：\(id)")
                return true
            }
            qaStateText = "创建问答回合失败"
            qaSystemImage = "exclamationmark.triangle"
            uploadState = "创建失败"
            log("实时语音问答回合创建失败：后端未返回回合 ID", level: "error")
        } catch {
            let message = networkErrorUserMessage(error)
            qaStateText = "创建问答回合失败"
            qaSystemImage = "exclamationmark.triangle"
            uploadState = "创建失败"
            qaAnswer = message
            log("实时语音问答回合创建失败：\(networkErrorDescription(error))", level: "error")
        }
        return false
    }

    private func submitQuestion(image: UIImage?, qaFrame: QAFrameCandidate?, currentTurn: Int, reusedAnchorFrame: Bool = false, generation: Int) async {
        guard let sessionId else {
            guard submissionIsCurrent(generation) else { return }
            finishQuestionSubmissionFailure(
                "没有创建成功学习回合，问题还没有提交到 AI。请再问一次。",
                stateText: "请先开始学习回合",
                systemImage: "exclamationmark.triangle",
                shouldSpeak: false
            )
            log("AI 问答需要先创建学习回合", level: "warning")
            return
        }
        let question = qaQuestionTextForSubmission()
        pendingQAQuestion = question
        guard !question.isEmpty else {
            guard submissionIsCurrent(generation) else { return }
            finishQuestionSubmissionFailure(
                "没有识别到问题内容，所以没有提交给 AI。请再说一次或切到文字输入。",
                stateText: "问题为空",
                systemImage: "exclamationmark.triangle",
                shouldSpeak: false
            )
            log("AI 问答问题为空：语音文本未冻结，已取消提交", level: "error")
            return
        }
        do {
            var files: [MultipartFile] = []
            if let image {
                files.append(MultipartFile(field: "image", name: "qa-\(currentTurn).jpg", mime: "image/jpeg", data: try jpegData(image)))
            }
            let data = try await postForm(
                path: "/api/sessions/\(sessionId)/qa",
                fields: [
                    "question": question,
                    "trigger_type": pendingQATrigger,
                    "source": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                    "focus": focusJSON(pendingQAFocus),
                    "context": qaContextJSON(qaFrame: qaFrame, turn: currentTurn, submittedCurrentFrame: image != nil, reusedAnchorFrame: reusedAnchorFrame),
                    "gesture": gestureJSON(pendingQAFocus)
                ],
                files: files
            )
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard submissionIsCurrent(generation) else { return }
            let answer = (json?["answer"] as? String) ?? "我收到了，但没有解析到回答。"
            let event = json?["event"] as? [String: Any]
            let qaEventId = (event?["id"] as? String) ?? ""
            let visualizationCandidate = (event?["visualization_candidate"] as? Bool) ?? false
            let visualizationReason = (event?["visualization_reason"] as? String) ?? ""
            let actionable = (event?["actionable"] as? Bool) ?? true
            let visualization = TeachingVisualization(json: event?["visualization"] as? [String: Any])
            let usedImageContext = (json?["used_image_context"] as? Bool) ?? (image != nil)
            let imageContextMode = (json?["image_context_mode"] as? String) ?? (image != nil ? "current_frame" : "text_only")
            qaTurnIndex = currentTurn
            qaAnswer = answer
            finalizeTurnContext(
                trace: json?["context_trace"] as? [String: Any],
                fallbackMemories: RetrievedMemory.list(from: json?["agent_memories"]),
                turn: currentTurn,
                usedImageContext: usedImageContext,
                imageContextMode: imageContextMode
            )
            appendAssistantChatMessage(
                answer,
                question: question,
                qaEventId: qaEventId,
                visualizationCandidate: visualizationCandidate,
                visualizationReason: visualizationReason,
                visualization: visualization,
                actionable: actionable
            )
            updateLongTermMemories(question: question, answer: answer)
            // 三期：真实 QA 路径收尾后被动拉取本轮记忆增量（异步、不阻塞答案渲染/播报）。
            // 抽取是 fire-and-forget，本轮可能尚未写入 delta；拉不到则下一轮补显。
            Task { await pullMemoryDeltas() }
            isThinking = false
            closeTransientCameraPreviewAfterQuestion()
            qaStateText = "正在回答"
            qaSystemImage = "speaker.wave.2"
            speak(answer) { [weak self] in
                Task { @MainActor in
                    guard let self else { return }
                    self.qaStateText = self.continuousVoiceActive ? "继续听" : "可继续追问"
                    self.qaSystemImage = self.continuousVoiceActive ? "waveform" : "bubble.left.and.bubble.right"
                    if self.continuousVoiceActive {
                        self.scheduleContinuousListeningRestart(after: 0.45)
                    } else {
                        self.scheduleQAOverlayHide(after: 15.0)
                    }
                }
            }
            log("AI 问答完成：\(question)（视觉上下文=\(usedImageContext ? imageContextMode : "none")）")
        } catch {
            guard submissionIsCurrent(generation) else { return }
            closeTransientCameraPreviewAfterQuestion()
            let message = networkErrorUserMessage(error)
            finishQuestionSubmissionFailure(
                message,
                stateText: "问答失败",
                systemImage: "exclamationmark.triangle",
                shouldSpeak: true
            )
            log("AI 问答失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    private func closeTransientCameraPreviewAfterQuestion() {
        cameraSheetVisible = false
        guard cameraTaskKind == .qaFrame, !isBursting, !continuousVoiceActive else {
            return
        }
        cameraTaskKind = .none
        cameraPreviewVisible = false
    }

    private func finishQuestionSubmissionFailure(
        _ message: String,
        stateText: String,
        systemImage: String,
        shouldSpeak: Bool
    ) {
        let cleanMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)
        isThinking = false
        qaStateText = stateText
        qaSystemImage = systemImage
        qaAnswer = cleanMessage
        if !cleanMessage.isEmpty {
            appendAssistantChatMessage(cleanMessage)
        }
        if shouldSpeak {
            speak("问答失败，请稍后再试。", completion: nil)
        }
        if continuousVoiceActive {
            scheduleContinuousListeningRestart(after: 1.2)
        }
    }

    private func qaQuestionTextForSubmission() -> String {
        let pending = pendingQAQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        if !pending.isEmpty {
            return pending
        }
        return recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func focusJSON(_ focus: QAFocus?) -> String {
        guard let focus else { return "{}" }
        var payload: [String: Any] = [
            "x": focus.x,
            "y": focus.y,
            "label": focus.label,
            "trigger": focus.trigger,
            "stable_frames": focus.stableFrames
        ]
        if let region = focus.region {
            payload["region"] = [
                "x": Double(region.minX),
                "y": Double(region.minY),
                "w": Double(region.width),
                "h": Double(region.height)
            ]
        }
        if let questionIndex = focus.questionIndex { payload["question_index"] = questionIndex }
        if let questionText = focus.questionText, !questionText.isEmpty {
            payload["question_text"] = String(questionText.prefix(200))
        }
        if focus.crop { payload["crop"] = true }
        return jsonString(payload)
    }

    private func gestureJSON(_ focus: QAFocus?) -> String {
        guard let focus else { return "{}" }
        return jsonString(["name": focus.trigger, "stable_frames": focus.stableFrames])
    }

    private func qaIntentHint(for question: String) -> String {
        let compact = question
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        if compact.isEmpty { return "" }
        let correctionTerms = ["改完", "改好了", "改好", "修改", "改了", "订正", "修正", "重写", "重新写", "写完", "做完", "算完", "填完", "改正"]
        let checkTerms = ["看看", "看下", "看一下", "再看", "检查", "核对", "批改", "对不对", "对了吗", "对了没", "是否正确", "是不是对", "有没有错", "还错", "还对", "正确吗", "可以吗", "行不行"]
        let answerTerms = ["答案", "结果", "步骤", "过程", "这道", "这题", "这个", "这里", "这步"]
        let visualTerms = ["图片", "照片", "拍照", "画面", "镜头", "图上", "这张", "这道", "这题", "这个", "这里", "这步"]
        if qaQuestionHasNewProblemReference(compact) {
            return "new_question"
        }
        if correctionTerms.contains(where: { compact.contains($0) }) && checkTerms.contains(where: { compact.contains($0) }) {
            return "correction_check"
        }
        if checkTerms.contains(where: { compact.contains($0) }) && answerTerms.contains(where: { compact.contains($0) }) {
            return "answer_check"
        }
        if checkTerms.contains(where: { compact.contains($0) }) && (qaTurnIndex >= 1 || visualTerms.contains(where: { compact.contains($0) })) {
            return "visual_check"
        }
        if pendingQATrigger.lowercased().contains("follow") || pendingQATrigger.lowercased().contains("ok") || qaTurnIndex >= 1 {
            return "followup_explain"
        }
        return "new_question"
    }

    private func qaQuestionHasNewProblemReference(_ question: String) -> Bool {
        let compact = question
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        if compact.isEmpty { return false }
        if ["下一题", "上一题", "换题", "新题", "另一题", "另外一题"].contains(where: { compact.contains($0) }) {
            return true
        }
        return compact.range(
            of: #"(第)?[0-9一二三四五六七八九十]+[题页]"#,
            options: .regularExpression
        ) != nil
    }

    private func qaContextJSON(qaFrame: QAFrameCandidate?, turn: Int, submittedCurrentFrame: Bool, reusedAnchorFrame: Bool = false) -> String {
        jsonString(qaContextPayload(qaFrame: qaFrame, turn: turn, submittedCurrentFrame: submittedCurrentFrame, reusedAnchorFrame: reusedAnchorFrame))
    }

    private func qaContextPayload(
        qaFrame: QAFrameCandidate?,
        turn: Int,
        submittedCurrentFrame: Bool,
        reusedAnchorFrame: Bool = false,
        transcriptOverride: String? = nil
    ) -> [String: Any] {
        let overrideText = transcriptOverride?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let transcript = overrideText.isEmpty ? (pendingQAQuestion.isEmpty ? recognizedText : pendingQAQuestion) : overrideText
        let intentHint = qaIntentHint(for: transcript)
        let structuredAssets = contextAssets(for: transcript)
        var payload: [String: Any] = [
            "turn": turn,
            "transcript": transcript,
            "intent_hint": intentHint,
            "context_inclusion": [
                "visual": contextInclusionSettings.visual,
                "observation": contextInclusionSettings.observation,
                "history": contextInclusionSettings.history,
                "mistakes": contextInclusionSettings.mistakes,
                "knowledge": contextInclusionSettings.knowledge,
                "memory": contextInclusionSettings.memory,
                "strategy": contextInclusionSettings.strategy
            ],
            "student_goal": trimmedStudentGoal,
            "learning_mode": learningMode.rawValue,
            "learning_mode_title": learningMode.title,
            "coach_depth": coachDepth.rawValue,
            "coach_depth_title": coachDepth.title,
            "coach_preference": coachAssistantFocus,
            "strategy_context": [
                "student_goal": trimmedStudentGoal,
                "learning_mode": learningMode.rawValue,
                "learning_mode_title": learningMode.title,
                "coach_depth": coachDepth.rawValue,
                "coach_depth_title": coachDepth.title,
                "report_style": coachReportStyle,
                "assistant_focus": coachAssistantFocus
            ],
            "context_use_policy": [
                "priority": [
                    "current_or_anchor_frame",
                    "current_question_and_recent_qa",
                    "active_review_mistake",
                    "related_mistakes",
                    "knowledge_points",
                    "formed_memories_and_user_profile",
                    "background_observation"
                ],
                "rules": [
                    "Use mistake assets for review, error diagnosis, and similar-problem warnings only when relevant.",
                    "Use knowledge assets to explain concepts, summarize key points, and generate transfer questions.",
                    "Use memory assets for personalization, preferences, recurring mistakes, and recent goals; never use them as factual proof for the current answer.",
                    "If assets are weakly related, answer from the current question/frame and ignore weak assets.",
                    "When an asset materially affects the answer, mention the reason naturally in Chinese."
                ]
            ],
            "is_bursting": isBursting,
            "is_observing": isBursting,
            "continuous_voice_active": continuousVoiceActive,
            "conversation_mode": continuousVoiceActive ? "continuous_observation_voice" : "turn_based_chat",
            "device_interaction_presence": userOperationPresencePayload(),
            "qa_thread_policy": "In follow-up turns, keep answering about the first-turn captured problem unless the user clearly starts a new problem or taps New Conversation."
        ]
        // Memory selection now happens server-side: the backend semantically retrieves
        // the relevant durable memories for this question (see memory_store.retrieve_for_turn)
        // and returns them as `agent_memories`. The old client-side keyword candidates were
        // redundant with that, so they are no longer sent.
        if !structuredAssets.isEmpty {
            payload["structured_context_assets"] = structuredAssets.map { $0.compactPayload }
        }
        if contextInclusionSettings.strategy {
            payload["dynamic_strategy"] = dynamicStrategyPayload(transcript: transcript, turn: turn, qaFrame: qaFrame)
        }
        if contextInclusionSettings.observation {
            payload["observation_context"] = observationContextPayload()
        }
        if contextInclusionSettings.visual {
            payload["current_frame_submitted"] = submittedCurrentFrame
            payload["current_frame_policy"] = reusedAnchorFrame ? "reused_first_turn_anchor_frame" : (submittedCurrentFrame ? "submitted_current_frame" : (qaFrame == nil ? "capture_unavailable" : "reuse_existing_context"))
            payload["anchor_frame_reused"] = reusedAnchorFrame
            payload["anchor_frame_available"] = anchoredQAFrame != nil
        } else {
            payload["current_frame_policy"] = "disabled_by_context_settings"
        }
        if contextInclusionSettings.history, let carriedHistoryContext {
            payload["carried_history_context"] = carriedHistoryContext.payload
        }
        if contextInclusionSettings.mistakes, let reviewContext = activeReviewContextPayload() {
            payload["review_context"] = reviewContext
        }
        if contextInclusionSettings.visual, let qaFrame {
            let signals = qaFrame.analysis.signals
            var qualityReasons = qaFrame.assessment.reasons
            if !qaFrame.shouldUseAsContext && !qualityReasons.contains("no_explicit_study_evidence") {
                qualityReasons.append("no_explicit_study_evidence")
            }
            payload["qa_frame_quality"] = [
                "qa_context_eligible": qaFrame.shouldUseAsContext,
                "status": qaFrame.assessment.status,
                "reasons": qualityReasons,
                "message": qaFrame.userMessage,
                "has_study_material": signals.hasStudyMaterial,
                "has_explicit_study_evidence": signals.hasExplicitStudyEvidence,
                "qa_reliable_context": signals.isReliableQAContext,
                "text_tokens": Array(qaFrame.analysis.textTokens).sorted(),
                "text_count": signals.textCount,
                "rectangle_count": signals.rectangleCount,
                "light_coverage": signals.lightCoverage,
                "edge_density": signals.edgeDensity,
                "contrast": signals.contrast,
                "blur_score": qaFrame.assessment.blurScore,
                "material_confidence": qaFrame.assessment.materialConfidence,
                "occlusion_score": qaFrame.assessment.occlusionScore,
                "should_upload_for_qa": submittedCurrentFrame,
                "signal_summary": signals.summary,
                "presence_summary": signals.presenceSummary,
                "activity_summary": signals.activitySummary,
                "hand_count": signals.handCount,
                "face_count": signals.faceCount,
                "body_count": signals.bodyCount,
                "device_interaction_presence": userOperationPresencePayload()
            ]
        }
        payload["device_interaction_presence"] = userOperationPresencePayload()
        // Per-memory opt-outs for this turn (I-4): backend filters these before mark_used (B-2).
        if !memoryOverrides.isEmpty {
            payload["memory_excludes"] = Array(memoryOverrides)
        }
        return payload
    }

    private func activeReviewContextPayload() -> [String: Any]? {
        guard let item = activeReviewItem else { return nil }
        return [
            "mode": "today_review",
            "selected_at": CaptureTimeFormatter.string(from: Date()),
            "instruction": "围绕这道错题进行复习。先让学生回忆思路，再给一个小提示，最后用一个小检查判断是否掌握；不要一开始直接给完整答案。",
            "item": item.compactContext
        ]
    }

    private func jsonString(_ value: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private func imageURL(filename: String) -> URL? {
        guard let encoded = filename.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return nil }
        return URL(string: "/images/\(encoded)", relativeTo: serverBaseURL)?.absoluteURL
    }

    private func imageThumbnailURL(filename: String) -> URL? {
        guard let encoded = filename.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return nil }
        return URL(string: "/api/images/\(encoded)/thumbnail", relativeTo: serverBaseURL)?.absoluteURL
    }

    private func speak(_ text: String, completion: (() -> Void)?) {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else {
            completion?()
            return
        }
        guard voicePlaybackEnabled else {
            completion?()
            return
        }
        stopSpeaking()
        isSpeaking = true
        ttsPlaybackPhase = .generating
        let segments = ttsSegments(from: cleanText)
        ttsCurrentSegmentIndex = segments.isEmpty ? 0 : 1
        ttsTotalSegmentCount = segments.count
        qaStateText = "正在生成语音"
        qaSystemImage = "speaker.wave.2"
        let generation = ttsPlaybackGeneration
        ttsPlaybackTask = Task { @MainActor [weak self] in
            guard let self else { return }
            guard !segments.isEmpty else {
                if generation == self.ttsPlaybackGeneration {
                    self.finishSpeechPlayback(generation: generation, completion: completion)
                }
                return
            }
            do {
                var nextAudio: (data: Data, service: TTSService)?
                for index in segments.indices {
                    try Task.checkCancellation()
                    guard generation == self.ttsPlaybackGeneration else { return }
                    self.ttsCurrentSegmentIndex = index + 1
                    self.ttsTotalSegmentCount = segments.count
                    do {
                        self.ttsPrefetchTask?.cancel()
                        self.ttsPrefetchTask = nil
                        self.ttsServiceText = TTSService.local.displayName
                        self.ttsPlaybackPhase = .generating
                        self.qaStateText = segments.count > 1 ? "本机语音准备中 \(index + 1)/\(segments.count)" : "本机语音准备中"
                        self.qaSystemImage = "speaker.wave.2"
                        try await self.speakLocalSegment(segments[index], generation: generation, segmentIndex: index + 1, totalSegments: segments.count)
                        continue
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        guard generation == self.ttsPlaybackGeneration else { return }
                        self.log("TTS 本机语音失败，切换网络语音：\(networkErrorDescription(error))", level: "warning")
                    }

                    do {
                        let audioData: Data
                        let service: TTSService
                        if let prefetched = nextAudio {
                            audioData = prefetched.data
                            service = prefetched.service
                            nextAudio = nil
                        } else {
                            let startedAt = Date()
                            guard generation == self.ttsPlaybackGeneration else { return }
                            self.ttsCurrentSegmentIndex = index + 1
                            self.ttsPlaybackPhase = .generating
                            self.ttsServiceText = TTSService.primary.displayName
                            self.qaStateText = self.ttsTotalSegmentCount > 1 ? "正在合成语音 \(index + 1)/\(segments.count)" : "正在合成语音"
                            self.qaSystemImage = "speaker.wave.2"
                            let result = try await self.fetchSpeechAudio(text: segments[index], generation: generation, segmentIndex: index + 1, totalSegments: segments.count)
                            audioData = result.data
                            service = result.service
                            let elapsed = Date().timeIntervalSince(startedAt)
                            guard generation == self.ttsPlaybackGeneration else { return }
                            self.ttsServiceText = service.displayName
                            self.log("TTS \(service.displayName)第 \(index + 1)/\(segments.count) 段合成完成：\(String(format: "%.2f", elapsed))s，\(audioData.count) bytes")
                        }

                        if index + 1 < segments.count {
                            self.ttsPrefetchTask?.cancel()
                            let nextText = segments[index + 1]
                            self.ttsPrefetchTask = Task {
                                try await self.fetchSpeechAudio(text: nextText, generation: generation, segmentIndex: index + 2, totalSegments: segments.count)
                            }
                        } else {
                            self.ttsPrefetchTask = nil
                        }

                        try Task.checkCancellation()
                        self.ttsServiceText = service.displayName
                        try await self.playSpeechAudioAsync(audioData, generation: generation, segmentIndex: index + 1, totalSegments: segments.count)
                        try Task.checkCancellation()

                        if let prefetchTask = self.ttsPrefetchTask {
                            do {
                                nextAudio = try await prefetchTask.value
                            } catch is CancellationError {
                                throw CancellationError()
                            } catch {
                                guard generation == self.ttsPlaybackGeneration else { return }
                                self.log("TTS 下一段预生成失败，将改用本机语音：\(networkErrorDescription(error))", level: "warning")
                                nextAudio = nil
                            }
                            self.ttsPrefetchTask = nil
                        }
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        guard generation == self.ttsPlaybackGeneration else { return }
                        self.ttsPrefetchTask?.cancel()
                        self.ttsPrefetchTask = nil
                        nextAudio = nil
                        self.log("TTS 网络语音失败：\(networkErrorDescription(error))", level: "warning")
                        throw error
                    }
                }
                guard generation == self.ttsPlaybackGeneration else { return }
                self.finishSpeechPlayback(generation: generation, completion: completion)
            } catch is CancellationError {
                if generation == self.ttsPlaybackGeneration {
                    self.ttsPrefetchTask?.cancel()
                    self.ttsPrefetchTask = nil
                    self.isSpeaking = false
                    self.ttsPlaybackPhase = .idle
                    self.ttsCurrentSegmentIndex = 0
                    self.ttsTotalSegmentCount = 0
                }
            } catch {
                guard generation == self.ttsPlaybackGeneration else { return }
                self.ttsPrefetchTask?.cancel()
                self.ttsPrefetchTask = nil
                self.isSpeaking = false
                self.ttsPlaybackPhase = .idle
                self.ttsCurrentSegmentIndex = 0
                self.ttsTotalSegmentCount = 0
                self.ttsServiceText = TTSService.local.displayName
                self.qaStateText = "语音合成失败，文字可读"
                self.qaSystemImage = "speaker.slash"
                self.log("TTS 语音合成/播放失败，已保留文字答案：\(networkErrorDescription(error))", level: "error")
                completion?()
            }
        }
    }

    private func finishSpeechPlayback(generation: Int, completion: (() -> Void)?) {
        guard generation == ttsPlaybackGeneration else { return }
        ttsPrefetchTask?.cancel()
        ttsPrefetchTask = nil
        ttsPlayer = nil
        isSpeaking = false
        ttsPlaybackPhase = .idle
        ttsCurrentSegmentIndex = 0
        ttsTotalSegmentCount = 0
        ttsServiceText = TTSService.local.displayName
        completion?()
    }

    private func stopSpeaking() {
        ttsPlaybackGeneration += 1
        ttsPlaybackTask?.cancel()
        ttsPlaybackTask = nil
        ttsPrefetchTask?.cancel()
        ttsPrefetchTask = nil
        ttsPlaybackDelegate.setCompletion(nil)
        if localSpeechSynthesizer.isSpeaking {
            localSpeechSynthesizer.stopSpeaking(at: .immediate)
        }
        ttsPlayer?.stop()
        ttsPlayer = nil
        isSpeaking = false
        ttsPlaybackPhase = .idle
        ttsCurrentSegmentIndex = 0
        ttsTotalSegmentCount = 0
        ttsServiceText = TTSService.local.displayName
    }

    private nonisolated func ttsSegments(from text: String) -> [String] {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\n\n+", with: "\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return [] }

        var segments: [String] = []
        var buffer = ""
        let delimiters = CharacterSet(charactersIn: "。！？!?；;\n")
        for scalar in normalized.unicodeScalars {
            buffer.unicodeScalars.append(scalar)
            if delimiters.contains(scalar) || buffer.count >= ttsSegmentMaxCharacters {
                let part = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !part.isEmpty {
                    segments.append(part)
                }
                buffer = ""
            }
        }
        let tail = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty {
            segments.append(tail)
        }
        if segments.isEmpty {
            return [normalized]
        }
        return segments
    }

    private func speakLocalSegment(_ text: String, generation: Int, segmentIndex: Int, totalSegments: Int) async throws {
        guard generation == ttsPlaybackGeneration else { throw CancellationError() }
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else { return }

        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try audioSession.setActive(true, options: [])

        let utterance = AVSpeechUtterance(string: cleanText)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN") ?? AVSpeechSynthesisVoice(language: "zh-Hans")
        utterance.rate = localSpeechRate(for: voicePlaybackRate)
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0

        ttsServiceText = TTSService.local.displayName
        ttsCurrentSegmentIndex = segmentIndex
        ttsTotalSegmentCount = totalSegments
        qaStateText = totalSegments > 1 ? "正在本机语音播放 \(segmentIndex)/\(totalSegments) \(voicePlaybackRateText)" : "正在本机语音播放 \(voicePlaybackRateText)"
        qaSystemImage = "speaker.wave.2"
        ttsPlaybackPhase = .playing
        isSpeaking = true
        localSpeechSynthesizer.speak(utterance)

        while generation == ttsPlaybackGeneration && (localSpeechSynthesizer.isSpeaking || localSpeechSynthesizer.isPaused) {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 80_000_000)
        }
        guard generation == ttsPlaybackGeneration else { throw CancellationError() }
    }

    private nonisolated func localSpeechRate(for playbackRate: Double) -> Float {
        let clamped = normalizedVoicePlaybackRate(playbackRate)
        let base = Double(AVSpeechUtteranceDefaultSpeechRate)
        let value = min(max(base * clamped, Double(AVSpeechUtteranceMinimumSpeechRate)), Double(AVSpeechUtteranceMaximumSpeechRate))
        return Float(value)
    }

    private func fetchSpeechAudio(text: String, generation: Int, segmentIndex: Int, totalSegments: Int) async throws -> (data: Data, service: TTSService) {
        do {
            guard generation == ttsPlaybackGeneration else { throw CancellationError() }
            ttsServiceText = TTSService.primary.displayName
            qaStateText = totalSegments > 1 ? "快速语音合成中 \(segmentIndex)/\(totalSegments)" : "快速语音合成中"
            let data = try await fetchPrimarySpeechAudio(text: text)
            return (data, .primary)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            guard generation == ttsPlaybackGeneration else { throw CancellationError() }
            log("TTS 快速语音失败，切换备用语音：\(networkErrorDescription(error))", level: "warning")
            ttsServiceText = TTSService.fallback.displayName
            qaStateText = totalSegments > 1 ? "备用语音合成中 \(segmentIndex)/\(totalSegments)" : "备用语音合成中"
            let data = try await fetchFallbackSpeechAudio(text: text)
            return (data, .fallback)
        }
    }

    private nonisolated func fetchPrimarySpeechAudio(text: String) async throws -> Data {
        var request = URLRequest(url: ttsPrimarySpeechURL)
        request.httpMethod = "POST"
        request.timeoutInterval = ttsRequestTimeoutSeconds
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("audio/mpeg", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": ttsPrimaryModel,
            "voice": ttsPrimaryVoice,
            "input": text,
            "response_format": ttsPrimaryResponseFormat
        ])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        guard !data.isEmpty else {
            throw URLError(.zeroByteResource)
        }
        return data
    }

    private nonisolated func fetchFallbackSpeechAudio(text: String) async throws -> Data {
        let boundary = "pxj-tts-\(UUID().uuidString)"
        var request = URLRequest(url: ttsFallbackGenerateURL)
        request.httpMethod = "POST"
        request.timeoutInterval = ttsFallbackTimeoutSeconds
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = makeMultipartBody(boundary: boundary, fields: [
            "text": text,
            "demo_id": ttsFallbackDemoId,
            "tts_max_batch_size": "1",
            "codec_max_batch_size": "0",
            "enable_text_normalization": "1",
            "enable_normalize_tts_text": "1",
            "cpu_threads": "4"
        ], files: [])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        let decoded = try JSONDecoder().decode(TTSGenerateResponse.self, from: data)
        guard let audioData = Data(base64Encoded: decoded.audioBase64), !audioData.isEmpty else {
            throw URLError(.cannotDecodeContentData)
        }
        return audioData
    }

    private func playSpeechAudioAsync(_ data: Data, generation: Int, segmentIndex: Int, totalSegments: Int) async throws {
        let player = try startSpeechAudio(data, generation: generation, segmentIndex: segmentIndex, totalSegments: totalSegments)
        while generation == ttsPlaybackGeneration && ttsPlayer === player {
            try Task.checkCancellation()
            if !player.isPlaying && ttsPlaybackPhase != .paused {
                break
            }
            try await Task.sleep(nanoseconds: 80_000_000)
        }
        guard generation == ttsPlaybackGeneration else {
            throw CancellationError()
        }
        ttsPlayer = nil
    }

    private func startSpeechAudio(_ data: Data, generation: Int, segmentIndex: Int, totalSegments: Int) throws -> AVAudioPlayer {
        guard generation == ttsPlaybackGeneration else { throw CancellationError() }
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try audioSession.setActive(true, options: [])
            let player = try AVAudioPlayer(data: data)
            player.enableRate = true
            player.rate = Float(voicePlaybackRate)
            player.prepareToPlay()
            ttsPlayer = player
            ttsCurrentSegmentIndex = segmentIndex
            ttsTotalSegmentCount = totalSegments
            qaStateText = totalSegments > 1 ? "正在用语音播放 \(segmentIndex)/\(totalSegments) \(voicePlaybackRateText)" : "正在用语音播放 \(voicePlaybackRateText)"
            qaSystemImage = "speaker.wave.2"
            if !player.play() {
                throw URLError(.cannotOpenFile)
            }
            ttsPlaybackPhase = .playing
            return player
        } catch {
            guard generation == ttsPlaybackGeneration else { throw CancellationError() }
            ttsPlayer = nil
            log("TTS 播放失败：\(networkErrorDescription(error))", level: "error")
            throw error
        }
    }

    private func scheduleQAOverlayHide(after seconds: TimeInterval) {
        qaHideTimer?.invalidate()
        qaHideTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.isListening, !self.isThinking, !self.ttsPlaybackPhase.isActive else { return }
                self.qaOverlayVisible = false
            }
        }
    }

    func cameraHostDidAttach(id: UUID) {
        currentCameraHostId = id
        isCameraReady = false
    }

    func isCurrentCameraHost(id: UUID) -> Bool {
        currentCameraHostId == id
    }

    private func suspendBackgroundCameraBriefly() {
        backgroundCameraResumeTask?.cancel()
        backgroundCameraEnabled = false
        backgroundCameraResumeTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 450_000_000)
            await MainActor.run {
                self?.backgroundCameraEnabled = true
            }
        }
    }

    func openSingleCaptureCamera() {
        activeReviewItem = nil
        pendingSingleCapture = false
        cameraTaskKind = .singleCapture
        cameraPreviewVisible = true
        cameraSheetVisible = false
        uploadState = "准备拍题"
        qualityFeedback = .idle
        log("用户打开拍题相机，等待确认拍照")
    }

    func requestSingleCapture() {
        activeReviewItem = nil
        log("用户点击单张拍题，准备抓拍当前画面")
        cameraTaskKind = .singleCapture
        cameraPreviewVisible = true
        cameraSheetVisible = false
        pendingSingleCapture = true
        uploadState = "拍照中"
        qualityFeedback = CaptureQualityFeedback.uploading("拍照中")
        if isCameraReady {
            triggerPendingSingleCapture()
        }
    }

    func captureSingleFromCamera() {
        activeReviewItem = nil
        cameraTaskKind = .singleCapture
        pendingSingleCapture = false
        cameraPreviewVisible = true
        uploadState = "拍照中"
        qualityFeedback = CaptureQualityFeedback.uploading("拍照中")
        captureSingle?()
    }

    func performCameraPrimaryAction() {
        switch cameraTaskKind {
        case .burst:
            stopBurst()
        case .qaFrame:
            if isListening {
                stopListening(submit: true)
            } else {
                cameraSheetVisible = false
                cameraPreviewVisible = true
            }
        case .extract:
            finishExtraction()
        case .singleCapture, .none:
            captureSingleFromCamera()
        }
    }

    func openActiveCameraTask() {
        openRuntimeTask(runtimeTasks.first?.id ?? "")
    }

    func stopActiveTaskFromOverlay() {
        closeRuntimeTask(runtimeTasks.first?.id ?? "")
    }

    func openRuntimeTask(_ id: String) {
        switch id {
        case "observation":
            cameraTaskKind = .burst
            cameraPreviewVisible = true
            cameraSheetVisible = false
        case "qa-frame":
            cameraTaskKind = .qaFrame
            cameraPreviewVisible = true
            cameraSheetVisible = false
        default:
            break
        }
    }

    func closeRuntimeTask(_ id: String) {
        if id.hasPrefix("server-task:") {
            let taskId = String(id.dropFirst("server-task:".count))
            Task { await cancelServerTask(taskId) }
            return
        }
        switch id {
        case "voice":
            cancelHoldToTalk()
        case "speaking", "tts-generating":
            stopSpeaking()
            qaStateText = "朗读已停止"
            qaSystemImage = "speaker.slash"
        case "qa-frame":
            cameraTaskKind = .none
            cameraSheetVisible = false
            cameraPreviewVisible = false
        case "observation":
            stopBurst(reason: "用户从任务浮层停止智能观察")
        case "camera-error":
            uploadState = isBursting ? "智能观察中" : "待机"
            qualityFeedback = .idle
        case "report", "upload-error":
            uploadState = isBursting ? "智能观察中" : "待机"
            qualityFeedback = .idle
        default:
            break
        }
    }

    // #8 后端后台任务：账号隔离地拉取自己在跑的生成任务，并可取消自己的任务。
    func refreshServerTasks() async {
        do {
            let data = try await getData(path: "/api/tasks")
            let decoded = try JSONDecoder().decode(ServerTasksResponse.self, from: data)
            serverTasks = decoded.tasks
        } catch {
            // 辅助信息，失败静默（不打扰主流程）。
        }
    }

    func cancelServerTask(_ id: String) async {
        do {
            _ = try await postJSON(path: "/api/tasks/\(id)/cancel", payload: [:])
            log("已请求取消后台任务")
            await refreshServerTasks()
        } catch {
            log("取消后台任务失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    func hideInlineCameraPreview() {
        if isBursting {
            cameraPreviewVisible = false
            log("相机预览已隐藏，智能观察继续在后台运行")
            return
        }
        cameraPreviewVisible = false
        cameraSheetVisible = false
        if cameraTaskKind == .singleCapture || (cameraTaskKind == .qaFrame && !isListening && !isPreparingVoiceInput && !isThinking) {
            cameraTaskKind = .none
        }
    }

    func openInlineCameraPreview() {
        if isBursting {
            cameraTaskKind = .burst
        } else if cameraTaskKind == .none {
            cameraTaskKind = .qaFrame
        }
        cameraPreviewVisible = true
        cameraSheetVisible = false
    }

    private func stopLegacyActiveTaskFromOverlay() {
        if isListening {
            stopListening(submit: false)
            qaStateText = "语音已取消"
            qaSystemImage = "mic.slash"
            if cameraTaskKind == .qaFrame {
                cameraTaskKind = .none
                cameraSheetVisible = false
                cameraPreviewVisible = false
            }
            return
        }
        if ttsPlaybackPhase.isActive {
            stopSpeaking()
            qaStateText = "朗读已停止"
            qaSystemImage = "speaker.slash"
            if continuousVoiceActive {
                scheduleContinuousListeningRestart(after: 0.35)
            }
            return
        }
        if cameraTaskKind == .qaFrame {
            cameraTaskKind = .none
            cameraSheetVisible = false
            cameraPreviewVisible = false
            return
        }
        if isBursting {
            stopContinuousVoiceConversation(reason: "用户从任务浮层停止智能观察")
            stopBurst(reason: "用户从任务浮层停止智能观察")
            return
        }
        if uploadState == "相机错误" {
            uploadState = isBursting ? "智能观察中" : "待机"
            qualityFeedback = .idle
        }
    }

    func toggleBurst() {
        recordUserOperation(isBursting ? "stop_observation" : "start_observation")
        if isBursting {
            stopContinuousVoiceConversation(reason: "用户停止智能观察")
            stopBurst()
        } else {
            prepareForObservationStart()
            startBurst()
        }
    }

    func prepareForObservationStart() {
        guard !isBursting else { return }
        observationStopNoticeTask?.cancel()
        observationStopNoticeTask = nil
        observationStopNotice = nil
        if hasChatStarted || sessionId != nil || !qaAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            startNewConversation()
            chatMessages.removeAll()
            lastSubmittedContextItems.removeAll()
            carriedHistoryContext = nil
            qaOverlayVisible = false
            qaAnswer = ""
            recognizedText = ""
            log("开启观察前已清空当前对话，让相机预览保持可见")
        }
        cameraTaskKind = .burst
        cameraSheetVisible = false
        cameraPreviewVisible = true
    }

    func startBurst(showGuide: Bool = true) {
        prepareForObservationStart()
        burstGeneration += 1
        let now = Date()
        strategySyncTask?.cancel()
        lastSyncedStrategySignature = ""
        strategySyncState = "待同步"
        disableIdleTimerForBurst()
        cameraTaskKind = .burst
        cameraSheetVisible = false
        cameraPreviewVisible = true
        isBursting = true
        observationGuideVisible = showGuide
        if showGuide {
            scheduleObservationGuideAutoHide(generation: burstGeneration)
        } else {
            observationGuideHideTask?.cancel()
            observationGuideHideTask = nil
        }
        activeReviewItem = nil
        burstBuffer.removeAll()
        lastAcceptedFingerprint = nil
        lastAcceptedTextTokens.removeAll()
        lastAcceptedAt = nil
        lastSceneActivityAt = now
        lastFrameHadStudyMaterial = nil
        emptySceneCount = 0
        similarSceneCount = 0
        isAnalyzingBurstFrame = false
        isFlushingBurst = false
        shouldFinishAfterFlush = false
        nextBurstSequenceIndex = 0
        observationAcceptedFrameCount = 0
        resetObservationQuestionTracking()
        uploadState = sessionId == nil ? "创建学习回合" : "智能观察中"
        qualityFeedback = .preparing
        log("智能观察启动：从视频预览取关键帧，不触发拍照快门；画面有变化就缓存，并作为问答背景上下文")
        if !trimmedStudentGoal.isEmpty {
            log("本轮帮助目标：\(trimmedStudentGoal)")
        }
        log("学习教练：\(learningMode.title) · \(coachDepth.title)")
        let generation = burstGeneration
        Task {
            let ready = await ensureObservationSession(generation: generation)
            guard ready, isBursting, generation == burstGeneration else { return }
            startTimers(generation: generation)
        }
    }

    private func resetObservationForNewConversationBoundary() {
        guard isBursting else { return }
        burstGeneration += 1
        let generation = burstGeneration
        burstTimer?.invalidate()
        secondTimer?.invalidate()
        burstTimer = nil
        secondTimer = nil
        burstBuffer.removeAll()
        lastAcceptedFingerprint = nil
        lastAcceptedTextTokens.removeAll()
        lastAcceptedAt = nil
        lastSceneActivityAt = Date()
        lastFrameHadStudyMaterial = nil
        emptySceneCount = 0
        similarSceneCount = 0
        isAnalyzingBurstFrame = false
        shouldFinishAfterFlush = false
        nextBurstSequenceIndex = 0
        observationAcceptedFrameCount = 0
        resetObservationQuestionTracking()
        qualityFeedback = .waitingForMaterial
        startTimers(generation: generation)
    }

    private func resetObservationQuestionTracking() {
        observationQuestionSeenKeys.removeAll()
        observationQuestionCandidateCount = 0
        observationQuestionUniqueCount = 0
        observationQuestionDuplicateCount = 0
        observationQuestionLastAction = ""
        observationQuestionEvents.removeAll()
        observationQuestionOverlayCandidates.removeAll()
        observationQuestionOverlayImageSize = .zero
        observationQuestionOverlayUpdatedAt = nil
        observationQuestionScanBusy = false
        observationQuestionScanSerial += 1
    }

    private func appendObservationQuestionEvent(_ message: String) {
        observationQuestionLastAction = message
        observationQuestionEvents.append(message)
        if observationQuestionEvents.count > 12 {
            observationQuestionEvents.removeFirst(observationQuestionEvents.count - 12)
        }
        log("题目指纹：\(message)")
    }

    private func scanObservationQuestionsIfNeeded(_ image: UIImage, frameSequence: Int, generation: Int, textCount: Int, rectangleCount: Int) {
        guard generation == burstGeneration, isBursting, !observationQuestionScanBusy else { return }
        guard textCount > 0 || rectangleCount > 0 else { return }
        observationQuestionScanBusy = true
        observationQuestionScanSerial += 1
        let scanSerial = observationQuestionScanSerial
        let seenKeys = observationQuestionSeenKeys
        Task.detached(priority: .utility) {
            let regions = QuestionSegmenter.segment(image, fast: true)
            let candidates = regions.compactMap { region -> ObservationQuestionCandidate? in
                guard let text = region.ocrText?.trimmingCharacters(in: .whitespacesAndNewlines),
                      let key = Self.extractScanKey(text) else { return nil }
                let preview = text
                    .replacingOccurrences(of: "\n", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let isDuplicate = seenKeys.contains { Self.extractKeySimilar($0, key) }
                return ObservationQuestionCandidate(
                    index: region.index,
                    key: key,
                    fingerprint: Self.observationQuestionFingerprint(key),
                    preview: String(preview.prefix(36)),
                    normalizedRect: region.normalizedRect,
                    isDuplicate: isDuplicate
                )
            }
            await MainActor.run {
                self.finishObservationQuestionScan(
                    candidates,
                    imageSize: image.size,
                    frameSequence: frameSequence,
                    generation: generation,
                    scanSerial: scanSerial
                )
            }
        }
    }

    private func finishObservationQuestionScan(
        _ candidates: [ObservationQuestionCandidate],
        imageSize: CGSize,
        frameSequence: Int,
        generation: Int,
        scanSerial: Int
    ) {
        guard scanSerial == observationQuestionScanSerial else { return }
        observationQuestionScanBusy = false
        guard generation == burstGeneration, isBursting else { return }
        guard !candidates.isEmpty else {
            if observationQuestionEvents.isEmpty {
                appendObservationQuestionEvent("关键帧 #\(frameSequence) 暂未切出稳定题目指纹，继续观察")
            }
            return
        }
        observationQuestionOverlayCandidates = candidates
        observationQuestionOverlayImageSize = imageSize
        observationQuestionOverlayUpdatedAt = Date()

        observationQuestionCandidateCount += candidates.count
        var newCount = 0
        var duplicateCount = 0
        var firstNew: ObservationQuestionCandidate?
        for candidate in candidates {
            if candidate.isDuplicate {
                duplicateCount += 1
            } else {
                observationQuestionSeenKeys.append(candidate.key)
                newCount += 1
                if firstNew == nil {
                    firstNew = candidate
                }
            }
        }
        observationQuestionUniqueCount = observationQuestionSeenKeys.count
        observationQuestionDuplicateCount += duplicateCount

        if let firstNew {
            let prefix = String(firstNew.fingerprint.prefix(8))
            appendObservationQuestionEvent(
                "关键帧 #\(frameSequence) 新题 \(newCount) 道、重复 \(duplicateCount) 道；Q\(firstNew.index) fp=\(prefix)；继续移动到下一题"
            )
        } else {
            appendObservationQuestionEvent(
                "关键帧 #\(frameSequence) 未发现新题，按题文指纹合并重复 \(duplicateCount) 道"
            )
        }
    }

    nonisolated private static func observationQuestionFingerprint(_ key: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    func stopBurst() {
        stopBurst(reason: "用户停止观察")
    }

    private func stopBurst(reason: String) {
        guard isBursting else { return }
        let collectedCount = observationAcceptedFrameCount
        let pendingCount = burstBuffer.count
        continuousVoiceActive = false
        observationGuideVisible = false
        observationGuideHideTask?.cancel()
        observationGuideHideTask = nil
        observationStopNoticeTask?.cancel()
        observationStopNoticeTask = nil
        stopListening(submit: false)
        isBursting = false
        if cameraTaskKind == .burst {
            cameraTaskKind = .none
        }
        if pendingQAFrameContinuation == nil {
            cameraSheetVisible = false
            cameraPreviewVisible = false
        }
        burstTimer?.invalidate()
        secondTimer?.invalidate()
        burstTimer = nil
        secondTimer = nil
        restoreIdleTimerAfterBurst()
        uploadState = "已停止"
        qualityFeedback = .observationStopped(collectedCount: collectedCount, pendingCount: pendingCount)
        showObservationStopNotice(collectedCount: collectedCount, pendingCount: pendingCount)
        qaStateText = collectedCount > 0 ? "观察已停止，后台生成报告" : "观察已停止"
        qaSystemImage = "icloud.and.arrow.up"
        log("智能观察停止：\(reason)，本轮采集 \(collectedCount) 张，剩余关键帧 \(pendingCount) 张")
        shouldFinishAfterFlush = true
        Task {
            await syncStudentGoalIfNeeded()
            await flushBurst(reason: reason)
            await finishBurstSessionIfReady()
        }
    }

    private func scheduleObservationGuideAutoHide(generation: Int) {
        observationGuideHideTask?.cancel()
        observationGuideHideTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            await MainActor.run {
                guard let self, self.isBursting, self.burstGeneration == generation else { return }
                withAnimation(.easeInOut(duration: 0.2)) {
                    self.observationGuideVisible = false
                }
            }
        }
    }

    private func showObservationStopNotice(collectedCount: Int, pendingCount: Int) {
        let countText = collectedCount == 0 ? "尚未采集到学习相关图片" : "共采集了 \(collectedCount) 张与学习相关的图片"
        let pendingText = pendingCount > 0 ? "，正异步传输到服务器" : "，正交给服务器后台处理"
        observationStopNotice = ObservationStopNotice(message: "\(countText)\(pendingText)，后台会自动生成该回合的报告。")
        observationStopNoticeTask?.cancel()
        observationStopNoticeTask = nil
    }

    func cameraDidOpen(hostId: UUID) {
        guard currentCameraHostId == hostId else { return }
        isCameraReady = true
        qualityFeedback = .idle
        log("相机已打开，开始预览桌面课本/屏幕/试卷")
        triggerPendingSingleCapture()
        if isBursting {
            let generation = burstGeneration
            if captureBurstFrame?() != true {
                scheduleNextBurstCapture(generation: generation, interval: 0.8)
            }
        }
    }

    func cameraFailed(_ message: String, hostId: UUID) {
        guard currentCameraHostId == hostId else { return }
        isCameraReady = false
        pendingSingleCapture = false
        uploadState = "相机错误"
        qualityFeedback = .cameraError(message)
        completePendingQAFrame(nil)
        if isBursting {
            stopBurst(reason: "相机不可用：\(message)")
        }
        log("相机错误：\(message)", level: "error")
    }

    func cameraDidClose(hostId: UUID) {
        guard currentCameraHostId == hostId else { return }
        currentCameraHostId = nil
        let shouldKeepWaitingForReplacementCamera = (isBursting || cameraTaskKind == .qaFrame || cameraTaskKind == .extract) && (cameraPreviewVisible || cameraSheetVisible || backgroundCameraActive)
        let wasWaitingForSingleCapture = pendingSingleCapture
        captureSingle = nil
        captureBurstFrame = nil
        captureQAFrame = nil
        isCameraReady = false
        pendingSingleCapture = false
        if !shouldKeepWaitingForReplacementCamera {
            completePendingQAFrame(nil)
        }
        if wasWaitingForSingleCapture {
            uploadState = "待机"
        }
        if !isBursting && cameraTaskKind != .qaFrame {
            cameraTaskKind = .none
            cameraPreviewVisible = false
            qualityFeedback = .idle
        }
        log("相机任务面板已关闭")
    }

    func closeCameraTask() {
        if isBursting {
            suspendBackgroundCameraBriefly()
            cameraSheetVisible = false
            cameraPreviewVisible = true
            log("智能观察已收起到浮层，后台继续观察")
            return
        }
        if cameraTaskKind == .qaFrame {
            suspendBackgroundCameraBriefly()
            cameraPreviewVisible = true
        }
        cameraSheetVisible = false
        if cameraTaskKind == .singleCapture {
            cameraTaskKind = .none
            cameraPreviewVisible = false
        }
    }

    private func triggerPendingSingleCapture() {
        guard pendingSingleCapture, isCameraReady else { return }
        pendingSingleCapture = false
        captureSingle?()
    }

    func studentGoalDidChange() {
        guard sessionId != nil else {
            strategySyncState = "待开始"
            return
        }
        strategySyncState = "待同步"
        strategySyncTask?.cancel()
        strategySyncTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 900_000_000)
            guard !Task.isCancelled else { return }
            await self?.syncStudentGoalIfNeeded()
        }
    }

    func coachPreferenceDidChange() {
        // 偏好改为自然语言文本（coachPreferenceText），通过 strategySignature 触发同步。
        studentGoalDidChange()
    }

    private func fetchReviewQueue(dueOnly: Bool, pageSize: Int) async throws -> ReviewQueueResponse {
        var components = URLComponents(url: serverBaseURL.appending(path: "/api/review-queue"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "due_only", value: dueOnly ? "true" : "false"),
            URLQueryItem(name: "page_size", value: String(pageSize))
        ]
        guard let url = components.url else {
            throw URLError(.badURL)
        }
        let data = try await getData(url: url)
        return try JSONDecoder().decode(ReviewQueueResponse.self, from: data)
    }

    private func reviewGoal(for item: ReviewMistakeItem, isDue: Bool) -> String {
        var parts = ["今日复习"]
        if !isDue {
            parts.append("提前复习")
        }
        let location = item.locationText
        if !location.isEmpty {
            parts.append(location)
        }
        let focus = firstNonEmpty(item.errorReason, item.errorType, item.nextAction)
        if !focus.isEmpty {
            parts.append("重点：\(focus)")
        }
        return parts.joined(separator: "；")
    }

    private func reviewQuestion(for item: ReviewMistakeItem) -> String {
        var lines = ["带我复习这道错题。"]
        let title = item.displayTitle
        if !title.isEmpty {
            lines.append("题目：\(shortText(title, limit: 140))")
        }
        if !item.locationText.isEmpty {
            lines.append("位置：\(item.locationText)")
        }
        if !item.errorReason.isEmpty {
            lines.append("上次错因：\(shortText(item.errorReason, limit: 120))")
        } else if !item.errorType.isEmpty {
            lines.append("错误类型：\(item.errorType)")
        }
        if !item.nextAction.isEmpty {
            lines.append("建议下一步：\(shortText(item.nextAction, limit: 120))")
        }
        if !item.knowledgePoints.isEmpty {
            lines.append("知识点：\(item.knowledgePoints.prefix(4).joined(separator: "、"))")
        }
        lines.append("请先问我一个回忆问题或给一个小提示，再根据我的回答判断是否掌握。")
        return lines.joined(separator: "\n")
    }

    private func reviewQueueSummary(for item: ReviewMistakeItem) -> String {
        let location = item.locationText
        let title = shortText(item.displayTitle, limit: 32)
        if !location.isEmpty {
            return "\(location) · \(title)"
        }
        return title
    }

    func startDeviceControlPolling() {
        guard deviceControlTask == nil else { return }
        log("多屏控制已启用：网页按钮会同步到本机执行")
        deviceControlTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.pollDeviceControlOnce()
                let interval = await MainActor.run { self?.remotePollIntervalSeconds ?? 1.0 }
                let nanoseconds = UInt64(max(0.5, min(interval, 5.0)) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: nanoseconds)
            }
        }
    }

    func didCaptureSingle(_ image: UIImage) {
        log("单张照片已捕获，准备压缩上传")
        Task { await uploadSingle(image) }
    }

    func didCaptureBurstFrame(_ image: UIImage) {
        guard isBursting, !isAnalyzingBurstFrame else { return }
        isAnalyzingBurstFrame = true
        let generation = burstGeneration
        burstAnalysisQueue.async { [weak self] in
            let analysis = BurstFrameAnalyzer.analyze(image)
            Task { @MainActor in
                self?.finishBurstFrameAnalysis(image, analysis: analysis, generation: generation)
            }
        }
    }

    private func finishBurstFrameAnalysis(_ image: UIImage, analysis: BurstFrameAnalysis, generation: Int) {
        guard generation == burstGeneration else { return }
        isAnalyzingBurstFrame = false
        guard isBursting else { return }
        let now = Date()
        let hasStudyMaterial = analysis.signals.hasStudyMaterial
        recordStudyMaterialPresence(hasStudyMaterial, at: now)
        let distance = lastAcceptedFingerprint.map { analysis.fingerprint.distance(to: $0) }
        let textDistance = textDifference(from: lastAcceptedTextTokens, to: analysis.textTokens)
        let secondsSinceAccepted = lastAcceptedAt.map { now.timeIntervalSince($0) } ?? .greatestFiniteMagnitude
        let isFirstUsefulFrame = lastAcceptedFingerprint == nil
        let visualChanged = (distance ?? .greatestFiniteMagnitude) >= 4.5
        let textChanged = analysis.textTokens.count >= 2 && lastAcceptedTextTokens.count >= 2 && textDistance >= 0.35
        let isMeaningfullyChanged = (visualChanged || textChanged) && secondsSinceAccepted >= 0.8
        let assessment = CaptureQualityAssessment.evaluate(
            signals: analysis.signals,
            visualDistance: distance,
            textDistance: textDistance,
            isFirstUsefulFrame: isFirstUsefulFrame
        )
        guard assessment.shouldUpload else {
            if hasStudyMaterial {
                similarSceneCount += 1
            } else {
                emptySceneCount += 1
            }
            uploadState = "未上传低质量画面"
            qualityFeedback = .lowQuality(assessment)
            if emptySceneCount == 1 || similarSceneCount == 1 || (emptySceneCount + similarSceneCount) % 4 == 0 {
                log("未上传低质量画面：\(assessment.userMessage)（blur=\(String(format: "%.2f", assessment.blurScore))，material=\(String(format: "%.2f", assessment.materialConfidence))）")
            }
            if stopBurstIfSceneIdle(at: now, generation: generation) { return }
            scheduleNextBurstCapture(generation: generation, interval: hasStudyMaterial ? intervalAfterSimilarScene() : intervalAfterEmptyScene())
            return
        }

        emptySceneCount = 0
        let shouldProbeSimilarScene = hasStudyMaterial && !isFirstUsefulFrame && !isMeaningfullyChanged && secondsSinceAccepted >= similarSceneProbeInterval
        let shouldAccept = isFirstUsefulFrame || isMeaningfullyChanged || shouldProbeSimilarScene

        if shouldAccept {
            let frame = BurstFrame(
                image: image,
                capturedAt: now,
                sequenceIndex: nextBurstSequenceIndex,
                signalSummary: analysis.signals.summary,
                rectangleCount: analysis.signals.rectangleCount,
                textCount: analysis.signals.textCount,
                lightCoverage: analysis.signals.lightCoverage,
                edgeDensity: analysis.signals.edgeDensity,
                contrast: analysis.signals.contrast,
                hasStudyMaterial: analysis.signals.hasStudyMaterial,
                hasExplicitStudyEvidence: analysis.signals.hasExplicitStudyEvidence,
                visualHash: analysis.fingerprint.hash,
                visualSample: analysis.fingerprint.values,
                textTokens: Array(analysis.textTokens).sorted(),
                visualDistance: distance,
                textDistance: textDistance,
                qualityStatus: assessment.status,
                qualityReasons: assessment.reasons,
                shouldUpload: assessment.shouldUpload,
                blurScore: assessment.blurScore,
                materialConfidence: assessment.materialConfidence,
                occlusionScore: assessment.occlusionScore,
                motionScore: assessment.motionScore,
                hasStudentPresence: analysis.signals.hasStudentPresence,
                studentPresenceStatus: analysis.signals.studentPresenceStatus,
                presenceSummary: analysis.signals.presenceSummary,
                activitySummary: analysis.signals.activitySummary,
                handCount: analysis.signals.handCount,
                faceCount: analysis.signals.faceCount,
                bodyCount: analysis.signals.bodyCount
            )
            nextBurstSequenceIndex += 1
            burstBuffer.append(frame)
            observationAcceptedFrameCount += 1
            lastAcceptedFingerprint = analysis.fingerprint
            lastAcceptedTextTokens = analysis.textTokens
            lastAcceptedAt = now
            if isFirstUsefulFrame || isMeaningfullyChanged {
                recordSceneActivity(at: now)
            }
            similarSceneCount = 0
            uploadState = "已缓存 \(burstBuffer.count) 张"
            if isFirstUsefulFrame || shouldProbeSimilarScene {
                qualityFeedback = .materialVisible(signals: analysis.signals, cachedCount: burstBuffer.count)
            } else {
                qualityFeedback = .sceneChanged(signals: analysis.signals, cachedCount: burstBuffer.count, distance: distance)
            }
            let changeText: String
            if shouldProbeSimilarScene {
                let distanceText = distance.map { String(format: "%.1f", $0) } ?? "0.0"
                changeText = String(format: "同页抽检 %.0f 秒，变化 %@", secondsSinceAccepted, distanceText)
            } else {
                changeText = distance.map { String(format: "%.1f", $0) } ?? "首张"
            }
            let actionText = shouldProbeSimilarScene ? "保留同页观察帧" : "检测到学习关键画面"
            log("\(actionText)，加入缓存：\(burstBuffer.count) 张（\(analysis.signals.summary)，变化 \(changeText)）")
            scanObservationQuestionsIfNeeded(
                image,
                frameSequence: frame.sequenceIndex,
                generation: generation,
                textCount: frame.textCount,
                rectangleCount: frame.rectangleCount
            )
            if shouldProbeSimilarScene, stopBurstIfSceneIdle(at: now, generation: generation) { return }
            scheduleNextBurstCapture(generation: generation, interval: activeCaptureInterval)
        } else {
            similarSceneCount += 1
            uploadState = "等待画面变化"
            qualityFeedback = .waitingForChange(signals: analysis.signals, distance: distance, similarCount: similarSceneCount)
            if similarSceneCount == 1 || similarSceneCount % 4 == 0 {
                let changeText = distance.map { String(format: "%.1f", $0) } ?? "0.0"
                log("学习材料仍在，但与上一关键帧相似，降低观察频率（变化 \(changeText)）")
            }
            if stopBurstIfSceneIdle(at: now, generation: generation) { return }
            scheduleNextBurstCapture(generation: generation, interval: intervalAfterSimilarScene())
        }

        if burstBuffer.count >= burstBatchSize {
            Task { await flushBurst(reason: "批次达到 \(burstBatchSize) 张关键帧") }
        }
    }

    private func textDifference(from oldTokens: Set<String>, to newTokens: Set<String>) -> Double {
        guard !oldTokens.isEmpty, !newTokens.isEmpty else { return 0 }
        let unionCount = oldTokens.union(newTokens).count
        guard unionCount > 0 else { return 0 }
        let intersectionCount = oldTokens.intersection(newTokens).count
        return 1.0 - (Double(intersectionCount) / Double(unionCount))
    }

    private func recordSceneActivity(at date: Date) {
        lastSceneActivityAt = date
    }

    private func recordStudyMaterialPresence(_ hasStudyMaterial: Bool, at date: Date) {
        if let previous = lastFrameHadStudyMaterial {
            if previous != hasStudyMaterial {
                recordSceneActivity(at: date)
            }
        } else {
            recordSceneActivity(at: date)
        }
        lastFrameHadStudyMaterial = hasStudyMaterial
    }

    @discardableResult
    private func stopBurstIfSceneIdle(at date: Date = Date(), generation: Int) -> Bool {
        guard isBursting, generation == burstGeneration, let lastSceneActivityAt else { return false }
        guard date.timeIntervalSince(lastSceneActivityAt) >= stillSceneAutoStopInterval else { return false }
        let idleMinutes = Int(stillSceneAutoStopInterval / 60)
        stopBurst(reason: "连续 \(idleMinutes) 分钟未检测到画面变化，自动停止智能观察")
        return true
    }

    private func intervalAfterEmptyScene() -> TimeInterval {
        let index = min(max(emptySceneCount - 1, 0), emptyCaptureIntervals.count - 1)
        return emptyCaptureIntervals[index]
    }

    private func intervalAfterSimilarScene() -> TimeInterval {
        similarSceneCount >= 4 ? longSimilarCaptureInterval : similarCaptureInterval
    }

    private func scheduleNextBurstCapture(generation: Int, interval: TimeInterval) {
        burstTimer?.invalidate()
        burstTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isBursting, generation == self.burstGeneration, !self.isAnalyzingBurstFrame else { return }
                if self.stopBurstIfSceneIdle(generation: generation) { return }
                if self.captureBurstFrame?() != true {
                    self.scheduleNextBurstCapture(generation: generation, interval: 0.8)
                }
            }
        }
    }

    private func startTimers(generation: Int) {
        burstTimer?.invalidate()
        secondTimer?.invalidate()
        if captureBurstFrame?() != true {
            scheduleNextBurstCapture(generation: generation, interval: 0.8)
        }
        secondTimer = Timer.scheduledTimer(withTimeInterval: 8.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isBursting, generation == self.burstGeneration else { return }
                if !self.isAnalyzingBurstFrame, self.stopBurstIfSceneIdle(generation: generation) { return }
                self.log("智能观察运行中：观察学习材料，等待新的关键变化")
            }
        }
    }

    private func ensureObservationSession(generation: Int) async -> Bool {
        guard isBursting, generation == burstGeneration else { return false }
        if sessionId != nil {
            uploadState = "智能观察中"
            qualityFeedback = .waitingForMaterial
            await syncStudentGoalIfNeeded()
            return true
        }
        return await createSession(generation: generation)
    }

    private func createSession(generation: Int) async -> Bool {
        do {
            var fields = [
                "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                "mode": "burst",
                "title": "智能观察学习回合",
                "report_style": coachReportStyle,
                "assistant_focus": coachAssistantFocus
            ]
            let goal = trimmedStudentGoal
            if !goal.isEmpty {
                fields["student_goal"] = goal
            }
            if !AuthSession.shared.activeStudentId.isEmpty {
                fields["student_profile_id"] = AuthSession.shared.activeStudentId
            }
            let data = try await postForm(path: "/api/sessions", fields: fields, files: [])
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any], let id = json["session_id"] as? String {
                guard isBursting, generation == burstGeneration else { return false }
                sessionId = id
                lastSyncedStrategySignature = strategySignature
                strategySyncState = "已同步"
                uploadState = "等待学习材料"
                qualityFeedback = .waitingForMaterial
                log("学习回合创建成功：\(id)")
                return true
            }
            guard generation == burstGeneration else { return false }
            uploadState = "创建失败"
            qualityFeedback = CaptureQualityFeedback(
                title: "创建学习回合失败",
                detail: "请稍后重试，当前不会上传无效画面",
                systemImage: "exclamationmark.triangle",
                tone: .warning
            )
            isBursting = false
            cameraSheetVisible = false
            cameraPreviewVisible = false
            restoreIdleTimerAfterBurst()
            log("创建学习回合失败：后端未返回回合 ID", level: "error")
        } catch {
            guard generation == burstGeneration else { return false }
            uploadState = "创建失败"
            qualityFeedback = CaptureQualityFeedback(
                title: "创建学习回合失败",
                detail: networkErrorUserMessage(error),
                systemImage: "exclamationmark.triangle",
                tone: .warning
            )
            isBursting = false
            cameraSheetVisible = false
            cameraPreviewVisible = false
            restoreIdleTimerAfterBurst()
            log("创建学习回合失败：\(networkErrorDescription(error))", level: "error")
        }
        return false
    }

    private func disableIdleTimerForBurst() {
        if idleTimerDisabledBeforeBurst == nil {
            idleTimerDisabledBeforeBurst = UIApplication.shared.isIdleTimerDisabled
        }
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func restoreIdleTimerAfterBurst() {
        guard let idleTimerDisabledBeforeBurst else { return }
        UIApplication.shared.isIdleTimerDisabled = idleTimerDisabledBeforeBurst
        self.idleTimerDisabledBeforeBurst = nil
    }

    private func uploadSingle(_ image: UIImage) async {
        uploadState = "上传单张"
        qualityFeedback = .uploading("上传单张")
        do {
            let jpeg = try jpegData(image)
            var fields = [
                "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                "report_style": coachReportStyle,
                "assistant_focus": coachAssistantFocus
            ]
            let goal = trimmedStudentGoal
            if !goal.isEmpty {
                fields["student_goal"] = goal
            }
            _ = try await postForm(path: "/api/solve-single", fields: fields, files: [MultipartFile(field: "image", name: "question.jpg", mime: "image/jpeg", data: jpeg)])
            uploadState = "后台解析中"
            qualityFeedback = CaptureQualityFeedback(
                title: "已上传单张",
                detail: "后台正在解析当前题目画面",
                systemImage: "checkmark.circle",
                tone: .good
            )
            log("单张拍题上传完成，后端 dashboard 将自动更新解析结果")
        } catch {
            uploadState = "上传失败"
            qualityFeedback = CaptureQualityFeedback(
                title: "上传失败",
                detail: networkErrorUserMessage(error),
                systemImage: "exclamationmark.triangle",
                tone: .warning
            )
            log("单张拍题失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    private func flushBurst(reason: String) async {
        guard let sessionId, !burstBuffer.isEmpty, !isFlushingBurst else { return }
        isFlushingBurst = true
        var uploadSucceeded = false
        let generation = burstGeneration
        defer {
            isFlushingBurst = false
            if uploadSucceeded, generation == burstGeneration {
                if !burstBuffer.isEmpty {
                    if isBursting, burstBuffer.count >= burstBatchSize {
                        Task { await flushBurst(reason: "上传期间继续积累关键帧") }
                    } else if !isBursting {
                        Task { await flushBurst(reason: "停止前剩余关键帧") }
                    }
                } else if shouldFinishAfterFlush, !isBursting {
                    Task { await finishBurstSessionIfReady() }
                }
            }
        }
        let frames = burstBuffer
        burstBuffer.removeAll()
        uploadState = "上传批次"
        qualityFeedback = .uploading("上传关键帧")
        log("上传智能观察批次：\(reason)，共 \(frames.count) 张")
        do {
            let files = try frames.map { frame in
                MultipartFile(field: "images", name: "burst-\(frame.sequenceIndex).jpg", mime: "image/jpeg", data: try jpegData(frame.image))
            }
            let meta = captureMetaJSON(for: frames)
            _ = try await postForm(
                path: "/api/sessions/\(sessionId)/batches",
                fields: [
                    "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
                    "environment": "iPhone 固定机位，桌面课本/试卷学习场景",
                    "capture_meta": meta
                ],
                files: files
            )
            uploadSucceeded = true
            if generation == burstGeneration {
                uploadState = isBursting ? "智能观察中" : "已停止"
                if !isBursting {
                    qualityFeedback = .observationStopped(collectedCount: observationAcceptedFrameCount, pendingCount: burstBuffer.count)
                }
                log("批次上传完成，后端将异步汇总学习报告")
            }
        } catch {
            if generation == burstGeneration {
                burstBuffer.insert(contentsOf: frames, at: 0)
                uploadState = "批次失败"
                qualityFeedback = CaptureQualityFeedback(
                    title: "关键帧上传失败",
                    detail: "已保留本批画面，网络恢复后可继续尝试",
                    systemImage: "exclamationmark.triangle",
                    tone: .warning
                )
                log("批次上传失败：\(networkErrorDescription(error))", level: "error")
            }
        }
    }

    private func captureMetaJSON(for frames: [BurstFrame]) -> String {
        let payload = frames.map { frame -> [String: Any] in
            var item: [String: Any] = [
                "sequence_index": frame.sequenceIndex,
                "captured_at": CaptureTimeFormatter.string(from: frame.capturedAt),
                "signal_summary": frame.signalSummary,
                "rectangle_count": frame.rectangleCount,
                "rectangleCount": frame.rectangleCount,
                "text_count": frame.textCount,
                "textCount": frame.textCount,
                "light_coverage": frame.lightCoverage,
                "lightCoverage": frame.lightCoverage,
                "edge_density": frame.edgeDensity,
                "edgeDensity": frame.edgeDensity,
                "contrast": frame.contrast,
                "has_study_material": frame.hasStudyMaterial,
                "hasStudyMaterial": frame.hasStudyMaterial,
                "has_explicit_study_evidence": frame.hasExplicitStudyEvidence,
                "hasExplicitStudyEvidence": frame.hasExplicitStudyEvidence,
                "has_student_presence": frame.hasStudentPresence,
                "student_presence_status": frame.studentPresenceStatus,
                "presence_summary": frame.presenceSummary,
                "activity_summary": frame.activitySummary,
                "hand_count": frame.handCount,
                "face_count": frame.faceCount,
                "body_count": frame.bodyCount,
                "device_interaction_presence": userOperationPresencePayload(referenceDate: frame.capturedAt),
                "has_device_interaction_presence": userOperationPresencePayload(referenceDate: frame.capturedAt)["present"] as? Bool ?? false,
                "quality_status": frame.qualityStatus,
                "qualityStatus": frame.qualityStatus,
                "quality_reasons": frame.qualityReasons,
                "qualityReasons": frame.qualityReasons,
                "should_upload": frame.shouldUpload,
                "shouldUpload": frame.shouldUpload,
                "blur_score": frame.blurScore,
                "blurScore": frame.blurScore,
                "material_confidence": frame.materialConfidence,
                "materialConfidence": frame.materialConfidence,
                "occlusion_score": frame.occlusionScore,
                "occlusionScore": frame.occlusionScore,
                "motion_score": frame.motionScore,
                "motionScore": frame.motionScore,
                "visual_hash": frame.visualHash,
                "visual_sample": frame.visualSample.map(Int.init),
                "text_tokens": frame.textTokens,
                "source": "ios-video-preview"
            ]
            if let visualDistance = frame.visualDistance, visualDistance.isFinite {
                item["visual_distance"] = visualDistance
            }
            if let textDistance = frame.textDistance, textDistance.isFinite {
                item["text_distance"] = textDistance
            }
            return item
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return text
    }

    private func finishBurstSessionIfReady() async {
        guard shouldFinishAfterFlush, !isBursting, !isFlushingBurst else { return }
        guard let sessionId else { return }
        guard burstBuffer.isEmpty else {
            await flushBurst(reason: "结束前剩余关键帧")
            return
        }
        await syncStudentGoalIfNeeded()
        shouldFinishAfterFlush = false
        uploadState = "生成报告"
        qualityFeedback = CaptureQualityFeedback.uploading("生成报告")
        log("通知后端结束学习回合，生成总结报告")
        do {
            _ = try await postForm(
                path: "/api/sessions/\(sessionId)/finish",
                fields: ["device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone"],
                files: []
            )
            uploadState = "报告生成中"
            qualityFeedback = CaptureQualityFeedback(
                title: "报告生成中",
                detail: "已停止采集，后端正在整理学习回合",
                systemImage: "doc.text.magnifyingglass",
                tone: .good
            )
            log("后端已开始生成学习回合总结报告")
        } catch {
            shouldFinishAfterFlush = true
            uploadState = "报告触发失败"
            qualityFeedback = CaptureQualityFeedback(
                title: "报告触发失败",
                detail: networkErrorUserMessage(error),
                systemImage: "exclamationmark.triangle",
                tone: .warning
            )
            log("结束学习回合失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    func generateObservationReportNow() {
        guard sessionId != nil else {
            appendStatusChatMessage(title: "生成报告", text: "当前还没有学习回合，先开始一次智能观察。", systemImage: "doc.text.magnifyingglass")
            return
        }
        shouldFinishAfterFlush = true
        upsertStatusChatMessage(
            key: "observation-final-report",
            title: "生成报告",
            text: "已提交生成学习报告；如果还有未上传关键帧，会先补传再汇总。",
            systemImage: "doc.text.magnifyingglass",
            showsProgress: true
        )
        Task {
            if !burstBuffer.isEmpty {
                await flushBurst(reason: "用户点击生成报告")
            }
            if shouldFinishAfterFlush {
                await finishBurstSessionIfReady()
            } else {
                await requestObservationFinalReport(statusKey: "observation-final-report")
            }
            await refreshServerTasks()
        }
    }

    func extractAllObservationQuestions() {
        guard let sessionId else {
            appendStatusChatMessage(title: "提取题目", text: "当前还没有学习回合，先开始一次智能观察。", systemImage: "rectangle.stack.badge.plus")
            return
        }
        upsertStatusChatMessage(
            key: "observation-question-extract",
            title: "提取题目",
            text: "已把本轮关键图排入题目提取队列，后台会按指纹合并重复题。",
            systemImage: "rectangle.stack.badge.plus",
            showsProgress: true
        )
        Task {
            do {
                let data = try await postForm(
                    path: "/api/sessions/\(sessionId)/extract-all-questions",
                    fields: ["limit": "80"],
                    files: []
                )
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                let imageCount = (json?["image_count"] as? Int) ?? 0
                let setCount = (json?["question_set_count"] as? Int) ?? 0
                extractSessionId = sessionId
                extractUniqueCount = setCount
                upsertStatusChatMessage(
                    key: "observation-question-extract",
                    title: "题目提取已排队",
                    text: "将处理 \(imageCount) 张关键图；当前题集已有 \(setCount) 道题。后台完成后可从历史回合打开还原页/空白卷。",
                    systemImage: "checkmark.circle"
                )
                if setCount > 0, let html = await fetchRestoreHTML(view: "restore") {
                    extractRestoreHTML = html
                    extractResultVisible = true
                }
                await refreshServerTasks()
            } catch {
                upsertStatusChatMessage(
                    key: "observation-question-extract",
                    title: "提取题目失败",
                    text: networkErrorUserMessage(error),
                    systemImage: "exclamationmark.triangle"
                )
            }
        }
    }

    func extractObservationMistakes() {
        guard sessionId != nil else {
            appendStatusChatMessage(title: "提取错题", text: "当前还没有学习回合，先开始一次智能观察。", systemImage: "book.closed")
            return
        }
        upsertStatusChatMessage(
            key: "observation-mistake-extract",
            title: "提取错题",
            text: "已请求整理本轮错题候选；系统会先生成/刷新学习报告，再把可疑错题写入候选区。",
            systemImage: "book.closed",
            showsProgress: true
        )
        Task {
            await requestObservationFinalReport(statusKey: "observation-mistake-extract")
            await refreshReviewQueuePreview()
            await refreshServerTasks()
        }
    }

    private func requestObservationFinalReport(statusKey: String? = nil) async {
        guard let sessionId else { return }
        await syncStudentGoalIfNeeded()
        uploadState = "生成报告"
        qualityFeedback = CaptureQualityFeedback.uploading("生成报告")
        do {
            _ = try await postForm(
                path: "/api/sessions/\(sessionId)/finish",
                fields: ["device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone"],
                files: []
            )
            uploadState = "报告生成中"
            qualityFeedback = CaptureQualityFeedback(
                title: "报告生成中",
                detail: "后端正在整理学习回合、题目线索和错题候选",
                systemImage: "doc.text.magnifyingglass",
                tone: .good
            )
            log("后端已开始生成学习回合总结报告")
            if let statusKey {
                upsertStatusChatMessage(
                    key: statusKey,
                    title: statusKey == "observation-mistake-extract" ? "错题提取处理中" : "报告生成中",
                    text: statusKey == "observation-mistake-extract" ? "报告任务已排队，完成后会自动沉淀错题候选。" : "报告任务已排队，稍后可在历史回合查看。",
                    systemImage: statusKey == "observation-mistake-extract" ? "book.closed" : "doc.text.magnifyingglass",
                    showsProgress: true
                )
            }
        } catch {
            uploadState = "报告触发失败"
            qualityFeedback = CaptureQualityFeedback(
                title: "报告触发失败",
                detail: networkErrorUserMessage(error),
                systemImage: "exclamationmark.triangle",
                tone: .warning
            )
            if let statusKey {
                upsertStatusChatMessage(
                    key: statusKey,
                    title: "后台任务触发失败",
                    text: networkErrorUserMessage(error),
                    systemImage: "exclamationmark.triangle"
                )
            }
            log("结束学习回合失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    private func syncStudentGoalIfNeeded() async {
        guard let sessionId else { return }
        let goal = trimmedStudentGoal
        let signature = strategySignature
        guard signature != lastSyncedStrategySignature else {
            strategySyncState = "已同步"
            return
        }
        do {
            _ = try await patchJSON(
                path: "/api/sessions/\(sessionId)/strategy",
                payload: [
                    "student_goal": goal,
                    "report_style": coachReportStyle,
                    "assistant_focus": coachAssistantFocus
                ]
            )
            lastSyncedStrategySignature = signature
            strategySyncState = "已同步"
            if !goal.isEmpty {
                log("学习目标已同步：\(goal)")
            } else {
                log("学习教练偏好已同步：\(trimmedCoachPreferenceText.isEmpty ? "默认策略" : trimmedCoachPreferenceText)")
            }
        } catch {
            strategySyncState = "同步失败"
            log("学习目标同步失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    private func remoteControlStatePayload() -> [String: Any] {
        [
            "app": "pxj",
            "platform": "ios",
            "session_id": sessionId ?? "",
            "session_title": modeTitle,
            "student_goal": trimmedStudentGoal,
            "learning_mode": learningMode.rawValue,
            "learning_mode_title": learningMode.title,
            "coach_depth": coachDepth.rawValue,
            "coach_depth_title": coachDepth.title,
            "coach_preference": coachAssistantFocus,
            "mode_title": modeTitle,
            "upload_state": uploadState,
            "is_bursting": isBursting,
            "is_observing": isBursting,
            "is_listening": isListening,
            "continuous_voice_active": continuousVoiceActive,
            "is_thinking": isThinking,
            "is_speaking": isSpeaking,
            "qa_state": qaStateText,
            "qa_answer": qaAnswer,
            "recognized_text": recognizedText,
            "strategy_sync_state": strategySyncState,
            "updated_at": CaptureTimeFormatter.string(from: Date())
        ]
    }

    private func pollDeviceControlOnce() async {
        do {
            var request = URLRequest(url: serverBaseURL.appending(path: "/api/device-control/poll"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            setControlTokenHeader(&request)
            if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
            request.timeoutInterval = 8
            let state = await MainActor.run { remoteControlStatePayload() }
            let payload: [String: Any] = [
                "device_id": remoteDeviceId,
                "session_id": sessionId ?? "",
                "source": "ios",
                "state": state
            ]
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { return }
            let decoded = try JSONDecoder().decode(DeviceControlPollResponse.self, from: data)
            if let interval = decoded.pollIntervalSeconds {
                remotePollIntervalSeconds = interval
            }
            for command in decoded.commands {
                await handleRemoteControlCommand(command)
            }
        } catch {
            // Keep this quiet; regular app logs are still uploaded through /api/logs.
        }
    }

    private func handleRemoteControlCommand(_ command: RemoteControlCommand) async {
        guard !handledRemoteCommandIds.contains(command.id) else { return }
        handledRemoteCommandIds.insert(command.id)
        let result: (String, String) = await MainActor.run {
            executeRemoteControlCommand(command)
        }
        await acknowledgeRemoteControlCommand(command, status: result.0, error: result.1)
        if handledRemoteCommandIds.count > 80 {
            handledRemoteCommandIds.removeAll(keepingCapacity: true)
        }
    }

    private func executeRemoteControlCommand(_ command: RemoteControlCommand) -> (String, String) {
        switch command.commandType {
        case "single_capture":
            requestSingleCapture()
            log("网页远程触发：拍题解析")
            return ("applied", "")
        case "start_burst":
            if isBursting {
                log("网页远程触发：智能观察已在运行")
                return ("ignored", "already observing")
            }
            startBurst()
            log("网页远程触发：开始智能观察")
            return ("applied", "")
        case "stop_burst":
            guard isBursting else {
                log("网页远程触发：当前未在智能观察")
                return ("ignored", "not observing")
            }
            stopBurst(reason: "网页远程停止智能观察")
            log("网页远程触发：停止智能观察")
            return ("applied", "")
        case "voice_question":
            startVoiceQuestion(trigger: "dashboard_voice")
            log("网页远程触发：语音提问")
            return ("applied", "")
        case "ok_followup":
            interruptForFollowUp()
            log("网页远程触发：OK 追问")
            return ("applied", "")
        case "end_qa":
            endQARound()
            log("网页远程触发：结束问答")
            return ("applied", "")
        case "set_goal":
            let goal = command.payload["student_goal"] ?? command.payload["goal"] ?? ""
            studentGoal = goal
            if let rawMode = command.payload["learning_mode"], let mode = LearningModePreference(rawValue: rawMode) {
                learningMode = mode
                UserDefaults.standard.set(mode.rawValue, forKey: learningModeDefaultsKey)
            }
            if let rawDepth = command.payload["coach_depth"], let depth = CoachDepthPreference(rawValue: rawDepth) {
                coachDepth = depth
                UserDefaults.standard.set(depth.rawValue, forKey: coachDepthDefaultsKey)
            }
            studentGoalDidChange()
            log("网页远程同步学习目标：\(goal.isEmpty ? "已清空" : goal)")
            return ("applied", "")
        default:
            return ("failed", "unknown command \(command.commandType)")
        }
    }

    private func acknowledgeRemoteControlCommand(_ command: RemoteControlCommand, status: String, error: String) async {
        do {
            var request = URLRequest(url: serverBaseURL.appending(path: "/api/control-commands/\(command.id)/ack"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            setControlTokenHeader(&request)
            if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
            request.timeoutInterval = 8
            let state = await MainActor.run { remoteControlStatePayload() }
            let payload: [String: Any] = [
                "device_id": remoteDeviceId,
                "session_id": sessionId ?? "",
                "status": status,
                "error": error,
                "state": state
            ]
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            _ = try await URLSession.shared.data(for: request)
        } catch {
            log("远程命令回执失败：\(networkErrorDescription(error))", level: "error")
        }
    }

    private func setControlTokenHeader(_ request: inout URLRequest) {
        guard !controlToken.isEmpty else { return }
        request.setValue(controlToken, forHTTPHeaderField: "X-PAI-Control-Token")
    }

    func log(_ message: String, level: String = "info") {
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        logs.append(LogLine(text: "[\(stamp)] \(message)"))
        if logs.count > 160 {
            logs.removeFirst(logs.count - 160)
        }
        Task { await uploadLog(message, level: level) }
    }

    private func uploadLog(_ message: String, level: String) async {
        var request = URLRequest(url: serverBaseURL.appending(path: "/api/logs"))
        request.httpMethod = "POST"
        if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = [
            "session_id": sessionId ?? NSNull(),
            "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
            "source": "ios",
            "level": level,
            "message": message
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        _ = try? await URLSession.shared.data(for: request)
    }

    /// 发送请求，并对「-1005 网络连接被中途断开」自动重试一次。
    /// 这类错误多发生在复用的 keep-alive 连接被服务器/中间层关闭时，请求通常还没到达服务器，
    /// 重试一次即可恢复——正是追问偶发「网络连接失败」的主因（#3）。其它错误原样抛出。
    private func dataWithConnectionLostRetry(for request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await URLSession.shared.data(for: request)
        } catch {
            let nsError = error as NSError
            guard nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorNetworkConnectionLost else {
                throw error
            }
            log("网络连接被中途断开（-1005），自动重试一次：\(request.url?.path ?? "")", level: "warning")
            return try await URLSession.shared.data(for: request)
        }
    }

    private func postForm(path: String, fields: [String: String], files: [MultipartFile]) async throws -> Data {
        let boundary = "pxj-\(UUID().uuidString)"
        var request = URLRequest(url: serverBaseURL.appending(path: path))
        request.httpMethod = "POST"
        // 问答常因「上下文构建+模型生成」较慢（纯文字也可能十几秒~几十秒），放宽超时避免误报「问答失败」
        request.timeoutInterval = files.isEmpty ? 90 : 180
        if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = makeMultipartBody(boundary: boundary, fields: fields, files: files)
        let (data, response) = try await dataWithConnectionLostRetry(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        return data
    }

    private func patchJSON(path: String, payload: [String: Any]) async throws -> Data {
        var request = URLRequest(url: serverBaseURL.appending(path: path))
        request.httpMethod = "PATCH"
        if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        return data
    }

    private func postJSON(path: String, payload: [String: Any], timeout: TimeInterval = 45) async throws -> Data {
        var request = URLRequest(url: serverBaseURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await dataWithConnectionLostRetry(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        return data
    }

    private func getData(path: String) async throws -> Data {
        try await getData(url: serverBaseURL.appending(path: path))
    }

    private func getData(url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 12
        if let auth = AuthSession.shared.authHeader { request.setValue(auth, forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NetworkRequestError.missingHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { AuthSession.shared.handleUnauthorized() }
            let body = String(data: data.prefix(600), encoding: .utf8) ?? ""
            throw NetworkRequestError.badStatus(http.statusCode, body)
        }
        return data
    }

    private func jpegData(_ image: UIImage) throws -> Data {
        let maxSide: CGFloat = 1600
        let size = image.size
        let ratio = min(1, maxSide / max(size.width, size.height))
        let target = CGSize(width: size.width * ratio, height: size.height * ratio)
        let renderer = UIGraphicsImageRenderer(size: target)
        let resized = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: target)) }
        guard let data = resized.jpegData(compressionQuality: 0.78) else {
            throw URLError(.cannotDecodeContentData)
        }
        return data
    }

}

struct MultipartFile {
    let field: String
    let name: String
    let mime: String
    let data: Data
}

enum CameraGesture {
    case point
    case ok
    case victory
}

final class AudioPlaybackFinishDelegate: NSObject, AVAudioPlayerDelegate {
    private var completion: (() -> Void)?

    func setCompletion(_ completion: (() -> Void)?) {
        self.completion = completion
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        finish()
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        finish()
    }

    private func finish() {
        let completion = completion
        self.completion = nil
        completion?()
    }
}

private enum NetworkRequestError: LocalizedError {
    case missingHTTPResponse
    case badStatus(Int, String)

    var errorDescription: String? {
        switch self {
        case .missingHTTPResponse:
            return "服务器未返回 HTTP 响应"
        case .badStatus(let statusCode, let body):
            if body.isEmpty {
                return "服务器返回 HTTP \(statusCode)"
            }
            return "服务器返回 HTTP \(statusCode)：\(body)"
        }
    }
}

private func networkErrorDescription(_ error: Error) -> String {
    let nsError = error as NSError
    var parts = [error.localizedDescription]
    parts.append("domain=\(nsError.domain)")
    parts.append("code=\(nsError.code)")

    if let failingURL = nsError.userInfo[NSURLErrorFailingURLErrorKey] as? URL {
        parts.append("url=\(failingURL.absoluteString)")
    } else if let failingURLString = nsError.userInfo[NSURLErrorFailingURLStringErrorKey] as? String {
        parts.append("url=\(failingURLString)")
    }

    if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? NSError {
        parts.append("underlying=\(underlying.domain) \(underlying.code) \(underlying.localizedDescription)")
    }

    return parts.joined(separator: "；")
}

private func networkErrorUserMessage(_ error: Error) -> String {
    let nsError = error as NSError
    if nsError.domain == NSURLErrorDomain {
        switch nsError.code {
        case NSURLErrorSecureConnectionFailed,
             NSURLErrorServerCertificateHasBadDate,
             NSURLErrorServerCertificateUntrusted,
             NSURLErrorServerCertificateHasUnknownRoot,
             NSURLErrorServerCertificateNotYetValid:
            return "网络安全连接失败，暂时没有拿到 AI 回复。请换个网络或稍后重试。"
        case NSURLErrorTimedOut:
            return "网络超时，暂时没有拿到 AI 回复。请稍后重试。"
        case NSURLErrorNotConnectedToInternet,
             NSURLErrorNetworkConnectionLost,
             NSURLErrorCannotConnectToHost,
             NSURLErrorCannotFindHost,
             NSURLErrorDNSLookupFailed:
            return "网络连接失败，暂时没有拿到 AI 回复。请检查网络后重试。"
        default:
            break
        }
    }

    if let requestError = error as? NetworkRequestError {
        switch requestError {
        case .badStatus(let statusCode, _):
            return "服务器返回 HTTP \(statusCode)，问题暂时没有完成处理。"
        case .missingHTTPResponse:
            return "服务器没有返回有效响应，暂时没有拿到 AI 回复。"
        }
    }
    return "网络请求失败，暂时没有拿到 AI 回复。请稍后重试。"
}

private func runtimeTaskDetail(from detail: String, fallback: String? = nil) -> String {
    let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        return fallback ?? "状态更新中"
    }
    if trimmed.contains("NSURLErrorDomain") ||
        trimmed.contains("CFNetwork") ||
        trimmed.contains("domain=") ||
        trimmed.contains("code=") ||
        trimmed.contains("url=http") ||
        trimmed.contains("TLS error") {
        return fallback ?? "网络连接失败，暂时没有拿到 AI 回复。"
    }
    return trimmed
}

private func firstNonEmpty(_ values: String...) -> String {
    values
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .first { !$0.isEmpty } ?? ""
}

private func shortText(_ text: String, limit: Int) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count > limit else { return trimmed }
    return String(trimmed.prefix(max(0, limit - 1))) + "…"
}

private func cleanLogText(_ text: String) -> String {
    text.replacingOccurrences(of: #"^\[[^\]]+\]\s*"#, with: "", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private func prettyJSONString(_ value: [String: Any]) -> String {
    guard JSONSerialization.isValidJSONObject(value),
          let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

private func makeMultipartBody(boundary: String, fields: [String: String], files: [MultipartFile]) -> Data {
    var body = Data()
    for (key, value) in fields {
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
        body.append("\(value)\r\n")
    }
    for file in files {
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"\(file.field)\"; filename=\"\(file.name)\"\r\n")
        body.append("Content-Type: \(file.mime)\r\n\r\n")
        body.append(file.data)
        body.append("\r\n")
    }
    body.append("--\(boundary)--\r\n")
    return body
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}

// 注：访问级别由 private 放宽为 internal，供 QuestionSegmenter.swift / QuestionRegionOverlay.swift 复用
// `resizedForVision(maxSide:)` 与 `cgImagePropertyOrientation`（行为不变）。
extension UIImage {
    var cgImagePropertyOrientation: CGImagePropertyOrientation {
        switch imageOrientation {
        case .up:
            return .up
        case .upMirrored:
            return .upMirrored
        case .down:
            return .down
        case .downMirrored:
            return .downMirrored
        case .left:
            return .left
        case .leftMirrored:
            return .leftMirrored
        case .right:
            return .right
        case .rightMirrored:
            return .rightMirrored
        @unknown default:
            return .up
        }
    }

    func resizedForVision(maxSide: CGFloat) -> UIImage {
        let longestSide = max(size.width, size.height)
        guard longestSide > maxSide else { return self }
        let ratio = maxSide / longestSide
        let target = CGSize(width: size.width * ratio, height: size.height * ratio)
        let renderer = UIGraphicsImageRenderer(size: target)
        return renderer.image { _ in draw(in: CGRect(origin: .zero, size: target)) }
    }

    func thumbnail(maxSide: CGFloat) -> UIImage {
        let longestSide = max(size.width, size.height)
        guard longestSide > maxSide else { return self }
        let ratio = maxSide / longestSide
        let target = CGSize(width: size.width * ratio, height: size.height * ratio)
        let renderer = UIGraphicsImageRenderer(size: target)
        return renderer.image { _ in draw(in: CGRect(origin: .zero, size: target)) }
    }
}

struct CameraView: UIViewControllerRepresentable {
    @ObservedObject var state: AppState

    func makeUIViewController(context: Context) -> CameraViewController {
        let controller = CameraViewController()
        controller.delegate = context.coordinator
        state.cameraHostDidAttach(id: context.coordinator.id)
        state.captureSingle = { [weak controller] in controller?.capture(kind: .single) }
        state.captureBurstFrame = { [weak controller] in controller?.capture(kind: .burst) ?? false }
        state.captureQAFrame = { [weak controller] in controller?.capture(kind: .qa) ?? false }
        state.captureSegment = { [weak controller] in controller?.capture(kind: .segment) ?? false }
        state.captureExtract = { [weak controller] in controller?.capture(kind: .extract) ?? false }
        state.latestPreviewFrame = { [weak controller] in controller?.currentPreviewFrame() }
        state.setTorch = { [weak controller] on in controller?.setTorch(on) }
        return controller
    }

    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {}

    static func dismantleUIViewController(_ uiViewController: CameraViewController, coordinator: Coordinator) {
        uiViewController.stopCamera()
        Task { @MainActor in
            coordinator.state.cameraDidClose(hostId: coordinator.id)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(state: state)
    }

    final class Coordinator: NSObject, CameraViewControllerDelegate {
        let id = UUID()
        let state: AppState

        init(state: AppState) {
            self.state = state
        }

        func cameraDidOpen() {
            Task { @MainActor in state.cameraDidOpen(hostId: id) }
        }

        func cameraFailed(_ message: String) {
            Task { @MainActor in state.cameraFailed(message, hostId: id) }
        }

        func didCapture(image: UIImage, kind: CaptureKind) {
            Task { @MainActor in
                guard state.isCurrentCameraHost(id: id) else { return }
                switch kind {
                case .single:
                    state.didCaptureSingle(image)
                case .burst:
                    state.didCaptureBurstFrame(image)
                case .qa:
                    state.didCaptureQAFrame(image)
                case .segment:
                    state.didCaptureSegmentationPhoto(image)
                case .extract:
                    state.didCaptureExtractionPhoto(image)
                }
            }
        }

        func didRecognizeGesture(_ gesture: CameraGesture, stableFrames: Int, point: CGPoint?) {
            Task { @MainActor in
                guard state.isCurrentCameraHost(id: id) else { return }
                state.cameraDidRecognizeGesture(gesture, stableFrames: stableFrames, point: point)
            }
        }

        func didUpdateAimQuality(brightness: Double, sharpness: Double, material: Double) {
            Task { @MainActor in
                guard state.isCurrentCameraHost(id: id) else { return }
                state.updateCameraAim(brightness: brightness, sharpness: sharpness, material: material)
            }
        }
    }
}

enum CaptureKind {
    case single
    case burst
    case qa
    case segment   // P0：分题/批改取景后抓拍（全分辨率真照片）
    case extract   // 题目提取取景后抓拍（全分辨率真照片，连拍）
}

protocol CameraViewControllerDelegate: AnyObject {
    func cameraDidOpen()
    func cameraFailed(_ message: String)
    func didCapture(image: UIImage, kind: CaptureKind)
    func didRecognizeGesture(_ gesture: CameraGesture, stableFrames: Int, point: CGPoint?)
    func didUpdateAimQuality(brightness: Double, sharpness: Double, material: Double)  // P0：取景实时质量（亮度/清晰度/是否有题目印刷体）
}

final class CameraViewController: UIViewController, AVCapturePhotoCaptureDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    weak var delegate: CameraViewControllerDelegate?

    private let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoOutputQueue = DispatchQueue(label: "com.pxj.camera-video-output", qos: .userInitiated)
    private let ciContext = CIContext()
    private var captureDevice: AVCaptureDevice?      // P0：留引用以配置对焦/曝光/点按对焦
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var pendingKinds: [Int64: CaptureKind] = [:]
    private var latestVideoImage: UIImage?
    private var latestVideoImageAt: Date?
    private var lastVideoImageUpdateAt = Date.distantPast
    private var stableGesture: CameraGesture?
    private var stableGestureCount = 0
    private var lastGestureFireAt = Date.distantPast
    private var lastMaterialCheckAt = Date.distantPast   // P0：印刷体检测节流
    private var cachedMaterialScore = 0.0

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configure()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updatePreviewGeometry()
    }

    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        coordinator.animate(alongsideTransition: { [weak self] _ in
            self?.updatePreviewGeometry()
        }, completion: { [weak self] _ in
            self?.updatePreviewGeometry()
        })
    }

    func stopCamera() {
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
        if session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [session] in
                session.stopRunning()
            }
        }
        latestVideoImage = nil
        latestVideoImageAt = nil
        pendingKinds.removeAll()
        resetGestureStability()
    }

    /// 当前预览帧（题目提取「自动扫描」端上跟踪用，轻量、不触发快门）。
    func currentPreviewFrame() -> UIImage? {
        guard let latestVideoImage, let latestVideoImageAt,
              Date().timeIntervalSince(latestVideoImageAt) < 2 else { return nil }
        return latestVideoImage
    }

    @discardableResult
    func capture(kind: CaptureKind) -> Bool {
        applyCurrentVideoOrientation()
        if kind == .burst {
            guard let latestVideoImage, let latestVideoImageAt, Date().timeIntervalSince(latestVideoImageAt) < 2 else { return false }
            delegate?.didCapture(image: latestVideoImage, kind: kind)
            return true
        }
        // P0：拍照答题/分题/批改（.qa）与单拍（.single）都走全分辨率 photoOutput 真照片，
        // 不再用低清预览帧——那正是题目分割/梯形校正失败的根因。仅 .burst 智能观察继续用预览帧（求速度）。
        let settings = photoSettings()
        pendingKinds[settings.uniqueID] = kind
        photoOutput.capturePhoto(with: settings, delegate: self)
        return true
    }

    /// P0：高质量照片设置（优先画质、开启高分辨率/最大尺寸）。
    private func photoSettings() -> AVCapturePhotoSettings {
        let settings: AVCapturePhotoSettings
        if photoOutput.availablePhotoCodecTypes.contains(.jpeg) {
            settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        } else {
            settings = AVCapturePhotoSettings()
        }
        settings.photoQualityPrioritization = .quality
        if #available(iOS 16.0, *) {
            settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions
        } else {
            settings.isHighResolutionPhotoEnabled = photoOutput.isHighResolutionCaptureEnabled
        }
        return settings
    }

    /// P0：点按对焦/测光——把对焦点设到用户点的位置。
    func focus(atDevicePoint devicePoint: CGPoint) {
        guard let device = captureDevice else { return }
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = devicePoint
                device.focusMode = device.isFocusModeSupported(.autoFocus) ? .autoFocus : device.focusMode
            }
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = devicePoint
                device.exposureMode = device.isExposureModeSupported(.continuousAutoExposure) ? .continuousAutoExposure : device.exposureMode
            }
            device.unlockForConfiguration()
        } catch {
            // 对焦失败不致命，忽略
        }
    }

    /// P0：开机即配置连续自动对焦/曝光，文档近距离更稳。
    private func configureDeviceFocus(_ device: AVCaptureDevice) {
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isSmoothAutoFocusSupported {
                device.isSmoothAutoFocusEnabled = true
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isAutoFocusRangeRestrictionSupported {
                device.autoFocusRangeRestriction = .none
            }
            device.unlockForConfiguration()
        } catch {
            // 配置失败不致命
        }
    }

    @objc private func handleFocusTap(_ recognizer: UITapGestureRecognizer) {
        guard let previewLayer else { return }
        let point = recognizer.location(in: view)
        let devicePoint = previewLayer.captureDevicePointConverted(fromLayerPoint: point)
        focus(atDevicePoint: devicePoint)
    }

    private func configure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    granted ? self?.setupSession() : self?.delegate?.cameraFailed("用户未授权相机")
                }
            }
        default:
            delegate?.cameraFailed("相机权限不可用")
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        // P0：全分辨率照片预设（拍清楚的地基），而非 .high(~720p)。
        if session.canSetSessionPreset(.photo) {
            session.sessionPreset = .photo
        } else if session.canSetSessionPreset(.high) {
            session.sessionPreset = .high
        }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input),
              session.canAddOutput(photoOutput),
              session.canAddOutput(videoOutput) else {
            delegate?.cameraFailed("无法初始化后置摄像头")
            return
        }
        captureDevice = device
        session.addInput(input)
        session.addOutput(photoOutput)
        if #available(iOS 16.0, *) {
            photoOutput.maxPhotoDimensions = device.activeFormat.supportedMaxPhotoDimensions.last ?? photoOutput.maxPhotoDimensions
        } else {
            photoOutput.isHighResolutionCaptureEnabled = true
        }
        photoOutput.maxPhotoQualityPrioritization = .quality
        configureDeviceFocus(device)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
        session.addOutput(videoOutput)

        applyCurrentVideoOrientation()
        session.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(layer, at: 0)
        previewLayer = layer
        updatePreviewGeometry()
        // P0：点按对焦/测光
        let focusTap = UITapGestureRecognizer(target: self, action: #selector(handleFocusTap(_:)))
        view.addGestureRecognizer(focusTap)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.applyCurrentVideoOrientation()
                self?.delegate?.cameraDidOpen()
            }
        }
    }

    private func updatePreviewGeometry() {
        previewLayer?.frame = view.bounds
        applyCurrentVideoOrientation()
    }

    private func applyCurrentVideoOrientation() {
        let orientation = currentVideoOrientation()
        let connections = [
            previewLayer?.connection,
            videoOutput.connection(with: .video),
            photoOutput.connection(with: .video)
        ]
        for connection in connections {
            guard let connection, connection.isVideoOrientationSupported else { continue }
            connection.videoOrientation = orientation
        }
    }

    private func currentVideoOrientation() -> AVCaptureVideoOrientation {
        if let interfaceOrientation = view.window?.windowScene?.interfaceOrientation {
            switch interfaceOrientation {
            case .landscapeLeft:
                return .landscapeLeft
            case .landscapeRight:
                return .landscapeRight
            default:
                break
            }
        }
        return .landscapeRight
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let kind = pendingKinds.removeValue(forKey: photo.resolvedSettings.uniqueID) ?? .single
        if let error {
            delegate?.cameraFailed(error.localizedDescription)
            return
        }
        guard let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else {
            delegate?.cameraFailed("照片数据不可用")
            return
        }
        delegate?.didCapture(image: image, kind: kind)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = Date()
        guard now.timeIntervalSince(lastVideoImageUpdateAt) >= 0.35 else { return }
        lastVideoImageUpdateAt = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage, scale: UIScreen.main.scale, orientation: .up)
        DispatchQueue.main.async { [weak self] in
            self?.latestVideoImage = image
            self?.latestVideoImageAt = now
        }
        let aim = aimQuality(cgImage: cgImage)
        // P0：印刷体题目检测（书本/试卷/屏幕）——较重，单独节流到 ~0.7s，缓存复用。
        if now.timeIntervalSince(lastMaterialCheckAt) >= 0.7 {
            lastMaterialCheckAt = now
            cachedMaterialScore = materialScore(cgImage: cgImage)
        }
        let material = cachedMaterialScore
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.didUpdateAimQuality(brightness: aim.brightness, sharpness: aim.sharpness, material: material)
        }
        detectGesture(cgImage: cgImage)
    }

    /// P0：画面里是否有「题目印刷体」（书本/试卷/平板屏幕）。用快速文字识别数文本行 + 矩形，
    /// 给出 0..1 置信度——没对准题目就不自动快门、也不去分割。
    private func materialScore(cgImage: CGImage) -> Double {
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .fast
        textRequest.usesLanguageCorrection = false
        textRequest.recognitionLanguages = ["zh-Hans", "en-US"]
        textRequest.minimumTextHeight = 0.02
        let rectRequest = VNDetectRectanglesRequest()
        rectRequest.maximumObservations = 4
        rectRequest.minimumConfidence = 0.4
        rectRequest.minimumSize = 0.2
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        guard (try? handler.perform([textRequest, rectRequest])) != nil else { return 0 }
        let lines = (textRequest.results ?? []).filter { obs in
            (obs.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines).count ?? 0) >= 2
        }.count
        let rects = (rectRequest.results ?? []).filter { $0.confidence >= 0.4 }.count
        // 题目通常有多行文字；3+ 行视为强证据；矩形（纸张/屏幕边框）加成。
        let textScore = min(1.0, Double(lines) / 4.0)
        let rectBonus = rects > 0 ? 0.15 : 0.0
        return min(1.0, textScore + rectBonus)
    }

    /// P0：取景实时质量——平均亮度 + 边缘锐度（清晰度代理）。轻量，跑在视频队列。
    private func aimQuality(cgImage: CGImage) -> (brightness: Double, sharpness: Double) {
        let w = 40, h = 40
        var pixels = [UInt8](repeating: 0, count: w * h)
        let cs = CGColorSpaceCreateDeviceGray()
        pixels.withUnsafeMutableBytes { ptr in
            if let ctx = CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: cs, bitmapInfo: 0) {
                ctx.interpolationQuality = .medium
                ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
        }
        var lum = 0.0
        var edge = 0.0
        var checks = 0
        for y in 2..<(h - 2) {
            for x in 2..<(w - 2) {
                let v = Int(pixels[y * w + x])
                lum += Double(v)
                edge += Double(abs(v - Int(pixels[y * w + x + 1]))) + Double(abs(v - Int(pixels[(y + 1) * w + x])))
                checks += 1
            }
        }
        let brightness = lum / Double(max(1, checks)) / 255.0
        let sharpness = min(1.0, edge / Double(max(1, checks)) / 30.0)
        return (brightness, sharpness)
    }

    func setTorch(_ on: Bool) {
        guard let device = captureDevice, device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            device.torchMode = on ? .on : .off
            device.unlockForConfiguration()
        } catch {
            // 手电失败不致命
        }
    }

    private func detectGesture(cgImage: CGImage) {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        do {
            try VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:]).perform([request])
        } catch {
            resetGestureStability()
            return
        }
        guard let observation = request.results?.first,
              let points = try? observation.recognizedPoints(.all),
              let classified = classifyGesture(points: points) else {
            resetGestureStability()
            return
        }
        DispatchQueue.main.async { [weak self] in
            self?.recordGesture(classified.gesture, point: classified.point)
        }
    }

    private func classifyGesture(points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]) -> (gesture: CameraGesture, point: CGPoint?)? {
        func point(_ name: VNHumanHandPoseObservation.JointName) -> VNRecognizedPoint? {
            guard let value = points[name], value.confidence >= 0.35 else { return nil }
            return value
        }
        guard let wrist = point(.wrist),
              let indexTip = point(.indexTip),
              let indexMCP = point(.indexMCP),
              let middleTip = point(.middleTip),
              let middleMCP = point(.middleMCP),
              let ringTip = point(.ringTip),
              let ringMCP = point(.ringMCP),
              let littleTip = point(.littleTip),
              let littleMCP = point(.littleMCP),
              let thumbTip = point(.thumbTip) else {
            return nil
        }
        let indexExtended = distance(indexTip.location, wrist.location) > distance(indexMCP.location, wrist.location) * 1.18
        let middleExtended = distance(middleTip.location, wrist.location) > distance(middleMCP.location, wrist.location) * 1.15
        let ringExtended = distance(ringTip.location, wrist.location) > distance(ringMCP.location, wrist.location) * 1.05
        let littleExtended = distance(littleTip.location, wrist.location) > distance(littleMCP.location, wrist.location) * 1.05
        let thumbIndexDistance = distance(thumbTip.location, indexTip.location)

        if thumbIndexDistance < 0.08 && middleExtended {
            return (.ok, nil)
        }
        if indexExtended && middleExtended && !ringExtended && !littleExtended {
            return (.victory, nil)
        }
        if indexExtended && !middleExtended && !ringExtended && !littleExtended {
            return (.point, CGPoint(x: indexTip.location.x, y: 1 - indexTip.location.y))
        }
        return nil
    }

    private func distance(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat {
        hypot(lhs.x - rhs.x, lhs.y - rhs.y)
    }

    private func recordGesture(_ gesture: CameraGesture, point: CGPoint?) {
        if stableGesture == gesture {
            stableGestureCount += 1
        } else {
            stableGesture = gesture
            stableGestureCount = 1
        }
        guard stableGestureCount >= 4, Date().timeIntervalSince(lastGestureFireAt) > 2.0 else { return }
        lastGestureFireAt = Date()
        delegate?.didRecognizeGesture(gesture, stableFrames: stableGestureCount, point: point)
    }

    private func resetGestureStability() {
        DispatchQueue.main.async { [weak self] in
            self?.stableGesture = nil
            self?.stableGestureCount = 0
        }
    }
}
