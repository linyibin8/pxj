import SwiftUI

// MARK: - iPad 原生端
// 与 iPhone 共享同一套核心：AppState / AuthSession / 所有数据模型 / 所有 API 调用。
// 这里只重写「容器 + 导航 + 布局」，把 iPhone 的横屏沉浸工作台换成 iPad 原生的
// NavigationSplitView 主从结构（侧栏 + 详情）。每个功能仍然调用 AppState 现有方法，
// 所以后端契约一变，两端自动同步。详见 docs/MULTI_CLIENT_GUIDE.md。

enum iPadSection: String, CaseIterable, Identifiable {
    case learn, history, mistakes, memory, settings
    var id: String { rawValue }

    var title: String {
        switch self {
        case .learn: return "学习"
        case .history: return "历史"
        case .mistakes: return "错题本"
        case .memory: return "记忆"
        case .settings: return "设置"
        }
    }

    var systemImage: String {
        switch self {
        case .learn: return "sparkles"
        case .history: return "clock.arrow.circlepath"
        case .mistakes: return "exclamationmark.bubble"
        case .memory: return "brain.head.profile"
        case .settings: return "gearshape"
        }
    }
}

struct iPadRootView: View {
    @StateObject private var state = AppState()
    @ObservedObject private var auth = AuthSession.shared
    @State private var section: iPadSection?
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var showBackgroundTasks = false   // #8 后台任务面板

    init() {
        #if DEBUG
        if ProcessInfo.processInfo.environment["PXJ_IPAD_COLLAPSE"] == "1" {
            _columnVisibility = State(initialValue: .detailOnly)
        }
        if let s = ProcessInfo.processInfo.environment["PXJ_IPAD_SECTION"],
           let sec = iPadSection(rawValue: s) {
            _section = State(initialValue: sec)
            return
        }
        #endif
        _section = State(initialValue: .learn)
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebar
        } detail: {
            detail
        }
        .navigationSplitViewStyle(.balanced)
        .task {
            state.startDeviceControlPolling()
            await auth.refreshOps()
        }
        .sheet(isPresented: $showBackgroundTasks) {
            iPadBackgroundTasksSheet(state: state, auth: auth) { showBackgroundTasks = false }
        }
        .fullScreenCover(isPresented: Binding(
            get: { state.segmentationVisible },
            set: { if !$0 { state.dismissSegmentation() } }
        )) {
            QuestionSegmentationSheet(state: state)
        }
        .overlay {
            if state.captureAimingVisible {
                QuestionCaptureOverlay(state: state)
            }
        }
        .sheet(isPresented: $state.extractResultVisible) {
            ExtractResultSheet(state: state)
        }
    }

    private var sidebar: some View {
        List(selection: $section) {
            Section {
                ForEach(iPadSection.allCases) { sec in
                    Label(sec.title, systemImage: sec.systemImage)
                        .tag(sec)
                        .accessibilityIdentifier("ipad-section-\(sec.rawValue)")
                }
            } header: {
                iPadStudentSwitcher(auth: auth)
                    .textCase(nil)
                    .padding(.bottom, 4)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("知进拍学")
        .safeAreaInset(edge: .bottom) {
            iPadAccountFooter(state: state, auth: auth) { showBackgroundTasks = true }
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch section ?? .learn {
        case .learn: iPadLearnView(state: state)
        case .history: iPadHistoryView(state: state, onContinue: { section = .learn })
        case .mistakes: iPadMistakesView(state: state, onReview: { section = .learn })
        case .memory: iPadMemoryView(state: state)
        case .settings: iPadSettingsView(state: state, auth: auth)
        }
    }
}

// MARK: - 学生档案切换（侧栏顶部）

struct iPadStudentSwitcher: View {
    @ObservedObject var auth: AuthSession

    private var students: [IdentityProfile] {
        auth.profiles.filter { $0.type == "student" }
    }

    private var activeName: String {
        students.first(where: { $0.id == auth.activeStudentId })?.name
            ?? students.first?.name
            ?? "未选择学生"
    }

    var body: some View {
        Menu {
            ForEach(students) { profile in
                Button {
                    auth.setActiveStudent(profile.id)
                } label: {
                    if profile.id == auth.activeStudentId {
                        Label(profile.name, systemImage: "checkmark")
                    } else {
                        Text(profile.name)
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "person.crop.circle")
                    .font(.title3)
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(activeName)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                    Text("当前学生")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 6)
        }
        .disabled(students.isEmpty)
    }
}

// MARK: - 账号 + 配额（侧栏底部）

struct iPadAccountFooter: View {
    @ObservedObject var state: AppState
    @ObservedObject var auth: AuthSession
    let onOpenTasks: () -> Void

    // #8 指示器文案：本机有正在运行的任务时优先报数量，否则报服务端队列 ETA。
    private var backgroundIndicatorText: String {
        let running = state.runtimeTasks.count
        if running > 0 {
            return "运行中 \(running) 项" + (auth.bgActive && auth.bgEtaSeconds > 0 ? " · 后台约\(auth.bgEtaSeconds)s" : "")
        }
        return "后台生成中" + (auth.bgEtaSeconds > 0 ? " · 约\(auth.bgEtaSeconds)s" : "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            HStack(spacing: 8) {
                Image(systemName: "envelope")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(auth.email.isEmpty ? auth.accountName : auth.email)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            if auth.quotaEnabled {
                HStack(spacing: 6) {
                    Image(systemName: "bolt.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("今日免费额度剩余 \(max(auth.quotaRemaining, 0))/\(auth.quotaLimit)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            // #8 后台/运行中任务：可点击查看在跑的任务并关闭。服务端队列(bgActive)或本机任务(runtimeTasks)任一活跃就显示。
            if auth.bgActive || !state.runtimeTasks.isEmpty {
                Button {
                    onOpenTasks()
                } label: {
                    HStack(spacing: 6) {
                        ProgressView().scaleEffect(0.7)
                        Text(backgroundIndicatorText)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 2)
                        Image(systemName: "chevron.up.circle")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.bar)
    }
}

// MARK: - #8 后台任务面板（看到在跑的任务 + 关闭可取消的任务）
// 数据源：本机 state.runtimeTasks（观察/语音/朗读/上传/报告等，含可关闭项）+ 服务端生成队列（auth.bgAhead/bgEtaSeconds）。
// 服务端队列在模型空闲时自动生成（可视化/报告），会自行完成；本机任务里 canClose 的可一键停止。
struct iPadBackgroundTasksSheet: View {
    @ObservedObject var state: AppState
    @ObservedObject var auth: AuthSession
    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section("正在运行（本机）") {
                    if state.runtimeTasks.isEmpty {
                        Text("本机暂无正在运行的任务。")
                            .font(.callout).foregroundStyle(.secondary)
                    } else {
                        ForEach(state.runtimeTasks) { task in
                            taskRow(task)
                        }
                    }
                }

                Section {
                    if state.serverTasks.isEmpty {
                        Text(auth.bgActive ? "后端正在生成，但当前账号下没有可单独取消的任务。"
                                           : "后端暂无在跑的生成任务。")
                            .font(.callout).foregroundStyle(.secondary)
                    } else {
                        ForEach(state.serverTasks) { task in
                            serverTaskRow(task)
                        }
                    }
                } header: {
                    Text("后端生成任务（本账号 · 可取消）")
                } footer: {
                    Text("可视化/报告/记忆整理在模型空闲时生成；取消只影响你自己账号的任务。"
                         + (auth.bgEtaSeconds > 0 ? "整体预计约 \(auth.bgEtaSeconds) 秒。" : ""))
                }
            }
            .navigationTitle("后台任务")
            .navigationBarTitleDisplayMode(.inline)
            .refreshable { await state.refreshServerTasks() }
            .task { await state.refreshServerTasks() }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { Task { await state.refreshServerTasks() } } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { onClose() }
                }
            }
        }
    }

    private func serverTaskRow(_ task: ServerBackgroundTask) -> some View {
        HStack(alignment: .top, spacing: 12) {
            if task.state == "running" || task.state == "cancelling" {
                ProgressView().frame(width: 22)
            } else {
                Image(systemName: "hourglass").foregroundStyle(.secondary).frame(width: 22)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(task.title).font(.callout.weight(.semibold))
                Text("\(task.stateText) · 已 \(task.ageSeconds) 秒")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(role: .destructive) {
                Task { await state.cancelServerTask(task.id) }
            } label: {
                Label("取消", systemImage: "xmark.circle")
                    .labelStyle(.iconOnly).font(.title3)
            }
            .buttonStyle(.borderless)
            .disabled(task.state == "cancelling")
        }
        .padding(.vertical, 2)
    }

    private func taskRow(_ task: RuntimeTaskItem) -> some View {
        HStack(alignment: .top, spacing: 12) {
            if task.showsProgress {
                ProgressView().frame(width: 22)
            } else {
                Image(systemName: task.systemImage).foregroundStyle(.tint).frame(width: 22)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(task.title).font(.callout.weight(.semibold))
                if !task.detail.isEmpty {
                    Text(task.detail).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            if task.canClose {
                Button(role: .destructive) {
                    state.closeRuntimeTask(task.id)
                } label: {
                    Label(task.closeTitle, systemImage: task.closeSystemImage)
                        .labelStyle(.iconOnly)
                        .font(.title3)
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.vertical, 2)
    }
}
