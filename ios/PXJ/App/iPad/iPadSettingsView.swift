import SwiftUI

// iPad 设置：表单式。一句话辅导偏好 / 提问开关 / 提示词 / 学生档案 / 账号。
// 全部复用 AppState 与 AuthSession 现有的已持久化 setter。
// 学习模式/讲解深度 Picker 与 AI 上下文 7 开关已移除（上下文能力归到上下文面板）。

struct iPadSettingsView: View {
    @ObservedObject var state: AppState
    @ObservedObject var auth: AuthSession

    @State private var showAddStudent = false
    @State private var newStudentName = ""

    private var students: [IdentityProfile] {
        auth.profiles.filter { $0.type == "student" }
    }

    var body: some View {
        NavigationStack {
            Form {
                coachPreferenceSection
                questionSection
                voiceSection
                PromptsSettingsSection(state: state)
                studentSection
                accountSection
            }
            .formStyle(.grouped)
            .navigationTitle("设置")
            .navigationBarTitleDisplayMode(.inline)
        }
        .alert("添加学生", isPresented: $showAddStudent) {
            TextField("学生姓名", text: $newStudentName)
            Button("取消", role: .cancel) { newStudentName = "" }
            Button("添加") {
                let name = newStudentName.trimmingCharacters(in: .whitespacesAndNewlines)
                newStudentName = ""
                guard !name.isEmpty else { return }
                Task { try? await auth.addProfile(type: "student", name: name) }
            }
        }
    }

    // 1. 一句话辅导偏好：多行 TextField，失焦/提交时持久化 + 触发策略同步。
    private var coachPreferenceSection: some View {
        Section {
            TextField("例如：先给提示别直接报答案，讲慢一点，多举例子",
                      text: Binding(get: { state.coachPreferenceText }, set: { state.coachPreferenceText = $0 }),
                      axis: .vertical)
                .lineLimit(2...5)
                .submitLabel(.done)
                .onSubmit { state.coachPreferenceTextDidChange(state.coachPreferenceText) }
        } header: {
            Text("一句话辅导偏好")
        } footer: {
            Text("用一句话告诉 AI 你希望它怎么辅导。留空则使用中性默认策略。提交后随上下文发送。")
        }
    }

    // 2. 提问开关
    private var questionSection: some View {
        Section {
            Toggle(isOn: Binding(get: { state.textOnlyQuestion }, set: { state.textOnlyQuestionDidChange($0) })) {
                Label("纯文字提问（不开相机）", systemImage: "keyboard")
            }
            Toggle(isOn: Binding(get: { state.questionSegmentationEnabled }, set: { state.questionSegmentationEnabledDidChange($0) })) {
                Label("多题自动分割", systemImage: "rectangle.split.3x1")
            }
            Toggle(isOn: Binding(get: { state.autoKeystoneCorrection }, set: { state.autoKeystoneCorrectionDidChange($0) })) {
                Label("自动梯形校正", systemImage: "perspective")
            }
        } header: {
            Text("提问")
        } footer: {
            Text("多题自动分割：拍照后把每道题盖一个浮层，点一道只问那道。自动梯形校正：把拍歪的纸张扶正。纯文字提问：打字/快捷追问不开相机。")
        }
    }

    private var voiceSection: some View {
        Section("语音朗读") {
            Toggle(isOn: Binding(get: { state.voicePlaybackEnabled }, set: { state.voicePlaybackEnabledDidChange($0) })) {
                Label("自动朗读回答", systemImage: "speaker.wave.2")
            }
            if state.voicePlaybackEnabled {
                VStack(alignment: .leading) {
                    Text("朗读语速 \(String(format: "%.2f", state.voicePlaybackRate))×")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Slider(value: Binding(get: { state.voicePlaybackRate }, set: { state.voicePlaybackRateDidChange($0) }),
                           in: 0.75...2.0, step: 0.25)
                }
            }
        }
    }

    private var studentSection: some View {
        Section {
            ForEach(students) { profile in
                HStack {
                    Label(profile.name, systemImage: "graduationcap")
                    Spacer()
                    if profile.id == auth.activeStudentId {
                        Text("当前").font(.caption).foregroundStyle(.tint)
                    } else {
                        Button("设为当前") { auth.setActiveStudent(profile.id) }
                            .font(.caption)
                            .buttonStyle(.borderless)
                    }
                }
            }
            .onDelete { indexSet in
                for index in indexSet {
                    let id = students[index].id
                    Task { try? await auth.deleteProfile(id: id) }
                }
            }
            Button {
                showAddStudent = true
            } label: {
                Label("添加学生", systemImage: "plus.circle")
            }
        } header: {
            Text("学生档案")
        }
    }

    private var accountSection: some View {
        Section("账号") {
            LabeledContent("邮箱", value: auth.email)
            if !auth.accountName.isEmpty {
                LabeledContent("账号名", value: auth.accountName)
            }
            Button(role: .destructive) {
                Task { await auth.signOut() }
            } label: {
                Label("退出登录", systemImage: "rectangle.portrait.and.arrow.right")
            }
        }
    }
}

// 3. 提示词：复用 GET /api/prompts 列表；每条只读预览 + 「恢复默认」按钮调 POST /api/prompts/{key}/reset。
// 不新增端点、不做自由编辑。
private struct PromptsSettingsSection: View {
    @ObservedObject var state: AppState

    var body: some View {
        Section {
            if state.isLoadingPrompts && state.coachPrompts.isEmpty {
                HStack {
                    ProgressView()
                    Text("加载提示词…").font(.caption).foregroundStyle(.secondary)
                }
            } else if state.coachPrompts.isEmpty {
                Button {
                    Task { await state.loadCoachPrompts() }
                } label: {
                    Label("加载提示词", systemImage: "arrow.clockwise")
                }
            } else {
                ForEach(state.coachPrompts) { prompt in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(prompt.label).font(.subheadline.weight(.semibold))
                            if prompt.isCustom {
                                Text("已自定义").font(.caption2).foregroundStyle(.tint)
                            }
                            Spacer()
                            if prompt.isCustom {
                                Button("恢复默认") {
                                    Task { await state.resetCoachPrompt(key: prompt.key) }
                                }
                                .font(.caption)
                                .buttonStyle(.borderless)
                            }
                        }
                        if !prompt.description.isEmpty {
                            Text(prompt.description).font(.caption).foregroundStyle(.secondary)
                        }
                        Text(prompt.content)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                    .padding(.vertical, 2)
                }
            }
        } header: {
            Text("提示词")
        } footer: {
            Text("查看 AI 使用的提示词。自定义过的可一键恢复默认。如需编辑请在后台管理台操作。")
        }
        .task { await state.loadCoachPrompts() }
    }
}
