import Foundation
import Security
import SwiftUI

// Shared server base (kept in sync with ContentView.serverBaseURL)
let authServerBaseURL = URL(string: "https://pxj.evowit.com")!

// MARK: - Keychain (stores the JWT access token)

enum AuthKeychain {
    private static let service = "com.linyibin8.pxj.auth"
    private static let account = "access_token"

    static func save(_ token: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        var add = query
        add[kSecValueData as String] = Data(token.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(add as CFDictionary, nil)
    }

    static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let token = String(data: data, encoding: .utf8) else { return nil }
        return token
    }

    static func clear() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - Identity profile

struct IdentityProfile: Identifiable, Hashable {
    let id: String
    let type: String   // student / parent / teacher
    let name: String

    init?(_ json: [String: Any]) {
        guard let id = json["id"] as? String else { return nil }
        self.id = id
        self.type = (json["profile_type"] as? String) ?? "student"
        self.name = (json["display_name"] as? String) ?? ""
    }

    var typeLabel: String {
        switch type {
        case "parent": return "家长"
        case "teacher": return "老师"
        default: return "学生"
        }
    }
}

enum AuthError: LocalizedError {
    case network
    case badResponse
    case server(Int, String)

    var errorDescription: String? {
        switch self {
        case .network: return "网络错误，请检查连接后重试"
        case .badResponse: return "服务器返回异常"
        case .server(_, let message): return message
        }
    }
}

// MARK: - Auth session (single source of truth for login state)

@MainActor
final class AuthSession: ObservableObject {
    static let shared = AuthSession()

    @Published private(set) var token: String?
    @Published private(set) var email = ""
    @Published private(set) var accountName = ""
    @Published private(set) var profiles: [IdentityProfile] = []
    @Published private(set) var activeStudentId = ""
    @Published private(set) var isAuthenticated = false
    @Published private(set) var bootstrapping = true

    // Free-tier quota (today, on our default model)
    @Published private(set) var quotaEnabled = false
    @Published private(set) var quotaUsed = 0
    @Published private(set) var quotaLimit = 0
    @Published private(set) var quotaRemaining = -1
    // Background generation status (visualization/report) from the LLM gate
    @Published private(set) var bgActive = false
    @Published private(set) var bgAhead = 0
    @Published private(set) var bgEtaSeconds = 0
    @Published private(set) var bgRealtimeBusy = false

    private let activeStudentKey = "pxj.activeStudentId"
    private var opsPolling = false

    private init() {}

    var quotaText: String? {
        guard quotaEnabled, quotaRemaining >= 0 else { return nil }
        return "今日剩余 \(quotaRemaining) 次"
    }
    var generationStatusText: String? {
        guard bgActive else { return nil }
        if bgRealtimeBusy { return "生成已让行：语音/问答优先，空闲后自动继续" }
        let eta = bgEtaSeconds >= 90 ? "约 \(bgEtaSeconds / 60) 分钟" : "约 \(max(1, bgEtaSeconds)) 秒"
        return "后台生成中：前方 \(bgAhead) 个任务，预计还需\(eta)"
    }

    var authHeader: String? { token.map { "Bearer \($0)" } }
    var studentProfiles: [IdentityProfile] { profiles.filter { $0.type == "student" } }
    var activeStudentName: String {
        profiles.first(where: { $0.id == activeStudentId })?.name ?? ""
    }

    func bootstrap() async {
        activeStudentId = UserDefaults.standard.string(forKey: activeStudentKey) ?? ""
        if let saved = AuthKeychain.load(), !saved.isEmpty {
            token = saved
            isAuthenticated = true
            await refreshMe()
            if isAuthenticated { startOpsPolling() }
        }
        bootstrapping = false
    }

    func refreshMe() async {
        guard let token else { return }
        var request = URLRequest(url: authServerBaseURL.appending(path: "/api/auth/me"))
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 15
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return }
            if http.statusCode == 401 {
                await signOut()
                return
            }
            if http.statusCode == 200,
               let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                apply(json)
                isAuthenticated = true
            }
        } catch {
            // offline: keep existing token/session
        }
    }

    private func apply(_ json: [String: Any]) {
        if let user = json["user"] as? [String: Any] {
            email = (user["email"] as? String) ?? email
        }
        if let account = json["account"] as? [String: Any] {
            accountName = (account["name"] as? String) ?? accountName
        }
        if let rawProfiles = json["profiles"] as? [[String: Any]] {
            profiles = rawProfiles.compactMap { IdentityProfile($0) }
            let students = profiles.filter { $0.type == "student" }
            if activeStudentId.isEmpty || !students.contains(where: { $0.id == activeStudentId }) {
                setActiveStudent(students.first?.id ?? "")
            }
        }
    }

    func setActiveStudent(_ id: String) {
        activeStudentId = id
        UserDefaults.standard.set(id, forKey: activeStudentKey)
    }

    func login(email: String, password: String) async throws {
        try await submit(path: "/api/auth/login", body: ["email": email, "password": password])
    }

    func register(email: String, password: String, studentName: String) async throws {
        var body: [String: Any] = ["email": email, "password": password]
        let trimmed = studentName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { body["student_name"] = trimmed }
        try await submit(path: "/api/auth/register", body: body)
    }

    private func submit(path: String, body: [String: Any]) async throws {
        var request = URLRequest(url: authServerBaseURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthError.network
        }
        guard let http = response as? HTTPURLResponse else { throw AuthError.network }
        guard (200..<300).contains(http.statusCode) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
            throw AuthError.server(http.statusCode, detail ?? "请求失败（\(http.statusCode)）")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let newToken = json["access_token"] as? String else {
            throw AuthError.badResponse
        }
        token = newToken
        AuthKeychain.save(newToken)
        apply(json)
        isAuthenticated = true
        startOpsPolling()
    }

    @discardableResult
    private func api(_ method: String, _ path: String, json: [String: Any]? = nil) async throws -> Any? {
        guard let token else { throw AuthError.network }
        var req = URLRequest(url: authServerBaseURL.appending(path: path))
        req.httpMethod = method
        req.timeoutInterval = 20
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let json {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: json)
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw AuthError.network }
        if http.statusCode == 401 { await signOut(); throw AuthError.server(401, "登录已过期，请重新登录") }
        guard (200..<300).contains(http.statusCode) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
            throw AuthError.server(http.statusCode, detail ?? "请求失败（\(http.statusCode)）")
        }
        return try? JSONSerialization.jsonObject(with: data)
    }

    private func applyProfilesResponse(_ r: Any?) {
        guard let j = r as? [String: Any], let raw = j["profiles"] as? [[String: Any]] else { return }
        profiles = raw.compactMap { IdentityProfile($0) }
        let students = profiles.filter { $0.type == "student" }
        if activeStudentId.isEmpty || !students.contains(where: { $0.id == activeStudentId }) {
            setActiveStudent(students.first?.id ?? "")
        }
    }

    func addProfile(type: String, name: String) async throws {
        applyProfilesResponse(try await api("POST", "/api/profiles", json: ["profile_type": type, "display_name": name]))
    }
    func renameProfile(id: String, name: String) async throws {
        applyProfilesResponse(try await api("PATCH", "/api/profiles/\(id)", json: ["display_name": name]))
    }
    func deleteProfile(id: String) async throws {
        applyProfilesResponse(try await api("DELETE", "/api/profiles/\(id)"))
    }

    func refreshOps() async {
        guard isAuthenticated, let j = (try? await api("GET", "/api/observability")) as? [String: Any] else { return }
        if let q = j["quota"] as? [String: Any] {
            quotaEnabled = (q["enabled"] as? Bool) ?? false
            quotaUsed = (q["used"] as? Int) ?? 0
            quotaLimit = (q["limit"] as? Int) ?? 0
            quotaRemaining = (q["remaining"] as? Int) ?? -1
        }
        let usage = j["llm_usage"] as? [String: Any]
        if let g = j["llm_gate"] as? [String: Any] {
            let ahead = ((g["waiting_background"] as? Int) ?? 0) + ((g["inflight_background"] as? Int) ?? 0)
            let realtimeBusy = (((g["inflight_realtime"] as? Int) ?? 0) + ((g["waiting_realtime"] as? Int) ?? 0)) > 0
            let avgSec = max(20, Int(((usage?["avg_duration_ms"] as? Double) ?? 0) / 1000.0))
            bgAhead = ahead
            bgRealtimeBusy = realtimeBusy
            bgActive = ahead > 0
            bgEtaSeconds = (ahead + 1) * avgSec
        }
    }

    func startOpsPolling() {
        guard !opsPolling else { return }
        opsPolling = true
        Task { @MainActor [weak self] in
            while let self, self.isAuthenticated {
                await self.refreshOps()
                try? await Task.sleep(nanoseconds: 8_000_000_000)
            }
            self?.opsPolling = false
        }
    }

    func signOut() async {
        token = nil
        isAuthenticated = false
        profiles = []
        email = ""
        accountName = ""
        quotaEnabled = false
        bgActive = false
        AuthKeychain.clear()
    }

    /// Called by the networking layer on a 401 to force re-login.
    func handleUnauthorized() {
        Task { await signOut() }
    }
}

// MARK: - Login / register gate

struct AuthGateView: View {
    @ObservedObject private var auth = AuthSession.shared
    @State private var isRegister = false
    @State private var email = ""
    @State private var password = ""
    @State private var studentName = ""
    @State private var busy = false
    @State private var errorText = ""

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.12, blue: 0.18), Color(red: 0.02, green: 0.07, blue: 0.12)],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 18) {
                    VStack(spacing: 6) {
                        Text("知进拍学")
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                        Text(isRegister ? "创建账号 · 开启学习陪伴" : "登录 · 继续你的学习")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.75))
                    }
                    .padding(.top, 40)

                    VStack(spacing: 12) {
                        field(title: "邮箱", text: $email, keyboard: .emailAddress)
                        secureField(title: "密码（至少 8 位）", text: $password)
                        if isRegister {
                            field(title: "学生姓名（可选）", text: $studentName, keyboard: .default)
                        }

                        if !errorText.isEmpty {
                            Text(errorText)
                                .font(.footnote)
                                .foregroundStyle(.red)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        Button(action: submit) {
                            HStack {
                                if busy { ProgressView().tint(.white) }
                                Text(isRegister ? "注册并进入" : "登录")
                                    .fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(busy || email.isEmpty || password.count < 8)

                        Button(isRegister ? "已有账号？去登录" : "没有账号？去注册") {
                            withAnimation { isRegister.toggle(); errorText = "" }
                        }
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.8))
                    }
                    .padding(18)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal, 22)

                    Text("pxj.evowit.com")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.4))
                }
            }
        }
    }

    private func field(title: String, text: Binding<String>, keyboard: UIKeyboardType) -> some View {
        TextField("", text: text, prompt: Text(title).foregroundColor(.white.opacity(0.5)))
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .keyboardType(keyboard)
            .foregroundStyle(.white)
            .padding(12)
            .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
    }

    private func secureField(title: String, text: Binding<String>) -> some View {
        SecureField("", text: text, prompt: Text(title).foregroundColor(.white.opacity(0.5)))
            .foregroundStyle(.white)
            .padding(12)
            .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
    }

    private func submit() {
        busy = true
        errorText = ""
        Task {
            do {
                if isRegister {
                    try await auth.register(email: email.trimmingCharacters(in: .whitespaces), password: password, studentName: studentName)
                } else {
                    try await auth.login(email: email.trimmingCharacters(in: .whitespaces), password: password)
                }
            } catch {
                errorText = error.localizedDescription
            }
            busy = false
        }
    }
}

// MARK: - Account badge (identity switch + logout) shown in the workbench

struct AccountBadge: View {
    @ObservedObject private var auth = AuthSession.shared
    @State private var showIdentities = false

    var body: some View {
        Menu {
            if !auth.email.isEmpty {
                Section(auth.accountName.isEmpty ? auth.email : "\(auth.accountName) · \(auth.email)") {
                    let students = auth.studentProfiles
                    ForEach(students) { profile in
                        Button {
                            auth.setActiveStudent(profile.id)
                        } label: {
                            Label(profile.name.isEmpty ? "学生" : profile.name,
                                  systemImage: profile.id == auth.activeStudentId ? "checkmark.circle.fill" : "person.circle")
                        }
                    }
                }
            }
            if let q = auth.quotaText {
                Section { Label(q, systemImage: "bolt.horizontal.circle") }
            }
            Button { showIdentities = true } label: {
                Label("管理学生 / 家长 / 老师", systemImage: "person.2.badge.gearshape")
            }
            Button(role: .destructive) {
                Task { await auth.signOut() }
            } label: {
                Label("退出登录", systemImage: "rectangle.portrait.and.arrow.right")
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "person.crop.circle")
                let title = auth.activeStudentName.isEmpty ? (auth.email.isEmpty ? "账号" : auth.email) : auth.activeStudentName
                Text(title).lineLimit(1)
                if let q = auth.quotaText {
                    Text(q).foregroundStyle(auth.quotaRemaining <= 10 ? .orange : .secondary)
                }
            }
            .font(.caption.weight(.medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
        .sheet(isPresented: $showIdentities) { IdentityManagerView() }
    }
}

// MARK: - Identity management (students / parents / teachers)

struct IdentityManagerView: View {
    @ObservedObject private var auth = AuthSession.shared
    @Environment(\.dismiss) private var dismiss
    @State private var newType = "student"
    @State private var newName = ""
    @State private var busy = false
    @State private var errorText = ""

    private let types: [(String, String)] = [("student", "学生"), ("parent", "家长"), ("teacher", "老师")]

    var body: some View {
        NavigationView {
            Form {
                if !errorText.isEmpty {
                    Section { Text(errorText).foregroundStyle(.red).font(.footnote) }
                }
                ForEach(types, id: \.0) { (type, label) in
                    let items = auth.profiles.filter { $0.type == type }
                    Section("\(label)（\(items.count)）") {
                        ForEach(items) { p in
                            HStack {
                                if type == "student" {
                                    Button {
                                        auth.setActiveStudent(p.id)
                                    } label: {
                                        Image(systemName: p.id == auth.activeStudentId ? "checkmark.circle.fill" : "circle")
                                    }.buttonStyle(.plain)
                                }
                                Text(p.name.isEmpty ? label : p.name)
                                Spacer()
                                Menu {
                                    Button("重命名") { rename(p) }
                                    Button("删除", role: .destructive) { remove(p) }
                                } label: { Image(systemName: "ellipsis.circle") }
                            }
                        }
                    }
                }
                Section("新增身份") {
                    Picker("类型", selection: $newType) {
                        ForEach(types, id: \.0) { Text($0.1).tag($0.0) }
                    }.pickerStyle(.segmented)
                    TextField("姓名", text: $newName)
                    Button {
                        add()
                    } label: {
                        HStack { if busy { ProgressView() }; Text("添加") }
                    }.disabled(busy || newName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .navigationTitle("身份管理")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } } }
            .task { await auth.refreshOps() }
        }
    }

    private func run(_ op: @escaping () async throws -> Void) {
        busy = true; errorText = ""
        Task { do { try await op() } catch { errorText = error.localizedDescription }; busy = false }
    }
    private func add() {
        let name = newName.trimmingCharacters(in: .whitespaces)
        run { try await auth.addProfile(type: newType, name: name); await MainActor.run { newName = "" } }
    }
    private func remove(_ p: IdentityProfile) { run { try await auth.deleteProfile(id: p.id) } }
    private func rename(_ p: IdentityProfile) {
        // simple inline rename via prompt-less flow: reuse newName field value if set, else append marker
        let name = newName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { errorText = "请在下方「姓名」框输入新名字后再点重命名"; return }
        run { try await auth.renameProfile(id: p.id, name: name); await MainActor.run { newName = "" } }
    }
}

// MARK: - First-run onboarding

struct OnboardingView: View {
    var onDone: () -> Void
    private let steps: [(String, String, String)] = [
        ("camera.viewfinder", "拍题解析", "对准课本/试卷拍一张，AI 帮你识别题目并分步讲解。"),
        ("mic.circle", "语音问答", "按住说话直接问，支持连续追问，语音优先实时响应。"),
        ("rectangle.stack.badge.person.crop", "智能观察 & 错题本", "固定机位自动观察学习过程，错题与知识点自动沉淀，按艾宾浩斯复习。"),
    ]
    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.05, green: 0.12, blue: 0.18), Color(red: 0.02, green: 0.07, blue: 0.12)],
                           startPoint: .topLeading, endPoint: .bottomTrailing).ignoresSafeArea()
            VStack(spacing: 22) {
                Text("欢迎使用 知进拍学").font(.title.bold()).foregroundStyle(.white).padding(.top, 36)
                ForEach(steps, id: \.0) { (icon, title, desc) in
                    HStack(alignment: .top, spacing: 14) {
                        Image(systemName: icon).font(.title2).foregroundStyle(.cyan).frame(width: 34)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(title).font(.headline).foregroundStyle(.white)
                            Text(desc).font(.subheadline).foregroundStyle(.white.opacity(0.78))
                        }
                        Spacer()
                    }.padding(.horizontal, 26)
                }
                Spacer()
                Button(action: onDone) {
                    Text("开始使用").fontWeight(.semibold).frame(maxWidth: .infinity).padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal, 26).padding(.bottom, 30)
            }
        }
    }
}

// MARK: - Background generation banner (queue / ETA, voice-priority)

struct GenerationBanner: View {
    @ObservedObject private var auth = AuthSession.shared
    var body: some View {
        if let text = auth.generationStatusText {
            HStack(spacing: 6) {
                Image(systemName: "hourglass")
                Text(text).lineLimit(2)
            }
            .font(.caption2)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
            .transition(.opacity)
        }
    }
}
