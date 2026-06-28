import SwiftUI

@main
struct PXJApp: App {
    @StateObject private var auth = AuthSession.shared
    @AppStorage("pxj.onboarded") private var onboarded = false

    var body: some Scene {
        WindowGroup {
            Group {
                if auth.bootstrapping {
                    ZStack {
                        Color.black.ignoresSafeArea()
                        ProgressView().tint(.white)
                    }
                } else if auth.isAuthenticated {
                    Group {
                        if UIDevice.current.userInterfaceIdiom == .pad {
                            iPadRootView()
                        } else {
                            ContentView()
                        }
                    }
                    .fullScreenCover(isPresented: Binding(get: { !onboarded }, set: { if !$0 { onboarded = true } })) {
                        OnboardingView { onboarded = true }
                    }
                } else {
                    AuthGateView()
                }
            }
            .task {
                await auth.bootstrap()
                #if DEBUG
                // 仅 DEBUG：模拟器自动登录，便于截图验证（Release/TestFlight 不含此代码）
                if !auth.isAuthenticated,
                   let e = ProcessInfo.processInfo.environment["PXJ_AUTOLOGIN_EMAIL"],
                   let p = ProcessInfo.processInfo.environment["PXJ_AUTOLOGIN_PW"] {
                    onboarded = true
                    try? await auth.login(email: e, password: p)
                }
                #endif
            }
        }
    }
}

