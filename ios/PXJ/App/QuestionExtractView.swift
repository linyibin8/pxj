import SwiftUI
import WebKit
import UIKit

/// 题目提取的还原页 / 空白卷：服务端渲染 HTML，iOS 用同一个 WKWebView 叶子显示（两端复用）。
struct RestorePageWebView: UIViewRepresentable {
    let html: String

    func makeUIView(context: Context) -> WKWebView {
        let web = WKWebView(frame: .zero)
        web.isOpaque = true
        web.backgroundColor = .systemBackground
        web.scrollView.contentInsetAdjustmentBehavior = .always
        return web
    }

    func updateUIView(_ web: WKWebView, context: Context) {
        web.loadHTMLString(html, baseURL: nil)
    }
}

/// 题目提取结果 sheet：还原页 + 「打印空白卷」（系统打印面板自带「存储为 PDF / 共享」）。两端共用。
struct ExtractResultSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var printing = false

    var body: some View {
        NavigationStack {
            Group {
                if let html = state.extractRestoreHTML, !html.isEmpty {
                    RestorePageWebView(html: html)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 40)).foregroundStyle(.secondary)
                        Text("还没有题目").font(.headline)
                        Text("把试卷拍清楚、铺满取景框再试一次。")
                            .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding()
                }
            }
            .navigationTitle("题目还原（去重 \(state.extractUniqueCount) 道）")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("完成") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        printBlankPaper()
                    } label: {
                        if printing { ProgressView() } else { Label("打印空白卷", systemImage: "printer") }
                    }
                    .disabled(printing || state.extractUniqueCount == 0)
                }
            }
        }
    }

    private func printBlankPaper() {
        printing = true
        Task {
            let html = await state.fetchRestoreHTML(view: "blank")
            await MainActor.run {
                printing = false
                guard let html, !html.isEmpty else { return }
                BlankPaperPrinter.shared.print(html: html)
            }
        }
    }
}

/// 把空白卷 HTML 载入离屏 WKWebView，等渲染（含网络缩略图）完成后调用系统打印面板。
/// iPad 必须提供 popover 锚点，否则崩溃——这里用 key window 根视图兜底。
final class BlankPaperPrinter: NSObject, WKNavigationDelegate {
    static let shared = BlankPaperPrinter()
    private var webView: WKWebView?

    func print(html: String) {
        let web = WKWebView(frame: CGRect(x: 0, y: 0, width: 595, height: 842)) // A4 @72dpi
        web.navigationDelegate = self
        webView = web
        web.loadHTMLString(html, baseURL: nil)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        // 给网络缩略图一点加载时间，再出打印面板。
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
            self?.presentPrint(webView)
        }
    }

    private func presentPrint(_ webView: WKWebView) {
        let controller = UIPrintInteractionController.shared
        let info = UIPrintInfo(dictionary: nil)
        info.outputType = .grayscale
        info.jobName = "空白练习卷"
        controller.printInfo = info
        controller.printFormatter = webView.viewPrintFormatter()

        guard let anchorView = Self.topView() else {
            controller.present(animated: true) { [weak self] _, _, _ in self?.webView = nil }
            return
        }
        // iPad 用 popover 锚点；iPhone 该 API 也兼容。
        controller.present(from: CGRect(x: anchorView.bounds.midX, y: anchorView.bounds.midY, width: 1, height: 1),
                           in: anchorView, animated: true) { [weak self] _, _, _ in
            self?.webView = nil
        }
    }

    private static func topView() -> UIView? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let window = scenes.flatMap { $0.windows }.first(where: { $0.isKeyWindow }) ?? scenes.first?.windows.first
        return window?.rootViewController?.view
    }
}
