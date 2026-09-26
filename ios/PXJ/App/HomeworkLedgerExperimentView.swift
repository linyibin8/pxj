import AVFoundation
import CoreImage
import CoreMedia
import SwiftUI
import UIKit
import WebKit

enum HomeworkLedgerPhase: Equatable {
    case idle
    case observing
    case syncing
    case ready
    case failed(String)
}

struct HomeworkLedgerKeyframe: Identifiable {
    let id: String
    let image: UIImage
    let capturedAt: Date
    let sequenceIndex: Int
    let sharpness: Double
    let brightness: Double
    let contrast: Double
    let fingerprint: UInt64
    let reason: String
    var pageEpisodeId: String = ""
    var candidateCount: Int = 0
}

struct HomeworkLedgerEvidence: Identifiable {
    let id: String
    let pageEpisodeId: String
    let canonicalRect: CGRect
    let bestFrameId: String
    let cropImage: UIImage
    let cropHash: UInt64
    let layoutKey: String
    let cropKind: String
    let confidence: Double
    let seenCount: Int
    let sourceFrameIds: [String]
    let mergeReasons: [String]
}

private struct HomeworkLedgerEvidenceCandidate {
    var pageEpisodeId: String
    var canonicalRect: CGRect
    var bestFrameId: String
    var cropImage: UIImage
    var cropHash: UInt64
    var layoutKey: String
    var cropKind: String
    var confidence: Double
    var seenCount: Int
    var sourceFrameIds: [String]
    var mergeReasons: [String]
    var score: Double

    mutating func merge(with other: HomeworkLedgerEvidenceCandidate) {
        seenCount += other.seenCount
        sourceFrameIds = Array(Set(sourceFrameIds + other.sourceFrameIds)).sorted()
        var reasons = Set(mergeReasons)
        reasons.formUnion(other.mergeReasons)
        reasons.insert("duplicate_layout_or_overlap")
        if other.score > score {
            canonicalRect = other.canonicalRect
            bestFrameId = other.bestFrameId
            cropImage = other.cropImage
            cropHash = other.cropHash
            confidence = max(confidence, other.confidence)
            score = other.score
            reasons.insert("best_evidence_replaced")
        }
        mergeReasons = Array(reasons).sorted()
    }

    func evidence(id: String) -> HomeworkLedgerEvidence {
        HomeworkLedgerEvidence(
            id: id,
            pageEpisodeId: pageEpisodeId,
            canonicalRect: canonicalRect,
            bestFrameId: bestFrameId,
            cropImage: cropImage,
            cropHash: cropHash,
            layoutKey: layoutKey,
            cropKind: cropKind,
            confidence: confidence,
            seenCount: seenCount,
            sourceFrameIds: sourceFrameIds,
            mergeReasons: mergeReasons
        )
    }
}

struct HomeworkLedgerExperimentView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var state = HomeworkLedgerExperimentState()

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                HomeworkLedgerCameraView(state: state)
                    .ignoresSafeArea(edges: .bottom)

                VStack(spacing: 10) {
                    statusPanel
                    controlBar
                }
                .padding()
            }
            .navigationTitle("作业账本实验")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") {
                        state.stopObservingLocally()
                        dismiss()
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    if state.htmlURL != nil {
                        Button {
                            state.resultVisible = true
                        } label: {
                            Label("结果", systemImage: "doc.richtext")
                        }
                    }
                }
            }
            .sheet(isPresented: $state.resultVisible) {
                HomeworkLedgerResultSheet(state: state)
            }
        }
    }

    private var statusPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(state.phaseTitle, systemImage: state.phaseSystemImage)
                    .font(.headline)
                Spacer()
                Text(state.elapsedText)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3), spacing: 8) {
                metric("预览", "\(state.previewFrames)")
                metric("关键帧", "\(state.keyframes.count)")
                metric("证据", "\(state.evidence.count)")
                metric("跳过", "\(state.skippedFrames)")
                metric("上传", state.uploadRatioText)
                metric("题卡", "\(state.cardCount)")
            }

            if !state.notice.isEmpty {
                Text(state.notice)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(12)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(.separator), lineWidth: 1))
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(.systemBackground).opacity(0.78))
        .clipShape(RoundedRectangle(cornerRadius: 7))
    }

    private var controlBar: some View {
        HStack(spacing: 10) {
            Button {
                state.reset()
            } label: {
                Label("重置", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.bordered)
            .disabled(state.phase == .observing || state.phase == .syncing)

            Button {
                if state.phase == .observing {
                    state.finishAndSync()
                } else {
                    state.startObserving()
                }
            } label: {
                Label(state.phase == .observing ? "停止并汇总" : "开始观察", systemImage: state.phase == .observing ? "stop.circle" : "record.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(state.phase == .syncing || !state.cameraReady)

            Button {
                state.resultVisible = true
            } label: {
                Image(systemName: "doc.text.magnifyingglass")
                    .frame(width: 34, height: 34)
            }
            .buttonStyle(.bordered)
            .disabled(state.htmlURL == nil)
            .accessibilityLabel("查看账本结果")
        }
        .controlSize(.large)
    }
}

private struct HomeworkLedgerResultSheet: View {
    @ObservedObject var state: HomeworkLedgerExperimentState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let url = state.htmlURL {
                    HomeworkLedgerRemoteWebView(url: url, authorization: state.resultAuthHeader)
                } else if let html = state.resultHTML, !html.isEmpty {
                    RestorePageWebView(html: html)
                } else {
                    ContentUnavailableCompat(
                        title: "还没有账本",
                        systemImage: "doc.text.magnifyingglass",
                        message: "开始观察并停止汇总后，会在这里显示原图、证据裁剪和数字版对比。"
                    )
                }
            }
            .navigationTitle("作业题目对比")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}

private struct HomeworkLedgerRemoteWebView: UIViewRepresentable {
    let url: URL
    let authorization: String?

    final class Coordinator {
        var loadedURL: URL?
        var loadedAuthorization: String?
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> WKWebView {
        let web = WKWebView(frame: .zero)
        web.scrollView.contentInsetAdjustmentBehavior = .always
        return web
    }

    func updateUIView(_ web: WKWebView, context: Context) {
        guard context.coordinator.loadedURL != url || context.coordinator.loadedAuthorization != authorization else {
            return
        }
        var request = URLRequest(url: url)
        if let authorization = authorization, !authorization.isEmpty {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }
        context.coordinator.loadedURL = url
        context.coordinator.loadedAuthorization = authorization
        web.load(request)
    }
}

private struct HomeworkLedgerCameraView: UIViewControllerRepresentable {
    @ObservedObject var state: HomeworkLedgerExperimentState

    func makeUIViewController(context: Context) -> HomeworkLedgerCameraController {
        let controller = HomeworkLedgerCameraController()
        controller.delegate = context.coordinator
        state.previewFrameProvider = { [weak controller] in controller?.currentPreviewFrame() }
        return controller
    }

    func updateUIViewController(_ uiViewController: HomeworkLedgerCameraController, context: Context) {}

    static func dismantleUIViewController(_ uiViewController: HomeworkLedgerCameraController, coordinator: Coordinator) {
        uiViewController.stopCamera()
        Task { @MainActor in
            coordinator.state.cameraReady = false
            coordinator.state.previewFrameProvider = nil
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(state: state)
    }

    final class Coordinator: NSObject, HomeworkLedgerCameraControllerDelegate {
        let state: HomeworkLedgerExperimentState

        init(state: HomeworkLedgerExperimentState) {
            self.state = state
        }

        func cameraDidOpen() {
            Task { @MainActor in
                state.cameraReady = true
                state.notice = "相机已就绪，开始后会低频筛选关键帧。"
            }
        }

        func cameraFailed(_ message: String) {
            Task { @MainActor in
                state.cameraReady = false
                state.phase = .failed(message)
                state.notice = message
            }
        }
    }
}

private protocol HomeworkLedgerCameraControllerDelegate: AnyObject {
    func cameraDidOpen()
    func cameraFailed(_ message: String)
}

private final class HomeworkLedgerCameraController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    weak var delegate: HomeworkLedgerCameraControllerDelegate?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "com.pxj.homework-ledger-camera", qos: .userInitiated)
    private let ciContext = CIContext()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var latestImage: UIImage?
    private var latestImageAt: Date?
    private var lastImageUpdateAt = Date.distantPast

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configure()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        applyCurrentVideoOrientation()
    }

    func stopCamera() {
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
        if session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [session] in session.stopRunning() }
        }
        latestImage = nil
        latestImageAt = nil
    }

    func currentPreviewFrame() -> UIImage? {
        guard let latestImage, let latestImageAt, Date().timeIntervalSince(latestImageAt) < 2 else { return nil }
        return latestImage
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
        session.sessionPreset = session.canSetSessionPreset(.high) ? .high : .medium
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input),
              session.canAddOutput(videoOutput) else {
            delegate?.cameraFailed("无法初始化后置摄像头")
            return
        }
        session.addInput(input)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        session.addOutput(videoOutput)
        session.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(layer, at: 0)
        previewLayer = layer
        previewLayer?.frame = view.bounds
        applyCurrentVideoOrientation()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.applyCurrentVideoOrientation()
                self?.delegate?.cameraDidOpen()
            }
        }
    }

    private func applyCurrentVideoOrientation() {
        let orientation: AVCaptureVideoOrientation
        if let interfaceOrientation = view.window?.windowScene?.interfaceOrientation {
            switch interfaceOrientation {
            case .landscapeLeft: orientation = .landscapeLeft
            case .landscapeRight: orientation = .landscapeRight
            case .portraitUpsideDown: orientation = .portraitUpsideDown
            default: orientation = .portrait
            }
        } else {
            orientation = .portrait
        }
        for connection in [previewLayer?.connection, videoOutput.connection(with: .video)] {
            guard let connection, connection.isVideoOrientationSupported else { continue }
            connection.videoOrientation = orientation
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(lastImageUpdateAt) >= 0.35,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastImageUpdateAt = Date()
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage, scale: UIScreen.main.scale, orientation: .right)
        DispatchQueue.main.async { [weak self] in
            self?.latestImage = image
            self?.latestImageAt = Date()
        }
    }
}

@MainActor
final class HomeworkLedgerExperimentState: ObservableObject {
    @Published var phase: HomeworkLedgerPhase = .idle
    @Published var cameraReady = false
    @Published var previewFrames = 0
    @Published var skippedFrames = 0
    @Published var keyframes: [HomeworkLedgerKeyframe] = []
    @Published var evidence: [HomeworkLedgerEvidence] = []
    @Published var notice = "准备相机中"
    @Published var htmlURL: URL?
    @Published var resultHTML: String?
    @Published var resultAuthHeader: String?
    @Published var resultVisible = false

    var previewFrameProvider: (() -> UIImage?)?

    private var startedAt: Date?
    private var observeTask: Task<Void, Never>?
    private var sequenceIndex = 0
    private var lastAcceptedAt: Date?
    private var lastAcceptedFingerprint: UInt64?
    private var lastPageFingerprint: UInt64?
    private var pageEpisodeIndex = 0

    var phaseTitle: String {
        switch phase {
        case .idle: return cameraReady ? "待开始" : "相机准备中"
        case .observing: return "观察中"
        case .syncing: return "汇总上传中"
        case .ready: return "账本已生成"
        case .failed: return "需要处理"
        }
    }

    var phaseSystemImage: String {
        switch phase {
        case .idle: return "camera.viewfinder"
        case .observing: return "dot.radiowaves.left.and.right"
        case .syncing: return "arrow.triangle.2.circlepath"
        case .ready: return "doc.richtext"
        case .failed: return "exclamationmark.triangle"
        }
    }

    var elapsedText: String {
        guard let startedAt else { return "00:00" }
        let seconds = max(0, Int(Date().timeIntervalSince(startedAt)))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }

    var uploadRatioText: String {
        let naive = keyframes.reduce(0) { $0 + jpegSizeEstimate($1.image, quality: 0.72) }
        let cropBytes = evidence.reduce(0) { $0 + jpegSizeEstimate($1.cropImage, quality: 0.78) }
        guard naive > 0 else { return "0%" }
        return "\(Int(round(Double(cropBytes + naive) / Double(max(naive * max(previewFrames, 1), 1)) * 100)))%"
    }

    var cardCount: Int { evidence.count }

    func startObserving() {
        guard cameraReady, phase != .observing else { return }
        reset(keepCameraNotice: true)
        phase = .observing
        startedAt = Date()
        notice = "低频观察中：稳定、新区域或更清晰时才保留关键帧。"
        observeTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.samplePreviewFrame()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func stopObservingLocally() {
        observeTask?.cancel()
        observeTask = nil
        if phase == .observing { phase = .idle }
    }

    func finishAndSync() {
        guard phase == .observing else { return }
        observeTask?.cancel()
        observeTask = nil
        phase = .syncing
        notice = "正在汇总本轮关键帧和证据图。"
        Task {
            await buildEvidenceAndSync()
        }
    }

    func reset(keepCameraNotice: Bool = false) {
        observeTask?.cancel()
        observeTask = nil
        phase = .idle
        previewFrames = 0
        skippedFrames = 0
        keyframes = []
        evidence = []
        htmlURL = nil
        resultHTML = nil
        resultAuthHeader = nil
        sequenceIndex = 0
        lastAcceptedAt = nil
        lastAcceptedFingerprint = nil
        lastPageFingerprint = nil
        pageEpisodeIndex = 0
        startedAt = nil
        if !keepCameraNotice { notice = cameraReady ? "相机已就绪。" : "准备相机中" }
    }

    private func samplePreviewFrame() async {
        guard phase == .observing, let image = previewFrameProvider?() else {
            notice = "等待稳定预览帧。"
            return
        }
        previewFrames += 1
        sequenceIndex += 1
        let metrics = HomeworkLedgerImageMetrics.measure(image)
        guard metrics.sharpness >= 0.08, metrics.contrast >= 0.05 else {
            skippedFrames += 1
            notice = "跳过模糊或低对比画面。"
            return
        }
        let fingerprint = HomeworkLedgerImageMetrics.averageHash(image)
        let distance = lastAcceptedFingerprint.map { HomeworkLedgerImageMetrics.hamming($0, fingerprint) } ?? 64
        let secondsSinceAccepted = lastAcceptedAt.map { Date().timeIntervalSince($0) } ?? .infinity
        let shouldAccept = keyframes.isEmpty || distance >= 10 || secondsSinceAccepted >= 18 || metrics.sharpness > (keyframes.last?.sharpness ?? 0) + 0.16
        guard shouldAccept else {
            skippedFrames += 1
            notice = "画面变化小，已跳过重复帧。"
            return
        }
        let pageEpisodeId: String
        if let lastPageFingerprint = lastPageFingerprint,
           HomeworkLedgerImageMetrics.hamming(lastPageFingerprint, fingerprint) < 22,
           pageEpisodeIndex > 0 {
            pageEpisodeId = String(format: "page_%04d", pageEpisodeIndex)
        } else {
            pageEpisodeIndex += 1
            pageEpisodeId = String(format: "page_%04d", pageEpisodeIndex)
            lastPageFingerprint = fingerprint
        }
        let reason = keyframes.isEmpty ? "first_keyframe" : (distance >= 22 ? "page_turn_or_new_region" : "better_or_periodic_evidence")
        var frame = HomeworkLedgerKeyframe(
            id: String(format: "frame_%04d", sequenceIndex),
            image: image.resizedToPixelMaxSide(maxSide: 1600),
            capturedAt: Date(),
            sequenceIndex: sequenceIndex,
            sharpness: metrics.sharpness,
            brightness: metrics.brightness,
            contrast: metrics.contrast,
            fingerprint: fingerprint,
            reason: reason,
            pageEpisodeId: pageEpisodeId
        )
        frame.candidateCount = QuestionSegmenter.segmentForPreviewOverlay(image, fast: true).count
        keyframes.append(frame)
        lastAcceptedAt = Date()
        lastAcceptedFingerprint = fingerprint
        notice = "保留关键帧 \(keyframes.count)：\(frame.reason)"
    }

    private func buildEvidenceAndSync() async {
        let builtEvidence = buildEvidence()
        evidence = builtEvidence
        guard !keyframes.isEmpty else {
            phase = .failed("本轮没有保留关键帧")
            notice = "没有可用关键帧，请把作业纸放稳再试。"
            return
        }
        do {
            let result = try await HomeworkLedgerAPI.sync(keyframes: keyframes, evidence: builtEvidence, previewFrames: previewFrames, skippedFrames: skippedFrames)
            htmlURL = result.htmlURL
            resultHTML = result.html
            resultAuthHeader = AuthSession.shared.authHeader
            phase = .ready
            resultVisible = true
            notice = "已生成 \(builtEvidence.count) 个端上证据，已打开后端对比页。"
        } catch {
            phase = .failed(error.localizedDescription)
            notice = "同步失败：\(error.localizedDescription)"
        }
    }

    private func buildEvidence() -> [HomeworkLedgerEvidence] {
        var candidates: [HomeworkLedgerEvidenceCandidate] = []
        for frame in keyframes {
            let regions = QuestionSegmenter.segment(frame.image, fast: true)
            let questionRects = regions.map(\.normalizedRect).prefix(8)
            for rect in questionRects {
                candidates.append(evidenceCandidate(frame: frame, rect: rect, cropKind: "question", confidence: 0.66, mergeReasons: []))
            }
            for rect in fallbackSectionRects() {
                let overlapsKnownQuestion = regions.contains { $0.normalizedRect.intersectionRatio(with: rect) > 0.72 }
                let reasons = overlapsKnownQuestion ? ["section_coverage_guard", "overlaps_question_candidate"] : ["section_coverage_guard"]
                candidates.append(evidenceCandidate(frame: frame, rect: rect, cropKind: "section", confidence: overlapsKnownQuestion ? 0.42 : 0.54, mergeReasons: reasons))
            }
        }
        let merged = mergeEvidenceCandidates(candidates)
        return merged.enumerated().map { index, candidate in
            candidate.evidence(id: String(format: "qev_%04d", index + 1))
        }
    }

    private func evidenceCandidate(
        frame: HomeworkLedgerKeyframe,
        rect: CGRect,
        cropKind: String,
        confidence: Double,
        mergeReasons: [String]
    ) -> HomeworkLedgerEvidenceCandidate {
        let expanded = rect.insetBy(dx: -rect.width * 0.04, dy: -rect.height * 0.12).clampedUnitRect
        let crop = QuestionSegmenter.crop(frame.image, to: expanded).resizedToPixelMaxSide(maxSide: 1200)
        let cropHash = HomeworkLedgerImageMetrics.averageHash(crop)
        let area = max(0.01, expanded.width * expanded.height)
        let kindBonus = cropKind == "question" ? 0.22 : 0.04
        let balancedArea = cropKind == "question" ? min(area / 0.18, 1.0) : min(area / 0.30, 1.0)
        let score = frame.sharpness * 1.8 + frame.contrast * 0.6 + confidence + balancedArea * 0.28 + kindBonus
        return HomeworkLedgerEvidenceCandidate(
            pageEpisodeId: frame.pageEpisodeId,
            canonicalRect: expanded,
            bestFrameId: frame.id,
            cropImage: crop,
            cropHash: cropHash,
            layoutKey: layoutKey(pageEpisodeId: frame.pageEpisodeId, rect: expanded),
            cropKind: cropKind,
            confidence: confidence,
            seenCount: 1,
            sourceFrameIds: [frame.id],
            mergeReasons: mergeReasons,
            score: score
        )
    }

    private func mergeEvidenceCandidates(_ candidates: [HomeworkLedgerEvidenceCandidate]) -> [HomeworkLedgerEvidenceCandidate] {
        var merged: [HomeworkLedgerEvidenceCandidate] = []
        for candidate in candidates.sorted(by: { $0.score > $1.score }) {
            if let index = merged.firstIndex(where: { shouldMergeEvidence($0, candidate) }) {
                merged[index].merge(with: candidate)
            } else {
                merged.append(candidate)
            }
        }
        return merged.sorted {
            if $0.pageEpisodeId == $1.pageEpisodeId {
                if abs($0.canonicalRect.minY - $1.canonicalRect.minY) > 0.04 {
                    return $0.canonicalRect.minY < $1.canonicalRect.minY
                }
                return $0.canonicalRect.minX < $1.canonicalRect.minX
            }
            return $0.pageEpisodeId < $1.pageEpisodeId
        }
    }

    private func shouldMergeEvidence(_ lhs: HomeworkLedgerEvidenceCandidate, _ rhs: HomeworkLedgerEvidenceCandidate) -> Bool {
        guard lhs.pageEpisodeId == rhs.pageEpisodeId else { return false }
        if lhs.layoutKey == rhs.layoutKey { return true }
        let overlap = lhs.canonicalRect.intersectionRatio(with: rhs.canonicalRect)
        if lhs.cropKind == "section" || rhs.cropKind == "section" {
            return overlap > 0.82
        }
        return overlap > 0.58
    }

    private func fallbackSectionRects() -> [CGRect] {
        [
            CGRect(x: 0.02, y: 0.02, width: 0.96, height: 0.31),
            CGRect(x: 0.02, y: 0.34, width: 0.96, height: 0.31),
            CGRect(x: 0.02, y: 0.66, width: 0.96, height: 0.31)
        ]
    }

    private func layoutKey(pageEpisodeId: String, rect: CGRect) -> String {
        let x = Int((rect.minX * 20).rounded())
        let y = Int((rect.minY * 20).rounded())
        let w = Int((rect.width * 20).rounded())
        let h = Int((rect.height * 20).rounded())
        return "\(pageEpisodeId):\(x):\(y):\(w):\(h)"
    }
}

private enum HomeworkLedgerAPI {
    static func sync(
        keyframes: [HomeworkLedgerKeyframe],
        evidence: [HomeworkLedgerEvidence],
        previewFrames: Int,
        skippedFrames: Int
    ) async throws -> (runId: String, htmlURL: URL, html: String?) {
        let runId = "ios_ledger_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        let activeStudentId = await MainActor.run { AuthSession.shared.activeStudentId }
        _ = try await postJSON(path: "/api/homework-ledger/runs", payload: [
            "run_id": runId,
            "title": "作业账本实验",
            "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "iphone",
            "student_profile_id": activeStudentId
        ])

        let manifest = manifestPayload(runId: runId, keyframes: keyframes, evidence: evidence, previewFrames: previewFrames, skippedFrames: skippedFrames)
        var files: [MultipartFile] = []
        for frame in keyframes {
            files.append(MultipartFile(field: "frames", name: "frames/\(frame.id).jpg", mime: "image/jpeg", data: try jpegData(frame.image, maxSide: 1600, quality: 0.74)))
        }
        for item in evidence {
            files.append(MultipartFile(field: "crops", name: "crops/\(item.id).jpg", mime: "image/jpeg", data: try jpegData(item.cropImage, maxSide: 1200, quality: 0.80)))
        }
        _ = try await postForm(path: "/api/homework-ledger/runs/\(runId)/sync", fields: [
            "manifest": jsonString(manifest)
        ], files: files)
        _ = try await postJSON(path: "/api/homework-ledger/runs/\(runId)/finish", payload: [
            "metrics": manifest["metrics"] as? [String: Any] ?? [:]
        ])
        _ = try? await postJSON(path: "/api/homework-ledger/runs/\(runId)/refine?max_frames=8&max_regions_per_frame=6&use_vlm=true", payload: [:])
        let compareURL = ledgerURL(path: "/api/homework-ledger/runs/\(runId)/compare")
        return (runId, compareURL, nil)
    }

    private static func manifestPayload(
        runId: String,
        keyframes: [HomeworkLedgerKeyframe],
        evidence: [HomeworkLedgerEvidence],
        previewFrames: Int,
        skippedFrames: Int
    ) -> [String: Any] {
        let pageIds = Array(Set(keyframes.map(\.pageEpisodeId))).sorted()
        let framePayloads = keyframes.map { frame in
            [
                "id": frame.id,
                "frame_id": frame.id,
                "index": frame.sequenceIndex,
                "filename": "frames/\(frame.id).jpg",
                "upload_ref": "frames/\(frame.id).jpg",
                "width": Int(frame.image.size.width * frame.image.scale),
                "height": Int(frame.image.size.height * frame.image.scale),
                "source_bytes": jpegSizeEstimate(frame.image, quality: 0.74),
                "normalized_bytes": jpegSizeEstimate(frame.image, quality: 0.74),
                "sharpness": frame.sharpness,
                "brightness": frame.brightness,
                "contrast": frame.contrast,
                "phash": String(frame.fingerprint, radix: 16),
                "accepted": true,
                "reason": frame.reason,
                "page_episode_id": frame.pageEpisodeId,
                "candidate_count": frame.candidateCount
            ] as [String: Any]
        }
        let pages = pageIds.map { pageId -> [String: Any] in
            let ids = keyframes.filter { $0.pageEpisodeId == pageId }.map(\.id)
            return [
                "id": pageId,
                "first_frame_id": ids.first ?? "",
                "last_frame_id": ids.last ?? "",
                "frame_ids": ids,
                "fingerprint": pageId,
                "coverage_cells": []
            ]
        }
        let evidencePayloads = evidence.map { item in
            [
                "id": item.id,
                "page_episode_id": item.pageEpisodeId,
                "canonical_rect": rectPayload(item.canonicalRect),
                "best_frame_id": item.bestFrameId,
                "best_crop_filename": "crops/\(item.id).jpg",
                "upload_ref": "crops/\(item.id).jpg",
                "crop_hash": String(item.cropHash, radix: 16),
                "layout_key": item.layoutKey,
                "ocr_key": "",
                "seen_count": item.seenCount,
                "status": "ready",
                "quality": ["confidence": item.confidence, "area": item.canonicalRect.width * item.canonicalRect.height],
                "source_frames": item.sourceFrameIds,
                "crop_kind": item.cropKind,
                "merge_reasons": item.mergeReasons
            ] as [String: Any]
        }
        let cards = evidence.map { item in
            [
                "id": "q_\(item.id)",
                "evidence_ids": [item.id],
                "number": "",
                "subject": "unknown",
                "question_type": item.cropKind == "section" ? "section_evidence" : "evidence",
                "stem_text": "Evidence \(item.id)",
                "editable_html": "<article><p>待识别题目，可在此编辑。</p><figure><img src=\"crops/\(item.id).jpg\" alt=\"\(item.id)\"></figure></article>",
                "figure_assets": [["type": "raster", "filename": "crops/\(item.id).jpg", "source_evidence_id": item.id]],
                "confidence": item.confidence,
                "review_flags": item.cropKind == "section" ? ["needs_review", "section_crop", "raster_figure_evidence"] : ["needs_text_extraction", "raster_figure_evidence"],
                "source_input_index": 0,
                "extraction_notes": "iOS ledger experiment evidence card"
            ] as [String: Any]
        }
        let keyframeBytes = keyframes.reduce(0) { $0 + jpegSizeEstimate($1.image, quality: 0.74) }
        let cropBytes = evidence.reduce(0) { $0 + jpegSizeEstimate($1.cropImage, quality: 0.80) }
        let naiveBytes = max(1, previewFrames) * max(1, keyframeBytes / max(1, keyframes.count))
        let sectionEvidenceCount = evidence.filter { $0.cropKind == "section" }.count
        let mergedEvidenceCount = evidence.filter { $0.seenCount > 1 || !$0.mergeReasons.isEmpty }.count
        let metrics: [String: Any] = [
            "frames_total": previewFrames,
            "keyframes": keyframes.count,
            "skipped_frames": skippedFrames,
            "page_episode_count": pageIds.count,
            "question_evidence_count": evidence.count,
            "question_card_count": cards.count,
            "section_evidence_count": sectionEvidenceCount,
            "merged_evidence_count": mergedEvidenceCount,
            "keyframe_upload_bytes": keyframeBytes,
            "crop_jpeg_upload_bytes": cropBytes,
            "full_frame_upload_bytes_if_naive": naiveBytes,
            "estimated_upload_bytes": keyframeBytes + cropBytes,
            "estimated_upload_ratio": Double(keyframeBytes + cropBytes) / Double(naiveBytes),
            "cards_with_raster_figure": cards.count,
            "cards_with_svg_draft": 0,
            "vlm_enabled": false,
            "source": "ios_homework_ledger_experiment"
        ]
        return [
            "source": "ios_homework_ledger_experiment",
            "simulator_version": "ios-ledger-v0",
            "run_id": runId,
            "frames": framePayloads,
            "page_episodes": pages,
            "question_evidence": evidencePayloads,
            "question_cards": cards,
            "metrics": metrics
        ]
    }

    private static func rectPayload(_ rect: CGRect) -> [String: Any] {
        ["x": rect.minX, "y": rect.minY, "w": rect.width, "h": rect.height, "width": rect.width, "height": rect.height]
    }

    private static func postJSON(path: String, payload: [String: Any]) async throws -> Data {
        var request = URLRequest(url: ledgerURL(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let auth = await MainActor.run(body: { AuthSession.shared.authHeader }) {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        return try await checkedData(for: request)
    }

    private static func postForm(path: String, fields: [String: String], files: [MultipartFile]) async throws -> Data {
        let boundary = "pxj-ledger-\(UUID().uuidString)"
        var request = URLRequest(url: ledgerURL(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        if let auth = await MainActor.run(body: { AuthSession.shared.authHeader }) {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = makeHomeworkLedgerMultipartBody(boundary: boundary, fields: fields, files: files)
        return try await checkedData(for: request)
    }

    private static func getData(path: String) async throws -> Data {
        var request = URLRequest(url: ledgerURL(path: path))
        request.httpMethod = "GET"
        request.timeoutInterval = 30
        if let auth = await MainActor.run(body: { AuthSession.shared.authHeader }) {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        return try await checkedData(for: request)
    }

    private static func checkedData(for request: URLRequest) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 {
                await MainActor.run { AuthSession.shared.handleUnauthorized() }
            }
            throw URLError(.badServerResponse)
        }
        return data
    }

    private static func jsonString(_ value: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    private static func ledgerURL(path: String) -> URL {
        URL(string: path, relativeTo: authServerBaseURL)!.absoluteURL
    }
}

private enum HomeworkLedgerImageMetrics {
    static func measure(_ image: UIImage) -> (sharpness: Double, brightness: Double, contrast: Double) {
        guard let cg = image.resizedToPixelMaxSide(maxSide: 320).cgImage,
              let data = cg.dataProvider?.data,
              let pointer = CFDataGetBytePtr(data) else { return (0, 0, 0) }
        let bytesPerPixel = max(1, cg.bitsPerPixel / 8)
        let bytesPerRow = cg.bytesPerRow
        var values: [Double] = []
        values.reserveCapacity(320 * 240)
        var sum = 0.0
        for y in stride(from: 0, to: cg.height, by: 2) {
            for x in stride(from: 0, to: cg.width, by: 2) {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let b = Double(pointer[offset])
                let g = bytesPerPixel > 1 ? Double(pointer[offset + 1]) : b
                let r = bytesPerPixel > 2 ? Double(pointer[offset + 2]) : g
                let lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                values.append(lum)
                sum += lum
            }
        }
        guard !values.isEmpty else { return (0, 0, 0) }
        let mean = sum / Double(values.count)
        let variance = values.reduce(0.0) { $0 + pow($1 - mean, 2) } / Double(values.count)
        let contrast = sqrt(variance)
        var edge = 0.0
        var edgeCount = 0
        let width = max(1, cg.width / 2)
        for index in 1..<values.count {
            if index % width == 0 { continue }
            edge += abs(values[index] - values[index - 1])
            edgeCount += 1
        }
        let sharpness = edgeCount > 0 ? min(1.0, edge / Double(edgeCount) * 8.0) : 0
        return (sharpness, mean, contrast)
    }

    static func averageHash(_ image: UIImage) -> UInt64 {
        guard let cg = image.resizedToExactPixelSize(CGSize(width: 8, height: 8)).cgImage,
              let data = cg.dataProvider?.data,
              let pointer = CFDataGetBytePtr(data) else { return 0 }
        let bytesPerPixel = max(1, cg.bitsPerPixel / 8)
        let bytesPerRow = cg.bytesPerRow
        var values: [Double] = []
        var sum = 0.0
        for y in 0..<8 {
            for x in 0..<8 {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let b = Double(pointer[offset])
                let g = bytesPerPixel > 1 ? Double(pointer[offset + 1]) : b
                let r = bytesPerPixel > 2 ? Double(pointer[offset + 2]) : g
                let lum = 0.299 * r + 0.587 * g + 0.114 * b
                values.append(lum)
                sum += lum
            }
        }
        let mean = sum / 64.0
        var hash: UInt64 = 0
        for (index, value) in values.enumerated() where value >= mean {
            hash |= (UInt64(1) << UInt64(index))
        }
        return hash
    }

    static func hamming(_ left: UInt64, _ right: UInt64) -> Int {
        Int((left ^ right).nonzeroBitCount)
    }
}

private func jpegData(_ image: UIImage, maxSide: CGFloat, quality: CGFloat) throws -> Data {
    let resized = image.resizedToPixelMaxSide(maxSide: maxSide)
    guard let data = resized.jpegData(compressionQuality: quality) else {
        throw URLError(.cannotDecodeContentData)
    }
    return data
}

private func jpegSizeEstimate(_ image: UIImage, quality: CGFloat) -> Int {
    (try? jpegData(image, maxSide: 1600, quality: quality).count) ?? 0
}

private func makeHomeworkLedgerMultipartBody(boundary: String, fields: [String: String], files: [MultipartFile]) -> Data {
    var body = Data()
    for (key, value) in fields {
        body.appendLedgerString("--\(boundary)\r\n")
        body.appendLedgerString("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
        body.appendLedgerString("\(value)\r\n")
    }
    for file in files {
        body.appendLedgerString("--\(boundary)\r\n")
        body.appendLedgerString("Content-Disposition: form-data; name=\"\(file.field)\"; filename=\"\(file.name)\"\r\n")
        body.appendLedgerString("Content-Type: \(file.mime)\r\n\r\n")
        body.append(file.data)
        body.appendLedgerString("\r\n")
    }
    body.appendLedgerString("--\(boundary)--\r\n")
    return body
}

private extension Data {
    mutating func appendLedgerString(_ string: String) {
        append(Data(string.utf8))
    }
}

private extension UIImage {
    func resizedToExactPixelSize(_ target: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: target, format: format)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
    }
}

private extension CGRect {
    var clampedUnitRect: CGRect {
        let x1 = max(0, min(1, minX))
        let y1 = max(0, min(1, minY))
        let x2 = max(x1, min(1, maxX))
        let y2 = max(y1, min(1, maxY))
        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }

    func intersectionRatio(with other: CGRect) -> CGFloat {
        let intersection = self.intersection(other)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else {
            return 0
        }
        let intersectionArea = intersection.width * intersection.height
        let smallerArea = max(0.0001, min(width * height, other.width * other.height))
        return intersectionArea / smallerArea
    }
}
