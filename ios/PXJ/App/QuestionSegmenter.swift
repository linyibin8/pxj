import UIKit
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins

/// 单道题目的归一化区域。
/// 坐标系：归一化 [0,1]、原点左上、x 右 y 下；定义在“校正后图”坐标系中。
struct QuestionRegion: Identifiable, Equatable {
    let id = UUID()
    var normalizedRect: CGRect   // 左上原点、[0,1]，定义在“校正后图”坐标系
    var index: Int               // 序号 1…n（上→下、左→右）
    var ocrText: String?
    var confidence: Double
}

/// 端上题目分割工具集：梯形校正 / 题块分割 / 子图裁剪。
/// 组织方式仿 BurstFrameAnalyzer：纯 static、无状态、异常即降级。
enum QuestionSegmenter {

    /// 共享 CIContext（CoreImage 渲染较重，复用一个即可）。
    static let ciContext = CIContext()

    // MARK: - 1. 梯形校正

    /// 文档梯形校正。
    /// - Returns: (校正后图, 是否真的做了校正)。任何不确定/失败一律返回 (近似原图, false)，避免越矫越歪。
    static func rectify(_ image: UIImage) -> (image: UIImage, didCorrect: Bool) {
        let base = normalizedUp(image)
        guard let quad = detectDocumentQuad(base) else { return (base, false) }

        // sanity：四角面积过小 / 过钝 / 几乎无形变 → 不矫正
        if quadArea(quad) < 0.35 { return (base, false) }
        if cornersTooObtuse(quad) { return (base, false) }
        if deformationTooSmall(quad) { return (base, false) }

        guard let cg = base.cgImage else { return (base, false) }
        let ci = CIImage(cgImage: cg)
        let w = ci.extent.width
        let h = ci.extent.height
        guard w > 0, h > 0 else { return (base, false) }

        // CIImage 左下原点，与 Vision 一致：归一化点直接乘像素尺寸即可。
        let filter = CIFilter.perspectiveCorrection()
        filter.inputImage = ci
        filter.topLeft = CGPoint(x: quad.topLeft.x * w, y: quad.topLeft.y * h)
        filter.topRight = CGPoint(x: quad.topRight.x * w, y: quad.topRight.y * h)
        filter.bottomLeft = CGPoint(x: quad.bottomLeft.x * w, y: quad.bottomLeft.y * h)
        filter.bottomRight = CGPoint(x: quad.bottomRight.x * w, y: quad.bottomRight.y * h)

        guard let output = filter.outputImage,
              output.extent.width >= 1, output.extent.height >= 1,
              let outCG = ciContext.createCGImage(output, from: output.extent) else {
            return (base, false)
        }
        return (UIImage(cgImage: outCG, scale: base.scale, orientation: .up), true)
    }

    // MARK: - 2. 题目分割（文字行聚类成大题块）

    /// 把整页拍摄图分割成若干“大题块”（小问 (1)(2)(3) 不单独拆）。
    /// - Returns: [QuestionRegion]，归一化矩形定义在 orientation 归一后的图坐标系；异常返回 []。
    static func segment(_ image: UIImage, fast: Bool = false) -> [QuestionRegion] {
        let base = normalizedUp(image)
        // 实时扫描用更小图 + fast 识别（省功耗、可达每秒数帧）；静态精提取用 accurate。
        let visionImage = base.resizedForVision(maxSide: fast ? 1000 : 1400)
        guard let cg = visionImage.cgImage else { return [] }
        let orientation = visionImage.cgImagePropertyOrientation

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = fast ? .fast : .accurate
        request.usesLanguageCorrection = !fast
        request.recognitionLanguages = ["zh-Hans", "en-US"]

        let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])
        guard (try? handler.perform([request])) != nil,
              let observations = request.results, !observations.isEmpty else {
            return []
        }

        // 每个 observation → 左上原点 rect 的文字行
        var lines: [TextLine] = []
        for obs in observations {
            guard let candidate = obs.topCandidates(1).first else { continue }
            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let bb = obs.boundingBox // 左下原点
            let rect = CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height) // 翻成左上原点
            lines.append(TextLine(rect: rect, text: text, confidence: Double(candidate.confidence)))
        }
        guard !lines.isEmpty else { return [] }

        lines.sort { $0.rect.minY < $1.rect.minY }
        let avgHeight = lines.map { $0.rect.height }.reduce(0, +) / CGFloat(lines.count)

        let pad: CGFloat = 0.012
        var regions: [QuestionRegion] = []

        /// 由一组文字行生成题块矩形；yBottomOverride 把竖直下界延伸到下一题题号，
        /// 从而把「题号到下一题之间」的图形/留白也圈进来（密排题切得更全更准）。
        func makeRegion(_ block: [TextLine], yBottomOverride: CGFloat?) -> QuestionRegion? {
            guard let first = block.first else { return nil }
            var union = first.rect
            for line in block.dropFirst() { union = union.union(line.rect) }
            if let yb = yBottomOverride, yb > union.minY {
                union = CGRect(x: union.minX, y: union.minY, width: union.width, height: yb - union.minY)
            }
            let padded = clampRect(union.insetBy(dx: -pad, dy: -pad))
            guard padded.width * padded.height >= 0.012 else { return nil }
            let text = block.map { $0.text }.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            let confidence = block.map { $0.confidence }.reduce(0, +) / Double(block.count)
            return QuestionRegion(normalizedRect: padded, index: 0, ocrText: text, confidence: confidence)
        }

        // 题号锚行（行首命中「N、/N，/N./第N题」等）。密排题靠题号逐题切，最准。
        let anchors = lines.indices.filter { matchesQuestionNumber(lines[$0].text) }
        if anchors.count >= 2 {
            // 每道题 = [本题号行, 下一题号行) 的文字行，竖直下界延伸到下一题题号顶部。
            for (a, start) in anchors.enumerated() {
                let end = (a + 1 < anchors.count) ? anchors[a + 1] : lines.count
                let yBottom = (a + 1 < anchors.count) ? lines[anchors[a + 1]].rect.minY - pad : nil
                if let region = makeRegion(Array(lines[start..<end]), yBottomOverride: yBottom) {
                    regions.append(region)
                }
            }
        } else {
            // 回退：识别不到足够题号时，按垂直间隙 > 行高*1.6 或命中题号 聚类。
            var blocks: [[TextLine]] = []
            var current: [TextLine] = []
            for line in lines {
                if current.isEmpty { current = [line]; continue }
                let prevMaxY = current.map { $0.rect.maxY }.max() ?? line.rect.minY
                let gap = line.rect.minY - prevMaxY
                if gap > avgHeight * 1.6 || matchesQuestionNumber(line.text) {
                    blocks.append(current); current = [line]
                } else { current.append(line) }
            }
            if !current.isEmpty { blocks.append(current) }
            for block in blocks {
                if let region = makeRegion(block, yBottomOverride: nil) { regions.append(region) }
            }
        }

        // 阅读顺序：上→下、左→右
        regions.sort { lhs, rhs in
            if abs(lhs.normalizedRect.minY - rhs.normalizedRect.minY) > 0.04 {
                return lhs.normalizedRect.minY < rhs.normalizedRect.minY
            }
            return lhs.normalizedRect.minX < rhs.normalizedRect.minX
        }
        for i in regions.indices { regions[i].index = i + 1 }
        return regions
    }

    static func segmentForPreviewOverlay(_ image: UIImage) -> [QuestionRegion] {
        let base = normalizedUp(image)
        guard let quad = detectDocumentQuad(base),
              quadArea(quad) >= 0.35,
              !cornersTooObtuse(quad),
              !deformationTooSmall(quad) else {
            return segment(base, fast: false)
        }
        let corrected = rectify(base)
        guard corrected.didCorrect else {
            return segment(base, fast: false)
        }
        return segment(corrected.image, fast: false).compactMap { region in
            let mapped = mapCorrectedRectToOriginal(region.normalizedRect, quad: quad)
            guard mapped.width * mapped.height >= 0.006 else { return nil }
            var copy = region
            copy.normalizedRect = mapped
            return copy
        }
    }

    // MARK: - 3. 裁剪子图

    /// 按归一化 rect（左上原点）裁出子图。clamp 到图内；失败返回原图。
    /// 留白由调用方在 rect 上自行预留。
    static func crop(_ image: UIImage, to normalizedRect: CGRect) -> UIImage {
        let base = normalizedUp(image)
        guard let cg = base.cgImage else { return image }
        let w = CGFloat(cg.width)
        let h = CGFloat(cg.height)
        let clamped = clampRect(normalizedRect)
        // cg 像素坐标为左上原点，与本约定一致
        let pixelRect = CGRect(x: clamped.minX * w,
                               y: clamped.minY * h,
                               width: clamped.width * w,
                               height: clamped.height * h).integral
        guard pixelRect.width >= 1, pixelRect.height >= 1,
              let cropped = cg.cropping(to: pixelRect) else { return base }
        return UIImage(cgImage: cropped, scale: base.scale, orientation: .up)
    }

    // MARK: - 私有辅助

    private struct TextLine {
        var rect: CGRect
        var text: String
        var confidence: Double
    }

    /// 四角，归一化、左下原点（Vision 原生）。topLeft 等指“视觉上的”角，与 CIFilter 一致。
    private struct Quad {
        var topLeft: CGPoint
        var topRight: CGPoint
        var bottomLeft: CGPoint
        var bottomRight: CGPoint
    }

    /// orientation 归一：重绘成 .up，后续坐标处理才一致。
    private static func normalizedUp(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: image.size)) }
    }

    /// 文档四角检测：iOS16+ 先用 DocumentSegmentation，失败回退 Rectangles。
    private static func detectDocumentQuad(_ image: UIImage) -> Quad? {
        let small = image.resizedForVision(maxSide: 900)
        guard let cg = small.cgImage else { return nil }
        let orientation = small.cgImagePropertyOrientation
        let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])

        if #available(iOS 16.0, *) {
            let docRequest = VNDetectDocumentSegmentationRequest()
            if (try? handler.perform([docRequest])) != nil,
               let obs = docRequest.results?.first {
                return Quad(topLeft: obs.topLeft, topRight: obs.topRight,
                            bottomLeft: obs.bottomLeft, bottomRight: obs.bottomRight)
            }
        }

        // 回退：参数仿 ContentView 现有矩形检测，取 confidence 最高的 quad
        let rectRequest = VNDetectRectanglesRequest()
        rectRequest.maximumObservations = 6
        rectRequest.minimumConfidence = 0.45
        rectRequest.minimumAspectRatio = 0.20
        rectRequest.maximumAspectRatio = 1.0
        rectRequest.minimumSize = 0.16
        rectRequest.quadratureTolerance = 28
        guard (try? handler.perform([rectRequest])) != nil,
              let best = rectRequest.results?.max(by: { $0.confidence < $1.confidence }) else {
            return nil
        }
        return Quad(topLeft: best.topLeft, topRight: best.topRight,
                    bottomLeft: best.bottomLeft, bottomRight: best.bottomRight)
    }

    /// 归一化坐标系下四边形面积（shoelace；方向无关）。
    private static func quadArea(_ q: Quad) -> CGFloat {
        let pts = [q.topLeft, q.topRight, q.bottomRight, q.bottomLeft]
        var sum: CGFloat = 0
        for i in 0..<pts.count {
            let a = pts[i]
            let b = pts[(i + 1) % pts.count]
            sum += a.x * b.y - b.x * a.y
        }
        return abs(sum) / 2
    }

    /// 任一内角过钝/过锐 → 形变离谱，放弃矫正。
    private static func cornersTooObtuse(_ q: Quad) -> Bool {
        let angles = [
            interiorAngle(at: q.topLeft, q.topRight, q.bottomLeft),
            interiorAngle(at: q.topRight, q.bottomRight, q.topLeft),
            interiorAngle(at: q.bottomRight, q.bottomLeft, q.topRight),
            interiorAngle(at: q.bottomLeft, q.topLeft, q.bottomRight)
        ]
        return angles.contains { $0 < 50 || $0 > 135 }
    }

    /// 四边形几乎就是其轴对齐外接框（接近矩形且未倾斜）→ 形变很小，无需矫正。
    private static func deformationTooSmall(_ q: Quad) -> Bool {
        let pts = [q.topLeft, q.topRight, q.bottomRight, q.bottomLeft]
        let minX = pts.map { $0.x }.min() ?? 0
        let maxX = pts.map { $0.x }.max() ?? 0
        let minY = pts.map { $0.y }.min() ?? 0
        let maxY = pts.map { $0.y }.max() ?? 0
        let bboxArea = (maxX - minX) * (maxY - minY)
        guard bboxArea > 0 else { return true }
        return quadArea(q) / bboxArea > 0.97
    }

    private static func mapCorrectedRectToOriginal(_ rect: CGRect, quad: Quad) -> CGRect {
        let corners = [
            CGPoint(x: rect.minX, y: rect.minY),
            CGPoint(x: rect.maxX, y: rect.minY),
            CGPoint(x: rect.minX, y: rect.maxY),
            CGPoint(x: rect.maxX, y: rect.maxY)
        ].map { mapCorrectedPointToOriginal($0, quad: quad) }
        let minX = corners.map { $0.x }.min() ?? rect.minX
        let maxX = corners.map { $0.x }.max() ?? rect.maxX
        let minY = corners.map { $0.y }.min() ?? rect.minY
        let maxY = corners.map { $0.y }.max() ?? rect.maxY
        return clampRect(
            CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
                .insetBy(dx: -0.004, dy: -0.004)
        )
    }

    private static func mapCorrectedPointToOriginal(_ point: CGPoint, quad: Quad) -> CGPoint {
        let topLeft = topLeftOriginPoint(quad.topLeft)
        let topRight = topLeftOriginPoint(quad.topRight)
        let bottomLeft = topLeftOriginPoint(quad.bottomLeft)
        let bottomRight = topLeftOriginPoint(quad.bottomRight)
        let x = max(0, min(1, point.x))
        let y = max(0, min(1, point.y))
        let top = interpolate(topLeft, topRight, amount: x)
        let bottom = interpolate(bottomLeft, bottomRight, amount: x)
        return interpolate(top, bottom, amount: y)
    }

    private static func topLeftOriginPoint(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x, y: 1 - point.y)
    }

    private static func interpolate(_ a: CGPoint, _ b: CGPoint, amount: CGFloat) -> CGPoint {
        CGPoint(
            x: a.x + (b.x - a.x) * amount,
            y: a.y + (b.y - a.y) * amount
        )
    }

    private static func interiorAngle(at p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let v1 = CGPoint(x: a.x - p.x, y: a.y - p.y)
        let v2 = CGPoint(x: b.x - p.x, y: b.y - p.y)
        let m1 = hypot(v1.x, v1.y)
        let m2 = hypot(v2.x, v2.y)
        guard m1 > 0, m2 > 0 else { return 0 }
        let dot = v1.x * v2.x + v1.y * v2.y
        let cosine = max(-1, min(1, dot / (m1 * m2)))
        return acos(cosine) * 180 / .pi
    }

    /// 把矩形 clamp 到 [0,1]×[0,1]，并保证非负宽高。
    private static func clampRect(_ rect: CGRect) -> CGRect {
        let minX = max(0, min(1, rect.minX))
        let minY = max(0, min(1, rect.minY))
        let maxX = max(0, min(1, rect.maxX))
        let maxY = max(0, min(1, rect.maxY))
        return CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
    }

    /// 题号正则：第?[数字/中文数字]+[、，,.)）题]；含全/半角逗号（试卷常写「1，2，3，」）。
    /// (1)(2) 这类小问以括号起头不命中，确保“大题为一块”。
    private static let questionNumberRegex: NSRegularExpression? = {
        try? NSRegularExpression(pattern: "^\\s*(第)?[0-9一二三四五六七八九十]+\\s*[、，,.\\)）题]")
    }()

    private static func matchesQuestionNumber(_ text: String) -> Bool {
        guard let regex = questionNumberRegex else { return false }
        let range = NSRange(text.startIndex..., in: text)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }
}
