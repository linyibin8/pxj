import UIKit
import Vision
import CoreML
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

struct QuestionSegmentationDiagnostics {
    var segmenterVersion = "question_region_v4_diagnostic"
    var selectedSegmenter = "none"
    var didCorrect = false
    var textRecognitionScope = "none"
    var textRecognitionDurationMs = 0
    var ocrV3DurationMs = 0
    var coreMLV4DurationMs = 0
    var totalDurationMs = 0
    var ocrTextLineCount = 0
    var ocrV3CandidateCount = 0
    var coreMLModelLoaded = false
    var coreMLAttempted = false
    var fallbackToOCR = false
    var coreMLRawObservationCount = 0
    var coreMLCandidateBoxCount = 0
    var coreMLAfterNMSBoxCount = 0
    var coreMLMatchedRegionCount = 0
    var selectedRegionCount = 0
    var pairedMatchCount = 0
    var pairedMeanIoU = 0.0
    var pairedMedianIoU = 0.0
    var unmatchedOCRV3Count = 0
    var unmatchedCoreMLV4Count = 0

    var payload: [String: Any] {
        [
            "segmenter_version": segmenterVersion,
            "selected_segmenter": selectedSegmenter,
            "did_correct": didCorrect,
            "text_recognition_scope": textRecognitionScope,
            "text_recognition_duration_ms": textRecognitionDurationMs,
            "ocr_v3_duration_ms": ocrV3DurationMs,
            "coreml_v4_duration_ms": coreMLV4DurationMs,
            "segmentation_total_duration_ms": totalDurationMs,
            "ocr_text_line_count": ocrTextLineCount,
            "ocr_v3_candidate_count": ocrV3CandidateCount,
            "coreml_model_loaded": coreMLModelLoaded,
            "coreml_attempted": coreMLAttempted,
            "fallback_to_ocr": fallbackToOCR,
            "coreml_raw_observation_count": coreMLRawObservationCount,
            "coreml_candidate_box_count": coreMLCandidateBoxCount,
            "coreml_box_after_nms_count": coreMLAfterNMSBoxCount,
            "coreml_matched_region_count": coreMLMatchedRegionCount,
            "selected_region_count": selectedRegionCount,
            "paired_match_count": pairedMatchCount,
            "paired_mean_iou": pairedMeanIoU,
            "paired_median_iou": pairedMedianIoU,
            "unmatched_ocr_v3_count": unmatchedOCRV3Count,
            "unmatched_coreml_v4_count": unmatchedCoreMLV4Count
        ]
    }
}

struct QuestionSegmentationResult {
    var regions: [QuestionRegion]
    var diagnostics: QuestionSegmentationDiagnostics
}

/// 端上题目分割工具集：梯形校正 / 题块分割 / 子图裁剪。
/// 组织方式仿 BurstFrameAnalyzer：纯 static、无状态、异常即降级。
enum QuestionSegmenter {

    /// 共享 CIContext（CoreImage 渲染较重，复用一个即可）。
    static let ciContext = CIContext()
    private static let detectorModelLock = NSLock()
    private static var cachedDetectorModel: VNCoreMLModel?
    private static var didAttemptDetectorLoad = false
    private static let detectorCategoryName = "question_block"
    private static let detectorFastMinConfidence: VNConfidence = 0.32
    private static let detectorAccurateMinConfidence: VNConfidence = 0.38
    private static let detectorNMSOverlapThreshold: CGFloat = 0.72

    // MARK: - 1. 梯形校正

    /// 文档梯形校正。
    /// - Returns: (校正后图, 是否真的做了校正)。任何不确定/失败一律返回 (近似原图, false)，避免越矫越歪。
    static func rectify(_ image: UIImage) -> (image: UIImage, didCorrect: Bool) {
        let base = normalizedUp(image)
        guard let quad = detectDocumentQuad(base) else { return (base, false) }
        return rectify(base, quad: quad)
    }

    private static func rectify(_ base: UIImage, quad: Quad) -> (image: UIImage, didCorrect: Bool) {
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
        return segmentWithDiagnostics(image, fast: fast).regions
    }

    static func segmentForPreviewOverlay(_ image: UIImage, fast: Bool = false) -> [QuestionRegion] {
        let base = normalizedUp(image)
        guard let quad = detectDocumentQuad(base),
              quadArea(quad) >= 0.35,
              !cornersTooObtuse(quad),
              !deformationTooSmall(quad) else {
            return segment(base, fast: fast)
        }
        let corrected = rectify(base, quad: quad)
        guard corrected.didCorrect else {
            return segment(base, fast: fast)
        }
        return segment(corrected.image, fast: fast).compactMap { region in
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
    static func segmentWithDiagnostics(_ image: UIImage, fast: Bool = false) -> QuestionSegmentationResult {
        let totalStartedAt = Date()
        var diagnostics = QuestionSegmentationDiagnostics()
        let base = normalizedUp(image)
        let visionImage = base.resizedForVision(maxSide: fast ? 1000 : 1400)
        guard let cg = visionImage.cgImage else {
            diagnostics.totalDurationMs = durationMs(since: totalStartedAt)
            return QuestionSegmentationResult(regions: [], diagnostics: diagnostics)
        }
        let orientation = visionImage.cgImagePropertyOrientation

        let detectorStartedAt = Date()
        var detector = detectorQuestionRegionResult(in: visionImage, cgImage: cg, lines: [], fast: fast, allowTextless: true)
        diagnostics.coreMLV4DurationMs = durationMs(since: detectorStartedAt)
        diagnostics.coreMLModelLoaded = detector.modelLoaded
        diagnostics.coreMLAttempted = detector.attempted
        diagnostics.coreMLRawObservationCount = detector.rawObservationCount
        diagnostics.coreMLCandidateBoxCount = detector.candidateBoxCount
        diagnostics.coreMLAfterNMSBoxCount = detector.nmsBoxCount

        if !detector.boxes.isEmpty {
            let textStartedAt = Date()
            let localizedLines = recognizeTextLinesInDetectorBoxes(cgImage: cg, boxes: detector.boxes, fast: fast)
            diagnostics.textRecognitionScope = "detector_boxes"
            diagnostics.textRecognitionDurationMs = durationMs(since: textStartedAt)
            diagnostics.ocrTextLineCount = localizedLines.count
            detector.regions = readingOrderIndexed(detectorRegions(for: detector.boxes, lines: localizedLines, allowTextless: true))
            detector.matchedRegionCount = detector.regions.filter {
                !($0.ocrText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }.count
            diagnostics.coreMLMatchedRegionCount = detector.matchedRegionCount
            diagnostics.selectedSegmenter = "coreml_v4_detector_first"
            diagnostics.fallbackToOCR = false
            diagnostics.selectedRegionCount = detector.regions.count
            diagnostics.totalDurationMs = durationMs(since: totalStartedAt)
            return QuestionSegmentationResult(regions: detector.regions, diagnostics: diagnostics)
        }

        let textStartedAt = Date()
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = fast ? .fast : .accurate
        request.usesLanguageCorrection = !fast
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])
        guard (try? handler.perform([request])) != nil,
              let observations = request.results, !observations.isEmpty else {
            diagnostics.textRecognitionDurationMs = durationMs(since: textStartedAt)
            diagnostics.textRecognitionScope = "full_page"
            diagnostics.totalDurationMs = durationMs(since: totalStartedAt)
            return QuestionSegmentationResult(regions: [], diagnostics: diagnostics)
        }
        diagnostics.textRecognitionDurationMs = durationMs(since: textStartedAt)
        diagnostics.textRecognitionScope = "full_page"

        var lines: [TextLine] = []
        for obs in observations {
            guard let candidate = obs.topCandidates(1).first else { continue }
            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let bb = obs.boundingBox
            let rect = CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height)
            lines.append(TextLine(rect: rect, text: text, confidence: Double(candidate.confidence)))
        }
        diagnostics.ocrTextLineCount = lines.count
        guard !lines.isEmpty else {
            diagnostics.totalDurationMs = durationMs(since: totalStartedAt)
            return QuestionSegmentationResult(regions: [], diagnostics: diagnostics)
        }

        let ocrStartedAt = Date()
        let ocrRegions = ocrLayoutQuestionRegions(from: lines)
        diagnostics.ocrV3DurationMs = diagnostics.textRecognitionDurationMs + durationMs(since: ocrStartedAt)
        diagnostics.ocrV3CandidateCount = ocrRegions.count

        let coreMLStartedAt = Date()
        detector = detectorQuestionRegionResult(in: visionImage, cgImage: cg, lines: lines, fast: fast)
        diagnostics.coreMLV4DurationMs = durationMs(since: coreMLStartedAt)
        diagnostics.coreMLModelLoaded = detector.modelLoaded
        diagnostics.coreMLAttempted = detector.attempted
        diagnostics.coreMLRawObservationCount = detector.rawObservationCount
        diagnostics.coreMLCandidateBoxCount = detector.candidateBoxCount
        diagnostics.coreMLAfterNMSBoxCount = detector.nmsBoxCount
        diagnostics.coreMLMatchedRegionCount = detector.matchedRegionCount

        let pairing = regionPairingMetrics(ocrRegions: ocrRegions, coreMLRegions: detector.regions)
        diagnostics.pairedMatchCount = pairing.matched
        diagnostics.pairedMeanIoU = pairing.meanIoU
        diagnostics.pairedMedianIoU = pairing.medianIoU
        diagnostics.unmatchedOCRV3Count = pairing.unmatchedOCR
        diagnostics.unmatchedCoreMLV4Count = pairing.unmatchedCoreML

        let selected = detector.regions.isEmpty ? ocrRegions : detector.regions
        diagnostics.selectedSegmenter = detector.regions.isEmpty ? "ocr_v3" : "coreml_v4"
        diagnostics.fallbackToOCR = detector.regions.isEmpty
        diagnostics.selectedRegionCount = selected.count
        diagnostics.totalDurationMs = durationMs(since: totalStartedAt)
        return QuestionSegmentationResult(regions: selected, diagnostics: diagnostics)
    }

    static func segmentForPreviewOverlayWithDiagnostics(_ image: UIImage, fast: Bool = false) -> QuestionSegmentationResult {
        let base = normalizedUp(image)
        guard let quad = detectDocumentQuad(base),
              quadArea(quad) >= 0.35,
              !cornersTooObtuse(quad),
              !deformationTooSmall(quad) else {
            return segmentWithDiagnostics(base, fast: fast)
        }
        let corrected = rectify(base, quad: quad)
        guard corrected.didCorrect else {
            return segmentWithDiagnostics(base, fast: fast)
        }
        var result = segmentWithDiagnostics(corrected.image, fast: fast)
        result.regions = result.regions.compactMap { region in
            let mapped = mapCorrectedRectToOriginal(region.normalizedRect, quad: quad)
            guard mapped.width * mapped.height >= 0.006 else { return nil }
            var copy = region
            copy.normalizedRect = mapped
            return copy
        }
        result.diagnostics.didCorrect = true
        result.diagnostics.selectedRegionCount = result.regions.count
        return result
    }

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

    private struct DetectorBox {
        var rect: CGRect
        var confidence: Double
    }

    private struct DetectorQuestionRegionResult {
        var regions: [QuestionRegion] = []
        var boxes: [DetectorBox] = []
        var modelLoaded = false
        var attempted = false
        var rawObservationCount = 0
        var candidateBoxCount = 0
        var nmsBoxCount = 0
        var matchedRegionCount = 0
    }

    private static func durationMs(since start: Date) -> Int {
        max(0, Int(Date().timeIntervalSince(start) * 1000))
    }

    private static func makeTextRecognitionRequest(fast: Bool) -> VNRecognizeTextRequest {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = fast ? .fast : .accurate
        request.usesLanguageCorrection = !fast
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        return request
    }

    private static func textLines(from observations: [VNRecognizedTextObservation], mapRect: (CGRect) -> CGRect) -> [TextLine] {
        var lines: [TextLine] = []
        for obs in observations {
            guard let candidate = obs.topCandidates(1).first else { continue }
            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let bb = obs.boundingBox
            let rect = mapRect(CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height))
            lines.append(TextLine(rect: clampRect(rect), text: text, confidence: Double(candidate.confidence)))
        }
        return lines
    }

    private static func recognizeFullPageTextLines(cgImage: CGImage, orientation: CGImagePropertyOrientation, fast: Bool) -> [TextLine] {
        let request = makeTextRecognitionRequest(fast: fast)
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        guard (try? handler.perform([request])) != nil,
              let observations = request.results, !observations.isEmpty else {
            return []
        }
        return textLines(from: observations) { $0 }
    }

    private static func recognizeTextLinesInDetectorBoxes(cgImage: CGImage, boxes: [DetectorBox], fast: Bool) -> [TextLine] {
        guard !boxes.isEmpty else { return [] }
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        var lines: [TextLine] = []
        for box in boxes {
            let padded = clampRect(box.rect.insetBy(dx: -0.012, dy: -0.012))
            let pixelRect = CGRect(
                x: padded.minX * imageWidth,
                y: padded.minY * imageHeight,
                width: padded.width * imageWidth,
                height: padded.height * imageHeight
            ).integral
            guard pixelRect.width >= 4,
                  pixelRect.height >= 4,
                  let crop = cgImage.cropping(to: pixelRect) else {
                continue
            }
            let request = makeTextRecognitionRequest(fast: fast)
            let handler = VNImageRequestHandler(cgImage: crop, orientation: .up, options: [:])
            guard (try? handler.perform([request])) != nil,
                  let observations = request.results, !observations.isEmpty else {
                continue
            }
            lines.append(contentsOf: textLines(from: observations) { cropRect in
                CGRect(
                    x: padded.minX + cropRect.minX * padded.width,
                    y: padded.minY + cropRect.minY * padded.height,
                    width: cropRect.width * padded.width,
                    height: cropRect.height * padded.height
                )
            })
        }
        return lines
    }

    private static func ocrLayoutQuestionRegions(from inputLines: [TextLine]) -> [QuestionRegion] {
        var lines = inputLines
        guard !lines.isEmpty else { return [] }
        lines.sort { $0.rect.minY < $1.rect.minY }
        let avgHeight = lines.map { $0.rect.height }.reduce(0, +) / CGFloat(lines.count)
        let pad: CGFloat = 0.012
        let layout = questionLayout(for: lines, avgHeight: avgHeight)
        var regions: [QuestionRegion] = []

        func makeRegion(_ block: [TextLine], yBottomOverride: CGFloat?, isTerminalBlock: Bool) -> QuestionRegion? {
            guard let first = block.first else { return nil }
            var union = first.rect
            for line in block.dropFirst() { union = union.union(line.rect) }
            let expanded = expandedQuestionRect(
                for: union,
                block: block,
                layout: layout,
                yBottomOverride: yBottomOverride,
                isTerminalBlock: isTerminalBlock
            )
            let padded = clampRect(expanded.insetBy(dx: -pad, dy: -pad))
            guard padded.width * padded.height >= 0.012 else { return nil }
            let text = block.map { $0.text }.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            let confidence = block.map { $0.confidence }.reduce(0, +) / Double(block.count)
            return QuestionRegion(normalizedRect: padded, index: 0, ocrText: text, confidence: confidence)
        }

        let anchors = lines.indices.filter { matchesQuestionNumber(lines[$0].text) }
        if anchors.count >= 2 {
            for (anchorOffset, start) in anchors.enumerated() {
                let end = (anchorOffset + 1 < anchors.count) ? anchors[anchorOffset + 1] : lines.count
                let yBottom = (anchorOffset + 1 < anchors.count) ? lines[anchors[anchorOffset + 1]].rect.minY - pad : nil
                if let region = makeRegion(Array(lines[start..<end]), yBottomOverride: yBottom, isTerminalBlock: anchorOffset + 1 == anchors.count) {
                    regions.append(region)
                }
            }
        } else {
            var blocks: [[TextLine]] = []
            var current: [TextLine] = []
            for line in lines {
                if current.isEmpty {
                    current = [line]
                    continue
                }
                let previousMaxY = current.map { $0.rect.maxY }.max() ?? line.rect.minY
                let gap = line.rect.minY - previousMaxY
                if gap > avgHeight * 1.6 || matchesQuestionNumber(line.text) {
                    blocks.append(current)
                    current = [line]
                } else {
                    current.append(line)
                }
            }
            if !current.isEmpty { blocks.append(current) }
            for (index, block) in blocks.enumerated() {
                let nextTop = (index + 1 < blocks.count) ? blocks[index + 1].first?.rect.minY : nil
                let yBottom = nextTop.map { $0 - pad }
                if let region = makeRegion(block, yBottomOverride: yBottom, isTerminalBlock: index + 1 == blocks.count) {
                    regions.append(region)
                }
            }
        }
        return readingOrderIndexed(regions)
    }

    private static func detectorQuestionRegionResult(
        in image: UIImage,
        cgImage: CGImage,
        lines: [TextLine],
        fast: Bool,
        allowTextless: Bool = false
    ) -> DetectorQuestionRegionResult {
        var result = DetectorQuestionRegionResult()
        guard let model = optionalDetectorModel() else { return result }
        result.modelLoaded = true
        result.attempted = true
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: image.cgImagePropertyOrientation, options: [:])
        guard (try? handler.perform([request])) != nil else { return result }

        let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
        result.rawObservationCount = observations.count
        let minConfidence: VNConfidence = fast ? detectorFastMinConfidence : detectorAccurateMinConfidence
        let boxes = observations.compactMap { observation -> DetectorBox? in
            guard observation.confidence >= minConfidence else { return nil }
            if let label = observation.labels.first,
               !label.identifier.isEmpty,
               label.identifier != detectorCategoryName,
               label.identifier != "question",
               label.identifier != "question_region" {
                return nil
            }
            let bb = observation.boundingBox
            let rect = clampRect(CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height))
            guard rect.width * rect.height >= 0.012 else { return nil }
            return DetectorBox(rect: rect, confidence: Double(observation.confidence))
        }
        result.candidateBoxCount = boxes.count
        let selectedBoxes = nonMaxSuppressed(boxes, overlapThreshold: detectorNMSOverlapThreshold)
        result.nmsBoxCount = selectedBoxes.count
        result.boxes = selectedBoxes
        let regions = detectorRegions(for: selectedBoxes, lines: lines, allowTextless: allowTextless)
        result.regions = readingOrderIndexed(regions)
        result.matchedRegionCount = result.regions.filter {
            !($0.ocrText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.count
        return result
    }

    private static func detectorRegions(for boxes: [DetectorBox], lines: [TextLine], allowTextless: Bool) -> [QuestionRegion] {
        var regions: [QuestionRegion] = []
        for box in boxes {
            let matchedLines = linesForQuestionBox(box.rect, lines: lines)
            let text = matchedLines.map { $0.text }.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            let textConfidence = matchedLines.isEmpty ? 0 : matchedLines.map { $0.confidence }.reduce(0, +) / Double(matchedLines.count)
            if !text.isEmpty || allowTextless {
                regions.append(
                    QuestionRegion(
                        normalizedRect: box.rect,
                        index: 0,
                        ocrText: text.isEmpty ? nil : text,
                        confidence: max(box.confidence, textConfidence)
                    )
                )
            }
        }
        return regions
    }

    private static func regionPairingMetrics(
        ocrRegions: [QuestionRegion],
        coreMLRegions: [QuestionRegion]
    ) -> (matched: Int, meanIoU: Double, medianIoU: Double, unmatchedOCR: Int, unmatchedCoreML: Int) {
        guard !ocrRegions.isEmpty, !coreMLRegions.isEmpty else {
            return (0, 0, 0, ocrRegions.count, coreMLRegions.count)
        }
        var pairs: [(ocr: Int, core: Int, iou: CGFloat)] = []
        for (ocrIndex, ocr) in ocrRegions.enumerated() {
            for (coreIndex, core) in coreMLRegions.enumerated() {
                let iou = intersectionOverUnion(ocr.normalizedRect, core.normalizedRect)
                if iou >= 0.30 {
                    pairs.append((ocrIndex, coreIndex, iou))
                }
            }
        }
        pairs.sort { $0.iou > $1.iou }
        var usedOCR = Set<Int>()
        var usedCore = Set<Int>()
        var matchedIoUs: [Double] = []
        for pair in pairs {
            guard !usedOCR.contains(pair.ocr), !usedCore.contains(pair.core) else { continue }
            usedOCR.insert(pair.ocr)
            usedCore.insert(pair.core)
            matchedIoUs.append(Double(pair.iou))
        }
        let mean = matchedIoUs.isEmpty ? 0 : matchedIoUs.reduce(0, +) / Double(matchedIoUs.count)
        let sorted = matchedIoUs.sorted()
        let median: Double
        if sorted.isEmpty {
            median = 0
        } else if sorted.count % 2 == 0 {
            median = (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        } else {
            median = sorted[sorted.count / 2]
        }
        return (
            matchedIoUs.count,
            roundedDiagnosticDouble(mean),
            roundedDiagnosticDouble(median),
            max(0, ocrRegions.count - usedOCR.count),
            max(0, coreMLRegions.count - usedCore.count)
        )
    }

    private static func roundedDiagnosticDouble(_ value: Double) -> Double {
        guard value.isFinite else { return 0 }
        return (value * 10_000).rounded() / 10_000
    }

    private static func detectorQuestionRegions(in image: UIImage, cgImage: CGImage, lines: [TextLine], fast: Bool) -> [QuestionRegion]? {
        guard !lines.isEmpty, let model = optionalDetectorModel() else { return nil }
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: image.cgImagePropertyOrientation, options: [:])
        guard (try? handler.perform([request])) != nil else { return nil }

        let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
        let minConfidence: VNConfidence = fast ? detectorFastMinConfidence : detectorAccurateMinConfidence
        let boxes = observations.compactMap { observation -> DetectorBox? in
            guard observation.confidence >= minConfidence else { return nil }
            if let label = observation.labels.first,
               !label.identifier.isEmpty,
               label.identifier != detectorCategoryName,
               label.identifier != "question",
               label.identifier != "question_region" {
                return nil
            }
            let bb = observation.boundingBox
            let rect = clampRect(CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height))
            guard rect.width * rect.height >= 0.012 else { return nil }
            return DetectorBox(rect: rect, confidence: Double(observation.confidence))
        }
        guard !boxes.isEmpty else { return nil }

        var regions: [QuestionRegion] = []
        for box in nonMaxSuppressed(boxes, overlapThreshold: detectorNMSOverlapThreshold) {
            let matchedLines = linesForQuestionBox(box.rect, lines: lines)
            guard !matchedLines.isEmpty else { continue }
            let text = matchedLines.map { $0.text }.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let textConfidence = matchedLines.map { $0.confidence }.reduce(0, +) / Double(matchedLines.count)
            regions.append(
                QuestionRegion(
                    normalizedRect: box.rect,
                    index: 0,
                    ocrText: text,
                    confidence: max(box.confidence, textConfidence)
                )
            )
        }
        return regions.isEmpty ? nil : readingOrderIndexed(regions)
    }

    private static func optionalDetectorModel() -> VNCoreMLModel? {
        detectorModelLock.lock()
        defer { detectorModelLock.unlock() }
        if didAttemptDetectorLoad { return cachedDetectorModel }
        didAttemptDetectorLoad = true
        guard let url = Bundle.main.url(forResource: "QuestionRegionDetector", withExtension: "mlmodelc") else {
            return nil
        }
        guard let mlModel = try? MLModel(contentsOf: url),
              let visionModel = try? VNCoreMLModel(for: mlModel) else {
            return nil
        }
        cachedDetectorModel = visionModel
        return visionModel
    }

    private static func linesForQuestionBox(_ box: CGRect, lines: [TextLine]) -> [TextLine] {
        let expanded = clampRect(box.insetBy(dx: -0.012, dy: -0.012))
        return lines.filter { line in
            let center = CGPoint(x: line.rect.midX, y: line.rect.midY)
            if expanded.contains(center) { return true }
            return intersectionArea(expanded, line.rect) / max(0.0001, line.rect.width * line.rect.height) >= 0.45
        }
    }

    private static func nonMaxSuppressed(
        _ boxes: [DetectorBox],
        overlapThreshold: CGFloat
    ) -> [DetectorBox] {
        var selected: [DetectorBox] = []
        for box in boxes.sorted(by: { $0.confidence > $1.confidence }) {
            if selected.contains(where: { intersectionOverUnion(box.rect, $0.rect) >= overlapThreshold }) {
                continue
            }
            selected.append(box)
        }
        return selected
    }

    private static func readingOrderIndexed(_ input: [QuestionRegion]) -> [QuestionRegion] {
        var regions = input
        regions.sort { lhs, rhs in
            if abs(lhs.normalizedRect.minY - rhs.normalizedRect.minY) > 0.04 {
                return lhs.normalizedRect.minY < rhs.normalizedRect.minY
            }
            return lhs.normalizedRect.minX < rhs.normalizedRect.minX
        }
        for i in regions.indices { regions[i].index = i + 1 }
        return regions
    }

    // Layout estimate used to grow OCR-only boxes into practical question crops.
    private struct QuestionLayout {
        var contentRect: CGRect
        var columns: [CGRect]
        var medianLineHeight: CGFloat
    }

    private static func questionLayout(for lines: [TextLine], avgHeight: CGFloat) -> QuestionLayout {
        guard let textRect = unionRect(for: lines) else {
            let fallback = CGRect(x: 0.06, y: 0.04, width: 0.88, height: 0.88)
            return QuestionLayout(contentRect: fallback, columns: [fallback], medianLineHeight: 0.018)
        }

        let safeAvgHeight = max(avgHeight, 0.012)
        let medianHeight = max(median(lines.map { $0.rect.height }), safeAvgHeight)
        let xPad = max(0.026, min(0.06, medianHeight * 2.2))
        let safeMinX: CGFloat = 0.025
        let safeMaxX: CGFloat = 0.975
        let desiredWidth = min(safeMaxX - safeMinX, max(textRect.width + xPad * 2, 0.68))
        let xSpan = boundedSpan(center: textRect.midX, length: desiredWidth, lower: safeMinX, upper: safeMaxX)

        let topPad = max(0.012, medianHeight * 1.2)
        let bottomPad = max(0.09, medianHeight * 5.0)
        let minContentHeight = max(0.30, medianHeight * 10.0)
        let minY = max(0, textRect.minY - topPad)
        let maxY = min(0.985, max(textRect.maxY + bottomPad, minY + minContentHeight))
        let contentRect = CGRect(x: xSpan.min, y: minY, width: xSpan.max - xSpan.min, height: max(0, maxY - minY))
        let columns = questionColumns(for: lines, contentRect: contentRect, medianHeight: medianHeight)

        return QuestionLayout(
            contentRect: contentRect,
            columns: columns.isEmpty ? [contentRect] : columns,
            medianLineHeight: medianHeight
        )
    }

    private static func expandedQuestionRect(
        for textRect: CGRect,
        block: [TextLine],
        layout: QuestionLayout,
        yBottomOverride: CGFloat?,
        isTerminalBlock: Bool
    ) -> CGRect {
        let column = questionColumn(for: textRect, block: block, layout: layout)
        var minX = textRect.minX
        var maxX = textRect.maxX

        if textRect.width < column.width * 0.92 {
            minX = column.minX
            maxX = column.maxX
        } else {
            let sidePad = min(0.02, column.width * 0.035)
            minX = max(column.minX, textRect.minX - sidePad)
            maxX = min(column.maxX, textRect.maxX + sidePad)
        }

        let minY = textRect.minY
        var maxY = textRect.maxY
        var hardBottom: CGFloat?
        if let yb = yBottomOverride, yb > textRect.maxY {
            let boundary = min(0.985, yb)
            hardBottom = boundary
            maxY = max(maxY, boundary)
        }

        let terminalWithoutBoundary = isTerminalBlock && hardBottom == nil
        let minHeight = minimumQuestionHeight(for: textRect, column: column, layout: layout, terminal: terminalWithoutBoundary)
        if maxY - minY < minHeight {
            let desiredBottom = minY + minHeight
            if let hardBottom = hardBottom {
                maxY = min(hardBottom, max(maxY, desiredBottom))
            } else {
                maxY = max(maxY, desiredBottom)
            }
        }

        if terminalWithoutBoundary {
            maxY = max(maxY, terminalQuestionBottom(for: textRect, minY: minY, column: column, layout: layout))
            let generatedHeightCap = min(max(0.42, layout.medianLineHeight * 14.0), 0.56)
            maxY = min(maxY, minY + max(textRect.height, generatedHeightCap))
        }

        maxY = min(0.985, maxY)
        return clampRect(CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY)))
    }

    private static func questionColumns(for lines: [TextLine], contentRect: CGRect, medianHeight: CGFloat) -> [CGRect] {
        let candidates = lines.filter { line in
            line.rect.width <= contentRect.width * 0.72 &&
            line.rect.midX >= contentRect.minX &&
            line.rect.midX <= contentRect.maxX
        }
        guard candidates.count >= 4 else { return [contentRect] }

        let centers = candidates.map { $0.rect.midX }.sorted()
        var bestGap: CGFloat = 0
        var bestIndex: Int?
        for idx in 0..<(centers.count - 1) {
            let leftCount = idx + 1
            let rightCount = centers.count - leftCount
            guard leftCount >= 2, rightCount >= 2 else { continue }
            let gap = centers[idx + 1] - centers[idx]
            if gap > bestGap {
                bestGap = gap
                bestIndex = idx
            }
        }

        guard let splitIndex = bestIndex else { return [contentRect] }
        let minGap = max(0.16, contentRect.width * 0.22)
        guard bestGap >= minGap else { return [contentRect] }

        let splitX = (centers[splitIndex] + centers[splitIndex + 1]) / 2
        let crossingCount = lines.filter { $0.rect.minX < splitX && $0.rect.maxX > splitX }.count
        guard crossingCount <= max(1, lines.count / 5) else { return [contentRect] }

        let leftLines = lines.filter { $0.rect.midX < splitX }
        let rightLines = lines.filter { $0.rect.midX >= splitX }
        guard !leftLines.isEmpty, !rightLines.isEmpty else { return [contentRect] }

        let leftWidth = splitX - contentRect.minX
        let rightWidth = contentRect.maxX - splitX
        guard leftWidth >= 0.24, rightWidth >= 0.24 else { return [contentRect] }

        return [
            questionColumnRect(xMin: contentRect.minX, xMax: splitX, lines: leftLines, contentRect: contentRect, medianHeight: medianHeight),
            questionColumnRect(xMin: splitX, xMax: contentRect.maxX, lines: rightLines, contentRect: contentRect, medianHeight: medianHeight)
        ]
    }

    private static func questionColumnRect(xMin: CGFloat, xMax: CGFloat, lines: [TextLine], contentRect: CGRect, medianHeight: CGFloat) -> CGRect {
        let bottomPad = max(0.09, medianHeight * 5.0)
        let minColumnHeight = max(0.30, medianHeight * 10.0)
        let lineMaxY = lines.map { $0.rect.maxY }.max() ?? contentRect.maxY
        let maxY = min(0.985, max(lineMaxY + bottomPad, contentRect.minY + minColumnHeight))
        return CGRect(x: xMin, y: contentRect.minY, width: max(0, xMax - xMin), height: max(0, maxY - contentRect.minY))
    }

    private static func questionColumn(for textRect: CGRect, block: [TextLine], layout: QuestionLayout) -> CGRect {
        guard layout.columns.count > 1 else { return layout.contentRect }
        if textRect.width >= layout.contentRect.width * 0.70 { return layout.contentRect }

        var best = layout.columns[0]
        var bestScore = -CGFloat.greatestFiniteMagnitude
        for column in layout.columns {
            var score = horizontalOverlap(textRect, column) * 2.0
            for line in block {
                if line.rect.midX >= column.minX && line.rect.midX <= column.maxX {
                    score += 0.05
                }
            }
            score -= abs(textRect.midX - column.midX) * 0.1
            if score > bestScore {
                bestScore = score
                best = column
            }
        }
        return best
    }

    private static func minimumQuestionHeight(for textRect: CGRect, column: CGRect, layout: QuestionLayout, terminal: Bool) -> CGFloat {
        let byLine = layout.medianLineHeight * (terminal ? 8.0 : 5.0)
        let byColumn = column.width * (terminal ? 0.30 : 0.20)
        let floor: CGFloat = terminal ? 0.20 : 0.12
        let cap: CGFloat = terminal ? 0.38 : 0.28
        let desired = max(floor, max(byLine, byColumn))
        return min(max(textRect.height, desired), cap)
    }

    private static func terminalQuestionBottom(for textRect: CGRect, minY: CGFloat, column: CGRect, layout: QuestionLayout) -> CGFloat {
        let extra = max(0.07, layout.medianLineHeight * 3.5)
        let desired = max(column.maxY, textRect.maxY + extra)
        return min(0.985, max(desired, minY + minimumQuestionHeight(for: textRect, column: column, layout: layout, terminal: true)))
    }

    private static func unionRect(for lines: [TextLine]) -> CGRect? {
        guard let first = lines.first else { return nil }
        var rect = first.rect
        for line in lines.dropFirst() {
            rect = rect.union(line.rect)
        }
        return rect
    }

    private static func median(_ values: [CGFloat]) -> CGFloat {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    private static func boundedSpan(center: CGFloat, length: CGFloat, lower: CGFloat, upper: CGFloat) -> (min: CGFloat, max: CGFloat) {
        let available = max(0, upper - lower)
        let span = min(max(0, length), available)
        var minValue = center - span / 2
        var maxValue = center + span / 2
        if minValue < lower {
            maxValue += lower - minValue
            minValue = lower
        }
        if maxValue > upper {
            minValue -= maxValue - upper
            maxValue = upper
        }
        return (max(lower, minValue), min(upper, maxValue))
    }

    private static func horizontalOverlap(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        max(0, min(lhs.maxX, rhs.maxX) - max(lhs.minX, rhs.minX))
    }

    private static func intersectionArea(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let width = max(0, min(lhs.maxX, rhs.maxX) - max(lhs.minX, rhs.minX))
        let height = max(0, min(lhs.maxY, rhs.maxY) - max(lhs.minY, rhs.minY))
        return width * height
    }

    private static func intersectionOverUnion(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let intersection = intersectionArea(lhs, rhs)
        guard intersection > 0 else { return 0 }
        let union = max(0.0001, lhs.width * lhs.height + rhs.width * rhs.height - intersection)
        return intersection / union
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
