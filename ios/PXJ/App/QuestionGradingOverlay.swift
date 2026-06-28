import SwiftUI
import UIKit

/// 整页批改：后端 grade-page 返回的逐题判分（与端上 QuestionRegion 按阅读顺序 index 对齐）。
struct GradedQuestion: Identifiable, Equatable {
    let id = UUID()
    var index: Int
    var questionText: String
    var studentAnswer: String
    var verdict: String        // 对/错/部分对/不确定/未作答/未识别
    var gradable: Bool         // false=几何/读图题，不硬判对错，只给讲解
    var correctAnswer: String
    var correction: String
    var errorReason: String
    var knowledge: String

    /// 叠在题上的标记。仅 gradable 题显示对/错；其余一律「?」（不显红叉，避免误判）。
    var mark: GradeMark {
        guard gradable else { return .uncertain }
        switch verdict {
        case "对": return .correct
        case "错": return .wrong
        case "部分对": return .partial
        case "未作答": return .blank
        default: return .uncertain
        }
    }

    /// 从后端 JSON 字典解析（缺字段安全降级）。
    static func from(_ obj: [String: Any]) -> GradedQuestion {
        func s(_ k: String) -> String { (obj[k] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "" }
        let idx = (obj["index"] as? Int) ?? Int((obj["index"] as? Double) ?? 0)
        return GradedQuestion(
            index: idx,
            questionText: s("question_text"),
            studentAnswer: s("student_answer"),
            verdict: s("verdict"),
            gradable: (obj["gradable"] as? Bool) ?? true,
            correctAnswer: s("correct_answer"),
            correction: s("correction"),
            errorReason: s("error_reason"),
            knowledge: s("knowledge")
        )
    }
}

enum GradeMark {
    case correct, wrong, partial, uncertain, blank
    var symbol: String {
        switch self {
        case .correct: return "checkmark.circle.fill"
        case .wrong: return "xmark.circle.fill"
        case .partial: return "circle.lefthalf.filled"
        case .uncertain: return "questionmark.circle.fill"
        case .blank: return "minus.circle.fill"
        }
    }
    var color: Color {
        switch self {
        case .correct: return .green
        case .wrong: return .red
        case .partial: return .orange
        case .uncertain: return .gray
        case .blank: return .secondary
        }
    }
    var label: String {
        switch self {
        case .correct: return "对"
        case .wrong: return "错"
        case .partial: return "部分对"
        case .uncertain: return "待讲解"
        case .blank: return "未作答"
        }
    }
}

/// 在冻结的批改图上，按端上 QuestionRegion 的准确 bbox 叠加每题 ✓/✗/? 标记。
/// verdicts 以 region.index 为键（端上分割与后端判分都用阅读顺序 index 对齐）。
struct QuestionGradingOverlay: View {
    let image: UIImage
    let regions: [QuestionRegion]
    let verdicts: [Int: GradedQuestion]
    @Binding var selectedIndex: Int?
    var onSelect: (QuestionRegion) -> Void

    var body: some View {
        GeometryReader { geo in
            let display = QuestionRegionOverlay.imageDisplayRect(imageSize: image.size, container: geo.size)
            ZStack(alignment: .topLeading) {
                Image(uiImage: image).resizable().scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)
                ForEach(regions) { region in
                    markBox(region, in: display)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }

    @ViewBuilder
    private func markBox(_ region: QuestionRegion, in display: CGRect) -> some View {
        let rect = region.normalizedRect
        let w = max(rect.width * display.width, 1)
        let h = max(rect.height * display.height, 1)
        let x = display.minX + rect.minX * display.width
        let y = display.minY + rect.minY * display.height
        let graded = verdicts[region.index]
        let mark = graded?.mark
        let color = mark?.color ?? .accentColor
        let selected = selectedIndex == region.index

        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 6)
                .fill(color.opacity(selected ? 0.20 : 0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(color, lineWidth: selected ? 2.5 : 1.5)
                )
                .frame(width: w, height: h)
                .overlay(alignment: .topLeading) {
                    if let mark {
                        Image(systemName: mark.symbol)
                            .font(.system(size: selected ? 22 : 18, weight: .bold))
                            .foregroundColor(.white)
                            .background(Circle().fill(color).frame(width: selected ? 26 : 22, height: selected ? 26 : 22))
                            .padding(3)
                    } else {
                        // 还没拿到判分：转圈占位
                        ProgressView().scaleEffect(0.6).padding(4)
                    }
                }
        }
        .frame(width: max(w, 44), height: max(h, 44))
        .contentShape(Rectangle())
        .onTapGesture { onSelect(region); selectedIndex = region.index }
        .position(x: x + w / 2, y: y + h / 2)
    }
}
