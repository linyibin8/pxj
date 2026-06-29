import SwiftUI
import UIKit

/// 在 scaledToFit 显示的题图上叠加可点选的题目区域框。
/// 坐标系：region.normalizedRect 为左上原点、[0,1]，相对“校正后图”。
struct QuestionRegionOverlay: View {
    let image: UIImage
    let regions: [QuestionRegion]
    @Binding var selectedID: UUID?
    var onSelect: (QuestionRegion) -> Void

    var body: some View {
        GeometryReader { geo in
            // 先算 scaledToFit 后图片实际显示矩形（含 letterbox 黑边偏移）
            let display = Self.imageDisplayRect(imageSize: image.size, container: geo.size)
            ZStack(alignment: .topLeading) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)

                ForEach(regions) { region in
                    regionBox(region, in: display)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }

    // MARK: - 单个区域

    @ViewBuilder
    private func regionBox(_ region: QuestionRegion, in display: CGRect) -> some View {
        let rect = region.normalizedRect
        let w = max(rect.width * display.width, 1)
        let h = max(rect.height * display.height, 1)
        let x = display.minX + rect.minX * display.width
        let y = display.minY + rect.minY * display.height

        let selected = selectedID == region.id
        let dimmed = (selectedID != nil) && !selected

        // 最小命中热区 ~44pt
        let hitW = max(w, 44)
        let hitH = max(h, 44)

        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.accentColor.opacity(selected ? 0.18 : 0.06))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(Color.accentColor, lineWidth: selected ? 2.5 : 1.5)
                )
                .frame(width: w, height: h)
                .overlay(alignment: .topLeading) {
                    Text("\(region.index)")
                        .font(.system(size: selected ? 13 : 11, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor))
                        .scaleEffect(selected ? 1.18 : 1.0)
                        .padding(4)
                }
        }
        .frame(width: hitW, height: hitH)
        .contentShape(Rectangle())
        .onTapGesture {
            onSelect(region)
            selectedID = region.id
        }
        .opacity(dimmed ? 0.5 : 1.0)
        .position(x: x + w / 2, y: y + h / 2)
    }

    // MARK: - scaledToFit 显示矩形

    /// 计算 Image(.scaledToFit) 在给定容器内的实际显示矩形（含居中 letterbox 偏移）。
    static func imageDisplayRect(imageSize: CGSize, container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              container.width > 0, container.height > 0 else {
            return CGRect(origin: .zero, size: container)
        }
        let imageAspect = imageSize.width / imageSize.height
        let containerAspect = container.width / container.height
        let displaySize: CGSize
        if imageAspect > containerAspect {
            // 宽度受限，上下留黑边
            displaySize = CGSize(width: container.width, height: container.width / imageAspect)
        } else {
            // 高度受限，左右留黑边
            displaySize = CGSize(width: container.height * imageAspect, height: container.height)
        }
        let originX = (container.width - displaySize.width) / 2
        let originY = (container.height - displaySize.height) / 2
        return CGRect(x: originX, y: originY, width: displaySize.width, height: displaySize.height)
    }

    /// 计算 AVCaptureVideoPreviewLayer(.resizeAspectFill) 在容器内的实际图像矩形。
    /// 智能观察实时绿框叠在相机预览层上，必须用 fill 坐标，否则横屏 iPad 上会偏。
    static func imageAspectFillRect(imageSize: CGSize, container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              container.width > 0, container.height > 0 else {
            return CGRect(origin: .zero, size: container)
        }
        let imageAspect = imageSize.width / imageSize.height
        let containerAspect = container.width / container.height
        let displaySize: CGSize
        if imageAspect > containerAspect {
            displaySize = CGSize(width: container.height * imageAspect, height: container.height)
        } else {
            displaySize = CGSize(width: container.width, height: container.width / imageAspect)
        }
        let originX = (container.width - displaySize.width) / 2
        let originY = (container.height - displaySize.height) / 2
        return CGRect(x: originX, y: originY, width: displaySize.width, height: displaySize.height)
    }
}
