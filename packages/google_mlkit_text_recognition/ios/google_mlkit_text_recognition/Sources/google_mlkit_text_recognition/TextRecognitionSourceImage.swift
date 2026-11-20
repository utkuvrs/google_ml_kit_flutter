import Flutter
import MLKitVision
import UIKit
import CoreGraphics
import CoreImage
import CoreVideo

/// The decoded input image, shared by ML Kit and Apple Vision when `enableIosConfidence` is on.
///
/// Decoding mirrors `VisionImage.visionImage(from:)` in `MLKVisionImage+FlutterPlugin.swift`, so ML Kit
/// receives the same pixels and orientation as with the flag off, but the image is decoded/rendered once
/// and the resulting `CGImage` is reused for the Vision pass instead of being rebuilt.
struct TextRecognitionSourceImage {
  let visionImage: VisionImage
  let cgImage: CGImage
  /// The orientation ML Kit applies to `cgImage`; Vision must use the same one so boxes line up.
  let orientation: CGImagePropertyOrientation

  /// Creating a `CIContext` is expensive; Apple recommends reusing one.
  private static let ciContext = CIContext(options: nil)

  private static let releasePixelBufferBytes: CVPixelBufferReleaseBytesCallback = { _, baseAddress in
    guard let baseAddress = baseAddress else { return }
    UnsafeMutableRawPointer(mutating: baseAddress).deallocate()
  }

  static func make(from imageData: [String: Any]) -> TextRecognitionSourceImage? {
    guard let imageType = imageData["type"] as? String else { return nil }
    switch imageType {
    case "file":
      guard let path = imageData["path"] as? String,
            let uiImage = UIImage(contentsOfFile: path) else { return nil }
      return fromUIImage(uiImage)
    case "bytes":
      return fromBytes(imageData)
    case "bitmap":
      return fromBitmap(imageData)
    default:
      return nil
    }
  }

  /// `file` and `bitmap`: ML Kit honours the UIImage orientation, so Vision must too.
  private static func fromUIImage(_ uiImage: UIImage) -> TextRecognitionSourceImage? {
    guard let cgImage = uiImage.cgImage else { return nil }
    let visionImage = VisionImage(image: uiImage)
    visionImage.orientation = uiImage.imageOrientation
    return TextRecognitionSourceImage(
      visionImage: visionImage,
      cgImage: cgImage,
      orientation: cgOrientation(uiImage.imageOrientation)
    )
  }

  /// `bytes`: camera frames. Like commons, build an upright image and ignore rotation metadata.
  private static func fromBytes(_ imageData: [String: Any]) -> TextRecognitionSourceImage? {
    guard let byteData = imageData["bytes"] as? FlutterStandardTypedData,
          let metadata = imageData["metadata"] as? [String: Any],
          let width = metadata["width"] as? NSNumber,
          let height = metadata["height"] as? NSNumber,
          let rawFormat = metadata["image_format"] as? NSNumber,
          let bytesPerRow = metadata["bytes_per_row"] as? NSNumber else {
      return nil
    }
    let widthValue = Int(truncating: width)
    let heightValue = Int(truncating: height)
    let bytesPerRowValue = Int(truncating: bytesPerRow)
    let bufferSize = bytesPerRowValue * heightValue
    guard bufferSize > 0, byteData.data.count >= bufferSize else { return nil }

    let copy = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: 1)
    let copyBuffer = UnsafeMutableBufferPointer(start: copy.assumingMemoryBound(to: UInt8.self), count: bufferSize)
    byteData.data.copyBytes(to: copyBuffer)

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault,
      widthValue,
      heightValue,
      OSType(truncating: rawFormat),
      copy,
      bytesPerRowValue,
      releasePixelBufferBytes,
      nil,
      nil,
      &pixelBuffer
    )
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      copy.deallocate()
      return nil
    }
    // ARC releases `buffer` (and frees `copy` via the callback) once the CGImage has been rendered.
    let ciImage = CIImage(cvPixelBuffer: buffer)
    guard let cgImage = ciContext.createCGImage(
      ciImage,
      from: CGRect(x: 0, y: 0, width: widthValue, height: heightValue)
    ) else {
      return nil
    }
    return TextRecognitionSourceImage(
      visionImage: VisionImage(image: UIImage(cgImage: cgImage)),
      cgImage: cgImage,
      orientation: .up
    )
  }

  private static func fromBitmap(_ imageData: [String: Any]) -> TextRecognitionSourceImage? {
    guard let bitmapData = imageData["bitmapData"] as? FlutterStandardTypedData else { return nil }
    if let metadata = imageData["metadata"] as? [String: Any],
       let width = metadata["width"] as? NSNumber,
       let height = metadata["height"] as? NSNumber {
      var rendered: CGImage?
      bitmapData.data.withUnsafeBytes { rawBuffer in
        guard let rawData = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self),
              let context = CGContext(
                data: UnsafeMutableRawPointer(mutating: rawData),
                width: width.intValue,
                height: height.intValue,
                bitsPerComponent: 8,
                bytesPerRow: 4 * width.intValue,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
              ) else { return }
        rendered = context.makeImage()
      }
      if let rendered = rendered {
        return fromUIImage(UIImage(cgImage: rendered))
      }
    }
    guard let uiImage = UIImage(data: bitmapData.data) else { return nil }
    return fromUIImage(uiImage)
  }

  /// UIImage.Orientation and CGImagePropertyOrientation have different raw values; map explicitly.
  private static func cgOrientation(_ orientation: UIImage.Orientation) -> CGImagePropertyOrientation {
    switch orientation {
    case .up: return .up
    case .down: return .down
    case .left: return .left
    case .right: return .right
    case .upMirrored: return .upMirrored
    case .downMirrored: return .downMirrored
    case .leftMirrored: return .leftMirrored
    case .rightMirrored: return .rightMirrored
    @unknown default: return .up
    }
  }
}
