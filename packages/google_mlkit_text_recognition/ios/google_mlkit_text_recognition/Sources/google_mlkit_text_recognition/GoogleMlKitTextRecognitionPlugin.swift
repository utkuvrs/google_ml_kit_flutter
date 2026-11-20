import Flutter
import MLKitVision
import MLKitTextRecognition
import MLKitTextRecognitionCommon
import UIKit
#if canImport(MLKitTextRecognitionChinese)
import MLKitTextRecognitionChinese
#endif
#if canImport(MLKitTextRecognitionDevanagari)
import MLKitTextRecognitionDevanagari
#endif
#if canImport(MLKitTextRecognitionJapanese)
import MLKitTextRecognitionJapanese
#endif
#if canImport(MLKitTextRecognitionKorean)
import MLKitTextRecognitionKorean
#endif

@objc
public class GoogleMlKitTextRecognitionPlugin: NSObject, FlutterPlugin {
  private var instances: [String: TextRecognizer] = [:]

  /// Serial background queue for the optional Apple Vision confidence pass, so it never runs on the
  /// platform (main) thread and overlapping calls cannot pile up concurrent Vision requests.
  private static let visionQueue = DispatchQueue(
    label: "google_mlkit_text_recognition.vision",
    qos: .userInitiated
  )

  /// Carries the Vision result from `visionQueue` to the main queue. The `DispatchGroup` used with it
  /// orders the write (before `leave()`) before the read (inside `notify`).
  private final class VisionResultBox {
    var index: VisionConfidenceIndex?
  }

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "google_mlkit_text_recognizer",
      binaryMessenger: registrar.messenger()
    )
    let instance = GoogleMlKitTextRecognitionPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "vision#startTextRecognizer":
      handleDetection(call: call, result: result)
    case "vision#closeTextRecognizer":
      if let args = call.arguments as? [String: Any], let uid = args["id"] as? String {
        instances.removeValue(forKey: uid)
      }
      result(nil)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func initialize(call: FlutterMethodCall) -> TextRecognizer {
    let scriptIndex = (call.arguments as? [String: Any])?["script"] as? Int ?? 0
    switch scriptIndex {
    case 0:
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
    case 1:
      #if canImport(MLKitTextRecognitionChinese)
      return TextRecognizer.textRecognizer(options: ChineseTextRecognizerOptions())
      #else
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
      #endif
    case 2:
      #if canImport(MLKitTextRecognitionDevanagari)
      return TextRecognizer.textRecognizer(options: DevanagariTextRecognizerOptions())
      #else
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
      #endif
    case 3:
      #if canImport(MLKitTextRecognitionJapanese)
      return TextRecognizer.textRecognizer(options: JapaneseTextRecognizerOptions())
      #else
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
      #endif
    case 4:
      #if canImport(MLKitTextRecognitionKorean)
      return TextRecognizer.textRecognizer(options: KoreanTextRecognizerOptions())
      #else
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
      #endif
    default:
      return TextRecognizer.textRecognizer(options: TextRecognizerOptions())
    }
  }

  private func handleDetection(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let imageData = args["imageData"] as? [String: Any],
          let uid = args["id"] as? String else {
      result(FlutterError(code: "invalid_args", message: "Missing arguments", details: nil))
      return
    }
    let enableConfidence = args["enableIosConfidence"] as? Bool ?? false
    let scriptIndex = args["script"] as? Int ?? 0

    let visionGroup = DispatchGroup()
    let visionResult = VisionResultBox()
    let image: VisionImage
    if enableConfidence {
      // One decode/render shared by ML Kit and Apple Vision.
      guard let source = TextRecognitionSourceImage.make(from: imageData) else {
        result(FlutterError(code: "invalid_image", message: "Invalid or missing image data", details: nil))
        return
      }
      image = source.visionImage
      let languages = VisionTextConfidence.languages(forScript: scriptIndex)
      if !languages.isEmpty {
        visionGroup.enter()
        Self.visionQueue.async {
          visionResult.index = VisionTextConfidence.recognize(
            cgImage: source.cgImage,
            orientation: source.orientation,
            languages: languages
          )
          visionGroup.leave()
        }
      }
    } else {
      guard let plainImage = VisionImage.visionImage(from: imageData) else {
        result(FlutterError(code: "invalid_image", message: "Invalid or missing image data", details: nil))
        return
      }
      image = plainImage
    }

    let recognizer: TextRecognizer
    if let existing = instances[uid] {
      recognizer = existing
    } else {
      recognizer = initialize(call: call)
      instances[uid] = recognizer
    }

    // ML Kit runs concurrently with the Vision pass; results are joined on the main queue.
    recognizer.process(image) { visionText, error in
      visionGroup.notify(queue: .main) {
        if let error = error as NSError? {
          result(FlutterError(code: "Error \(error.code)", message: error.domain, details: error.localizedDescription))
          return
        }
        guard let visionText = visionText else {
          result(["text": "", "blocks": [] as [[String: Any]]])
          return
        }
        result(self.buildResult(visionText, confidenceIndex: visionResult.index))
      }
    }
  }

  private func buildResult(_ visionText: Text, confidenceIndex: VisionConfidenceIndex?) -> [String: Any] {
    var textBlocks: [[String: Any]] = []
    for block in visionText.blocks {
      var blockData = addData(
        cornerPoints: block.cornerPoints,
        frame: block.frame,
        languages: block.recognizedLanguages,
        text: block.text,
        confidence: nil,
        angle: nil
      )
      blockData["lines"] = block.lines.map { lineData(for: $0, confidenceIndex: confidenceIndex) }
      textBlocks.append(blockData)
    }
    return ["text": visionText.text, "blocks": textBlocks]
  }

  private func lineData(for line: TextLine, confidenceIndex: VisionConfidenceIndex?) -> [String: Any] {
    var elementsData: [[String: Any]] = []
    var elementConfidences: [Float] = []
    for element in line.elements {
      let elementConfidence = confidenceIndex?.confidence(for: element.frame, text: element.text)
      var elementData = addData(
        cornerPoints: element.cornerPoints,
        frame: element.frame,
        languages: element.recognizedLanguages,
        text: element.text,
        confidence: elementConfidence.map { NSNumber(value: $0) },
        angle: angle(from: element.cornerPoints)
      )
      elementData["symbols"] = [] as [[String: Any]]
      elementsData.append(elementData)
      if let confidence = elementConfidence {
        elementConfidences.append(confidence)
      }
    }
    // Line confidence is the mean of its matched elements; fall back to matching the whole line.
    let lineConfidence: Float?
    if elementConfidences.isEmpty {
      lineConfidence = confidenceIndex?.confidence(for: line.frame, text: line.text)
    } else {
      lineConfidence = elementConfidences.reduce(0, +) / Float(elementConfidences.count)
    }
    var data = addData(
      cornerPoints: line.cornerPoints,
      frame: line.frame,
      languages: line.recognizedLanguages,
      text: line.text,
      confidence: lineConfidence.map { NSNumber(value: $0) },
      angle: angle(from: line.cornerPoints)
    )
    data["elements"] = elementsData
    return data
  }

  private func addData(
    cornerPoints: [NSValue],
    frame: CGRect,
    languages: [TextRecognizedLanguage],
    text: String,
    confidence: NSNumber?,
    angle: NSNumber?
  ) -> [String: Any] {
    let points = cornerPoints.map { (point: NSValue) -> [String: Double] in
      let cgPoint = point.cgPointValue
      return ["x": Double(cgPoint.x), "y": Double(cgPoint.y)]
    }
    let allLanguageData = languages.compactMap { $0.languageCode }
    return [
      "points": points,
      "rect": [
        "left": Double(frame.origin.x),
        "top": Double(frame.origin.y),
        "right": Double(frame.origin.x + frame.size.width),
        "bottom": Double(frame.origin.y + frame.size.height)
      ] as [String: Double],
      "recognizedLanguages": allLanguageData,
      "text": text,
      "confidence": confidence ?? NSNull(),
      "angle": angle ?? NSNull()
    ]
  }

  /// Rotation angle in degrees from the top-left → top-right corner vector.
  /// ML Kit returns corner points clockwise starting top-left, so index 0→1 spans the top edge.
  private func angle(from cornerPoints: [NSValue]) -> NSNumber? {
    guard cornerPoints.count >= 2 else { return nil }
    let topLeft = cornerPoints[0].cgPointValue
    let topRight = cornerPoints[1].cgPointValue
    let radians = atan2(topRight.y - topLeft.y, topRight.x - topLeft.x)
    return NSNumber(value: Double(radians) * 180.0 / .pi)
  }
}
