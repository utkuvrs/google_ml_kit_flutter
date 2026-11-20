import CoreGraphics
import Foundation
import Vision

/// Optional Apple Vision pass that supplies per-element/per-line confidence, which the ML Kit iOS SDK
/// does not expose. Scores come from Apple's recognizer, so they are not calibrated like Android's
/// ML Kit confidence.
enum VisionTextConfidence {
  /// Vision languages to request per `TextRecognitionScript` index. The first entry is the script's
  /// primary language: if Vision does not support it, the pass is skipped and confidence stays nil.
  private static let desiredLanguages: [Int: [String]] = [
    0: ["en-US", "fr-FR", "de-DE", "it-IT", "es-ES", "pt-BR"], // latin
    1: ["zh-Hans", "zh-Hant", "en-US"], // chinese
    2: [], // devanagari: not supported by Apple Vision
    3: ["ja-JP", "en-US"], // japanese (Vision revision 3, iOS 16+)
    4: ["ko-KR", "en-US"] // korean (Vision revision 3, iOS 16+)
  ]

  /// Cache of `languages(forScript:)`. Only accessed from the main thread.
  private static var languageCache: [Int: [String]] = [:]

  /// Languages to pass to Vision for a script, filtered to what this OS version supports.
  /// Empty means "skip the Vision pass". Call on the main thread.
  static func languages(forScript scriptIndex: Int) -> [String] {
    if let cached = languageCache[scriptIndex] {
      return cached
    }
    let desired = desiredLanguages[scriptIndex] ?? desiredLanguages[0] ?? []
    let supported = Set((try? makeRequest().supportedRecognitionLanguages()) ?? [])
    var languages: [String] = []
    if let primary = desired.first, supported.contains(primary) {
      languages = desired.filter { supported.contains($0) }
    }
    languageCache[scriptIndex] = languages
    return languages
  }

  /// Runs Vision text recognition and returns its observations in ML Kit's pixel, top-left coordinate
  /// space. Returns nil on failure, so confidence degrades to nil. Safe to call off the main thread.
  static func recognize(
    cgImage: CGImage,
    orientation: CGImagePropertyOrientation,
    languages: [String]
  ) -> VisionConfidenceIndex? {
    let request = makeRequest()
    request.recognitionLanguages = languages
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
    do {
      try handler.perform([request])
    } catch {
      return nil
    }
    guard let observations = request.results, !observations.isEmpty else { return nil }
    // After orientation correction Vision reports boxes in the upright image, whose width and height
    // are swapped for the 90°/270° (.left/.right) families.
    let rotatedOrientations: [CGImagePropertyOrientation] = [.left, .right, .leftMirrored, .rightMirrored]
    let swapsAxes = rotatedOrientations.contains(orientation)
    let width = CGFloat(swapsAxes ? cgImage.height : cgImage.width)
    let height = CGFloat(swapsAxes ? cgImage.width : cgImage.height)
    let entries = observations.map { observation -> VisionConfidenceIndex.Entry in
      let box = observation.boundingBox // normalized, bottom-left origin
      let candidate = observation.topCandidates(1).first
      return VisionConfidenceIndex.Entry(
        rect: CGRect(
          x: box.minX * width,
          y: (1.0 - box.maxY) * height,
          width: box.width * width,
          height: box.height * height
        ),
        confidence: candidate?.confidence ?? observation.confidence,
        normalizedText: normalized(candidate?.string ?? "")
      )
    }
    return VisionConfidenceIndex(entries: entries)
  }

  /// Lowercase and keep only letters/digits, so spacing and punctuation differences between the two
  /// engines don't break text comparison.
  static func normalized(_ string: String) -> String {
    let scalars = string.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }
    return String(String.UnicodeScalarView(scalars))
  }

  private static func makeRequest() -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    return request
  }
}

/// Vision observations sorted by `minY`, so a lookup only scans the boxes in the vertical band that can
/// overlap the query frame: `O(log V + k)` instead of `O(V)`.
struct VisionConfidenceIndex {
  struct Entry {
    let rect: CGRect
    let confidence: Float
    /// Pre-normalized once at build time (see `VisionTextConfidence.normalized`).
    let normalizedText: String
  }

  private let entries: [Entry]
  private let maxHeight: CGFloat

  init(entries: [Entry]) {
    self.entries = entries.sorted { $0.rect.minY < $1.rect.minY }
    self.maxHeight = entries.map { $0.rect.height }.max() ?? 0
  }

  /// Best Vision confidence for an ML Kit element or line frame, or nil if nothing matches well enough.
  ///
  /// Geometry score is max(IoU, containment). Containment (intersection / smaller area) handles a
  /// word-sized ML Kit box inside a line-sized Vision box. A text-agreement bonus lets a box whose text
  /// matches win over a purely geometric one.
  func confidence(for frame: CGRect, text: String) -> Float? {
    guard !entries.isEmpty, frame.width > 0, frame.height > 0 else { return nil }
    let query = VisionTextConfidence.normalized(text)
    let frameArea = frame.width * frame.height
    var bestScore: CGFloat = 0
    var bestGeometry: CGFloat = 0
    var bestTextMatch = false
    var bestConfidence: Float?
    // A box can only overlap the frame vertically if minY > frame.minY - maxHeight and minY < frame.maxY.
    var index = firstIndex(withMinYAbove: frame.minY - maxHeight)
    while index < entries.count, entries[index].rect.minY < frame.maxY {
      let entry = entries[index]
      index += 1
      let intersection = frame.intersection(entry.rect)
      if intersection.isNull || intersection.isEmpty { continue }
      let interArea = intersection.width * intersection.height
      let rectArea = entry.rect.width * entry.rect.height
      let unionArea = frameArea + rectArea - interArea
      let iou = unionArea > 0 ? interArea / unionArea : 0
      let minArea = min(frameArea, rectArea)
      let containment = minArea > 0 ? interArea / minArea : 0
      let geometry = max(iou, containment)
      let textMatch = Self.textsAgree(query, entry.normalizedText)
      let score = geometry + (textMatch ? 1.0 : 0.0)
      if score > bestScore {
        bestScore = score
        bestConfidence = entry.confidence
        bestGeometry = geometry
        bestTextMatch = textMatch
      }
    }
    // Accept when the texts agree with any real overlap, or when geometry alone is strong.
    if bestTextMatch && bestGeometry >= 0.1 { return bestConfidence }
    if bestGeometry >= 0.3 { return bestConfidence }
    return nil
  }

  /// Binary search: index of the first entry whose minY is greater than `value`.
  private func firstIndex(withMinYAbove value: CGFloat) -> Int {
    var low = 0
    var high = entries.count
    while low < high {
      let mid = (low + high) / 2
      if entries[mid].rect.minY > value {
        high = mid
      } else {
        low = mid + 1
      }
    }
    return low
  }

  /// True when one normalized string contains the other (ML Kit word inside a Vision line, or vice versa).
  private static func textsAgree(_ lhs: String, _ rhs: String) -> Bool {
    guard !lhs.isEmpty, !rhs.isEmpty else { return false }
    return lhs.contains(rhs) || rhs.contains(lhs)
  }
}
