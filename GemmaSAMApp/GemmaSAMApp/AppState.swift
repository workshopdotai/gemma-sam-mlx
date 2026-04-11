import SwiftUI
import Combine
import AVFoundation

@MainActor
class AppState: ObservableObject {
    @Published var currentImage: NSImage?
    @Published var currentImagePath: String?
    @Published var maskOverlay: NSImage?  // composited image with masks overlaid
    @Published var segmentationResult: SegmentationResult?
    @Published var analysisResult: AnalysisResult?
    @Published var isProcessing = false
    @Published var backendStatus: BackendStatus?
    @Published var errorMessage: String?
    @Published var promptText = ""
    
    // For video playback
    @Published var videoURL: URL?
    @Published var isVideoMode = false
    
    /// Computed property: true when there's content (image or video) to work with
    var hasContent: Bool {
        currentImage != nil || (isVideoMode && videoURL != nil)
    }
    
    let apiClient = APIClient()
    
    func checkStatus() async {
        do {
            backendStatus = try await apiClient.getStatus()
        } catch {
            errorMessage = "Backend not reachable: \(error.localizedDescription)"
        }
    }
    
    func segmentCurrentImage(prompt: String) async {
        guard let image = resolveCurrentImage() else { return }
        isProcessing = true
        errorMessage = nil
        defer { isProcessing = false }
        
        do {
            let result = try await apiClient.segment(image: image, prompt: prompt)
            segmentationResult = result
            // Create mask overlay
            maskOverlay = createMaskOverlay(original: image, result: result)
        } catch {
            errorMessage = "Segmentation failed: \(error.localizedDescription)"
        }
    }
    
    func analyzeCurrentImage(prompt: String? = nil) async {
        guard let image = resolveCurrentImage() else { return }
        isProcessing = true
        errorMessage = nil
        defer { isProcessing = false }
        
        do {
            let result = try await apiClient.analyze(image: image, prompt: prompt)
            analysisResult = result
            if let action = result.action, action.action == "segment", let segPrompt = action.prompt {
                promptText = segPrompt
            }
        } catch {
            errorMessage = "Analysis failed: \(error.localizedDescription)"
        }
    }
    
    func analyzeAndSegment(prompt: String? = nil) async {
        guard let image = resolveCurrentImage() else { return }
        isProcessing = true
        errorMessage = nil
        defer { isProcessing = false }
        
        do {
            let result = try await apiClient.analyzeAndSegment(image: image, prompt: prompt)
            analysisResult = AnalysisResult(text: result.text, action: result.action)
            if let seg = result.segmentation {
                segmentationResult = seg
                maskOverlay = createMaskOverlay(original: image, result: seg)
            }
        } catch {
            errorMessage = "Analysis + segmentation failed: \(error.localizedDescription)"
        }
    }
    
    /// Returns the current image — if in video mode, captures a frame first
    func resolveCurrentImage() -> NSImage? {
        if let image = currentImage {
            return image
        }
        if isVideoMode, let url = videoURL {
            // Capture frame at the current playback time (or 1s default)
            let time = currentVideoTime ?? CMTime(seconds: 1, preferredTimescale: 600)
            if let captured = captureFrame(from: url, at: time) {
                currentImage = captured
                return captured
            }
        }
        return nil
    }
    
    /// Current video playback time (set by VideoPlayerView)
    var currentVideoTime: CMTime?
    
    private func captureFrame(from url: URL, at time: CMTime) -> NSImage? {
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        do {
            let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
            return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        } catch {
            errorMessage = "Frame capture failed: \(error.localizedDescription)"
            return nil
        }
    }
    
    private func createMaskOverlay(original: NSImage, result: SegmentationResult) -> NSImage {
        let size = original.size
        // Use the actual pixel dimensions of the original image for proper rendering
        guard let originalCGImage = original.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return original
        }
        let pixelWidth = CGFloat(originalCGImage.width)
        let pixelHeight = CGFloat(originalCGImage.height)
        let scaleX = size.width / pixelWidth
        let scaleY = size.height / pixelHeight
        
        let composited = NSImage(size: size)
        composited.lockFocus()
        
        // Draw original image
        original.draw(in: NSRect(origin: .zero, size: size))
        
        // Overlay masks with colors
        let colors: [NSColor] = [
            NSColor(red: 1, green: 0.3, blue: 0.3, alpha: 0.4),
            NSColor(red: 0.3, green: 1, blue: 0.3, alpha: 0.4),
            NSColor(red: 0.3, green: 0.3, blue: 1, alpha: 0.4),
            NSColor(red: 1, green: 1, blue: 0.3, alpha: 0.4),
            NSColor(red: 1, green: 0.3, blue: 1, alpha: 0.4),
            NSColor(red: 0.3, green: 1, blue: 1, alpha: 0.4),
        ]
        
        let context = NSGraphicsContext.current!.cgContext
        
        for (i, maskB64) in result.masks.enumerated() {
            guard let maskData = Data(base64Encoded: maskB64),
                  let maskNSImage = NSImage(data: maskData),
                  let maskCGImage = maskNSImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                print("[⚠️] Failed to decode mask \(i)")
                continue
            }
            
            let color = colors[i % colors.count]
            color.setFill()
            
            // Scale mask from mask pixel coordinates to display coordinates
            let maskDisplayWidth = CGFloat(maskCGImage.width) * scaleX
            let maskDisplayHeight = CGFloat(maskCGImage.height) * scaleY
            let maskRect = NSRect(x: 0, y: 0, width: maskDisplayWidth, height: maskDisplayHeight)
            
            context.interpolationQuality = .none
            context.clip(to: maskRect, mask: maskCGImage)
            context.fill(NSRect(origin: .zero, size: size))
            context.resetClip()
        }
        
        composited.unlockFocus()
        return composited
    }
}

struct BackendStatus: Codable {
    let gemma_loaded: Bool
    let sam_loaded: Bool
    let status: String
}

struct SegmentationResult: Codable {
    let count: Int
    let masks: [String]  // base64 PNG
    let boxes: [[Double]]  // [[x1,y1,x2,y2], ...]
    let scores: [Double]
    let text_prompt: String
    let image_width: Int
    let image_height: Int
}

struct AnalysisResult {
    let text: String?
    let action: ActionData?
}

struct ActionData: Codable {
    let action: String
    let prompt: String?
    let summary: String?
}

struct AnalyzeAndSegmentResult {
    let text: String?
    let action: ActionData?
    let segmentation: SegmentationResult?
}
