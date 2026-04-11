import SwiftUI
import AVKit

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var isDragOver = false
    
    var body: some View {
        HSplitView {
            // Left: Image display
            imagePanel
                .frame(minWidth: 500)
            
            // Right: Controls
            controlPanel
                .frame(minWidth: 300, maxWidth: 400)
        }
        .onAppear {
            Task { await appState.checkStatus() }
        }
        .alert("Error", isPresented: .constant(appState.errorMessage != nil), presenting: appState.errorMessage) { _ in
            Button("OK") { appState.errorMessage = nil }
        } message: { msg in
            Text(msg)
        }
    }
    
    // MARK: - Image Panel
    private var imagePanel: some View {
        ZStack {
            if appState.isVideoMode, let videoURL = appState.videoURL {
                // Video mode: always show video player, overlay results on top
                ZStack {
                    VideoPlayerView(url: videoURL)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    
                    // Show captured frame with mask overlay
                    if let overlayImage = appState.maskOverlay {
                        Image(nsImage: overlayImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if let currentImage = appState.currentImage {
                        Image(nsImage: currentImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .opacity(0.9)
                    }
                }
            } else if let overlayImage = appState.maskOverlay {
                Image(nsImage: overlayImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let currentImage = appState.currentImage {
                Image(nsImage: currentImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Drop zone placeholder
                dropZonePlaceholder
            }
            
            // Processing overlay
            if appState.isProcessing {
                Color.black.opacity(0.3)
                ProgressView("Processing...")
                    .progressViewStyle(.circular)
                    .controlSize(.large)
                    .tint(.white)
                    .foregroundStyle(.white)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onDrop(of: [.image, .fileURL], isTargeted: $isDragOver) { providers in
            handleDrop(providers: providers)
        }
        .border(isDragOver ? Color.accentColor : Color.clear, width: 3)
    }
    
    private var dropZonePlaceholder: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 64))
                .foregroundStyle(.secondary)
            Text("Drop an image or video here")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("Supported: JPEG, PNG, MP4, MOV")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    // MARK: - Control Panel
    private var controlPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Status
            statusView
            
            Divider()
            
            // Prompt input
            promptView
            
            Divider()
            
            // Results
            resultsView
            
            Spacer()
            
            // Clear / Resume buttons
            HStack {
                if appState.isVideoMode && appState.currentImage != nil {
                    Button("Resume Video") {
                        appState.currentImage = nil
                        appState.maskOverlay = nil
                        appState.segmentationResult = nil
                        appState.analysisResult = nil
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Spacer()
                Button("Clear All") {
                    appState.currentImage = nil
                    appState.currentImagePath = nil
                    appState.maskOverlay = nil
                    appState.segmentationResult = nil
                    appState.analysisResult = nil
                    appState.videoURL = nil
                    appState.isVideoMode = false
                    appState.promptText = ""
                }
                .controlSize(.small)
            }
        }
        .padding()
        .frame(maxHeight: .infinity)
        .background(Color(nsColor: .controlBackgroundColor))
    }
    
    // MARK: - Status View
    private var statusView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Gemma 4 + SAM 3.1")
                .font(.headline)
            Text("Local Vision-Guided Segmentation")
                .font(.caption)
                .foregroundStyle(.secondary)
            
            HStack(spacing: 12) {
                Label("Gemma 4", systemImage: appState.backendStatus?.gemma_loaded == true ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(appState.backendStatus?.gemma_loaded == true ? .green : .secondary)
                    .font(.caption)
                
                Label("SAM 3.1", systemImage: appState.backendStatus?.sam_loaded == true ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(appState.backendStatus?.sam_loaded == true ? .green : .secondary)
                    .font(.caption)
            }
        }
    }
    
    // MARK: - Prompt View
    private var promptView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.subheadline)
                .fontWeight(.medium)
            
            TextField("e.g., all vehicles, people, white cars...", text: $appState.promptText)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    if !appState.promptText.isEmpty {
                        Task { await appState.segmentCurrentImage(prompt: appState.promptText) }
                    }
                }
            
            HStack(spacing: 8) {
                Button("Auto Analyze") {
                    Task { await appState.analyzeAndSegment(prompt: appState.promptText.isEmpty ? nil : appState.promptText) }
                }
                .disabled(!appState.hasContent || appState.isProcessing)
                .buttonStyle(.borderedProminent)
                
                Button("Segment") {
                    Task { await appState.segmentCurrentImage(prompt: appState.promptText) }
                }
                .disabled(!appState.hasContent || appState.promptText.isEmpty || appState.isProcessing)
                .buttonStyle(.bordered)
                
                Button("Analyze Only") {
                    Task { await appState.analyzeCurrentImage(prompt: appState.promptText.isEmpty ? nil : appState.promptText) }
                }
                .disabled(!appState.hasContent || appState.isProcessing)
                .buttonStyle(.bordered)
            }
        }
    }
    
    // MARK: - Results View
    private var resultsView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if let result = appState.segmentationResult {
                    segmentationResults(result)
                }
                
                if let analysis = appState.analysisResult {
                    analysisResults(analysis)
                }
            }
        }
    }
    
    private func segmentationResults(_ result: SegmentationResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Segmentation Results")
                .font(.subheadline)
                .fontWeight(.medium)
            
            HStack {
                Label("\(result.count) objects found", systemImage: "checkerboard.rectangle")
                Spacer()
                Text("Prompt: \"\(result.text_prompt)\"")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            if !result.scores.isEmpty {
                let avgScore = result.scores.reduce(0, +) / Double(result.scores.count)
                let maxScore = result.scores.max() ?? 0
                let minScore = result.scores.min() ?? 0
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Scores:")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 16) {
                        Text("Avg: \(String(format: "%.2f", avgScore))")
                        Text("Max: \(String(format: "%.2f", maxScore))")
                        Text("Min: \(String(format: "%.2f", minScore))")
                    }
                    .font(.caption2)
                }
            }
        }
        .padding(8)
        .background(Color(nsColor: .controlColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    
    private func analysisResults(_ analysis: AnalysisResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Gemma 4 Analysis")
                .font(.subheadline)
                .fontWeight(.medium)
            
            if let text = analysis.text {
                Text(text)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            
            if let action = analysis.action {
                HStack {
                    Label(action.action == "segment" ? "Segment" : "Done", systemImage: action.action == "segment" ? "rectangle.dashed" : "checkmark.circle")
                    if let prompt = action.prompt {
                        Text("\"\(prompt)\"")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
            }
        }
        .padding(8)
        .background(Color(nsColor: .controlColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    
    // MARK: - Drop Handling
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        
        // Try image first
        if provider.hasItemConformingToTypeIdentifier("public.image") {
            provider.loadDataRepresentation(forTypeIdentifier: "public.image") { data, error in
                guard let data = data, error == nil else { return }
                DispatchQueue.main.async {
                    if let nsImage = NSImage(data: data) {
                        appState.currentImage = nsImage
                        appState.isVideoMode = false
                        appState.videoURL = nil
                        appState.maskOverlay = nil
                        appState.segmentationResult = nil
                        appState.analysisResult = nil
                    }
                }
            }
            return true
        }
        
        // Try file URL (for videos and image files)
        if provider.hasItemConformingToTypeIdentifier("public.file-url") {
            provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { [weak appState] (item, error) in
                guard error == nil else { return }
                // Extract URL string on this thread, then dispatch Sendable data
                var fileURL: URL?
                if let data = item as? Data,
                   let urlString = String(data: data, encoding: .utf8) {
                    fileURL = URL(string: urlString)
                } else if let url = item as? URL {
                    fileURL = url
                }
                guard let url = fileURL else { return }
                
                DispatchQueue.main.async {
                    let ext = url.pathExtension.lowercased()
                    if ["mp4", "mov", "m4v", "avi"].contains(ext) {
                        appState?.videoURL = url
                        appState?.isVideoMode = true
                        appState?.currentImage = nil
                        appState?.maskOverlay = nil
                        appState?.segmentationResult = nil
                        appState?.analysisResult = nil
                    } else if ["jpg", "jpeg", "png", "tiff", "bmp", "gif", "heic"].contains(ext),
                              let nsImage = NSImage(contentsOf: url) {
                        appState?.currentImage = nsImage
                        appState?.currentImagePath = url.path
                        appState?.isVideoMode = false
                        appState?.videoURL = nil
                        appState?.maskOverlay = nil
                        appState?.segmentationResult = nil
                        appState?.analysisResult = nil
                    }
                }
            }
            return true
        }
        
        return false
    }
}
