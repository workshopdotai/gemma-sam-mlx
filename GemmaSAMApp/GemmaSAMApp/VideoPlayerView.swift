import SwiftUI
import AVKit
import AVFoundation

struct VideoPlayerView: NSViewRepresentable {
    let url: URL
    @EnvironmentObject var appState: AppState
    
    func makeNSView(context: Context) -> AVPlayerView {
        let playerView = AVPlayerView()
        let player = AVPlayer(url: url)
        playerView.player = player
        
        // Observe current time periodically
        let interval = CMTime(seconds: 0.5, preferredTimescale: 600)
        player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { time in
            Task { @MainActor in
                appState.currentVideoTime = time
            }
        }
        
        return playerView
    }
    
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        // Check if the current URL matches
        let currentURL = (nsView.player?.currentItem?.asset as? AVURLAsset)?.url
        if currentURL != url {
            let player = AVPlayer(url: url)
            nsView.player = player
            
            let interval = CMTime(seconds: 0.5, preferredTimescale: 600)
            player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { time in
                Task { @MainActor in
                    appState.currentVideoTime = time
                }
            }
        }
    }
}

// Frame capture button sits outside VideoPlayerView
struct VideoControlsOverlay: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                Button(action: captureCurrentFrame) {
                    Label("Capture Frame", systemImage: "camera.viewfinder")
                        .padding(8)
                        .background(Color.black.opacity(0.6))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .padding()
            }
        }
    }
    
    private func captureCurrentFrame() {
        // This now just delegates to AppState which has the current time
        guard appState.currentImage == nil else { return } // already have a captured frame
        _ = appState.resolveCurrentImage()
    }
}
