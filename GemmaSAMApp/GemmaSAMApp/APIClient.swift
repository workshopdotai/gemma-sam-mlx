import Foundation
import AppKit

@MainActor
class APIClient {
    private let baseURL = "http://localhost:8199"
    
    func getStatus() async throws -> BackendStatus {
        let url = URL(string: "\(baseURL)/api/status")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(BackendStatus.self, from: data)
    }
    
    func segment(image: NSImage, prompt: String) async throws -> SegmentationResult {
        let url = URL(string: "\(baseURL)/api/segment")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image
        if let imageData = image.jpegData {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // Add prompt
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n".data(using: .utf8)!)
        body.append(prompt.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            let errorText = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIClientError.serverError("HTTP \(httpResponse.statusCode): \(errorText)")
        }
        
        return try JSONDecoder().decode(SegmentationResult.self, from: data)
    }
    
    func analyze(image: NSImage, prompt: String? = nil) async throws -> AnalysisResult {
        let url = URL(string: "\(baseURL)/api/analyze")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image
        if let imageData = image.jpegData {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // Add optional prompt
        if let prompt = prompt {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n".data(using: .utf8)!)
            body.append(prompt.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        
        let (data, _) = try await URLSession.shared.data(for: request)
        
        // Parse JSON manually since AnalysisResult has optional non-Codable fields
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let text = json["text"] as? String
            var action: ActionData? = nil
            if let actionDict = json["action"] as? [String: Any] {
                let actionData = try JSONSerialization.data(withJSONObject: actionDict)
                action = try JSONDecoder().decode(ActionData.self, from: actionData)
            }
            return AnalysisResult(text: text, action: action)
        }
        
        throw APIClientError.invalidResponse
    }
    
    func analyzeAndSegment(image: NSImage, prompt: String? = nil) async throws -> AnalyzeAndSegmentResult {
        let url = URL(string: "\(baseURL)/api/analyze-and-segment")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image
        if let imageData = image.jpegData {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // Add optional prompt
        if let prompt = prompt {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n".data(using: .utf8)!)
            body.append(prompt.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        
        let (data, _) = try await URLSession.shared.data(for: request)
        
        // Parse manually
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let text = json["text"] as? String
            var action: ActionData? = nil
            if let actionDict = json["action"] as? [String: Any] {
                let actionData = try JSONSerialization.data(withJSONObject: actionDict)
                action = try JSONDecoder().decode(ActionData.self, from: actionData)
            }
            var segmentation: SegmentationResult? = nil
            if let segDict = json["segmentation"] as? [String: Any] {
                let segData = try JSONSerialization.data(withJSONObject: segDict)
                segmentation = try JSONDecoder().decode(SegmentationResult.self, from: segData)
            }
            return AnalyzeAndSegmentResult(text: text, action: action, segmentation: segmentation)
        }
        
        throw APIClientError.invalidResponse
    }
}

enum APIClientError: LocalizedError {
    case serverError(String)
    case invalidResponse
    
    var errorDescription: String? {
        switch self {
        case .serverError(let msg): return msg
        case .invalidResponse: return "Invalid response from server"
        }
    }
}

extension NSImage {
    var jpegData: Data? {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        bitmapRep.size = self.size
        return bitmapRep.representation(using: .jpeg, properties: [.compressionFactor: 0.85])
    }
}
