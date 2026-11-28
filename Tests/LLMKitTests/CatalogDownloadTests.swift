import CryptoKit
import Foundation
import Testing
@testable import LLMKit

@Suite struct CatalogDownloadTests {
    @Test func verifiesTheManifestAndProviderBytesBeforeReturningModels() async throws {
        let catalog = try await ModelCatalog.download(
            version: "fixture", manifestSHA256: CatalogURLProtocol.manifestSHA256, session: CatalogURLProtocol.session())
        #expect(catalog.version == "fixture")
        #expect(catalog.models.map(\.id) == ["test-model"])
    }

    @Test func rejectsAManifestThatDoesNotMatchTheBundledPin() async {
        await #expect(throws: LLMError.self) {
            try await ModelCatalog.downloadPinned(session: CatalogURLProtocol.session())
        }
    }

    @Test func rejectsAProviderFileThatDoesNotMatchTheVerifiedManifest() async {
        await #expect(throws: LLMError.self) {
            try await ModelCatalog.download(
                version: "changed-fixture", manifestSHA256: CatalogURLProtocol.manifestSHA256, session: CatalogURLProtocol.session())
        }
    }

    @Test func stillDownloadsAnArbitraryVersionWithoutThePin() async throws {
        let catalog = try await ModelCatalog.download(version: "changed-fixture", session: CatalogURLProtocol.session())
        #expect(catalog.version == "changed-fixture")
        #expect(catalog.models.map(\.id) == ["other-model"])
    }
}

private final class CatalogURLProtocol: URLProtocol, @unchecked Sendable {
    private static let providerData = Data(
        #"{"openai-responses":{"test-model":{"api":"openai-responses","id":"test-model","provider":"openai","baseUrl":"https://example.com"}}}"#.utf8)
    private static let manifest: Data = {
        let providerSHA256 = SHA256.hash(data: providerData).map { String(format: "%02x", $0) }.joined()
        return Data(#"{"generatedAt":"2026-11-27T16:40:10.684Z","files":{"openai.json":"\#(providerSHA256)"}}"#.utf8)
    }()
    static let manifestSHA256 = SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined()

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CatalogURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "cdn.jsdelivr.net" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else { return }
        let data: Data
        switch url.lastPathComponent {
        case ".manifest.json":
            data = Self.manifest
        case "openai.json":
            data = url.path.contains("@changed-fixture/")
                ? Data(String(decoding: Self.providerData, as: UTF8.self).replacingOccurrences(of: "test-model", with: "other-model").utf8)
                : Self.providerData
        default:
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
