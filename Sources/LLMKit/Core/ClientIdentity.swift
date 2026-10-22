import Foundation

/// The `User-Agent` sent to providers:
/// - `detailed`: `pi/<version> (<platform>; <arch>)`. pi-ai's Node.js runtime field has no equivalent here.
/// - `short`: `pi`.
/// - `custom`: the given value.
public enum UserAgentMode: Codable, Sendable, Hashable {
    case short
    case detailed
    case custom(String)
}

/// Who requests are attributed to. `ClientIdentity.attributionHeaders(for:)` maps the fields to each provider's headers.
public struct ClientAttribution: Codable, Sendable, Hashable {
    /// App name: OpenRouter app title, NVIDIA invoke origin (first letter upper-cased), OpenCode client.
    public var name: String
    /// App URL: OpenRouter `HTTP-Referer`, the app's identity in OpenRouter's rankings, grouped by origin.
    public var url: String
    /// App categories, e.g. `["cli-agent"]`: OpenRouter `X-OpenRouter-Categories`, comma-separated.
    public var categories: [String]?

    public init(name: String, url: String, categories: [String]? = nil) {
        self.name = name
        self.url = url
        self.categories = categories
    }

    /// pi's attribution, used by pi-agent-core, pi-durable, and the coding agent.
    public static let pi = ClientAttribution(name: "pi", url: "https://pi.dev", categories: ["cli-agent"])
}

/// How LLMKit identifies the calling app to providers. Ports pi-ai's `utils/identity.ts`. Every field is
/// optional; absent fields use pi-ai's defaults: a detailed `User-Agent` with pi-ai's version and no attribution headers.
public struct ClientIdentity: Codable, Sendable, Hashable {
    /// Default: `.detailed`.
    public var userAgent: UserAgentMode?
    /// Provider attribution headers. Absent: none.
    public var attribution: ClientAttribution?
    /// Version in the detailed `User-Agent`. Default: the pi-ai version the port tracks.
    public var version: String?

    public init(userAgent: UserAgentMode? = nil, attribution: ClientAttribution? = nil, version: String? = nil) {
        self.userAgent = userAgent
        self.attribution = attribution
        self.version = version
    }

    /// The `User-Agent` value of this identity.
    public var formattedUserAgent: String {
        switch userAgent ?? .detailed {
        case .custom(let value): return value
        case .short: return "pi"
        case .detailed:
            #if os(macOS)
            let platform = "darwin"
            #elseif os(iOS)
            let platform = "ios"
            #elseif os(visionOS)
            let platform = "visionos"
            #elseif os(watchOS)
            let platform = "watchos"
            #elseif os(tvOS)
            let platform = "tvos"
            #else
            let platform = "linux"
            #endif
            #if arch(arm64)
            let arch = "arm64"
            #else
            let arch = "x64"
            #endif
            // pi-ai's PI_AI_VERSION; bump with each catalog sync.
            return "pi/\(version ?? "1.4.0") (\(platform); \(arch))"
        }
    }

    /// Provider attribution headers of this identity for a request target. Empty when attribution is off.
    public func attributionHeaders(for model: Model) -> [String: String] {
        guard let attribution else { return [:] }
        if Self.isOpenRouter(model) {
            var headers = ["HTTP-Referer": attribution.url, "X-OpenRouter-Title": attribution.name]
            if let categories = attribution.categories, !categories.isEmpty {
                headers["X-OpenRouter-Categories"] = categories.joined(separator: ",")
            }
            return headers
        }
        // NVIDIA expects a capitalized origin, such as `Pi`.
        if model.provider == "nvidia" || URL(string: model.baseURL)?.host == "integrate.api.nvidia.com" {
            return ["X-BILLING-INVOKE-ORIGIN": attribution.name.prefix(1).uppercased() + attribution.name.dropFirst()]
        }
        if Self.isOpenCode(model) { return ["x-opencode-client": attribution.name] }
        return [:]
    }

    /// Default headers this identity adds to a request: `User-Agent` and provider attribution. Requests apply
    /// them below adapter, model, and caller headers, so configured headers override them and `nil` caller
    /// headers remove them.
    public func headers(for model: Model) -> [String: String] {
        attributionHeaders(for: model).merging(["User-Agent": formattedUserAgent]) { $1 }
    }

    /// OpenCode's per-conversation routing header for requests to OpenCode, including custom providers pointing
    /// at it. `nil` without a session or for other targets.
    static func openCodeSessionHeaders(for model: Model, sessionId: String?) -> [String: String]? {
        guard let sessionId, !sessionId.isEmpty, isOpenCode(model) else { return nil }
        return ["x-opencode-session": sessionId]
    }

    private static func isOpenRouter(_ model: Model) -> Bool {
        model.provider == "openrouter" || model.baseURL.contains("openrouter.ai")
    }

    private static func isOpenCode(_ model: Model) -> Bool {
        ["opencode", "opencode-go"].contains(model.provider) || URL(string: model.baseURL)?.host == "opencode.ai"
    }
}
