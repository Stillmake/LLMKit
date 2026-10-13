import Foundation

/// Token budgets per thinking level, for providers that take a budget instead of an effort.
public struct ThinkingBudgets: Sendable, Hashable {
    public var minimal: Int
    public var low: Int
    public var medium: Int
    public var high: Int

    public init(minimal: Int = 1024, low: Int = 2048, medium: Int = 8192, high: Int = 16384) {
        self.minimal = minimal
        self.low = low
        self.medium = medium
        self.high = high
    }

    /// The budget for a level; `xhigh` and `max` use the `high` budget.
    public subscript(level: ThinkingLevel) -> Int {
        switch level {
        case .off, .minimal: minimal
        case .low: low
        case .medium: medium
        case .high, .xhigh, .max: high
        }
    }
}

/// Provider-neutral request options. Adapters translate each field into their wire format and
/// ignore fields their API has no equivalent for.
public struct StreamOptions: Sendable {
    /// API key or OAuth access token. Falls back to the provider's environment variable.
    public var apiKey: String?
    /// Extra HTTP headers, applied last. A `nil` value removes a default header.
    public var headers: [String: String?]
    public var temperature: Double?
    /// Output token cap. Defaults to the model's limit, clamped to the room left in the context window.
    public var maxTokens: Int?
    /// Thinking level, clamped to the model's supported levels. `nil` or `.off` disables thinking.
    public var reasoning: ThinkingLevel?
    /// Custom thinking budgets. `nil` uses `ThinkingBudgets()`, or Gemini 2.5's model-specific defaults.
    public var thinkingBudgets: ThinkingBudgets?
    public var toolChoice: ToolChoice?
    public var cacheRetention: CacheRetention
    /// Identifies a conversation for prompt caching and session-affinity routing. OpenCode requires it.
    public var sessionId: String?
    /// Arbitrary sampling parameters (`top_p`, `top_k`, `min_p`, ...), merged over the model's
    /// defaults per key. Applied by the OpenAI-compatible APIs.
    public var samplingParams: JSONObject?
    /// Deep-merged into the request body last by every adapter.
    public var extraBody: JSONObject?
    /// Final hook over the request body before it is sent.
    public var onPayload: (@Sendable (inout JSONValue, Model) -> Void)?
    /// Observes the HTTP response before its body is consumed.
    public var onResponse: (@Sendable (HTTPURLResponse, Model) -> Void)?
    /// Anthropic `metadata.user_id`.
    public var userId: String?
    /// OpenAI Responses and Codex service tier (`flex`, `priority`, ...).
    public var serviceTier: String?
    /// OpenAI Responses and Codex reasoning summary mode. Default `auto`.
    public var reasoningSummary: String?
    /// Codex text verbosity. Default `low`.
    public var textVerbosity: String?
    /// Anthropic thinking display: `summarized` (default) or `omitted`.
    public var thinkingDisplay: String?
    /// Idle timeout between bytes of the response.
    public var timeout: Duration
    /// Retries for failed requests before the response starts (408, 409, 429, 5xx, network errors).
    public var maxRetries: Int
    /// Longest server-requested retry delay to honor; longer requests fail immediately. `.zero` disables the cap.
    public var maxRetryDelay: Duration
    public var session: URLSession

    public init(
        apiKey: String? = nil,
        headers: [String: String?] = [:],
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        reasoning: ThinkingLevel? = nil,
        thinkingBudgets: ThinkingBudgets? = nil,
        toolChoice: ToolChoice? = nil,
        cacheRetention: CacheRetention = .short,
        sessionId: String? = nil,
        samplingParams: JSONObject? = nil,
        extraBody: JSONObject? = nil,
        onPayload: (@Sendable (inout JSONValue, Model) -> Void)? = nil,
        onResponse: (@Sendable (HTTPURLResponse, Model) -> Void)? = nil,
        userId: String? = nil,
        serviceTier: String? = nil,
        reasoningSummary: String? = nil,
        textVerbosity: String? = nil,
        thinkingDisplay: String? = nil,
        timeout: Duration = .seconds(600),
        maxRetries: Int = 2,
        maxRetryDelay: Duration = .seconds(60),
        session: URLSession = .shared
    ) {
        self.apiKey = apiKey
        self.headers = headers
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.reasoning = reasoning
        self.thinkingBudgets = thinkingBudgets
        self.toolChoice = toolChoice
        self.cacheRetention = cacheRetention
        self.sessionId = sessionId
        self.samplingParams = samplingParams
        self.extraBody = extraBody
        self.onPayload = onPayload
        self.onResponse = onResponse
        self.userId = userId
        self.serviceTier = serviceTier
        self.reasoningSummary = reasoningSummary
        self.textVerbosity = textVerbosity
        self.thinkingDisplay = thinkingDisplay
        self.timeout = timeout
        self.maxRetries = maxRetries
        self.maxRetryDelay = maxRetryDelay
        self.session = session
    }
}

/// Minimum tokens always left for the answer when thinking shares the output ceiling.
let minimumAnswerTokens = 1024

/// Options after the provider-neutral resolution every adapter shares.
struct ResolvedOptions {
    var raw: StreamOptions
    var apiKey: String?
    /// Thinking level clamped to the model; `nil` when thinking is off.
    var reasoning: ThinkingLevel?
    var maxTokens: Int
    /// Model defaults, then thinking-level overrides, then request values.
    var samplingParams: JSONObject?

    init(model: Model, transcript: [Message], options: StreamOptions) {
        raw = options
        apiKey = options.apiKey.flatMap { $0.isEmpty ? nil : $0 } ?? APIKeyEnvironment.apiKey(for: model.provider)
        let clamped = options.reasoning.map(model.clampThinkingLevel) ?? .off
        reasoning = clamped == .off ? nil : clamped
        // pi-ai's clampMaxTokensToContext: keep a safety margin under the context window.
        let requested = options.maxTokens ?? model.maxTokens
        maxTokens = model.contextWindow > 0
            ? min(requested, max(1, model.contextWindow - Transcript.estimatedTokens(transcript) - 4096))
            : max(1, requested)
        let levelParams = model.samplingParamsByThinkingLevel?[clamped]
        samplingParams = model.samplingParams == nil && levelParams == nil && options.samplingParams == nil
            ? nil
            : (model.samplingParams ?? [:])
                .merging(levelParams ?? [:]) { $1 }
                .merging(options.samplingParams ?? [:]) { $1 }
    }

    func requireAPIKey(for model: Model) throws -> String {
        guard let apiKey else { throw LLMError.missingAPIKey(provider: model.provider) }
        return apiKey
    }

    /// Whether the caller supplied its own authorization header.
    var hasAuthorizationHeader: Bool {
        raw.headers.contains { ["authorization", "x-api-key", "cf-aig-authorization"].contains($0.key.lowercased()) && $0.value != nil }
    }

    /// Applies `samplingParams` (OpenAI-compatible adapters only), then `extraBody`, then `onPayload`.
    func finalize(_ body: inout JSONObject, model: Model, applySamplingParams: Bool) -> JSONValue {
        if applySamplingParams, let samplingParams {
            body.merge(samplingParams) { $1 }
        }
        var payload = JSONValue.object(body)
        if let extraBody = raw.extraBody {
            payload.deepMerge(.object(extraBody))
        }
        raw.onPayload?(&payload, model)
        return payload
    }
}
