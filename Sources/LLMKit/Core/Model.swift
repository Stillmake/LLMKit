import Foundation

/// A wire format. About forty providers reduce to a handful of these, each served by one adapter.
public struct API: RawRepresentable, Codable, Sendable, Hashable, ExpressibleByStringLiteral, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }
    public var description: String { rawValue }

    public static let openAICompletions: API = "openai-completions"
    public static let openAIResponses: API = "openai-responses"
    public static let openAICodexResponses: API = "openai-codex-responses"
    public static let anthropicMessages: API = "anthropic-messages"
    public static let googleGenerativeAI: API = "google-generative-ai"
}

public enum ThinkingLevel: String, Codable, CodingKeyRepresentable, Sendable, Hashable, CaseIterable, Comparable {
    case off, minimal, low, medium, high, xhigh, max

    public static func < (lhs: Self, rhs: Self) -> Bool {
        allCases.firstIndex(of: lhs)! < allCases.firstIndex(of: rhs)!
    }
}

public enum InputModality: String, Codable, Sendable, Hashable {
    case text, image
}

public enum CacheRetention: String, Codable, Sendable, Hashable {
    case none, short, long
}

/// Prices in USD per million tokens.
public struct CostRates: Codable, Sendable, Hashable {
    public var input: Double
    public var output: Double
    public var cacheRead: Double
    public var cacheWrite: Double

    public init(input: Double, output: Double, cacheRead: Double = 0, cacheWrite: Double = 0) {
        self.input = input
        self.output = output
        self.cacheRead = cacheRead
        self.cacheWrite = cacheWrite
    }
}

public struct ModelCost: Codable, Sendable, Hashable {
    /// A request-wide tier: applies to the whole request when its input exceeds `inputTokensAbove`.
    public struct Tier: Codable, Sendable, Hashable {
        public var inputTokensAbove: Int
        public var rates: CostRates

        private enum CodingKeys: String, CodingKey { case inputTokensAbove }

        public init(inputTokensAbove: Int, rates: CostRates) {
            self.inputTokensAbove = inputTokensAbove
            self.rates = rates
        }

        public init(from decoder: Decoder) throws {
            inputTokensAbove = try decoder.container(keyedBy: CodingKeys.self).decode(Int.self, forKey: .inputTokensAbove)
            rates = try CostRates(from: decoder)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(inputTokensAbove, forKey: .inputTokensAbove)
            try rates.encode(to: encoder)
        }
    }

    public var rates: CostRates
    public var tiers: [Tier]?

    private enum CodingKeys: String, CodingKey { case tiers }

    public init(rates: CostRates, tiers: [Tier]? = nil) {
        self.rates = rates
        self.tiers = tiers
    }

    public init(from decoder: Decoder) throws {
        rates = try CostRates(from: decoder)
        tiers = try decoder.container(keyedBy: CodingKeys.self).decodeIfPresent([Tier].self, forKey: .tiers)
    }

    public func encode(to encoder: Encoder) throws {
        try rates.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(tiers, forKey: .tiers)
    }
}

/// Maps LLMKit thinking levels to provider- or model-specific values.
///
/// A missing level uses the provider default (the level's own name); a level mapped to
/// `.unsupported` is not offered for the model. `xhigh` and `max` are only offered when mapped.
public struct ThinkingLevelMap: Codable, Sendable, Hashable {
    public enum Mapping: Sendable, Hashable {
        case unsupported
        case value(String)
    }

    public var mappings: [ThinkingLevel: Mapping]

    public init(_ mappings: [ThinkingLevel: Mapping] = [:]) {
        self.mappings = mappings
    }

    public subscript(level: ThinkingLevel) -> Mapping? {
        mappings[level]
    }

    public init(from decoder: Decoder) throws {
        let raw = try [ThinkingLevel: String?](from: decoder)
        mappings = raw.mapValues { $0.map(Mapping.value) ?? .unsupported }
    }

    public func encode(to encoder: Encoder) throws {
        try mappings.mapValues { mapping -> String? in
            if case .value(let value) = mapping { value } else { nil }
        }.encode(to: encoder)
    }
}

/// Request-format quirks of OpenAI-compatible servers, selected by `ModelCompat.thinkingFormat`.
public struct ThinkingFormat: RawRepresentable, Codable, Sendable, Hashable, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    /// `reasoning_effort`.
    public static let openAI: Self = "openai"
    /// `reasoning: {effort}`.
    public static let openRouter: Self = "openrouter"
    /// `thinking: {type}` plus `reasoning_effort` when supported.
    public static let deepSeek: Self = "deepseek"
    /// `reasoning: {enabled}` plus `reasoning_effort` when supported.
    public static let together: Self = "together"
    /// `chat_template_args` plus `reasoning_effort` when supported.
    public static let baseten: Self = "baseten"
    /// `thinking: {type}`.
    public static let zai: Self = "zai"
    /// Top-level `enable_thinking`.
    public static let qwen: Self = "qwen"
    /// `chat_template_kwargs` from `ModelCompat.chatTemplateKwargs`.
    public static let chatTemplate: Self = "chat-template"
    /// `chat_template_kwargs: {enable_thinking, preserve_thinking}`.
    public static let qwenChatTemplate: Self = "qwen-chat-template"
    /// Top-level string `thinking`.
    public static let stringThinking: Self = "string-thinking"
    /// `reasoning: {effort}` only when the mapped effort exists.
    public static let antLing: Self = "ant-ling"
}

/// Per-model compatibility overrides, as data. Every field is optional: unset fields fall back to
/// defaults each adapter derives from the provider and base URL. Mirrors pi-ai's compat schema;
/// unknown keys in the catalog are ignored.
public struct ModelCompat: Codable, Sendable, Hashable {
    public enum MaxTokensField: String, Codable, Sendable, Hashable {
        case maxCompletionTokens = "max_completion_tokens"
        case maxTokens = "max_tokens"
    }

    public enum SessionAffinityFormat: String, Codable, Sendable, Hashable {
        case openAI = "openai"
        case openAINoSession = "openai-nosession"
        case openRouter = "openrouter"
    }

    public enum ThinkingTokenBudgetField: String, Codable, Sendable, Hashable {
        /// vLLM.
        case thinkingTokenBudget = "thinking_token_budget"
        /// Qwen, DashScope, SGLang.
        case thinkingBudget = "thinking_budget"
        /// llama.cpp.
        case thinkingBudgetTokens = "thinking_budget_tokens"
    }

    // Shared
    public var supportsDeveloperRole: Bool?
    public var supportsMidConvoSystemMessages: Bool?
    public var supportsStrictMode: Bool?
    public var supportsLongCacheRetention: Bool?
    public var sendSessionAffinityHeaders: Bool?
    public var sessionAffinityFormat: SessionAffinityFormat?

    // OpenAI Chat Completions
    public var supportsStore: Bool?
    public var supportsReasoningEffort: Bool?
    public var supportsUsageInStreaming: Bool?
    public var supportsFinishReason: Bool?
    public var maxTokensField: MaxTokensField?
    public var requiresToolResultName: Bool?
    public var requiresAssistantAfterToolResult: Bool?
    public var requiresThinkingAsText: Bool?
    public var requiresReasoningContentOnAssistantMessages: Bool?
    public var thinkingFormat: ThinkingFormat?
    /// Sent as `chat_template_kwargs` for `.chatTemplate`. Values may be
    /// `{"$var": "thinking.enabled" | "thinking.effort" | "thinking.budget", "omitWhenOff": Bool}`.
    public var chatTemplateKwargs: JSONObject?
    /// Sent as `chat_template_args` for `.baseten`, with the same `$var` substitution.
    public var chatTemplateArgs: JSONObject?
    /// OpenRouter provider routing, sent as the `provider` request field.
    public var openRouterRouting: JSONValue?
    /// Vercel AI Gateway routing: `{"only": [...], "order": [...]}`.
    public var vercelGatewayRouting: JSONValue?
    public var zaiToolStream: Bool?
    public var thinkingTokenBudgetField: ThinkingTokenBudgetField?
    public var supportsThinkingTokenBudget: Bool?
    /// `"anthropic"` applies Anthropic-style `cache_control` markers through an OpenAI-compatible gateway.
    public var cacheControlFormat: String?
    public var vllmPriority: Double?

    // OpenAI Responses
    public var supportsExplicitPromptCacheMode: Bool?
    public var supportsMaxOutputTokens: Bool?

    // Anthropic Messages
    public var supportsEagerToolInputStreaming: Bool?
    public var supportsCacheControlOnTools: Bool?
    public var supportsTemperature: Bool?
    public var forceAdaptiveThinking: Bool?
    public var allowEmptySignature: Bool?
    public var supportsStrictTools: Bool?

    public init() {}
}

/// A chat model and everything needed to call it.
public struct Model: Codable, Sendable, Hashable, Identifiable {
    public var id: String
    public var name: String
    public var api: API
    public var provider: String
    public var baseURL: String
    public var input: [InputModality]
    public var cost: ModelCost
    public var headers: [String: String]?
    /// Whether the model can think before answering.
    public var reasoning: Bool
    public var thinkingLevelMap: ThinkingLevelMap?
    public var contextWindow: Int
    public var maxTokens: Int
    /// Default sampling parameters merged into the request body (OpenAI-compatible APIs).
    public var samplingParams: JSONObject?
    /// Sampling parameter overrides for the effective thinking level.
    public var samplingParamsByThinkingLevel: [ThinkingLevel: JSONObject]?
    public var compat: ModelCompat?

    private enum CodingKeys: String, CodingKey {
        case id, name, api, provider, input, cost, headers, reasoning, thinkingLevelMap, contextWindow, maxTokens
        case samplingParams, samplingParamsByThinkingLevel, compat
        case baseURL = "baseUrl"
    }

    public init(
        id: String,
        name: String? = nil,
        api: API,
        provider: String,
        baseURL: String,
        input: [InputModality] = [.text],
        cost: ModelCost = ModelCost(rates: CostRates(input: 0, output: 0)),
        headers: [String: String]? = nil,
        reasoning: Bool = false,
        thinkingLevelMap: ThinkingLevelMap? = nil,
        contextWindow: Int = 128_000,
        maxTokens: Int = 16_384,
        samplingParams: JSONObject? = nil,
        samplingParamsByThinkingLevel: [ThinkingLevel: JSONObject]? = nil,
        compat: ModelCompat? = nil
    ) {
        self.id = id
        self.name = name ?? id
        self.api = api
        self.provider = provider
        self.baseURL = baseURL
        self.input = input
        self.cost = cost
        self.headers = headers
        self.reasoning = reasoning
        self.thinkingLevelMap = thinkingLevelMap
        self.contextWindow = contextWindow
        self.maxTokens = maxTokens
        self.samplingParams = samplingParams
        self.samplingParamsByThinkingLevel = samplingParamsByThinkingLevel
        self.compat = compat
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try container.decode(String.self, forKey: .id),
            name: try container.decodeIfPresent(String.self, forKey: .name),
            api: try container.decode(API.self, forKey: .api),
            provider: try container.decode(String.self, forKey: .provider),
            baseURL: try container.decode(String.self, forKey: .baseURL),
            input: try container.decodeIfPresent([InputModality].self, forKey: .input) ?? [.text],
            cost: try container.decodeIfPresent(ModelCost.self, forKey: .cost) ?? ModelCost(rates: CostRates(input: 0, output: 0)),
            headers: try container.decodeIfPresent([String: String].self, forKey: .headers),
            reasoning: try container.decodeIfPresent(Bool.self, forKey: .reasoning) ?? false,
            thinkingLevelMap: try container.decodeIfPresent(ThinkingLevelMap.self, forKey: .thinkingLevelMap),
            contextWindow: try container.decodeIfPresent(Int.self, forKey: .contextWindow) ?? 0,
            maxTokens: try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 0,
            samplingParams: try container.decodeIfPresent(JSONObject.self, forKey: .samplingParams),
            samplingParamsByThinkingLevel: try container.decodeIfPresent(
                [ThinkingLevel: JSONObject].self, forKey: .samplingParamsByThinkingLevel),
            compat: try container.decodeIfPresent(ModelCompat.self, forKey: .compat)
        )
    }

    public var supportsImages: Bool { input.contains(.image) }

    /// Thinking levels this model accepts, from `off` upward.
    public var supportedThinkingLevels: [ThinkingLevel] {
        guard reasoning else { return [.off] }
        return ThinkingLevel.allCases.filter { level in
            switch thinkingLevelMap?[level] {
            case .unsupported: false
            case .value: true
            case nil: level != .xhigh && level != .max
            }
        }
    }

    /// The nearest supported level: the requested one, else the next higher, else the next lower.
    public func clampThinkingLevel(_ level: ThinkingLevel) -> ThinkingLevel {
        let supported = supportedThinkingLevels
        return supported.first { $0 >= level } ?? supported.last ?? .off
    }

    /// The provider value for a level: its mapping when one is set, else the level's own name
    /// (`none` for `off`, which no API spells `off`); `nil` when the level is mapped as unsupported.
    func thinkingValue(for level: ThinkingLevel) -> String? {
        switch thinkingLevelMap?[level] {
        case .unsupported: nil
        case .value(let value): value
        case nil: level == .off ? "none" : level.rawValue
        }
    }

    /// Whether the map explicitly marks `level` as unsupported.
    func thinkingLevelIsUnsupported(_ level: ThinkingLevel) -> Bool {
        thinkingLevelMap?[level] == .unsupported
    }
}
