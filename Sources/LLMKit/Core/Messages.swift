import Foundation

// MARK: - Content

public struct TextContent: Codable, Sendable, Hashable {
    public var text: String
    /// Provider replay metadata: OpenAI Responses message id/phase, or a Gemini thought signature.
    public var textSignature: String?

    public init(_ text: String, textSignature: String? = nil) {
        self.text = text
        self.textSignature = textSignature
    }
}

public struct ThinkingContent: Codable, Sendable, Hashable {
    public var thinking: String
    /// Provider-specific opaque reasoning replay data (Anthropic signature, OpenAI reasoning item,
    /// OpenRouter reasoning details, or the Chat Completions field the reasoning arrived in).
    public var thinkingSignature: String?
    /// The reasoning was redacted by safety filters; `thinkingSignature` holds the encrypted payload.
    public var redacted: Bool

    public init(_ thinking: String, thinkingSignature: String? = nil, redacted: Bool = false) {
        self.thinking = thinking
        self.thinkingSignature = thinkingSignature
        self.redacted = redacted
    }
}

public struct ImageContent: Codable, Sendable, Hashable {
    /// Base64-encoded image bytes.
    public var data: String
    public var mimeType: String

    public init(data: String, mimeType: String) {
        self.data = data
        self.mimeType = mimeType
    }

    public init(data: Data, mimeType: String) {
        self.init(data: data.base64EncodedString(), mimeType: mimeType)
    }
}

public struct ToolCall: Codable, Sendable, Hashable, Identifiable {
    public var id: String
    public var name: String
    public var arguments: JSONObject
    /// Gemini: opaque signature that must be echoed back with this call.
    public var thoughtSignature: String?
    /// OpenAI Responses namespace for namespaced tools.
    public var namespace: String?
    /// Set when the model's complete arguments text was not a valid JSON object; `arguments` is then
    /// empty. Describes the problem and includes the text the model sent.
    public var argumentsError: String?

    public init(
        id: String,
        name: String,
        arguments: JSONObject,
        thoughtSignature: String? = nil,
        namespace: String? = nil,
        argumentsError: String? = nil
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.thoughtSignature = thoughtSignature
        self.namespace = namespace
        self.argumentsError = argumentsError
    }
}

/// Content a user or a tool result can carry.
public enum UserContent: Sendable, Hashable {
    case text(TextContent)
    case image(ImageContent)
}

/// Content an assistant response can carry.
public enum AssistantContent: Sendable, Hashable {
    case text(TextContent)
    case thinking(ThinkingContent)
    case toolCall(ToolCall)
}

// MARK: - Usage

public struct Cost: Codable, Sendable, Hashable {
    public var input: Double = 0
    public var output: Double = 0
    public var cacheRead: Double = 0
    public var cacheWrite: Double = 0
    public var total: Double = 0

    public init() {}
}

public struct Usage: Codable, Sendable, Hashable {
    /// Uncached input tokens.
    public var input = 0
    /// Output tokens, including reasoning tokens.
    public var output = 0
    public var cacheRead = 0
    public var cacheWrite = 0
    /// Subset of `cacheWrite` written with 1h retention (Anthropic only).
    public var cacheWrite1h: Int?
    /// Reasoning tokens as a subset of `output`, when the provider reports them.
    public var reasoning: Int?
    public var totalTokens = 0
    public var cost = Cost()

    public init() {}

    /// Prices this usage with the model's rates, honoring request-wide tiers and Anthropic's
    /// 2x-input rate for 1h cache writes.
    mutating func applyCost(of model: Model) {
        let inputTokens = input + cacheRead + cacheWrite
        let rates = model.cost.tiers?
            .filter { inputTokens > $0.inputTokensAbove }
            .max { $0.inputTokensAbove < $1.inputTokensAbove }?
            .rates ?? model.cost.rates
        let longWrite = Double(cacheWrite1h ?? 0)
        let shortWrite = Double(cacheWrite) - longWrite
        cost.input = rates.input / 1_000_000 * Double(input)
        cost.output = rates.output / 1_000_000 * Double(output)
        cost.cacheRead = rates.cacheRead / 1_000_000 * Double(cacheRead)
        cost.cacheWrite = (rates.cacheWrite * shortWrite + rates.input * 2 * longWrite) / 1_000_000
        cost.total = cost.input + cost.output + cost.cacheRead + cost.cacheWrite
    }
}

public enum StopReason: String, Codable, Sendable, Hashable {
    case pending, stop, length, toolUse, error, aborted
}

// MARK: - Tools

public struct Tool: Codable, Sendable, Hashable {
    public enum StrictMode: String, Codable, Sendable, Hashable {
        /// Use JSON-schema constrained sampling when the provider supports it and the schema fits its subset.
        case prefer
        /// Fail the request when constrained sampling is unavailable.
        case require
    }

    public var name: String
    public var description: String
    /// JSON Schema for the arguments; the root must be `{"type": "object", ...}`.
    public var parameters: JSONValue
    public var strict: StrictMode?

    public init(name: String, description: String, parameters: JSONValue, strict: StrictMode? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict
    }
}

public enum ToolChoice: String, Codable, Sendable, Hashable {
    case auto, none, required
}

// MARK: - Messages

/// System instructions and tool declarations at one point in the transcript.
///
/// The leading system message is the system prompt. Later ones change it: `content` adds
/// instructions, `sections` replace (or, with `nil` text, remove) named sections, and
/// `toolsAdded`/`toolsRemoved` change the tool set. Models that accept mid-conversation system
/// messages receive them in place; others get the leading message rebuilt from the replayed state.
public struct SystemMessage: Codable, Sendable, Hashable {
    public struct Section: Codable, Sendable, Hashable {
        public var name: String
        /// `nil` removes a previously declared section.
        public var text: String?

        public init(name: String, text: String?) {
            self.name = name
            self.text = text
        }
    }

    public var content: String
    public var sections: [Section]
    public var toolsAdded: [Tool]
    public var toolsRemoved: [String]
    public var timestamp: Date

    public init(
        _ content: String,
        sections: [Section] = [],
        toolsAdded: [Tool] = [],
        toolsRemoved: [String] = [],
        timestamp: Date = .now
    ) {
        self.content = content
        self.sections = sections
        self.toolsAdded = toolsAdded
        self.toolsRemoved = toolsRemoved
        self.timestamp = timestamp
    }

    /// The full prompt: content followed by the sections.
    var promptText: String {
        ([content] + sections.compactMap(\.text)).filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// A later system message rendered for models that accept system messages mid-conversation.
    var updateText: String {
        ((content.isEmpty ? [] : [content]) + sections.map { section in
            section.text.map { "Updated system prompt section \"\(section.name)\":\n\n\($0)" }
                ?? "Removed system prompt section \"\(section.name)\"."
        }).joined(separator: "\n\n")
    }
}

public struct UserMessage: Codable, Sendable, Hashable {
    public var content: [UserContent]
    public var timestamp: Date

    public init(_ content: [UserContent], timestamp: Date = .now) {
        self.content = content
        self.timestamp = timestamp
    }

    public init(_ text: String, images: [ImageContent] = [], timestamp: Date = .now) {
        self.init([.text(TextContent(text))] + images.map(UserContent.image), timestamp: timestamp)
    }
}

public struct AssistantMessage: Codable, Sendable, Hashable {
    public var content: [AssistantContent]
    public var api: API
    public var provider: String
    public var model: String
    /// Concrete model reported by the provider when it differs from `model`.
    public var responseModel: String?
    public var responseId: String?
    public var usage: Usage
    public var stopReason: StopReason
    public var errorMessage: String?
    /// The provider's own stop reason, for diagnostics.
    public var rawStopReason: String?
    public var timestamp: Date
    /// Time from request start until the response ended.
    public var duration: Duration?

    public init(model: Model, timestamp: Date = .now) {
        self.content = []
        self.api = model.api
        self.provider = model.provider
        self.model = model.id
        self.usage = Usage()
        self.stopReason = .pending
        self.timestamp = timestamp
    }

    public var text: String {
        content.compactMap { block -> String? in
            guard case .text(let text) = block else { return nil }
            return text.text
        }.joined()
    }

    public var toolCalls: [ToolCall] {
        content.compactMap { block -> ToolCall? in
            guard case .toolCall(let call) = block else { return nil }
            return call
        }
    }
}

public struct ToolResultMessage: Codable, Sendable, Hashable {
    public var toolCallId: String
    public var toolName: String
    public var content: [UserContent]
    public var isError: Bool
    /// Application data kept with the result and never sent to the model.
    public var details: JSONValue?
    public var timestamp: Date

    public init(
        toolCallId: String,
        toolName: String,
        content: [UserContent],
        isError: Bool = false,
        details: JSONValue? = nil,
        timestamp: Date = .now
    ) {
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.content = content
        self.isError = isError
        self.details = details
        self.timestamp = timestamp
    }

    var text: String {
        content.compactMap { block -> String? in
            guard case .text(let text) = block else { return nil }
            return text.text
        }.joined(separator: "\n")
    }

    var images: [ImageContent] {
        content.compactMap { block -> ImageContent? in
            guard case .image(let image) = block else { return nil }
            return image
        }
    }
}

public enum Message: Sendable, Hashable {
    case system(SystemMessage)
    case user(UserMessage)
    case assistant(AssistantMessage)
    case toolResult(ToolResultMessage)

    public var timestamp: Date {
        switch self {
        case .system(let message): message.timestamp
        case .user(let message): message.timestamp
        case .assistant(let message): message.timestamp
        case .toolResult(let message): message.timestamp
        }
    }
}

/// Request input: `systemPrompt` and `tools` are shorthand for a leading system message.
public struct Context: Sendable, Hashable {
    public var systemPrompt: String?
    public var messages: [Message]
    public var tools: [Tool]

    public init(systemPrompt: String? = nil, messages: [Message] = [], tools: [Tool] = []) {
        self.systemPrompt = systemPrompt
        self.messages = messages
        self.tools = tools
    }
}

// MARK: - Codable with pi-ai's discriminators

extension ThinkingContent {
    private enum CodingKeys: String, CodingKey { case thinking, thinkingSignature, redacted }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            try container.decode(String.self, forKey: .thinking),
            thinkingSignature: try container.decodeIfPresent(String.self, forKey: .thinkingSignature),
            redacted: try container.decodeIfPresent(Bool.self, forKey: .redacted) ?? false
        )
    }
}

extension SystemMessage {
    private enum CodingKeys: String, CodingKey { case content, sections, toolsAdded, toolsRemoved, timestamp }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            try container.decode(String.self, forKey: .content),
            sections: try container.decodeIfPresent([Section].self, forKey: .sections) ?? [],
            toolsAdded: try container.decodeIfPresent([Tool].self, forKey: .toolsAdded) ?? [],
            toolsRemoved: try container.decodeIfPresent([String].self, forKey: .toolsRemoved) ?? [],
            timestamp: try container.decode(Date.self, forKey: .timestamp)
        )
    }
}

extension UserContent: Codable {
    private enum CodingKeys: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        switch try decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .type) {
        case "image": self = .image(try ImageContent(from: decoder))
        default: self = .text(try TextContent(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let content):
            try container.encode("text", forKey: .type)
            try content.encode(to: encoder)
        case .image(let content):
            try container.encode("image", forKey: .type)
            try content.encode(to: encoder)
        }
    }
}

extension AssistantContent: Codable {
    private enum CodingKeys: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        switch try decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .type) {
        case "thinking": self = .thinking(try ThinkingContent(from: decoder))
        case "toolCall": self = .toolCall(try ToolCall(from: decoder))
        default: self = .text(try TextContent(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let content):
            try container.encode("text", forKey: .type)
            try content.encode(to: encoder)
        case .thinking(let content):
            try container.encode("thinking", forKey: .type)
            try content.encode(to: encoder)
        case .toolCall(let content):
            try container.encode("toolCall", forKey: .type)
            try content.encode(to: encoder)
        }
    }
}

extension Message: Codable {
    private enum CodingKeys: String, CodingKey { case role }

    public init(from decoder: Decoder) throws {
        switch try decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .role) {
        case "system": self = .system(try SystemMessage(from: decoder))
        case "assistant": self = .assistant(try AssistantMessage(from: decoder))
        case "toolResult": self = .toolResult(try ToolResultMessage(from: decoder))
        default: self = .user(try UserMessage(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .system(let message):
            try container.encode("system", forKey: .role)
            try message.encode(to: encoder)
        case .user(let message):
            try container.encode("user", forKey: .role)
            try message.encode(to: encoder)
        case .assistant(let message):
            try container.encode("assistant", forKey: .role)
            try message.encode(to: encoder)
        case .toolResult(let message):
            try container.encode("toolResult", forKey: .role)
            try message.encode(to: encoder)
        }
    }
}

// MARK: - Stream events

/// Events of one assistant response.
///
/// A successful stream emits `start`, block events, and ends with `done`. A failure ends with
/// `error`, whose message has `stopReason` `.error` or `.aborted` and an `errorMessage`. A
/// request that fails before generation starts may emit `error` without `start`. Each text,
/// thinking and tool-call block is opened by `*Start`, grows through `*Delta`, and is closed by
/// an authoritative `*End`; `index` is the block's position in the final `content`.
public enum AssistantMessageEvent: Sendable {
    case start
    case textStart(index: Int)
    case textDelta(index: Int, delta: String)
    case textEnd(index: Int, content: TextContent)
    case thinkingStart(index: Int)
    case thinkingDelta(index: Int, delta: String)
    case thinkingEnd(index: Int, content: ThinkingContent)
    case toolCallStart(index: Int, id: String, name: String)
    /// `arguments` is the best-effort parse of the JSON received so far.
    case toolCallDelta(index: Int, delta: String, arguments: JSONObject)
    case toolCallEnd(index: Int, toolCall: ToolCall)
    case done(AssistantMessage)
    case error(AssistantMessage)
}
