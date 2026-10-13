import Foundation

/// One Swift API over every supported LLM wire format.
///
/// ```swift
/// let model = ModelCatalog.builtin.model(provider: "anthropic", id: "claude-sonnet-4-5")!
/// let context = Context(systemPrompt: "Be brief.", messages: [.user(UserMessage("Hi"))])
/// for await event in LLM.stream(model: model, context: context, options: StreamOptions(reasoning: .medium)) {
///     if case .textDelta(_, let delta) = event { print(delta, terminator: "") }
/// }
/// ```
public enum LLM {
    /// Streams one assistant response. Failures never throw: they end the stream with `.error`.
    public static func stream(model: Model, context: Context, options: StreamOptions = StreamOptions()) -> AssistantMessageStream {
        let (events, continuation) = AsyncStream.makeStream(of: AssistantMessageEvent.self)
        let task = Task {
            let builder = ResponseBuilder(model: model, continuation: continuation)
            do {
                let transcript = Transcript.normalize(context)
                let resolved = ResolvedOptions(model: model, transcript: transcript, options: options)
                switch model.api {
                case .openAICompletions:
                    try await OpenAICompletions.run(model: model, transcript: transcript, options: resolved, builder: builder)
                case .openAIResponses:
                    try await OpenAIResponses.run(model: model, transcript: transcript, options: resolved, builder: builder)
                case .openAICodexResponses:
                    try await OpenAICodexResponses.run(model: model, transcript: transcript, options: resolved, builder: builder)
                case .anthropicMessages:
                    try await AnthropicMessages.run(model: model, transcript: transcript, options: resolved, builder: builder)
                case .googleGenerativeAI:
                    try await GoogleGenerativeAI.run(model: model, transcript: transcript, options: resolved, builder: builder)
                default:
                    throw LLMError.unsupportedAPI(model.api)
                }
                try builder.complete()
            } catch {
                builder.fail(error)
            }
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
        return AssistantMessageStream(events: events)
    }

    /// Runs one request to completion. Returns the final message, which carries `stopReason`
    /// `.error` or `.aborted` and an `errorMessage` when the request failed.
    public static func complete(model: Model, context: Context, options: StreamOptions = StreamOptions()) async throws -> AssistantMessage {
        try await stream(model: model, context: context, options: options).result()
    }
}

/// A provider's display name and its logo on models.dev.
public struct ProviderInfo: Codable, Sendable, Hashable {
    /// Display name, e.g. `DeepSeek`.
    public var name: String
    /// The models.dev provider id whose logo is at `https://models.dev/logos/<logo>.svg`.
    public var logo: String

    public init(name: String, logo: String) {
        self.name = name
        self.logo = logo
    }
}

/// The built-in model catalog, generated from pi-ai's (see `Scripts/sync-catalog.py`).
public struct ModelCatalog: Sendable {
    public let models: [Model]
    /// Keyed by provider id.
    public let providerInfo: [String: ProviderInfo]

    public init(models: [Model], providerInfo: [String: ProviderInfo] = [:]) {
        self.models = models
        self.providerInfo = providerInfo
    }

    public static let builtin: ModelCatalog = {
        func load<T: Decodable>(_ name: String, as type: T.Type) -> T? {
            guard let url = Bundle.module.url(forResource: name, withExtension: "json"),
                let data = try? Data(contentsOf: url),
                let value = try? JSONDecoder().decode(type, from: data)
            else {
                assertionFailure("The bundled \(name).json is missing or malformed")
                return nil
            }
            return value
        }
        return ModelCatalog(
            models: load("models", as: [Model].self) ?? [],
            providerInfo: load("providers", as: [String: ProviderInfo].self) ?? [:]
        )
    }()

    public var providers: [String] {
        Array(Set(models.map(\.provider))).sorted()
    }

    public func models(provider: String) -> [Model] {
        models.filter { $0.provider == provider }
    }

    public func model(provider: String, id: String) -> Model? {
        models.first { $0.provider == provider && $0.id == id }
    }
}
