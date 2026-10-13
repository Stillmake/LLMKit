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

/// The built-in model catalog, generated from pi-ai's (see `Scripts/sync-catalog.py`).
public struct ModelCatalog: Sendable {
    public let models: [Model]

    public init(models: [Model]) {
        self.models = models
    }

    public static let builtin: ModelCatalog = {
        guard let url = Bundle.module.url(forResource: "models", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let models = try? JSONDecoder().decode([Model].self, from: data)
        else {
            assertionFailure("The bundled model catalog is missing or malformed")
            return ModelCatalog(models: [])
        }
        return ModelCatalog(models: models)
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
