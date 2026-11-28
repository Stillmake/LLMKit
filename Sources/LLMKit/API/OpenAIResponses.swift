import Foundation

/// OpenAI Responses API (OpenAI, xAI, GitHub Copilot, OpenRouter, OpenCode). Requests run with
/// `store: false` and replay encrypted reasoning, so conversations stay stateless on the server.
/// Ports pi-ai's `openai-responses.ts`.
enum OpenAIResponses {
    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let overrides = model.compat ?? ModelCompat()
        let supportsMidConvoSystemMessages = overrides.supportsMidConvoSystemMessages ?? false
        let supportsLongCacheRetention = overrides.supportsLongCacheRetention ?? true
        let supportsStrictMode = overrides.supportsStrictMode ?? false
        let supportsExplicitPromptCacheMode = overrides.supportsExplicitPromptCacheMode ?? false
        let promptCacheKeyMode = overrides.promptCacheKeyMode ?? .auto
        let isOpenRouter = model.provider == "openrouter" || model.baseURL.contains("openrouter.ai")
        let sessionAffinityFormat = overrides.sessionAffinityFormat ?? (isOpenRouter ? .openRouter : .openAI)

        let messages = Transcript.resolve(transcript, supportsMidConvoSystemMessages: supportsMidConvoSystemMessages)
        if options.apiKey == nil, !options.hasAuthorizationHeader {
            throw LLMError.missingAPIKey(provider: model.provider)
        }
        let retention = options.raw.cacheRetention
        let sessionId = retention == .none ? nil : options.raw.sessionId

        var headers: [String: String?] = model.provider == "github-copilot" ? Transcript.copilotHeaders(messages) : [:]
        if let apiKey = options.apiKey { headers["Authorization"] = "Bearer \(apiKey)" }
        if let sessionId {
            switch sessionAffinityFormat {
            case .openRouter:
                headers["x-session-id"] = sessionId
            case .openAI:
                headers["session-id"] = sessionId
                headers["x-client-request-id"] = sessionId
            case .openAINoSession:
                headers["x-client-request-id"] = sessionId
            }
        }

        // A non-`sk-` credential sent straight to OpenAI is a Sign in with ChatGPT token, which
        // rejects the cache-retention, output-cap and temperature fields.
        let isChatGPTSignIn = model.provider == "openai" && model.baseURL == "https://api.openai.com/v1"
            && !(options.apiKey ?? "sk-").hasPrefix("sk-")
        // Changing the thinking level inside the conversation keeps the prompt cache: the request keeps
        // a fixed effort and each turn's level travels as a `configuration_update` input item.
        let configurationUpdateEffort = overrides.supportsMidConvoEffort == true && !isChatGPTSignIn
            ? options.reasoning.flatMap { model.thinkingValue(for: $0) ?? $0.rawValue }
            : nil
        builder.message.providerThinkingLevel = configurationUpdateEffort
        var body: JSONObject = [
            "model": .string(model.id),
            "input": .array(OpenAIResponsesShared.convertMessages(
                model: model,
                messages: messages,
                toolCallProviders: ["openai", "openai-codex", "opencode"],
                supportsMidConvoSystemMessages: supportsMidConvoSystemMessages,
                configurationUpdateEffort: configurationUpdateEffort
            ).input),
            "stream": true,
            "store": false,
        ]
        if let sessionId, retention != .none, promptCacheKeyMode != .disabled {
            body["prompt_cache_key"] = .string(String(sessionId.prefix(64)))
        }
        if !isChatGPTSignIn {
            if retention == .long, supportsLongCacheRetention, !supportsExplicitPromptCacheMode {
                body["prompt_cache_retention"] = "24h"
            }
            if supportsExplicitPromptCacheMode {
                if retention == .none {
                    body["prompt_cache_options"] = ["mode": "explicit"]
                } else if retention == .long, supportsLongCacheRetention {
                    body["prompt_cache_options"] = ["ttl": "30m"]
                }
            }
            if overrides.supportsMaxOutputTokens ?? true {
                // Responses rejects max_output_tokens below 16.
                body["max_output_tokens"] = .number(Double(max(options.maxTokens, 16)))
            }
            if let temperature = options.raw.temperature { body["temperature"] = .number(temperature) }
        }
        if let serviceTier = options.raw.serviceTier { body["service_tier"] = .string(serviceTier) }
        let tools = Transcript.currentTools(messages)
        if !tools.isEmpty {
            body["tools"] = .array(try OpenAIResponsesShared.convertTools(tools, defaultStrict: false, supportsStrictMode: supportsStrictMode))
        }
        if let toolChoice = options.raw.toolChoice { body["tool_choice"] = .string(toolChoice.rawValue) }

        if model.reasoning {
            if let effort = options.reasoning ?? (options.raw.reasoningSummary == nil ? nil : .medium) {
                // A fixed effort keeps the cache while configuration_update items carry the level.
                let value = configurationUpdateEffort == nil
                    ? options.reasoning.flatMap { model.thinkingValue(for: $0) } ?? effort.rawValue
                    : "medium"
                body["reasoning"] = ["effort": .string(value), "summary": .string(options.raw.reasoningSummary ?? "auto")]
                body["include"] = ["reasoning.encrypted_content"]
            } else if model.provider != "github-copilot", !model.thinkingLevelIsUnsupported(.off) {
                body["reasoning"] = ["effort": .string(model.thinkingValue(for: .off) ?? "none")]
            }
            if model.provider == "xai" { body["include"] = ["reasoning.encrypted_content"] }
        }

        let payload = options.finalize(&body, model: model, applySamplingParams: true)
        let url = try HTTP.url(model.baseURL, path: "/responses")
        // Sign in with ChatGPT shares the subscription's usage limit with other apps; on ChatGPT
        // Business plans, the workspace admin also sets a per-user limit. Both reset after hours.
        let usageLimitHints = [
            "subscription_sharing_usage_limit_exceeded": "Check your ChatGPT usage: https://chatgpt.com/settings/usage",
            "subscription_sharing_workplace_individual_usage_limit_exceeded":
                "You've reached your ChatGPT workspace's usage limit. Ask your workspace admin to increase it at admin.openai.com → Workspace → Usage limits.",
        ]
        let bytes: URLSession.AsyncBytes
        do {
            bytes = try await HTTP.send(
                HTTP.request(url: url, body: payload, headers: headers, model: model, options: options),
                model: model,
                options: options,
                isRetryable: { _, body in !usageLimitHints.keys.contains { body.contains($0) } }
            )
        } catch LLMError.http(let status, let message, let responseHeaders) {
            let hint = usageLimitHints.first { message.contains($0.key) }?.value
            throw LLMError.http(status: status, message: hint.map { "\(message)\n\($0)" } ?? message, headers: responseHeaders)
        }
        builder.start()
        try await OpenAIResponsesShared.process(
            HTTP.jsonEvents(bytes).map { $0.json },
            builder: builder,
            model: model,
            codex: false,
            serviceTier: { $0 ?? options.raw.serviceTier }
        )
    }
}
