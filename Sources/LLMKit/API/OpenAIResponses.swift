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
                headers["session_id"] = sessionId
                headers["x-client-request-id"] = sessionId
            case .openAINoSession:
                headers["x-client-request-id"] = sessionId
            }
        }

        // A non-`sk-` credential sent straight to OpenAI is a Sign in with ChatGPT token, which
        // rejects the cache-retention, output-cap and temperature fields.
        let isChatGPTSignIn = model.provider == "openai" && model.baseURL == "https://api.openai.com/v1"
            && !(options.apiKey ?? "sk-").hasPrefix("sk-")
        var body: JSONObject = [
            "model": .string(model.id),
            "input": .array(OpenAIResponsesShared.convertMessages(
                model: model,
                messages: messages,
                toolCallProviders: ["openai", "openai-codex", "opencode"],
                includeSystemPrompt: true,
                supportsMidConvoSystemMessages: supportsMidConvoSystemMessages
            )),
            "stream": true,
            "store": false,
        ]
        if let sessionId { body["prompt_cache_key"] = .string(String(sessionId.prefix(64))) }
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
                let value = options.reasoning.map { model.thinkingValue(for: $0) ?? $0.rawValue } ?? effort.rawValue
                body["reasoning"] = ["effort": .string(value), "summary": .string(options.raw.reasoningSummary ?? "auto")]
                body["include"] = ["reasoning.encrypted_content"]
            } else if model.provider != "github-copilot", !model.thinkingLevelIsUnsupported(.off) {
                body["reasoning"] = ["effort": .string(model.thinkingValue(for: .off) ?? "none")]
            }
            if model.provider == "xai" { body["include"] = ["reasoning.encrypted_content"] }
        }

        let payload = options.finalize(&body, model: model, applySamplingParams: true)
        let url = try HTTP.url(model.baseURL, path: "/responses")
        let bytes = try await HTTP.send(
            HTTP.request(url: url, body: payload, headers: headers, model: model, options: options),
            model: model,
            options: options
        )
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

/// ChatGPT's Codex backend (`chatgpt.com/backend-api/codex/responses`), authenticated with a
/// ChatGPT OAuth access token. Same Responses wire format with Codex-specific request fields.
/// Ports pi-ai's `openai-codex-responses.ts` over SSE (the WebSocket transport is not ported).
enum OpenAICodexResponses {
    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let token = try options.requireAPIKey(for: model)
        // The account id lives in the access token's claims.
        let claims = token.split(separator: ".", omittingEmptySubsequences: false)
        let encodedClaims = claims.count == 3
            ? claims[1].replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
            : ""
        guard claims.count == 3,
            let claimsData = Data(base64Encoded: encodedClaims + String(repeating: "=", count: (4 - encodedClaims.count % 4) % 4)),
            let accountId = (try? JSONValue(data: claimsData))?["https://api.openai.com/auth"]?["chatgpt_account_id"]?.stringValue
        else {
            throw LLMError.invalidRequest("Failed to extract the ChatGPT account id from the access token")
        }

        let overrides = model.compat ?? ModelCompat()
        let supportsMidConvoSystemMessages = overrides.supportsMidConvoSystemMessages ?? false
        let messages = Transcript.resolve(transcript, supportsMidConvoSystemMessages: supportsMidConvoSystemMessages)
        let sessionId = options.raw.cacheRetention == .none ? nil : options.raw.sessionId

        var headers: [String: String?] = [
            "originator": "llmkit",
            "Authorization": "Bearer \(token)",
            "chatgpt-account-id": accountId,
            "OpenAI-Beta": "responses=experimental",
        ]
        if let sessionId {
            headers["session-id"] = sessionId
            headers["x-client-request-id"] = sessionId
        }

        let instructions = Transcript.initialSystemMessage(messages)?.promptText ?? ""
        var body: JSONObject = [
            "model": .string(model.id),
            "store": false,
            "stream": true,
            "instructions": .string(instructions.isEmpty ? "You are a helpful assistant." : instructions),
            "input": .array(OpenAIResponsesShared.convertMessages(
                model: model,
                messages: messages,
                toolCallProviders: ["openai", "openai-codex", "opencode"],
                includeSystemPrompt: false,
                supportsMidConvoSystemMessages: supportsMidConvoSystemMessages
            )),
            "text": ["verbosity": .string(options.raw.textVerbosity ?? "low")],
            "include": ["reasoning.encrypted_content"],
            "tool_choice": .string(options.raw.toolChoice?.rawValue ?? "auto"),
            "parallel_tool_calls": true,
        ]
        if let sessionId { body["prompt_cache_key"] = .string(sessionId) }
        if let temperature = options.raw.temperature { body["temperature"] = .number(temperature) }
        if let serviceTier = options.raw.serviceTier { body["service_tier"] = .string(serviceTier) }
        let tools = Transcript.currentTools(messages)
        if !tools.isEmpty {
            body["tools"] = .array(try OpenAIResponsesShared.convertTools(
                tools, defaultStrict: nil, supportsStrictMode: overrides.supportsStrictMode ?? true))
        }
        if let effort = options.reasoning {
            if let value = model.thinkingValue(for: effort) {
                body["reasoning"] = ["effort": .string(value), "summary": .string(options.raw.reasoningSummary ?? "auto")]
            }
        } else if model.reasoning, !model.thinkingLevelIsUnsupported(.off) {
            body["reasoning"] = ["effort": .string(model.thinkingValue(for: .off) ?? "none")]
        }

        let payload = options.finalize(&body, model: model, applySamplingParams: false)
        var base = model.baseURL.isEmpty ? "https://chatgpt.com/backend-api" : model.baseURL
        while base.hasSuffix("/") { base.removeLast() }
        let path = base.hasSuffix("/codex/responses") ? "" : base.hasSuffix("/codex") ? "/responses" : "/codex/responses"
        let bytes = try await HTTP.send(
            HTTP.request(url: try HTTP.url(base, path: path), body: payload, headers: headers, model: model, options: options),
            model: model,
            options: options,
            isRetryable: { status, body in
                // Exhausted plans also answer 429; retrying cannot help.
                !(status == 429 && body.range(
                    of: "GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing",
                    options: [.regularExpression, .caseInsensitive]
                ) != nil)
            }
        )
        builder.start()
        let requestTier = options.raw.serviceTier
        try await OpenAIResponsesShared.process(
            HTTP.jsonEvents(bytes).map { $0.json },
            builder: builder,
            model: model,
            codex: true,
            serviceTier: { responseTier in
                // Codex reports "default" even when it honored a requested flex or priority tier.
                responseTier == "default" && (requestTier == "flex" || requestTier == "priority") ? requestTier : responseTier ?? requestTier
            }
        )
    }
}
