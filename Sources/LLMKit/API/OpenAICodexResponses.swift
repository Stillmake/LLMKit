import Foundation

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
        // Keep the initial effort in the request and replay changes in the input for cache reuse.
        let configurationUpdateEffort = (overrides.supportsMidConvoEffort == true && model.reasoning)
            ? options.reasoning.flatMap { model.thinkingValue(for: $0) } ?? model.thinkingValue(for: .off)
            : nil
        builder.message.providerThinkingLevel = configurationUpdateEffort

        var headers: [String: String?] = [
            // Codex expects an originator on every request; it is not attribution.
            "originator": "pi",
            "Authorization": "Bearer \(token)",
            "chatgpt-account-id": accountId,
            "OpenAI-Beta": "responses=experimental",
        ]
        if let sessionId {
            headers["session-id"] = sessionId
            headers["x-client-request-id"] = sessionId
        }

        let prompt = OpenAIResponsesShared.convertMessages(
            model: model,
            messages: messages,
            toolCallProviders: ["openai", "openai-codex", "opencode"],
            supportsMidConvoSystemMessages: supportsMidConvoSystemMessages,
            configurationUpdateEffort: configurationUpdateEffort,
            pinInitialEffort: true
        )
        var body: JSONObject = [
            "model": .string(model.id),
            "store": false,
            "stream": true,
            "input": .array(prompt.input),
            "text": ["verbosity": .string(options.raw.textVerbosity ?? "low")],
            "include": ["reasoning.encrypted_content"],
            "tool_choice": .string(options.raw.toolChoice?.rawValue ?? "auto"),
            "parallel_tool_calls": true,
        ]
        if let sessionId { body["prompt_cache_key"] = .string(sessionId) }
        if overrides.supportsMaxOutputTokens ?? true {
            // The Responses API rejects max_output_tokens below 16.
            body["max_output_tokens"] = .number(Double(max(options.maxTokens, 16)))
        }
        if let temperature = options.raw.temperature { body["temperature"] = .number(temperature) }
        if let serviceTier = options.raw.serviceTier { body["service_tier"] = .string(serviceTier) }
        let tools = Transcript.currentTools(messages)
        if !tools.isEmpty {
            body["tools"] = .array(try OpenAIResponsesShared.convertTools(
                tools, defaultStrict: nil, supportsStrictMode: overrides.supportsStrictMode ?? true))
        }
        if let effort = prompt.initialEffort ?? options.reasoning.flatMap({ model.thinkingValue(for: $0) }) {
            body["reasoning"] = ["effort": .string(effort), "summary": .string(options.raw.reasoningSummary ?? "auto")]
        } else if options.reasoning == nil, model.reasoning, !model.thinkingLevelIsUnsupported(.off) {
            body["reasoning"] = [
                "effort": .string(model.thinkingValue(for: .off) ?? "none"),
                "summary": .string(options.raw.reasoningSummary ?? "auto"),
            ]
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
