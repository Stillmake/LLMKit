import Foundation

/// Anthropic Messages API and compatible endpoints (Kimi Coding, MiniMax, Z.ai, OpenRouter,
/// Fireworks, GitHub Copilot, ...). Supports budget-based and adaptive thinking, signed and
/// redacted thinking replay, prompt caching, and eager tool-input streaming.
/// Ports pi-ai's `anthropic-messages.ts`.
enum AnthropicMessages {
    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let overrides = model.compat ?? ModelCompat()
        let isOpenRouter = model.provider == "openrouter" || model.baseURL.contains("openrouter.ai")
        let supportsMidConvoSystemMessages = overrides.supportsMidConvoSystemMessages ?? false
        let supportsEagerToolInputStreaming = overrides.supportsEagerToolInputStreaming ?? true
        let supportsLongCacheRetention = overrides.supportsLongCacheRetention ?? true
        let messages = Transcript.resolve(transcript, supportsMidConvoSystemMessages: supportsMidConvoSystemMessages)
        let tools = Transcript.currentTools(messages)
        if options.apiKey == nil, !options.hasAuthorizationHeader {
            throw LLMError.missingAPIKey(provider: model.provider)
        }

        let retention = options.raw.cacheRetention
        var cacheControl: JSONValue?
        if retention != .none {
            cacheControl = .object(retention == .long && supportsLongCacheRetention ? ["type": "ephemeral", "ttl": "1h"] : ["type": "ephemeral"])
        }

        // Thinking: adaptive (effort) for models that require it, otherwise a token budget that
        // shares max_tokens with the answer.
        var maxTokens = options.maxTokens
        var thinking: JSONValue?
        var effort: String?
        let display = JSONValue.string(options.raw.thinkingDisplay ?? "summarized")
        if let level = options.reasoning, model.reasoning {
            if overrides.forceAdaptiveThinking == true {
                thinking = ["type": "adaptive", "display": display]
                if case .value(let mapped)? = model.thinkingLevelMap?[level] {
                    effort = mapped
                } else {
                    effort = switch level {
                    case .minimal, .low: "low"
                    case .medium: "medium"
                    default: "high"
                    }
                }
            } else {
                var budget = (options.raw.thinkingBudgets ?? ThinkingBudgets())[level]
                maxTokens = min(maxTokens + budget, model.maxTokens)
                if maxTokens <= budget { budget = max(0, maxTokens - minimumAnswerTokens) }
                if model.contextWindow > 0 {
                    maxTokens = min(maxTokens, max(1, model.contextWindow - Transcript.estimatedTokens(transcript) - 4096))
                }
                budget = min(budget, max(0, maxTokens - minimumAnswerTokens))
                thinking = ["type": "enabled", "budget_tokens": .number(Double(budget > 0 ? budget : 1024)), "display": display]
            }
        } else if model.reasoning, !model.thinkingLevelIsUnsupported(.off) {
            thinking = ["type": "disabled"]
        }

        var betas: [String] = []
        if !tools.isEmpty, !supportsEagerToolInputStreaming { betas.append("fine-grained-tool-streaming-2025-05-14") }
        if thinking?["type"] == "enabled" { betas.append("interleaved-thinking-2025-05-14") }

        var headers: [String: String?] = model.provider == "github-copilot" ? Transcript.copilotHeaders(messages) : [:]
        headers["anthropic-version"] = "2023-06-01"
        if !betas.isEmpty { headers["anthropic-beta"] = betas.joined(separator: ",") }
        if let apiKey = options.apiKey {
            if model.provider == "github-copilot" {
                headers["Authorization"] = "Bearer \(apiKey)"
            } else {
                headers["x-api-key"] = apiKey
            }
        }
        if let sessionId = options.raw.sessionId, retention != .none, overrides.sendSessionAffinityHeaders ?? isOpenRouter {
            let format = overrides.sessionAffinityFormat ?? (isOpenRouter ? .openRouter : nil)
            headers[format == .openRouter ? "x-session-id" : "x-session-affinity"] = sessionId
        }

        let transformed = Transcript.transform(messages, for: model) { id, _ in Transcript.sanitizeIdentifier(id) }
        let allowEmptySignature = overrides.allowEmptySignature ?? false
        var wire: [JSONValue] = []
        // Later system messages wait until just before the next assistant message: tool results
        // must immediately follow their tool use.
        var pendingSystem: [JSONValue] = []
        var index = transformed.first.map { if case .system = $0 { 1 } else { 0 } } ?? 0

        while index < transformed.count {
            switch transformed[index] {
            case .system(let system):
                let text = system.updateText
                if !text.isEmpty { pendingSystem.append(["role": "system", "content": [["type": "text", "text": .string(text)]]]) }

            case .user(let user):
                let blocks: [JSONValue] = user.content.compactMap { block -> JSONValue? in
                    switch block {
                    case .text(let text):
                        text.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : ["type": "text", "text": .string(text.text)]
                    case .image(let image):
                        ["type": "image", "source": ["type": "base64", "media_type": .string(image.mimeType), "data": .string(image.data)]]
                    }
                }
                if !blocks.isEmpty { wire.append(["role": "user", "content": .array(blocks)]) }

            case .assistant(let assistant):
                wire += pendingSystem
                pendingSystem = []
                let blocks: [JSONValue] = assistant.content.compactMap { block -> JSONValue? in
                    switch block {
                    case .text(let text):
                        return text.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? nil : ["type": "text", "text": .string(text.text)]
                    case .thinking(let thinking):
                        if thinking.redacted { return ["type": "redacted_thinking", "data": .string(thinking.thinkingSignature ?? "")] }
                        let signature = thinking.thinkingSignature?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        if signature.isEmpty {
                            if thinking.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return nil }
                            // An unsigned block (e.g. from an aborted stream) cannot replay as thinking.
                            return allowEmptySignature
                                ? ["type": "thinking", "thinking": .string(thinking.thinking), "signature": ""]
                                : ["type": "text", "text": .string(thinking.thinking)]
                        }
                        return ["type": "thinking", "thinking": .string(thinking.thinking), "signature": .string(thinking.thinkingSignature ?? "")]
                    case .toolCall(let call):
                        return ["type": "tool_use", "id": .string(call.id), "name": .string(call.name), "input": .object(call.arguments)]
                    }
                }
                if !blocks.isEmpty { wire.append(["role": "assistant", "content": .array(blocks)]) }

            case .toolResult:
                var results: [JSONValue] = []
                while index < transformed.count, case .toolResult(let result) = transformed[index] {
                    let content: JSONValue
                    if result.images.isEmpty {
                        content = .string(result.text)
                    } else {
                        let blocks: [JSONValue] = result.content.map { block in
                            switch block {
                            case .text(let text): ["type": "text", "text": .string(text.text)]
                            case .image(let image):
                                ["type": "image", "source": ["type": "base64", "media_type": .string(image.mimeType), "data": .string(image.data)]]
                            }
                        }
                        content = .array(blocks.contains { $0["type"] == "text" } ? blocks : [["type": "text", "text": "(see attached image)"]] + blocks)
                    }
                    results.append(["type": "tool_result", "tool_use_id": .string(result.toolCallId), "content": content, "is_error": .bool(result.isError)])
                    index += 1
                }
                wire.append(["role": "user", "content": .array(results)])
                continue
            }
            index += 1
        }
        wire += pendingSystem

        // Cache the conversation prefix at the last user or system turn.
        if let cacheControl, case .object(var last)? = wire.last, ["user", "system"].contains(last["role"]?.stringValue),
            case .array(var content)? = last["content"], case .object(var block)? = content.last,
            ["text", "image", "tool_result"].contains(block["type"]?.stringValue)
        {
            block["cache_control"] = cacheControl
            content[content.count - 1] = .object(block)
            last["content"] = .array(content)
            wire[wire.count - 1] = .object(last)
        }

        var body: JSONObject = [
            "model": .string(model.id),
            "messages": .array(wire),
            "max_tokens": .number(Double(maxTokens)),
            "stream": true,
        ]
        if let system = Transcript.initialSystemMessage(messages)?.promptText, !system.isEmpty {
            var block: JSONObject = ["type": "text", "text": .string(system)]
            block["cache_control"] = cacheControl
            body["system"] = [.object(block)]
        }
        // Temperature is incompatible with extended thinking and rejected by newer models.
        if let temperature = options.raw.temperature, options.reasoning == nil, overrides.supportsTemperature ?? true {
            body["temperature"] = .number(temperature)
        }
        if !tools.isEmpty {
            let supportsStrictTools = overrides.supportsStrictTools ?? false
            let toolCacheControl = overrides.supportsCacheControlOnTools ?? true ? cacheControl : nil
            body["tools"] = .array(try tools.enumerated().map { index, tool in
                let strict = try StrictSchema.resolve(tool, supportsStrictMode: supportsStrictTools, keywordCheck: StrictSchema.anthropicKeywordCheck)
                let parameters = StrictSchema.parameters(of: tool, strict: strict)
                var inputSchema: JSONObject = strict == true ? parameters.objectValue ?? [:] : [:]
                inputSchema["type"] = "object"
                inputSchema["properties"] = parameters["properties"] ?? [:]
                inputSchema["required"] = parameters["required"] ?? []
                var definition: JSONObject = [
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "input_schema": .object(inputSchema),
                ]
                if supportsEagerToolInputStreaming { definition["eager_input_streaming"] = true }
                if strict == true { definition["strict"] = true }
                if index == tools.count - 1 { definition["cache_control"] = toolCacheControl }
                return .object(definition)
            })
        }
        body["thinking"] = thinking
        if let effort { body["output_config"] = ["effort": .string(effort)] }
        if let userId = options.raw.userId { body["metadata"] = ["user_id": .string(userId)] }
        switch options.raw.toolChoice {
        case .auto?: body["tool_choice"] = ["type": "auto"]
        case .none?: body["tool_choice"] = ["type": "none"]
        case .required?: body["tool_choice"] = ["type": "any"]
        case nil: break
        }

        let payload = options.finalize(&body, model: model, applySamplingParams: false)
        let url = try HTTP.url(model.baseURL, path: "/v1/messages")
        let bytes = try await HTTP.send(
            HTTP.request(url: url, body: payload, headers: headers, model: model, options: options),
            model: model,
            options: options
        )
        builder.start()

        var blocks: [Int: ResponseBuilder.Block] = [:]
        var sawStart = false
        var sawStop = false
        for try await event in ServerSentEvents(base: bytes) {
            if event.event == "error" { throw LLMError.provider(HTTP.errorMessage(from: event.data)) }
            guard let type = event.event,
                ["message_start", "message_delta", "message_stop", "content_block_start", "content_block_delta", "content_block_stop"]
                    .contains(type)
            else { continue }
            guard let data = try? JSONValue(parsing: event.data) else {
                throw LLMError.protocolViolation("Could not parse Anthropic SSE event \(type): \(event.data)")
            }

            switch type {
            case "message_start":
                sawStart = true
                let message = data["message"]
                builder.message.responseId = message?["id"]?.stringValue
                if let responseModel = message?["model"]?.stringValue, responseModel != model.id {
                    builder.message.responseModel = responseModel
                }
                // Input counts arrive here, so an early abort still reports them.
                let usage = message?["usage"]
                builder.message.usage.input = usage?["input_tokens"]?.intValue ?? 0
                builder.message.usage.output = usage?["output_tokens"]?.intValue ?? 0
                builder.message.usage.cacheRead = usage?["cache_read_input_tokens"]?.intValue ?? 0
                builder.message.usage.cacheWrite = usage?["cache_creation_input_tokens"]?.intValue ?? 0
                builder.message.usage.cacheWrite1h = usage?["cache_creation"]?["ephemeral_1h_input_tokens"]?.intValue ?? 0
                updateTotals(&builder.message.usage, model: model)

            case "content_block_start":
                guard let index = data["index"]?.intValue, let block = data["content_block"] else { continue }
                switch block["type"]?.stringValue {
                case "text":
                    blocks[index] = builder.openText(block["text"]?.stringValue ?? "")
                case "thinking":
                    blocks[index] = builder.openThinking(block["thinking"]?.stringValue ?? "", signature: block["signature"]?.stringValue ?? "")
                case "redacted_thinking":
                    blocks[index] = builder.openThinking("[Reasoning redacted]", signature: block["data"]?.stringValue, redacted: true)
                case "tool_use":
                    blocks[index] = builder.openToolCall(
                        id: block["id"]?.stringValue ?? "",
                        name: block["name"]?.stringValue ?? "",
                        arguments: block["input"]?.objectValue ?? [:]
                    )
                case "fallback" where builder.blocks.isEmpty:
                    continue
                case "fallback":
                    throw LLMError.provider("Anthropic performed an unsupported mid-output model fallback")
                default:
                    continue
                }

            case "content_block_delta":
                guard let index = data["index"]?.intValue, let block = blocks[index], let delta = data["delta"] else { continue }
                switch (delta["type"]?.stringValue, block.kind) {
                case ("text_delta", .text):
                    builder.appendText(delta["text"]?.stringValue ?? "", to: block)
                case ("thinking_delta", .thinking):
                    builder.appendText(delta["thinking"]?.stringValue ?? "", to: block)
                case ("input_json_delta", .toolCall):
                    builder.appendToolArguments(delta["partial_json"]?.stringValue ?? "", to: block)
                case ("signature_delta", .thinking):
                    block.signature = (block.signature ?? "") + (delta["signature"]?.stringValue ?? "")
                default:
                    continue
                }

            case "content_block_stop":
                if let index = data["index"]?.intValue, let block = blocks[index] { builder.close(block) }

            case "message_delta":
                if let reason = data["delta"]?["stop_reason"]?.stringValue {
                    builder.message.rawStopReason = reason
                    switch reason {
                    case "end_turn", "pause_turn", "stop_sequence": builder.message.stopReason = .stop
                    case "max_tokens": builder.message.stopReason = .length
                    case "tool_use": builder.message.stopReason = .toolUse
                    case "refusal":
                        builder.message.stopReason = .error
                        builder.message.errorMessage = data["delta"]?["stop_details"]?["explanation"]?.stringValue
                            ?? "The model refused to complete the request"
                    default:
                        builder.message.stopReason = .error
                        builder.message.errorMessage = "Provider stopped with: \(reason)"
                    }
                }
                // Only present fields update: proxies may omit input_tokens here.
                if let usage = data["usage"], !usage.isNull {
                    if let value = usage["input_tokens"]?.intValue { builder.message.usage.input = value }
                    if let value = usage["output_tokens"]?.intValue { builder.message.usage.output = value }
                    if let value = usage["cache_read_input_tokens"]?.intValue { builder.message.usage.cacheRead = value }
                    if let value = usage["cache_creation_input_tokens"]?.intValue { builder.message.usage.cacheWrite = value }
                    if let value = usage["cache_creation"]?["ephemeral_1h_input_tokens"]?.intValue { builder.message.usage.cacheWrite1h = value }
                    if let value = usage["output_tokens_details"]?["thinking_tokens"]?.intValue { builder.message.usage.reasoning = value }
                }
                updateTotals(&builder.message.usage, model: model)

            default:
                sawStop = true
            }
        }
        if sawStart, !sawStop { throw LLMError.protocolViolation("Anthropic stream ended before message_stop") }
    }

    private static func updateTotals(_ usage: inout Usage, model: Model) {
        usage.totalTokens = usage.input + usage.output + usage.cacheRead + usage.cacheWrite
        usage.applyCost(of: model)
    }
}
