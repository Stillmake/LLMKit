import Foundation

/// OpenAI Chat Completions and the many servers that speak it: DeepSeek, OpenRouter, Groq,
/// Cerebras, xAI, Together, Fireworks, Moonshot, Qwen, Z.ai, MiniMax, NVIDIA, Hugging Face,
/// vLLM, llama.cpp, and more. Provider differences live in `ModelCompat`, auto-detected from the
/// provider and base URL and overridden by the catalog. Ports pi-ai's `openai-completions.ts`.
enum OpenAICompletions {
    struct Compat {
        var supportsStore: Bool
        var supportsDeveloperRole: Bool
        var supportsReasoningEffort: Bool
        var supportsUsageInStreaming: Bool
        var supportsFinishReason: Bool
        var maxTokensField: ModelCompat.MaxTokensField
        var requiresToolResultName: Bool
        var requiresAssistantAfterToolResult: Bool
        var requiresThinkingAsText: Bool
        var requiresReasoningContentOnAssistantMessages: Bool
        var thinkingFormat: ThinkingFormat
        var chatTemplateKwargs: JSONObject
        var chatTemplateArgs: JSONObject
        var openRouterRouting: JSONValue?
        var vercelGatewayRouting: JSONValue?
        var zaiToolStream: Bool
        var thinkingTokenBudgetField: ModelCompat.ThinkingTokenBudgetField?
        var supportsStrictMode: Bool
        var supportsMidConvoSystemMessages: Bool
        var usesAnthropicCacheControl: Bool
        var sendSessionAffinityHeaders: Bool
        var sessionAffinityFormat: ModelCompat.SessionAffinityFormat
        var supportsLongCacheRetention: Bool
        var vllmPriority: Double?

        /// Detects defaults from the provider and base URL, then applies the model's overrides.
        init(model: Model) {
            let provider = model.provider
            let url = model.baseURL
            let isZai = provider == "zai" || provider == "zai-coding-cn" || url.contains("api.z.ai") || url.contains("open.bigmodel.cn")
            let isTogether = provider == "together" || url.contains("api.together.ai") || url.contains("api.together.xyz")
            let isMoonshot = provider == "moonshotai" || provider == "moonshotai-cn" || url.contains("api.moonshot.")
            let isOpenRouter = provider == "openrouter" || url.contains("openrouter.ai")
            let isCloudflareWorkersAI = provider == "cloudflare-workers-ai" || url.contains("api.cloudflare.com")
            let isCloudflareGateway = provider == "cloudflare-ai-gateway" || url.contains("gateway.ai.cloudflare.com")
            let isNvidia = provider == "nvidia" || url.contains("integrate.api.nvidia.com")
            let isAntLing = provider == "ant-ling" || url.contains("api.ant-ling.com")
            let isCerebras = provider == "cerebras" || url.contains("cerebras.ai")
            let isDeepSeek = provider == "deepseek" || url.lowercased().contains("deepseek.com")
            let isGrok = provider == "xai" || url.contains("api.x.ai")
            let isChutes = url.contains("chutes.ai")
            let isOpenCode = provider == "opencode" || url.contains("opencode.ai")
            let isNonStandard = isNvidia || isCerebras || isGrok || isTogether || isChutes || isDeepSeek || isZai
                || isMoonshot || isOpenCode || isCloudflareWorkersAI || isCloudflareGateway || isAntLing
            let usesMaxTokens = isChutes || isDeepSeek || isMoonshot || isCloudflareGateway || isTogether || isNvidia || isAntLing || isZai
            let isOpenRouterDeveloperRoleModel = isOpenRouter && (model.id.hasPrefix("anthropic/") || model.id.hasPrefix("openai/"))
            let detectedFormat: ThinkingFormat =
                isDeepSeek ? .deepSeek : isZai ? .zai : isTogether ? .together : isAntLing ? .antLing : isOpenRouter ? .openRouter : .openAI

            let overrides = model.compat ?? ModelCompat()
            supportsStore = overrides.supportsStore ?? !isNonStandard
            supportsDeveloperRole = overrides.supportsDeveloperRole ?? (isOpenRouterDeveloperRoleModel || (!isNonStandard && !isOpenRouter))
            supportsReasoningEffort = overrides.supportsReasoningEffort
                ?? !(isGrok || isZai || isMoonshot || isTogether || isCloudflareGateway || isNvidia || isAntLing)
            supportsUsageInStreaming = overrides.supportsUsageInStreaming ?? true
            supportsFinishReason = overrides.supportsFinishReason ?? true
            maxTokensField = overrides.maxTokensField ?? (usesMaxTokens ? .maxTokens : .maxCompletionTokens)
            requiresToolResultName = overrides.requiresToolResultName ?? false
            requiresAssistantAfterToolResult = overrides.requiresAssistantAfterToolResult ?? false
            requiresThinkingAsText = overrides.requiresThinkingAsText ?? false
            requiresReasoningContentOnAssistantMessages = overrides.requiresReasoningContentOnAssistantMessages ?? isDeepSeek
            thinkingFormat = overrides.thinkingFormat ?? detectedFormat
            chatTemplateKwargs = overrides.chatTemplateKwargs ?? [:]
            chatTemplateArgs = overrides.chatTemplateArgs ?? [:]
            openRouterRouting = overrides.openRouterRouting
            vercelGatewayRouting = overrides.vercelGatewayRouting
            zaiToolStream = overrides.zaiToolStream ?? false
            thinkingTokenBudgetField = overrides.thinkingTokenBudgetField
                ?? (overrides.supportsThinkingTokenBudget == true ? .thinkingTokenBudget : nil)
            supportsStrictMode = overrides.supportsStrictMode ?? false
            supportsMidConvoSystemMessages = overrides.supportsMidConvoSystemMessages ?? false
            usesAnthropicCacheControl = (overrides.cacheControlFormat
                ?? (provider == "openrouter" && model.id.hasPrefix("anthropic/") ? "anthropic" : nil)) == "anthropic"
            sendSessionAffinityHeaders = overrides.sendSessionAffinityHeaders ?? isOpenRouter
            sessionAffinityFormat = overrides.sessionAffinityFormat ?? (isOpenRouter ? .openRouter : .openAI)
            supportsLongCacheRetention = overrides.supportsLongCacheRetention
                ?? !(isTogether || isCloudflareWorkersAI || isCloudflareGateway || isNvidia || isAntLing)
            vllmPriority = overrides.vllmPriority
        }
    }

    /// Streamed delta fields that may carry reasoning text, in priority order.
    static let reasoningFields = ["reasoning_content", "reasoning", "reasoning_text"]

    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let compat = Compat(model: model)
        let messages = Transcript.resolve(transcript, supportsMidConvoSystemMessages: compat.supportsMidConvoSystemMessages)
        if options.apiKey == nil, !options.hasAuthorizationHeader {
            throw LLMError.missingAPIKey(provider: model.provider)
        }

        var headers: [String: String?] = model.provider == "github-copilot" ? Transcript.copilotHeaders(messages) : [:]
        if let apiKey = options.apiKey { headers["Authorization"] = "Bearer \(apiKey)" }
        if let sessionId = options.raw.sessionId, options.raw.cacheRetention != .none, compat.sendSessionAffinityHeaders {
            if compat.sessionAffinityFormat == .openRouter {
                headers["x-session-id"] = sessionId
            } else {
                if compat.sessionAffinityFormat == .openAI { headers["session_id"] = sessionId }
                headers["x-client-request-id"] = sessionId
                headers["x-session-affinity"] = sessionId
            }
        }

        var body = try requestBody(model: model, messages: messages, options: options, compat: compat)
        let payload = options.finalize(&body, model: model, applySamplingParams: true)
        let url = try HTTP.url(model.baseURL, path: "/chat/completions")
        let bytes = try await HTTP.send(
            HTTP.request(url: url, body: payload, headers: headers, model: model, options: options),
            model: model,
            options: options
        )
        builder.start()

        var textBlock: ResponseBuilder.Block?
        var thinkingBlock: ResponseBuilder.Block?
        var toolBlocksByIndex: [Int: ResponseBuilder.Block] = [:]
        var toolBlocksById: [String: ResponseBuilder.Block] = [:]
        var reasoningDetails: [JSONObject]?
        var hasFinishReason = false

        for try await (_, chunk) in HTTP.jsonEvents(bytes) {
            if let error = chunk["error"], !error.isNull {
                throw LLMError.provider(HTTP.errorMessage(from: JSONValue.object(["error": error]).jsonString()))
            }
            if builder.message.responseId == nil { builder.message.responseId = chunk["id"]?.stringValue }
            if let responseModel = chunk["model"]?.stringValue, !responseModel.isEmpty, responseModel != model.id,
                builder.message.responseModel == nil
            {
                builder.message.responseModel = responseModel
            }
            if let usage = chunk["usage"], !usage.isNull {
                builder.message.usage = parseUsage(usage, model: model)
            }
            guard let choice = chunk["choices"]?[0] else { continue }
            // Moonshot reports usage on the choice instead of the chunk.
            if chunk["usage"]?.isNull ?? true, let usage = choice["usage"], !usage.isNull {
                builder.message.usage = parseUsage(usage, model: model)
            }
            if let finishReason = choice["finish_reason"]?.stringValue {
                builder.message.rawStopReason = finishReason
                switch finishReason {
                case "stop", "end": builder.message.stopReason = .stop
                case "length": builder.message.stopReason = .length
                case "function_call", "tool_calls": builder.message.stopReason = .toolUse
                default:
                    builder.message.stopReason = .error
                    builder.message.errorMessage = "Provider finish_reason: \(finishReason)"
                }
                hasFinishReason = true
            }
            guard let delta = choice["delta"]?.objectValue else { continue }

            // Reasoning precedes content: some proxies (LiteLLM) send both in one delta, and the
            // thinking block must come before the text block.
            // Servers name the reasoning field differently; some send the same text under two names.
            if let field = reasoningFields.first(where: { !(delta[$0]?.stringValue ?? "").isEmpty }),
                let reasoning = delta[field]?.stringValue
            {
                let signature = model.provider == "opencode-go" && field == "reasoning" ? "reasoning_content" : field
                let block = thinkingBlock ?? builder.openThinking(signature: signature)
                thinkingBlock = block
                builder.appendText(reasoning, to: block)
            }

            // OpenRouter streams structured reasoning replay data as deltas: consecutive text and
            // summary entries merge into one logical entry; encrypted entries stay discrete.
            for detail in delta["reasoning_details"]?.arrayValue ?? [] {
                guard case .object(let entry) = detail, isReasoningDetail(entry) else { continue }
                if thinkingBlock == nil { thinkingBlock = builder.openThinking(signature: "") }
                var details = reasoningDetails ?? []
                let mergeKey: String? = switch entry["type"]?.stringValue {
                case "reasoning.text": "text"
                case "reasoning.summary": "summary"
                default: nil
                }
                if let mergeKey, var last = details.last, last["type"] == entry["type"] {
                    last[mergeKey] = .string((last[mergeKey]?.stringValue ?? "") + (entry[mergeKey]?.stringValue ?? ""))
                    if mergeKey == "text", (last["signature"]?.stringValue ?? "").isEmpty, let signature = entry["signature"] {
                        last["signature"] = signature
                    }
                    for key in ["id", "index"] where last[key] == nil || last[key] == .null {
                        last[key] = entry[key]
                    }
                    if (last["format"]?.stringValue ?? "").isEmpty { last["format"] = entry["format"] }
                    details[details.count - 1] = last
                } else {
                    details.append(entry)
                }
                reasoningDetails = details
            }

            if let content = delta["content"]?.stringValue, !content.isEmpty {
                let block = textBlock ?? builder.openText()
                textBlock = block
                builder.appendText(content, to: block)
            }

            for toolDelta in delta["tool_calls"]?.arrayValue ?? [] {
                let streamIndex = toolDelta["index"]?.intValue
                let id = toolDelta["id"]?.stringValue
                let name = toolDelta["function"]?["name"]?.stringValue
                let block = streamIndex.flatMap { toolBlocksByIndex[$0] } ?? id.flatMap { toolBlocksById[$0] }
                    ?? builder.openToolCall(id: id ?? "", name: name ?? "")
                if let streamIndex { toolBlocksByIndex[streamIndex] = block }
                if let id, !id.isEmpty {
                    toolBlocksById[id] = block
                    if block.toolCallId.isEmpty { block.toolCallId = id }
                }
                if block.toolName.isEmpty, let name { block.toolName = name }
                if let arguments = toolDelta["function"]?["arguments"]?.stringValue {
                    builder.appendToolArguments(arguments, to: block)
                }
            }
        }

        if let reasoningDetails, let thinkingBlock {
            thinkingBlock.signature = JSONValue.array(reasoningDetails.map(JSONValue.object)).jsonString()
        }
        if !hasFinishReason, !compat.supportsFinishReason {
            builder.message.stopReason = builder.hasToolCalls ? .toolUse : .stop
        } else if !hasFinishReason {
            throw LLMError.protocolViolation("Stream ended without finish_reason")
        }
    }

    private static func isReasoningDetail(_ detail: JSONObject) -> Bool {
        let commonFieldsValid = (detail["id"].map { $0 == .null || $0.stringValue != nil } ?? true)
            && (detail["format"].map { $0.stringValue != nil } ?? true)
            && (detail["index"].map { $0.doubleValue != nil } ?? true)
        guard commonFieldsValid else { return false }
        switch detail["type"]?.stringValue {
        case "reasoning.summary": return detail["summary"]?.stringValue != nil
        case "reasoning.encrypted": return detail["data"]?.stringValue != nil
        case "reasoning.text":
            return detail["text"]?.stringValue != nil && (detail["signature"].map { $0 == .null || $0.stringValue != nil } ?? true)
        default: return false
        }
    }

    private static func parseUsage(_ raw: JSONValue, model: Model) -> Usage {
        // cached_tokens counts cache hits: OpenAI/OpenRouter put it under prompt_tokens_details,
        // DeepSeek reports prompt_cache_hit_tokens, Kimi a top-level cached_tokens. Writes are separate.
        let promptTokens = raw["prompt_tokens"]?.intValue ?? 0
        let cacheRead = raw["prompt_tokens_details"]?["cached_tokens"]?.intValue
            ?? raw["prompt_cache_hit_tokens"]?.intValue ?? raw["cached_tokens"]?.intValue ?? 0
        let cacheWrite = raw["prompt_tokens_details"]?["cache_write_tokens"]?.intValue ?? 0
        var usage = Usage()
        usage.input = max(0, promptTokens - cacheRead - cacheWrite)
        usage.output = raw["completion_tokens"]?.intValue ?? 0
        usage.cacheRead = cacheRead
        usage.cacheWrite = cacheWrite
        usage.reasoning = raw["completion_tokens_details"]?["reasoning_tokens"]?.intValue ?? 0
        usage.totalTokens = usage.input + usage.output + cacheRead + cacheWrite
        usage.applyCost(of: model)
        return usage
    }

    // MARK: Request

    static func requestBody(model: Model, messages: [Message], options: ResolvedOptions, compat: Compat) throws -> JSONObject {
        var wireMessages = convertMessages(model: model, messages: messages, compat: compat)
        let retention = options.raw.cacheRetention
        let effort = options.reasoning
        var body: JSONObject = ["model": .string(model.id), "stream": true]

        if let sessionId = options.raw.sessionId,
            (model.baseURL.contains("api.openai.com") && retention != .none) || (retention == .long && compat.supportsLongCacheRetention)
        {
            body["prompt_cache_key"] = .string(String(sessionId.prefix(64)))
        }
        if retention == .long, compat.supportsLongCacheRetention { body["prompt_cache_retention"] = "24h" }
        if compat.supportsUsageInStreaming { body["stream_options"] = ["include_usage": true] }
        if compat.supportsStore { body["store"] = false }
        body[compat.maxTokensField.rawValue] = .number(Double(options.maxTokens))
        if let temperature = options.raw.temperature { body["temperature"] = .number(temperature) }

        let tools = Transcript.currentTools(messages)
        var wireTools: [JSONValue]?
        if !tools.isEmpty {
            wireTools = try tools.map { tool in
                let strict = try StrictSchema.resolve(tool, supportsStrictMode: compat.supportsStrictMode)
                var function: JSONObject = [
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": StrictSchema.parameters(of: tool, strict: strict),
                ]
                // Only send `strict` where supported: some servers reject unknown fields.
                if compat.supportsStrictMode { function["strict"] = .bool(strict ?? false) }
                return ["type": "function", "function": .object(function)]
            }
            if compat.zaiToolStream { body["tool_stream"] = true }
        } else if messages.contains(where: { message in
            switch message {
            case .toolResult: true
            case .assistant(let assistant): !assistant.toolCalls.isEmpty
            default: false
            }
        }) {
            // Anthropic behind LiteLLM-style proxies requires `tools` whenever the history has tool calls.
            wireTools = []
        }

        if compat.usesAnthropicCacheControl, retention != .none {
            var cacheControl: JSONObject = ["type": "ephemeral"]
            if retention == .long, compat.supportsLongCacheRetention { cacheControl["ttl"] = "1h" }
            if let system = wireMessages.firstIndex(where: { ["system", "developer"].contains($0["role"]?.stringValue) }) {
                _ = addCacheControl(.object(cacheControl), to: &wireMessages[system])
            }
            if var lastTool = wireTools?.last?.objectValue {
                lastTool["cache_control"] = .object(cacheControl)
                wireTools![wireTools!.count - 1] = .object(lastTool)
            }
            for index in wireMessages.indices.reversed()
            where ["user", "assistant", "tool"].contains(wireMessages[index]["role"]?.stringValue) {
                if addCacheControl(.object(cacheControl), to: &wireMessages[index]) { break }
            }
        }
        body["messages"] = .array(wireMessages)
        if let wireTools { body["tools"] = .array(wireTools) }
        if let toolChoice = options.raw.toolChoice { body["tool_choice"] = .string(toolChoice.rawValue) }
        if let priority = compat.vllmPriority { body["priority"] = .number(priority) }

        // Budget capped so at least `minimumAnswerTokens` remain under max_tokens.
        let budget: Int? = effort.flatMap { level in
            guard model.reasoning else { return nil }
            let value = min((options.raw.thinkingBudgets ?? ThinkingBudgets())[level], max(0, options.maxTokens - minimumAnswerTokens))
            return value > 0 ? value : nil
        }
        if model.reasoning {
            // A level mapped as unsupported falls back to its own name, as pi-ai's `map[level] ?? level` does.
            let effortValue = effort.map { JSONValue.string(model.thinkingValue(for: $0) ?? $0.rawValue) }
            let offMapping = model.thinkingLevelMap?[.off]
            let offValue: String? = if case .value(let value)? = offMapping { value } else { nil }

            switch compat.thinkingFormat {
            case .zai:
                body["thinking"] = effort == nil ? ["type": "disabled"] : ["type": "enabled", "clear_thinking": false]
                if let effort, compat.supportsReasoningEffort, let value = model.thinkingValue(for: effort) {
                    body["reasoning_effort"] = .string(value)
                }
            case .qwen:
                body["enable_thinking"] = .bool(effort != nil)
                if let effortValue, compat.supportsReasoningEffort { body["reasoning_effort"] = effortValue }
            case .qwenChatTemplate:
                body["chat_template_kwargs"] = ["enable_thinking": .bool(effort != nil), "preserve_thinking": true]
            case .chatTemplate:
                let values = chatTemplateValues(compat.chatTemplateKwargs, model: model, effort: effort, budget: budget)
                if !values.isEmpty { body["chat_template_kwargs"] = .object(values) }
            case .baseten:
                let values = chatTemplateValues(compat.chatTemplateArgs, model: model, effort: effort, budget: budget)
                if !values.isEmpty { body["chat_template_args"] = .object(values) }
                if compat.supportsReasoningEffort {
                    switch effort.map({ model.thinkingLevelMap?[$0] }) ?? offMapping {
                    case .value(let value)?: body["reasoning_effort"] = .string(value)
                    case nil: if let effort { body["reasoning_effort"] = .string(effort.rawValue) }
                    case .unsupported?: break
                    }
                }
            case .deepSeek:
                if effort != nil {
                    body["thinking"] = ["type": "enabled"]
                } else if offMapping != .unsupported {
                    body["thinking"] = ["type": "disabled"]
                }
                if let effortValue, compat.supportsReasoningEffort { body["reasoning_effort"] = effortValue }
            case .openRouter:
                if let effortValue {
                    body["reasoning"] = ["effort": effortValue]
                } else if offMapping != .unsupported {
                    body["reasoning"] = ["effort": .string(offValue ?? "none")]
                }
            case .together:
                body["reasoning"] = ["enabled": .bool(effort != nil)]
                if let effortValue, compat.supportsReasoningEffort { body["reasoning_effort"] = effortValue }
            case .stringThinking:
                if let effortValue {
                    body["thinking"] = effortValue
                } else if offMapping != .unsupported {
                    body["thinking"] = .string(offValue ?? "none")
                }
            case .antLing where effort != nil:
                if let effort, case .value(let value)? = model.thinkingLevelMap?[effort] { body["reasoning"] = ["effort": .string(value)] }
            default:
                if compat.supportsReasoningEffort, let value = effortValue ?? offValue.map(JSONValue.string) {
                    body["reasoning_effort"] = value
                }
            }
        }
        // A top-level budget caps reasoning independently of the format: reasoning and answer share
        // max_tokens here, so uncapped reasoning can consume the whole response.
        if let field = compat.thinkingTokenBudgetField, let budget { body[field.rawValue] = .number(Double(budget)) }

        if let routing = compat.openRouterRouting { body["provider"] = routing }
        if let routing = compat.vercelGatewayRouting, routing["only"] != nil || routing["order"] != nil {
            var gateway: JSONObject = [:]
            if let only = routing["only"] { gateway["only"] = only }
            if let order = routing["order"] { gateway["order"] = order }
            body["providerOptions"] = ["gateway": .object(gateway)]
        }
        return body
    }

    /// Resolves `{"$var": ...}` placeholders in chat-template values.
    private static func chatTemplateValues(_ values: JSONObject, model: Model, effort: ThinkingLevel?, budget: Int?) -> JSONObject {
        values.compactMapValues { value in
            guard case .object(let placeholder) = value else { return value }
            if effort == nil, placeholder["omitWhenOff"] == true { return nil }
            switch placeholder["$var"]?.stringValue {
            case "thinking.enabled":
                return .bool(effort != nil)
            case "thinking.budget":
                return budget.map { .number(Double($0)) }
            default:
                switch effort.map({ model.thinkingLevelMap?[$0] }) ?? model.thinkingLevelMap?[.off] {
                case .value(let mapped)?: return .string(mapped)
                case nil: return effort.map { .string($0.rawValue) }
                case .unsupported?: return nil
                }
            }
        }
    }

    /// Marks the last text part of a message with `cache_control`, converting string content to parts.
    private static func addCacheControl(_ cacheControl: JSONValue, to message: inout JSONValue) -> Bool {
        guard case .object(var object) = message else { return false }
        switch object["content"] {
        case .string(let text)? where !text.isEmpty:
            object["content"] = [["type": "text", "text": .string(text), "cache_control": cacheControl]]
        case .array(var parts)?:
            guard let index = parts.lastIndex(where: { $0["type"] == "text" }), case .object(var part) = parts[index] else {
                return false
            }
            part["cache_control"] = cacheControl
            parts[index] = .object(part)
            object["content"] = .array(parts)
        default:
            return false
        }
        message = .object(object)
        return true
    }

    static func convertMessages(model: Model, messages: [Message], compat: Compat) -> [JSONValue] {
        let transformed = Transcript.transform(messages, for: model) { id, _ in
            // Responses-style ids are `{call_id}|{item_id}` with 400+ characters; Chat Completions
            // needs distinct ids of at most 40 characters.
            guard let separator = id.firstIndex(of: "|") else {
                return model.provider == "openai" ? String(id.prefix(40)) : id
            }
            let callId = Transcript.sanitizeIdentifier(String(id[..<separator]), maxLength: .max)
            let itemId = Transcript.sanitizeIdentifier(String(id[id.index(after: separator)...]), maxLength: .max)
            let combined = itemId.isEmpty ? callId : "\(callId)_\(itemId)"
            guard combined.count > 40 else { return combined }
            let hash = Transcript.shortHash(id).prefix(8)
            return "\(callId.prefix(max(1, 40 - hash.count - 1)))_\(hash)"
        }
        let instructionRole: JSONValue = model.reasoning && compat.supportsDeveloperRole ? "developer" : "system"
        let bridge: JSONValue = ["role": "assistant", "content": "I have processed the tool results."]
        var wire: [JSONValue] = []
        var lastWasToolResult = false
        var index = 0

        while index < transformed.count {
            let message = transformed[index]
            if compat.requiresAssistantAfterToolResult, lastWasToolResult, case .user = message {
                wire.append(bridge)
            }
            lastWasToolResult = false

            switch message {
            case .system(let system):
                let text = index == 0 ? system.promptText : system.updateText
                if !text.isEmpty { wire.append(["role": instructionRole, "content": .string(text)]) }

            case .user(let user):
                let parts: [JSONValue] = user.content.compactMap { block -> JSONValue? in
                    switch block {
                    case .text(let text):
                        text.text.isEmpty ? nil : ["type": "text", "text": .string(text.text)]
                    case .image(let image):
                        ["type": "image_url", "image_url": ["url": .string("data:\(image.mimeType);base64,\(image.data)")]]
                    }
                }
                if parts.count == 1, let text = parts[0]["text"] {
                    wire.append(["role": "user", "content": text])
                } else if !parts.isEmpty {
                    wire.append(["role": "user", "content": .array(parts)])
                }

            case .assistant(let assistant):
                var thinkingBlocks: [ThinkingContent] = []
                var textParts: [String] = []
                var toolCalls: [ToolCall] = []
                for block in assistant.content {
                    switch block {
                    case .text(let text) where !text.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
                        textParts.append(text.text)
                    case .thinking(let thinking): thinkingBlocks.append(thinking)
                    case .toolCall(let call): toolCalls.append(call)
                    default: break
                    }
                }
                let text = textParts.joined()
                var converted: JSONObject = ["role": "assistant", "content": compat.requiresAssistantAfterToolResult ? "" : .null]

                // `reasoning_details` (OpenRouter) is the structured alternative to a raw reasoning field.
                let reasoningDetails = thinkingBlocks.lazy.compactMap { block -> [JSONValue]? in
                    guard let signature = block.thinkingSignature, signature.hasPrefix("["),
                        case .array(let details)? = try? JSONValue(parsing: signature),
                        !details.isEmpty, details.allSatisfy({ $0.objectValue.map(isReasoningDetail) ?? false })
                    else { return nil }
                    return details
                }.first

                let nonEmptyThinking = thinkingBlocks.filter { !$0.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                if !nonEmptyThinking.isEmpty, compat.requiresThinkingAsText {
                    // Plain text without tags, so the model does not learn to mimic them.
                    let thinkingText = nonEmptyThinking.map(\.thinking).joined(separator: "\n\n")
                    converted["content"] = .array(([thinkingText] + textParts).map { ["type": "text", "text": .string($0)] })
                } else {
                    // Always a plain string: array content makes some models (DeepSeek via NIM) echo the structure.
                    if !text.isEmpty { converted["content"] = .string(text) }
                    if !nonEmptyThinking.isEmpty, reasoningDetails == nil, var field = nonEmptyThinking[0].thinkingSignature {
                        if model.provider == "opencode-go", field == "reasoning" { field = "reasoning_content" }
                        if reasoningFields.contains(field) {
                            converted[field] = .string(nonEmptyThinking.map(\.thinking).joined(separator: "\n"))
                        }
                    }
                }
                if !toolCalls.isEmpty {
                    converted["tool_calls"] = .array(toolCalls.map { call in
                        [
                            "id": .string(call.id),
                            "type": "function",
                            "function": ["name": .string(call.name), "arguments": .string(JSONValue.object(call.arguments).jsonString())],
                        ]
                    })
                }
                if let reasoningDetails { converted["reasoning_details"] = .array(reasoningDetails) }
                if compat.requiresReasoningContentOnAssistantMessages, model.reasoning, converted["reasoning_content"] == nil {
                    converted["reasoning_content"] = ""
                }
                // Providers require content or tool calls; this also drops empty aborted responses.
                let hasContent = switch converted["content"] {
                case .string(let value)?: !value.isEmpty
                case .array(let value)?: !value.isEmpty
                default: false
                }
                if hasContent || !toolCalls.isEmpty { wire.append(.object(converted)) }

            case .toolResult:
                var images: [JSONValue] = []
                while index < transformed.count, case .toolResult(let result) = transformed[index] {
                    let text = result.text
                    let hasImages = !result.images.isEmpty
                    var toolMessage: JSONObject = [
                        "role": "tool",
                        "content": .string((text.allSatisfy(\.isWhitespace) ? (hasImages ? "(see attached image)" : "(no tool output)") : text)),
                        "tool_call_id": .string(result.toolCallId),
                    ]
                    if compat.requiresToolResultName, !result.toolName.isEmpty { toolMessage["name"] = .string(result.toolName) }
                    wire.append(.object(toolMessage))
                    if hasImages, model.supportsImages {
                        images += result.images.map { ["type": "image_url", "image_url": ["url": .string("data:\($0.mimeType);base64,\($0.data)")]] }
                    }
                    index += 1
                }
                if images.isEmpty {
                    lastWasToolResult = true
                } else {
                    // Tool messages cannot carry images; attach them in a following user message.
                    if compat.requiresAssistantAfterToolResult { wire.append(bridge) }
                    wire.append(["role": "user", "content": .array([["type": "text", "text": "Attached image(s) from tool result:"]] + images)])
                }
                continue
            }
            index += 1
        }
        return wire
    }

}
