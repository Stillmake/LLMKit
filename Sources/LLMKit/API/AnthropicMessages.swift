import Foundation

/// Anthropic Messages API and compatible endpoints (Kimi Coding, MiniMax, Z.ai, OpenRouter,
/// Fireworks, GitHub Copilot, ...). Supports budget-based and adaptive thinking, signed and
/// redacted thinking replay, prompt caching, and eager tool-input streaming.
/// Ports pi-ai's `anthropic-messages.ts`.
enum AnthropicMessages {
    static let claudeCodeVersion = "2.1.322"
    static let messageThreadsBeta = "message-threads-2026-08-12"

    static let claudeCodeTools = Dictionary(uniqueKeysWithValues: [
        "Read", "Write", "Edit", "Bash", "Grep", "Glob", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "KillShell",
        "NotebookEdit", "Skill", "Task", "TaskOutput", "TodoWrite", "WebFetch", "WebSearch",
    ].map { ($0.lowercased(), $0) })

    struct ThreadEntry {
        /// Number of messages in the last full request as sent plus the converted reply.
        var prefixCount: Int
        /// Hash of the request's fixed settings and those messages.
        var prefixHash: String
        /// Id of the reply to continue from.
        var replyId: String
    }

    enum ThreadState {
        case entry(ThreadEntry)
        case off
    }

    struct PreparedThreadRequest {
        var sessionId: String
        /// The full request the continuation was derived from.
        var params: JSONObject
        /// The body to send: either a new thread or only the messages after the prefix.
        var body: JSONObject
        var isContinuation: Bool
    }

    private static let threadState = ThreadStateStore()

    private final class ThreadStateStore: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [String: ThreadState] = [:]
        private var order: [String] = []
        private let maximumEntries = 10_000

        func get(_ sessionId: String) -> ThreadState? {
            lock.lock()
            defer { lock.unlock() }
            return entries[sessionId]
        }

        func set(_ sessionId: String, _ state: ThreadState) {
            lock.lock()
            defer { lock.unlock() }
            entries[sessionId] = state
            order.removeAll { $0 == sessionId }
            order.append(sessionId)
            while order.count > maximumEntries {
                entries.removeValue(forKey: order.removeFirst())
            }
        }

        func remove(_ sessionId: String) {
            lock.lock()
            defer { lock.unlock() }
            entries.removeValue(forKey: sessionId)
            order.removeAll { $0 == sessionId }
        }

        func removeAll() {
            lock.lock()
            defer { lock.unlock() }
            entries.removeAll()
            order.removeAll()
        }
    }

    private static func claudeCodeName(_ name: String, isOAuth: Bool) -> String {
        isOAuth ? claudeCodeTools[name.lowercased()] ?? name : name
    }

    private static func normalizeThreadValue(_ value: JSONValue, key: String? = nil) -> JSONValue {
        if key == "content", case .string(let text) = value {
            return .array([["type": "text", "text": .string(text)]])
        }
        switch value {
        case .array(let values):
            return .array(values.map { normalizeThreadValue($0) })
        case .object(let object):
            var normalized: JSONObject = [:]
            for (name, value) in object where name != "cache_control" {
                normalized[name] = normalizeThreadValue(value, key: name)
            }
            return .object(normalized)
        default:
            return value
        }
    }

    /// Hash of everything a continuation builds on: the fixed request settings and the messages
    /// the server already holds. Cache markers are ignored because they move between requests.
    static func hashThreadPrefix(model: Model, body: JSONObject, messages: [JSONValue], betas: [String]) -> String {
        let settings = JSONValue.array([
            .string(model.provider),
            .string(model.baseURL),
            body["model"] ?? .null,
            body["system"] ?? .null,
            body["tools"] ?? .null,
            body["thinking"] ?? .null,
            body["output_config"] ?? .null,
            body["tool_choice"] ?? .null,
            .array(betas.map(JSONValue.string)),
            body["fallbacks"] ?? .null,
        ])
        let conversation = JSONValue.array(messages.map { normalizeThreadValue($0) })
        return Transcript.shortHash(settings.jsonString() + conversation.jsonString())
    }

    /// Prepares a request to create or continue a thread. A continuation sends only messages after
    /// the recorded prefix and omits `system` and `tools`, which the server keeps for the thread.
    static func prepareThreadRequest(
        model: Model,
        sessionId: String,
        body: JSONObject,
        betas: [String]
    ) -> PreparedThreadRequest? {
        guard let messages = body["messages"]?.arrayValue else { return nil }
        if case .off? = threadState.get(sessionId) { return nil }
        if case .entry(let entry)? = threadState.get(sessionId),
            messages.count > entry.prefixCount,
            hashThreadPrefix(
                model: model,
                body: body,
                messages: Array(messages.prefix(entry.prefixCount)),
                betas: betas
            ) == entry.prefixHash
        {
            // Each reply can be continued once; removing it also keeps a concurrent request from trying.
            threadState.remove(sessionId)
            var continuation = body
            continuation["system"] = nil
            continuation["tools"] = nil
            continuation["messages"] = .array(Array(messages.dropFirst(entry.prefixCount)))
            continuation["thread"] = ["type": "continue", "previous_message_id": .string(entry.replyId)]
            return PreparedThreadRequest(
                sessionId: sessionId,
                params: body,
                body: continuation,
                isContinuation: true
            )
        }
        var created = body
        created["thread"] = ["type": "create"]
        return PreparedThreadRequest(sessionId: sessionId, params: body, body: created, isContinuation: false)
    }

    /// Records where the next request can continue: after `output`, the clean reply to the thread
    /// request, whose full request was `body`.
    static func recordThreadReply(
        model: Model,
        thread: PreparedThreadRequest,
        body: JSONObject,
        output: AssistantMessage,
        isOAuth: Bool,
        allowEmptySignature: Bool,
        betas: [String]
    ) {
        // A fallback model's reply is not stored by the server, so it cannot be continued.
        guard let replyId = output.responseId, output.responseModel == nil else { return }
        let reply = convertMessages(
            [.assistant(output)],
            model: model,
            isOAuth: isOAuth,
            allowEmptySignature: allowEmptySignature
        ).filter { $0["role"]?.stringValue == "assistant" }
        let prefix = (body["messages"]?.arrayValue ?? []) + reply
        threadState.set(thread.sessionId, .entry(ThreadEntry(
            prefixCount: prefix.count,
            prefixHash: hashThreadPrefix(model: model, body: body, messages: prefix, betas: betas),
            replyId: replyId
        )))
    }

    /// Sends a thread request. A rejection about the thread is resent once: as a new thread when
    /// the old one cannot be continued, or without `thread` when the endpoint does not support it.
    private static func sendThreadRequest(
        _ prepared: PreparedThreadRequest,
        url: URL,
        headers: [String: String?],
        model: Model,
        options: ResolvedOptions
    ) async throws -> (bytes: URLSession.AsyncBytes, thread: PreparedThreadRequest?) {
        func send(_ body: JSONObject) async throws -> URLSession.AsyncBytes {
            try await HTTP.send(
                HTTP.request(url: url, body: .object(body), headers: headers, model: model, options: options),
                model: model,
                options: options
            )
        }

        do {
            return (try await send(prepared.body), prepared)
        } catch let error as LLMError {
            guard case .http(let status, let message, _) = error,
                [400, 404, 409].contains(status)
            else { throw error }
            let lowercased = message.lowercased()
            let threadRelated = lowercased.contains("thread")
                || lowercased.contains("previous message")
                || (prepared.isContinuation && (status == 404 || status == 409))
            guard threadRelated else { throw error }

            if prepared.isContinuation {
                var created = prepared.params
                created["thread"] = ["type": "create"]
                let request = PreparedThreadRequest(
                    sessionId: prepared.sessionId,
                    params: prepared.params,
                    body: created,
                    isContinuation: false
                )
                return (try await send(created), request)
            }

            var body = prepared.params
            body["thread"] = nil
            let bytes = try await send(body)
            threadState.set(prepared.sessionId, .off)
            return (bytes, nil)
        }
    }

    /// Frees thread state for one session, or all sessions when `sessionId` is nil.
    static func clearSessionResources(_ sessionId: String?) {
        if let sessionId {
            threadState.remove(sessionId)
        } else {
            threadState.removeAll()
        }
    }

    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let overrides = model.compat ?? ModelCompat()
        let isOpenRouter = model.provider == "openrouter" || model.baseURL.contains("openrouter.ai")
        let supportsMidConvoSystemMessages = overrides.supportsMidConvoSystemMessages ?? false
        let supportsEagerToolInputStreaming = overrides.supportsEagerToolInputStreaming ?? true
        let supportsLongCacheRetention = overrides.supportsLongCacheRetention ?? true
        let supportsMessageThreads = overrides.supportsMessageThreads
            ?? model.baseURL.contains("api.anthropic.com")
        let messages = Transcript.resolve(transcript, supportsMidConvoSystemMessages: supportsMidConvoSystemMessages)
        let tools = Transcript.currentTools(messages)
        if options.apiKey == nil, !options.hasAuthorizationHeader {
            throw LLMError.missingAPIKey(provider: model.provider)
        }
        // A Claude Pro/Max OAuth token must present the Claude Code identity: its headers, betas,
        // system prompt, and Claude Code's canonical tool-name casing.
        let isOAuth = model.provider != "github-copilot" && options.apiKey?.contains("sk-ant-oat") == true
        let wireToolName = { (name: String) in claudeCodeName(name, isOAuth: isOAuth) }

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
                    maxTokens = ResolvedOptions.clampMaxTokensToContext(
                        contextWindow: model.contextWindow,
                        estimatedInput: Transcript.estimatedTokens(transcript),
                        maxTokens: maxTokens
                    )
                }
                budget = min(budget, max(0, maxTokens - minimumAnswerTokens))
                thinking = ["type": "enabled", "budget_tokens": .number(Double(budget > 0 ? budget : 1024)), "display": display]
            }
        } else if model.reasoning, !model.thinkingLevelIsUnsupported(.off) {
            thinking = ["type": "disabled"]
        }

        let threadSessionId = supportsMessageThreads && retention != .none ? options.raw.sessionId : nil
        var configuredBetas: String?? = nil
        for (name, value) in model.headers ?? [:] where name.lowercased() == "anthropic-beta" {
            configuredBetas = value
        }
        for (name, value) in options.raw.headers where name.lowercased() == "anthropic-beta" {
            configuredBetas = value
        }
        let betas: [String]
        if let configuredBetas {
            betas = configuredBetas?.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty } ?? []
        } else {
            var defaults: [String] = isOAuth ? ["claude-code-20250219", "oauth-2025-04-20"] : []
            if !tools.isEmpty, !supportsEagerToolInputStreaming { defaults.append("fine-grained-tool-streaming-2025-05-14") }
            if thinking?["type"] == "enabled" { defaults.append("interleaved-thinking-2025-05-14") }
            if threadSessionId != nil { defaults.append(messageThreadsBeta) }
            var seen: Set<String> = []
            betas = defaults.filter { seen.insert($0).inserted }
        }

        var headers: [String: String?] = model.provider == "github-copilot" ? Transcript.copilotHeaders(messages) : [:]
        headers["anthropic-version"] = "2023-06-01"
        if !betas.isEmpty { headers["anthropic-beta"] = betas.joined(separator: ",") }
        if let apiKey = options.apiKey {
            if model.provider == "github-copilot" {
                headers["Authorization"] = "Bearer \(apiKey)"
            } else if isOAuth {
                headers["Authorization"] = "Bearer \(apiKey)"
                headers["accept"] = "application/json"
                headers["anthropic-dangerous-direct-browser-access"] = "true"
                headers["user-agent"] = "claude-cli/\(claudeCodeVersion)"
                headers["x-app"] = "cli"
            } else {
                headers["x-api-key"] = apiKey
            }
        }
        if let sessionId = options.raw.sessionId, retention != .none, overrides.sendSessionAffinityHeaders ?? isOpenRouter {
            let format = overrides.sessionAffinityFormat ?? (isOpenRouter ? .openRouter : nil)
            headers[format == .openRouter ? "x-session-id" : "x-session-affinity"] = sessionId
        }

        let allowEmptySignature = overrides.allowEmptySignature ?? false
        var wire = convertMessages(
            messages,
            model: model,
            isOAuth: isOAuth,
            allowEmptySignature: allowEmptySignature
        )

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

        let systemTexts = (isOAuth ? ["You are Claude Code, Anthropic's official CLI for Claude."] : [])
            + [Transcript.initialSystemMessage(messages)?.promptText].compactMap { $0.flatMap { $0.isEmpty ? nil : $0 } }
        let url = try HTTP.url(model.baseURL, path: "/v1/messages")

        func buildBody(disablePreferredStrictTools: Bool) throws -> (body: JSONObject, hasPreferredStrictTools: Bool) {
            var body: JSONObject = [
                "model": .string(model.id),
                "messages": .array(wire),
                "max_tokens": .number(Double(maxTokens)),
                "stream": true,
            ]
            if !systemTexts.isEmpty {
                body["system"] = .array(systemTexts.map { text in
                    var block: JSONObject = ["type": "text", "text": .string(text)]
                    block["cache_control"] = cacheControl
                    return .object(block)
                })
            }
            // Temperature is incompatible with extended thinking and rejected by newer models.
            if let temperature = options.raw.temperature, options.reasoning == nil, overrides.supportsTemperature ?? true {
                body["temperature"] = .number(temperature)
            }
            var hasPreferredStrictTools = false
            if !tools.isEmpty {
                let supportsStrictTools = overrides.supportsStrictTools ?? false
                let toolCacheControl = overrides.supportsCacheControlOnTools ?? true ? cacheControl : nil
                body["tools"] = .array(try tools.enumerated().map { index, tool in
                    let strict: Bool?
                    if disablePreferredStrictTools, tool.strict == .prefer {
                        strict = nil
                    } else {
                        strict = try StrictSchema.resolve(
                            tool,
                            supportsStrictMode: supportsStrictTools,
                            keywordCheck: StrictSchema.anthropicKeywordCheck
                        )
                    }
                    if strict == true, tool.strict == .prefer { hasPreferredStrictTools = true }
                    let parameters = StrictSchema.parameters(of: tool, strict: strict)
                    var inputSchema: JSONObject = strict == true ? parameters.objectValue ?? [:] : [:]
                    inputSchema["type"] = "object"
                    inputSchema["properties"] = parameters["properties"] ?? [:]
                    inputSchema["required"] = parameters["required"] ?? []
                    // Keep shared definitions so property $refs still resolve, and additionalProperties so
                    // the model knows whether extra arguments are allowed. Other root keywords stay out:
                    // Anthropic rejects root anyOf/oneOf/allOf with a 400.
                    if let additionalProperties = parameters["additionalProperties"] {
                        inputSchema["additionalProperties"] = additionalProperties
                    }
                    if let definitions = parameters["$defs"] { inputSchema["$defs"] = definitions }
                    if let definitions = parameters["definitions"] { inputSchema["definitions"] = definitions }
                    var definition: JSONObject = [
                        "name": .string(wireToolName(tool.name)),
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
            return (body, hasPreferredStrictTools)
        }

        var disablePreferredStrictTools = StrictToolLimits.hasExceeded(options.raw.sessionId)
        var received: URLSession.AsyncBytes?
        var params: JSONObject = [:]
        var thread: PreparedThreadRequest?
        while true {
            var built = try buildBody(disablePreferredStrictTools: disablePreferredStrictTools)
            let payload = options.finalize(&built.body, model: model, applySamplingParams: false)
            guard let paramsValue = payload.objectValue else {
                throw LLMError.invalidRequest("Anthropic request payload must be a JSON object")
            }
            params = paramsValue
            let prepared = threadSessionId.flatMap { sessionId in
                options.raw.disableContinuation
                    ? nil
                    : prepareThreadRequest(model: model, sessionId: sessionId, body: paramsValue, betas: betas)
            }
            do {
                if let prepared {
                    let sent = try await sendThreadRequest(
                        prepared,
                        url: url,
                        headers: headers,
                        model: model,
                        options: options
                    )
                    received = sent.bytes
                    thread = sent.thread
                } else {
                    received = try await HTTP.send(
                        HTTP.request(url: url, body: payload, headers: headers, model: model, options: options),
                        model: model,
                        options: options
                    )
                    thread = nil
                }
                break
            } catch let error as LLMError {
                if case .http(_, let message, _) = error,
                    !disablePreferredStrictTools,
                    built.hasPreferredStrictTools,
                    StrictToolLimits.isLimitError(message)
                {
                    disablePreferredStrictTools = true
                    StrictToolLimits.markExceeded(options.raw.sessionId)
                    continue
                }
                throw error
            }
        }
        guard let bytes = received else {
            throw LLMError.protocolViolation("Anthropic request completed without a response")
        }
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
                    // Map Claude Code casing back to the caller's tool name.
                    let name = block["name"]?.stringValue ?? ""
                    blocks[index] = builder.openToolCall(
                        id: block["id"]?.stringValue ?? "",
                        name: isOAuth ? tools.first { $0.name.lowercased() == name.lowercased() }?.name ?? name : name,
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
                if isOpenRouter, let usage = data["usage"] {
                    OpenRouterCost.apply(usage, to: &builder.message.usage)
                }

            default:
                sawStop = true
            }
        }
        if sawStart, !sawStop { throw LLMError.protocolViolation("Anthropic stream ended before message_stop") }
        if builder.message.stopReason == .aborted || builder.message.stopReason == .error {
            throw LLMError.provider(builder.message.errorMessage ?? "An unknown error occurred")
        }
        if let thread {
            var output = builder.message
            output.content = builder.blocks.map(\.content)
            recordThreadReply(
                model: model,
                thread: thread,
                body: params,
                output: output,
                isOAuth: isOAuth,
                allowEmptySignature: allowEmptySignature,
                betas: betas
            )
        }
    }

    /// Converts stored messages to Anthropic wire messages. Shared with thread recording so the
    /// prefix hash sees exactly the assistant reply the next request will send.
    static func convertMessages(
        _ messages: [Message],
        model: Model,
        isOAuth: Bool,
        allowEmptySignature: Bool
    ) -> [JSONValue] {
        let transformed = Transcript.transform(messages, for: model) { id, _ in Transcript.sanitizeIdentifier(id) }
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
                        return [
                            "type": "tool_use",
                            "id": .string(call.id),
                            "name": .string(claudeCodeName(call.name, isOAuth: isOAuth)),
                            "input": .object(call.arguments),
                        ]
                    }
                }
                if !blocks.isEmpty {
                    wire += pendingSystem
                    pendingSystem = []
                    wire.append(["role": "assistant", "content": .array(blocks)])
                }

            case .toolResult:
                func imageBlock(_ image: ImageContent) -> JSONValue {
                    ["type": "image", "source": ["type": "base64", "media_type": .string(image.mimeType), "data": .string(image.data)]]
                }
                var results: [JSONValue] = []
                // Anthropic rejects error tool results that contain anything but text, so their images follow
                // all tool results of the turn, which must come first in the user message (#10794).
                var errorImages: [JSONValue] = []
                while index < transformed.count, case .toolResult(var result) = transformed[index] {
                    if result.isError, !result.images.isEmpty {
                        errorImages.append(["type": "text", "text": .string("Images from the error result of tool call \(result.toolCallId) (\(result.toolName)):")])
                        errorImages += result.images.map(imageBlock)
                        result.content = Transcript.replacingImages(result.content, placeholder: "(image attached after the tool results)")
                    }
                    let content: JSONValue
                    if result.images.isEmpty {
                        content = .string(result.text)
                    } else {
                        let blocks: [JSONValue] = result.content.map { block in
                            switch block {
                            case .text(let text): ["type": "text", "text": .string(text.text)]
                            case .image(let image): imageBlock(image)
                            }
                        }
                        content = .array(blocks.contains { $0["type"] == "text" } ? blocks : [["type": "text", "text": "(see attached image)"]] + blocks)
                    }
                    results.append(["type": "tool_result", "tool_use_id": .string(result.toolCallId), "content": content, "is_error": .bool(result.isError)])
                    index += 1
                }
                wire.append(["role": "user", "content": .array(results + errorImages)])
                continue
            }
            index += 1
        }
        wire += pendingSystem
        return wire
    }

    private static func updateTotals(_ usage: inout Usage, model: Model) {
        usage.totalTokens = usage.input + usage.output + usage.cacheRead + usage.cacheWrite
        usage.applyCost(of: model)
    }
}
