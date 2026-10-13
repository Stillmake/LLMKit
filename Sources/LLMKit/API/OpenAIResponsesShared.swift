import Foundation

/// Message conversion and stream processing shared by the OpenAI Responses and Codex adapters.
/// Ports pi-ai's `openai-responses-shared.ts`.
enum OpenAIResponsesShared {
    // MARK: Messages

    /// Converts the transcript to Responses `input` items.
    ///
    /// Reasoning items replay verbatim (from `thinkingSignature`) for the same model, assistant text
    /// keeps its message id and phase, and tool-call ids are `{call_id}|{item_id}`. Item ids from a
    /// different model or provider are dropped or re-derived so OpenAI's reasoning/item pairing
    /// validation does not reject the replay.
    static func convertMessages(
        model: Model,
        messages: [Message],
        toolCallProviders: Set<String>,
        includeSystemPrompt: Bool,
        supportsMidConvoSystemMessages: Bool
    ) -> [JSONValue] {
        func normalizePart(_ part: String) -> String {
            var normalized = Transcript.sanitizeIdentifier(part)
            while normalized.hasSuffix("_") { normalized.removeLast() }
            return normalized
        }
        let transformed = Transcript.transform(
            Transcript.resolve(messages, supportsMidConvoSystemMessages: supportsMidConvoSystemMessages),
            for: model
        ) { id, source in
            guard toolCallProviders.contains(model.provider), let separator = id.firstIndex(of: "|") else {
                return normalizePart(id)
            }
            let itemPart = id[id.index(after: separator)...].prefix { $0 != "|" }
            let isForeign = source.provider != model.provider || source.api != model.api
            var itemId = isForeign ? String("fc_\(Transcript.shortHash(String(itemPart)))".prefix(64)) : normalizePart(String(itemPart))
            // Responses requires function-call item ids to start with "fc".
            if !itemId.hasPrefix("fc_") { itemId = normalizePart("fc_\(itemId)") }
            return "\(normalizePart(String(id[..<separator])))|\(itemId)"
        }
        let instructionRole: JSONValue = model.reasoning && model.compat?.supportsDeveloperRole != false ? "developer" : "system"

        var input: [JSONValue] = []
        var messageIndex = 0
        for (sourceIndex, message) in transformed.enumerated() {
            switch message {
            case .system(let system):
                let isLeading = sourceIndex == 0
                if !isLeading || includeSystemPrompt {
                    let text = isLeading ? system.promptText : system.updateText
                    if !text.isEmpty { input.append(["role": instructionRole, "content": .string(text)]) }
                }
                if isLeading { continue }

            case .user(let user):
                let content: [JSONValue] = user.content.map { block in
                    switch block {
                    case .text(let text): ["type": "input_text", "text": .string(text.text)]
                    case .image(let image): ["type": "input_image", "detail": "auto", "image_url": .string("data:\(image.mimeType);base64,\(image.data)")]
                    }
                }
                guard !content.isEmpty else { continue }
                input.append(["role": "user", "content": .array(content)])

            case .assistant(let assistant):
                let isSameProviderAndAPI = assistant.provider == model.provider && assistant.api == model.api
                let isSameModel = isSameProviderAndAPI && assistant.model == model.id
                var output: [JSONValue] = []
                var textIndex = 0
                for block in assistant.content {
                    switch block {
                    case .thinking(let thinking):
                        guard let signature = thinking.thinkingSignature else { continue }
                        guard let item = try? JSONValue(parsing: signature) else {
                            assertionFailure("Responses reasoning signatures are serialized reasoning items")
                            continue
                        }
                        output.append(item)
                    case .text(let text):
                        // textSignature is `{"v":1,"id":...,"phase":...}`, or a legacy plain message id.
                        var id = textIndex == 0 ? "msg_pi_\(messageIndex)" : "msg_pi_\(messageIndex)_\(textIndex)"
                        var phase: String?
                        if let signature = text.textSignature, !signature.isEmpty {
                            if signature.hasPrefix("{"), let json = try? JSONValue(parsing: signature), json["v"] == 1,
                                let signedId = json["id"]?.stringValue
                            {
                                id = signedId
                                phase = json["phase"].flatMap { $0 == "commentary" || $0 == "final_answer" ? $0.stringValue : nil }
                            } else {
                                id = signature
                            }
                        }
                        textIndex += 1
                        if id.count > 64 { id = "msg_\(Transcript.shortHash(id))" }
                        var item: JSONObject = [
                            "type": "message",
                            "role": "assistant",
                            "content": [["type": "output_text", "text": .string(text.text), "annotations": []]],
                            "status": "completed",
                            "id": .string(id),
                        ]
                        if let phase { item["phase"] = .string(phase) }
                        output.append(.object(item))
                    case .toolCall(let call):
                        let parts = call.id.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false)
                        var item: JSONObject = [
                            "type": "function_call",
                            "call_id": .string(String(parts[0])),
                            "name": .string(call.name),
                            "arguments": .string(JSONValue.object(call.arguments).jsonString()),
                        ]
                        // OpenAI pairs item ids with the reasoning items that produced them; another
                        // model's ids (or ids of the wrong type) would fail that validation.
                        if isSameModel || !isSameProviderAndAPI, parts.count > 1, parts[1].hasPrefix("fc_") {
                            item["id"] = .string(String(parts[1]))
                        }
                        if isSameModel, let namespace = call.namespace { item["namespace"] = .string(namespace) }
                        output.append(.object(item))
                    }
                }
                guard !output.isEmpty else { continue }
                input += output

            case .toolResult(let result):
                let callId = String(result.toolCallId.prefix { $0 != "|" })
                let output: JSONValue
                if result.images.isEmpty || !model.supportsImages {
                    output = .string(result.text.isEmpty ? (result.images.isEmpty ? "(no tool output)" : "(see attached image)") : result.text)
                } else {
                    output = .array((result.text.isEmpty ? [] : [["type": "input_text", "text": .string(result.text)]])
                        + result.images.map { ["type": "input_image", "detail": "auto", "image_url": .string("data:\($0.mimeType);base64,\($0.data)")] })
                }
                input.append(["type": "function_call_output", "call_id": .string(callId), "output": output])
            }
            messageIndex += 1
        }
        return input
    }

    /// `defaultStrict` is sent for tools that do not opt into strict sampling; `nil` sends `null`.
    static func convertTools(_ tools: [Tool], defaultStrict: Bool?, supportsStrictMode: Bool) throws -> [JSONValue] {
        try tools.map { tool in
            let strict = try StrictSchema.resolve(tool, supportsStrictMode: supportsStrictMode) ?? defaultStrict
            var function: JSONObject = [
                "type": "function",
                "name": .string(tool.name),
                "description": .string(tool.description),
                "parameters": StrictSchema.parameters(of: tool, strict: strict),
            ]
            if supportsStrictMode { function["strict"] = strict.map(JSONValue.bool) ?? .null }
            return .object(function)
        }
    }

    // MARK: Stream

    /// Consumes Responses stream events into `builder`.
    ///
    /// - Parameters:
    ///   - codex: accept ChatGPT's `response.done` terminal event, treat unknown statuses as unknown,
    ///     and stop at the first terminal event.
    ///   - serviceTier: maps the response's and the request's service tier to the tier to bill.
    static func process<Events: AsyncSequence>(
        _ events: Events,
        builder: ResponseBuilder,
        model: Model,
        codex: Bool,
        serviceTier: (String?) -> String?
    ) async throws where Events.Element == JSONValue {
        var slots: [Int: ResponseBuilder.Block] = [:]
        var reasoningById: [String: ResponseBuilder.Block] = [:]
        var sawTerminalEvent = false

        func slot(_ event: JSONValue, _ kind: ResponseBuilder.Block.Kind) -> ResponseBuilder.Block? {
            guard let index = event["output_index"]?.intValue, let block = slots[index], block.kind == kind else { return nil }
            return block
        }

        @discardableResult
        func openSlot(_ index: Int, item: JSONValue) -> ResponseBuilder.Block? {
            if let existing = slots[index] { return existing }
            let block: ResponseBuilder.Block
            switch item["type"]?.stringValue {
            case "reasoning":
                block = builder.openThinking()
            case "message":
                if item["phase"] == "final_answer" { builder.message.stopReason = .stop }
                block = builder.openText()
            case "function_call":
                block = builder.openToolCall(
                    id: "\(item["call_id"]?.stringValue ?? "")|\(item["id"]?.stringValue ?? "")",
                    name: item["name"]?.stringValue ?? ""
                )
                block.namespace = item["namespace"]?.stringValue
                if let arguments = item["arguments"]?.stringValue, !arguments.isEmpty {
                    builder.appendToolArguments(arguments, to: block)
                }
            default:
                return nil
            }
            slots[index] = block
            return block
        }

        func finalize(_ response: JSONValue?) {
            sawTerminalEvent = true
            // Azure can omit encrypted reasoning from output_item.done and only send it here;
            // backfill it so store:false multi-turn replay stays stateless.
            for item in response?["output"]?.arrayValue ?? [] where item["type"] == "reasoning" {
                guard let encrypted = item["encrypted_content"], !encrypted.isNull,
                    let id = item["id"]?.stringValue, let block = reasoningById[id],
                    let signature = block.signature, case .object(var stored)? = try? JSONValue(parsing: signature),
                    stored["encrypted_content"] == nil || stored["encrypted_content"] == .null
                else { continue }
                stored["encrypted_content"] = encrypted
                block.signature = JSONValue.object(stored).jsonString()
            }
            if let id = response?["id"]?.stringValue { builder.message.responseId = id }
            if let usage = response?["usage"], !usage.isNull {
                // input_tokens includes cached and cache-write tokens.
                let cached = usage["input_tokens_details"]?["cached_tokens"]?.intValue ?? 0
                let cacheWrite = usage["input_tokens_details"]?["cache_write_tokens"]?.intValue ?? 0
                var parsed = Usage()
                parsed.input = max(0, (usage["input_tokens"]?.intValue ?? 0) - cached - cacheWrite)
                parsed.output = usage["output_tokens"]?.intValue ?? 0
                parsed.cacheRead = cached
                parsed.cacheWrite = cacheWrite
                parsed.reasoning = usage["output_tokens_details"]?["reasoning_tokens"]?.intValue ?? 0
                parsed.totalTokens = usage["total_tokens"]?.intValue ?? 0
                builder.message.usage = parsed
            }
            builder.message.usage.applyCost(of: model)
            let multiplier: Double = switch serviceTier(response?["service_tier"]?.stringValue) {
            case "flex": 0.5
            case "priority", "fast": model.id == "gpt-5.5" ? 2.5 : 2
            default: 1
            }
            if multiplier != 1 {
                var cost = builder.message.usage.cost
                cost.input *= multiplier
                cost.output *= multiplier
                cost.cacheRead *= multiplier
                cost.cacheWrite *= multiplier
                cost.total = cost.input + cost.output + cost.cacheRead + cost.cacheWrite
                builder.message.usage.cost = cost
            }

            var status = response?["status"]?.stringValue
            if codex, let value = status, !["completed", "incomplete", "failed", "cancelled", "queued", "in_progress"].contains(value) {
                status = nil
            }
            let incompleteReason = response?["incomplete_details"]?["reason"]?.stringValue
            builder.message.rawStopReason = incompleteReason.map { "\(status ?? "").\($0)" } ?? status
            builder.message.errorMessage = nil
            switch status {
            case nil, "completed", "in_progress", "queued":
                builder.message.stopReason = .stop
            case "incomplete" where incompleteReason == "max_output_tokens":
                builder.message.stopReason = .length
            case "incomplete":
                builder.message.stopReason = .error
                builder.message.errorMessage = incompleteReason.map { "Response incomplete: \($0)" } ?? "Response incomplete without a provider reason"
            default:
                builder.message.stopReason = .error
            }
            if builder.message.stopReason == .stop, builder.hasToolCalls { builder.message.stopReason = .toolUse }
        }

        eventLoop: for try await event in events {
            switch event["type"]?.stringValue {
            case "response.created":
                builder.message.responseId = event["response"]?["id"]?.stringValue
            case "response.output_item.added":
                if let index = event["output_index"]?.intValue, let item = event["item"] { openSlot(index, item: item) }
            case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
                if let block = slot(event, .thinking), let delta = event["delta"]?.stringValue { builder.appendText(delta, to: block) }
            case "response.reasoning_summary_part.done":
                if let block = slot(event, .thinking) { builder.appendText("\n\n", to: block) }
            case "response.output_text.delta", "response.refusal.delta":
                if let block = slot(event, .text), let delta = event["delta"]?.stringValue { builder.appendText(delta, to: block) }
            case "response.function_call_arguments.delta":
                if let block = slot(event, .toolCall), let delta = event["delta"]?.stringValue {
                    builder.appendToolArguments(delta, to: block)
                }
            case "response.function_call_arguments.done":
                if let block = slot(event, .toolCall), let arguments = event["arguments"]?.stringValue {
                    builder.setToolArguments(arguments, on: block)
                }
            case "response.output_item.done":
                guard let index = event["output_index"]?.intValue, let item = event["item"] else { continue }
                if item["phase"] == "final_answer" { builder.message.stopReason = .stop }
                guard let block = openSlot(index, item: item) else { continue }
                switch (item["type"]?.stringValue, block.kind) {
                case ("reasoning", .thinking):
                    let summary = (item["summary"]?.arrayValue ?? []).compactMap { $0["text"]?.stringValue }.joined(separator: "\n\n")
                    let content = (item["content"]?.arrayValue ?? []).compactMap { $0["text"]?.stringValue }.joined(separator: "\n\n")
                    block.text = summary.isEmpty ? (content.isEmpty ? block.text : content) : summary
                    block.signature = item.jsonString()
                    if let id = item["id"]?.stringValue { reasoningById[id] = block }
                case ("message", .text):
                    block.text = (item["content"]?.arrayValue ?? [])
                        .compactMap { $0["type"] == "output_text" ? $0["text"]?.stringValue : $0["refusal"]?.stringValue }
                        .joined()
                    var signature: JSONObject = ["v": 1, "id": item["id"] ?? ""]
                    if let phase = item["phase"], !phase.isNull { signature["phase"] = phase }
                    block.signature = JSONValue.object(signature).jsonString()
                case ("function_call", .toolCall):
                    builder.setToolArguments(item["arguments"]?.stringValue ?? (block.text.isEmpty ? "{}" : block.text), on: block)
                    if let namespace = item["namespace"]?.stringValue { block.namespace = namespace }
                default:
                    continue
                }
                builder.close(block)
                slots[index] = nil
            case "response.completed", "response.incomplete":
                finalize(event["response"])
                if codex { break eventLoop }
            case "response.done" where codex:
                finalize(event["response"])
                break eventLoop
            case "error":
                let code = event["code"]?.stringValue ?? event["error"]?["code"]?.stringValue
                let message = event["message"]?.stringValue ?? event["error"]?["message"]?.stringValue
                throw LLMError.provider(
                    codex ? "Codex error: \(message ?? code ?? event.jsonString())" : "Error Code \(code ?? "unknown"): \(message ?? "")"
                )
            case "response.failed":
                sawTerminalEvent = true
                let response = event["response"]
                builder.message.rawStopReason = response?["status"]?.stringValue
                if let error = response?["error"], !error.isNull {
                    throw LLMError.provider("\(error["code"]?.stringValue ?? "unknown"): \(error["message"]?.stringValue ?? "no message")")
                }
                if let reason = response?["incomplete_details"]?["reason"]?.stringValue {
                    throw LLMError.provider("incomplete: \(reason)")
                }
                throw LLMError.provider("Unknown error (no error details in response)")
            default:
                continue
            }
        }

        guard sawTerminalEvent else {
            throw LLMError.protocolViolation("OpenAI Responses stream ended before a terminal response event")
        }
        // Every call in the final message gets executed: refuse calls whose output_item.done never arrived.
        if builder.message.stopReason == .toolUse,
            let unfinished = builder.blocks.first(where: { $0.kind == .toolCall && !$0.isClosed })
        {
            throw LLMError.protocolViolation(
                "OpenAI Responses stream completed with an unfinished tool call: \(unfinished.toolName) (\(unfinished.toolCallId))"
            )
        }
    }
}
