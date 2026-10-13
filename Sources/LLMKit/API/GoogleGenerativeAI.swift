import Foundation

/// Google Gemini API (`generativelanguage.googleapis.com`). Gemini has no mid-conversation
/// system messages; the replayed prompt is sent as `systemInstruction`. Thought signatures are
/// echoed back on the parts that carried them, for the same model only.
/// Ports pi-ai's `google-generative-ai.ts` and `google-shared.ts`.
enum GoogleGenerativeAI {
    static func run(model: Model, transcript: [Message], options: ResolvedOptions, builder: ResponseBuilder) async throws {
        let apiKey = try options.requireAPIKey(for: model)
        let messages = Transcript.collapseSystemMessages(transcript)
        let tools = Transcript.currentTools(messages)
        let geminiMajorVersion = model.id.lowercased()
            .firstMatch(of: /^gemini(?:-live)?-(\d+)/).flatMap { Int($0.1) }
        let supportsStrictMode = (geminiMajorVersion ?? 0) >= 3

        var generationConfig: JSONObject = ["maxOutputTokens": .number(Double(options.maxTokens))]
        if let temperature = options.raw.temperature { generationConfig["temperature"] = .number(temperature) }
        if model.reasoning {
            // Gemini 3 and Gemma 4 take a discrete level; Gemini 2.5 takes a token budget.
            let id = model.id.lowercased()
            let usesThinkingLevel = id.contains(/gemini-3(?:\.\d+)?-(?:pro|flash)/)
                || id == "gemini-flash-latest" || id == "gemini-flash-lite-latest" || id.contains(/gemma-?4/)
            func googleLevel(_ level: ThinkingLevel) throws -> String {
                let mapped: String = if case .value(let value)? = model.thinkingLevelMap?[level] { value.lowercased() } else { level.rawValue }
                guard ["minimal", "low", "medium", "high"].contains(mapped) else {
                    throw LLMError.invalidRequest("Unsupported Google thinking level mapping for \(model.provider)/\(model.id): \(level) -> \(mapped)")
                }
                return mapped.uppercased()
            }

            if let level = options.reasoning {
                let resolved = try googleLevel(level)
                if usesThinkingLevel {
                    generationConfig["thinkingConfig"] = ["includeThoughts": true, "thinkingLevel": .string(resolved)]
                } else {
                    let budget: Int
                    if let custom = options.raw.thinkingBudgets {
                        budget = custom[level]
                    } else {
                        let budgets: [String: Int]? = if id.contains("2.5-pro") {
                            ["MINIMAL": 128, "LOW": 2048, "MEDIUM": 8192, "HIGH": 32768]
                        } else if id.contains("2.5-flash-lite") {
                            ["MINIMAL": 512, "LOW": 2048, "MEDIUM": 8192, "HIGH": 24576]
                        } else if id.contains("2.5-flash") {
                            ["MINIMAL": 128, "LOW": 2048, "MEDIUM": 8192, "HIGH": 24576]
                        } else {
                            nil
                        }
                        // -1 lets the model decide.
                        budget = budgets?[resolved] ?? -1
                    }
                    generationConfig["thinkingConfig"] = ["includeThoughts": true, "thinkingBudget": .number(Double(budget))]
                }
            } else {
                let fallback = model.clampThinkingLevel(.off)
                if usesThinkingLevel, fallback != .off {
                    generationConfig["thinkingConfig"] = ["thinkingLevel": .string(try googleLevel(fallback))]
                } else {
                    generationConfig["thinkingConfig"] = ["thinkingBudget": 0]
                }
            }
        }

        // Claude and gpt-oss behind Google APIs, and Gemini 3+, need explicit tool-call ids.
        let requiresToolCallId = model.id.hasPrefix("claude-") || model.id.hasPrefix("gpt-oss-") || (geminiMajorVersion ?? 0) >= 3
        let supportsMultimodalFunctionResponse = geminiMajorVersion.map { $0 >= 3 } ?? true
        let conversation = Transcript.initialSystemMessage(messages) == nil ? messages : Array(messages.dropFirst())
        let transformed = Transcript.transform(conversation, for: model) { id, _ in
            requiresToolCallId ? Transcript.sanitizeIdentifier(id) : id
        }

        var contents: [JSONValue] = []
        for message in transformed {
            switch message {
            case .system:
                continue

            case .user(let user):
                let parts: [JSONValue] = user.content.map { block in
                    switch block {
                    case .text(let text): ["text": .string(text.text)]
                    case .image(let image): ["inlineData": ["mimeType": .string(image.mimeType), "data": .string(image.data)]]
                    }
                }
                if !parts.isEmpty { contents.append(["role": "user", "parts": .array(parts)]) }

            case .assistant(let assistant):
                // Signatures are only valid for the model that produced them, and must be base64.
                let isSameModel = assistant.provider == model.provider && assistant.model == model.id
                func validSignature(_ signature: String?) -> String? {
                    guard isSameModel, let signature, !signature.isEmpty, signature.count % 4 == 0,
                        signature.wholeMatch(of: /[A-Za-z0-9+\/]+={0,2}/) != nil
                    else { return nil }
                    return signature
                }
                var parts: [JSONValue] = []
                for block in assistant.content {
                    switch block {
                    case .text(let text):
                        let signature = validSignature(text.textSignature)
                        // Gemini may sign a part with empty text and needs the signature echoed back.
                        guard !text.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || signature != nil else { continue }
                        var part: JSONObject = ["text": .string(text.text)]
                        if let signature { part["thoughtSignature"] = .string(signature) }
                        parts.append(.object(part))
                    case .thinking(let thinking):
                        let signature = validSignature(thinking.thinkingSignature)
                        let isEmpty = thinking.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        if isSameModel {
                            guard !isEmpty || signature != nil else { continue }
                            var part: JSONObject = ["thought": true, "text": .string(thinking.thinking)]
                            if let signature { part["thoughtSignature"] = .string(signature) }
                            parts.append(.object(part))
                        } else if !isEmpty {
                            parts.append(["text": .string(thinking.thinking)])
                        }
                    case .toolCall(let call):
                        var functionCall: JSONObject = ["name": .string(call.name), "args": .object(call.arguments)]
                        if requiresToolCallId { functionCall["id"] = .string(call.id) }
                        var part: JSONObject = ["functionCall": .object(functionCall)]
                        if let signature = validSignature(call.thoughtSignature) { part["thoughtSignature"] = .string(signature) }
                        parts.append(.object(part))
                    }
                }
                if !parts.isEmpty { contents.append(["role": "model", "parts": .array(parts)]) }

            case .toolResult(let result):
                let text = result.text
                let images = model.supportsImages ? result.images : []
                let value: JSONValue = .string(!text.isEmpty ? text : images.isEmpty ? "" : "(see attached image)")
                let imageParts: [JSONValue] = images.map { ["inlineData": ["mimeType": .string($0.mimeType), "data": .string($0.data)]] }
                var response: JSONObject = [
                    "name": .string(result.toolName),
                    "response": result.isError ? ["error": value] : ["output": value],
                ]
                if !images.isEmpty, supportsMultimodalFunctionResponse { response["parts"] = .array(imageParts) }
                if requiresToolCallId { response["id"] = .string(result.toolCallId) }
                let part: JSONValue = ["functionResponse": .object(response)]

                // All function responses of one turn belong in a single user content.
                if case .object(var last)? = contents.last, last["role"] == "user",
                    case .array(var parts)? = last["parts"], parts.contains(where: { $0["functionResponse"] != nil })
                {
                    parts.append(part)
                    last["parts"] = .array(parts)
                    contents[contents.count - 1] = .object(last)
                } else {
                    contents.append(["role": "user", "parts": [part]])
                }
                if !images.isEmpty, !supportsMultimodalFunctionResponse {
                    contents.append(["role": "user", "parts": .array([["text": "Tool result image:"]] + imageParts)])
                }
            }
        }

        var body: JSONObject = [
            "contents": .array(contents),
            "generationConfig": .object(generationConfig),
        ]
        if let system = Transcript.initialSystemMessage(messages)?.promptText, !system.isEmpty {
            body["systemInstruction"] = ["parts": [["text": .string(system)]]]
        }
        if !tools.isEmpty {
            var usesStrictTools = false
            let declarations: [JSONValue] = try tools.map { tool in
                let strict = try StrictSchema.resolve(tool, supportsStrictMode: supportsStrictMode)
                if strict == true { usesStrictTools = true }
                return [
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parametersJsonSchema": StrictSchema.parameters(of: tool, strict: strict),
                ]
            }
            body["tools"] = [["functionDeclarations": .array(declarations)]]
            let mode: String? = switch options.raw.toolChoice {
            case .none?: "NONE"
            case .required?: "ANY"
            case .auto?: usesStrictTools ? "VALIDATED" : "AUTO"
            case nil: usesStrictTools ? "VALIDATED" : nil
            }
            if let mode { body["toolConfig"] = ["functionCallingConfig": ["mode": .string(mode)]] }
        }

        let payload = options.finalize(&body, model: model, applySamplingParams: false)
        let url = try HTTP.url(model.baseURL, path: "/models/\(model.id):streamGenerateContent?alt=sse")
        let bytes = try await HTTP.send(
            HTTP.request(url: url, body: payload, headers: ["x-goog-api-key": apiKey], model: model, options: options),
            model: model,
            options: options
        )
        builder.start()

        var current: ResponseBuilder.Block?
        for try await (_, chunk) in HTTP.jsonEvents(bytes) {
            if let error = chunk["error"], !error.isNull {
                throw LLMError.provider(HTTP.errorMessage(from: JSONValue.object(["error": error]).jsonString()))
            }
            if builder.message.responseId == nil { builder.message.responseId = chunk["responseId"]?.stringValue }
            let candidate = chunk["candidates"]?[0]
            for part in candidate?["content"]?["parts"]?.arrayValue ?? [] {
                // `thought: true` marks thinking; a thoughtSignature may ride on any part and only
                // arrives on some deltas, so keep the last non-empty one per block.
                let signature = part["thoughtSignature"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
                if let text = part["text"]?.stringValue {
                    let kind: ResponseBuilder.Block.Kind = part["thought"] == true ? .thinking : .text
                    if current?.kind != kind {
                        if let current { builder.close(current) }
                        current = kind == .thinking ? builder.openThinking() : builder.openText()
                    }
                    guard let block = current else { continue }
                    builder.appendText(text, to: block)
                    if let signature { block.signature = signature }
                }
                if let call = part["functionCall"] {
                    if let block = current { builder.close(block) }
                    current = nil
                    let name = call["name"]?.stringValue ?? ""
                    let providedId = call["id"]?.stringValue ?? ""
                    let isDuplicate = builder.blocks.contains { $0.kind == .toolCall && $0.toolCallId == providedId }
                    let id = providedId.isEmpty || isDuplicate ? "\(name)_\(UUID().uuidString)" : providedId
                    let block = builder.openToolCall(id: id, name: name, arguments: call["args"]?.objectValue ?? [:])
                    block.thoughtSignature = signature
                    builder.close(block)
                }
            }
            if let reason = candidate?["finishReason"]?.stringValue {
                builder.message.rawStopReason = reason
                switch reason {
                case "STOP": builder.message.stopReason = builder.hasToolCalls ? .toolUse : .stop
                case "MAX_TOKENS": builder.message.stopReason = .length
                default:
                    builder.message.stopReason = .error
                    builder.message.errorMessage = "Provider stopped with: \(reason)"
                }
            }
            if let metadata = chunk["usageMetadata"], !metadata.isNull {
                let cached = metadata["cachedContentTokenCount"]?.intValue ?? 0
                let thoughts = metadata["thoughtsTokenCount"]?.intValue ?? 0
                var usage = Usage()
                usage.input = (metadata["promptTokenCount"]?.intValue ?? 0) - cached
                usage.output = (metadata["candidatesTokenCount"]?.intValue ?? 0) + thoughts
                usage.cacheRead = cached
                usage.reasoning = thoughts
                usage.totalTokens = metadata["totalTokenCount"]?.intValue ?? 0
                usage.applyCost(of: model)
                builder.message.usage = usage
            }
        }
    }
}
