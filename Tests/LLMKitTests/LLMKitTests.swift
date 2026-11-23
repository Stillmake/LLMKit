import Foundation
import Testing

@testable import LLMKit

@Suite struct Upstream150Tests {
    @Test func incrementalArgumentsHandleChunksAndKeepSnapshots() {
        let parser = StreamingJSON()
        parser.append(#"{"items":[{"text":"hello"#)
        let first = parser.snapshot()
        #expect(first == ["items": [["text": "hello"]]])
        parser.append(#"\nworld","n":1e"#)
        #expect(parser.snapshot() == ["items": [["text": "hello\nworld", "n": 1]]])
        #expect(first == ["items": [["text": "hello"]]])
        parser.append(#"2},true,null],"emoji":"\uD83D\uDE00"}"#)
        #expect(parser.snapshot() == ["items": [["text": "hello\nworld", "n": 100], true, .null], "emoji": "😀"])
    }

    @Test func incrementalArgumentsHandleSplitEscapesAndInvalidInput() {
        let parser = StreamingJSON()
        parser.append(#"{"a":"x\u00"#)
        #expect(parser.snapshot() == ["a": "x"])
        parser.append(#"e9\q","b":fal"#)
        #expect(parser.snapshot() == ["a": "xé\\q", "b": false])
        parser.append(#"se}"#)
        #expect(parser.snapshot() == ["a": "xé\\q", "b": false])
    }

    @Test func incrementalArgumentsParseOneCharacterAtATime() throws {
        let text = #"{"nested":[1,2,{"unicode":"😀","escaped":"a\\\"b","value":-1.25e+3}],"empty":{},"bool":true}"#
        let expected = try #require(try JSONValue(parsing: text).objectValue)
        let parser = StreamingJSON()
        for char in text {
            parser.append(String(char))
            _ = parser.snapshot()
        }
        #expect(parser.snapshot() == expected)
    }

    @Test func geminiConstRewritePreservesLiteralData() {
        let input: JSONValue = ["properties": ["kind": ["const": "x"], "const": ["const": 3]],
                                "default": ["const": "literal"], "examples": [["const": "example"]]]
        let output = GoogleGenerativeAI.rewriteConstAsEnum(input)
        #expect(output["properties"]?["kind"]?["enum"] == ["x"])
        #expect(output["properties"]?["const"]?["enum"] == [3])
        #expect(output["default"] == ["const": "literal"])
        #expect(output["examples"] == [["const": "example"]])
    }

    @Test func openRouterCostRequiresCompleteAccounting() {
        var usage = Usage()
        usage.cost.total = 42
        OpenRouterCost.apply(["cost": 0.5, "is_byok": true, "cost_details": ["upstream_inference_cost": .null, "upstream_inference_prompt_cost": 1, "upstream_inference_completions_cost": 2]], to: &usage)
        #expect(usage.cost.total == 3.5)
        #expect(usage.cost.totalFromProvider == true)
        usage.cost.total = 42
        usage.cost.totalFromProvider = false
        OpenRouterCost.apply(["cost": 0.5, "is_byok": true, "cost_details": [:]], to: &usage)
        #expect(usage.cost.total == 42)
        #expect(usage.cost.totalFromProvider == false)
        OpenRouterCost.apply(["cost": 0, "is_byok": false], to: &usage)
        #expect(usage.cost.total == 0)
        #expect(usage.cost.totalFromProvider == true)
    }
}

/// Ports pi-ai 1.6.0's constrained-sampling, compat, and Responses conversion behavior.
@Suite struct Upstream160Tests {
    @Test func strictSchemasRequireATypeUnlessTheyHaveAnyOfOrAnEnum() throws {
        func resolve(_ property: JSONValue, strict mode: Tool.StrictMode = .prefer) throws -> Bool? {
            let tool = Tool(
                name: "t", description: "",
                parameters: ["type": "object", "properties": ["value": property]], strict: mode)
            return try StrictSchema.resolve(tool, supportsStrictMode: true)
        }
        #expect(try resolve(["enum": ["a", "b"]]) == true)
        #expect(try resolve(["anyOf": [["type": "string"], ["type": "null"]]]) == true)
        // A bare const counts as untyped and falls back to non-strict mode, or fails when required.
        #expect(try resolve(["const": "a"]) == nil)
        #expect(throws: LLMError.self) { try resolve([:], strict: .require) }
    }

    @Test func sendsTheDeveloperRoleOnlyToOpenAIsAPI() {
        func supportsDeveloperRole(_ provider: String, _ baseURL: String, compat: ModelCompat? = nil) -> Bool {
            let model = Model(
                id: "m", api: .openAICompletions, provider: provider, baseURL: baseURL, reasoning: true, compat: compat)
            return OpenAICompletions.Compat(model: model).supportsDeveloperRole
        }
        #expect(supportsDeveloperRole("openai", "https://api.openai.com/v1"))
        #expect(!supportsDeveloperRole("azure", "https://acme.openai.azure.com/openai/v1"))
        #expect(!supportsDeveloperRole("openrouter", "https://openrouter.ai/api/v1"))
        #expect(!supportsDeveloperRole("groq", "https://api.groq.com/openai/v1"))
        var forced = ModelCompat()
        forced.supportsDeveloperRole = true
        #expect(supportsDeveloperRole("groq", "https://api.groq.com/openai/v1", compat: forced))
    }

    @Test func responsesSendTheSystemPromptAsALeadingDeveloperItem() {
        let model = Model(
            id: "gpt-6.2-sol", api: .openAICodexResponses, provider: "openai-codex",
            baseURL: "https://chatgpt.com/backend-api", reasoning: true)
        let messages = Transcript.normalize(
            Context(systemPrompt: "Be terse.", messages: [.user(UserMessage("hi"))]))
        let input = OpenAIResponsesShared.convertMessages(
            model: model, messages: messages, toolCallProviders: ["openai", "openai-codex", "opencode"],
            supportsMidConvoSystemMessages: false)
        #expect(input == [
            ["role": "developer", "content": "Be terse."],
            ["role": "user", "content": [["type": "input_text", "text": "hi"]]],
        ])
    }

    @Test func replaysThinkingLevelsAsConfigurationUpdates() {
        let model = Model(
            id: "gpt-6.2-sol", api: .openAIResponses, provider: "openai",
            baseURL: "https://api.openai.com/v1", reasoning: true)
        var previous = AssistantMessage(model: model)
        previous.content = [.text(TextContent("answer"))]
        previous.stopReason = .stop
        previous.providerThinkingLevel = "low"
        let messages = Transcript.normalize(Context(messages: [
            .user(UserMessage("first")), .assistant(previous), .user(UserMessage("second")),
        ]))
        let input = OpenAIResponsesShared.convertMessages(
            model: model, messages: messages, toolCallProviders: ["openai", "openai-codex", "opencode"],
            supportsMidConvoSystemMessages: false, configurationUpdateEffort: "high")
        #expect(input.map { $0["role"]?.stringValue ?? $0["type"]?.stringValue }
            == ["user", "configuration_update", "assistant", "user", "configuration_update"])
        #expect(input[1] == ["type": "configuration_update", "reasoning": ["effort": "low"]])
        #expect(input.last == ["type": "configuration_update", "reasoning": ["effort": "high"]])
        // Without the feature the effort stays in the request and no items are replayed.
        let plain = OpenAIResponsesShared.convertMessages(
            model: model, messages: messages, toolCallProviders: ["openai", "openai-codex", "opencode"],
            supportsMidConvoSystemMessages: false)
        #expect(plain.allSatisfy { $0["type"] != "configuration_update" })
    }

    @Test func recordsWithheldReasoningAsDiagnostics() throws {
        var output = AssistantMessage(model: Model(
            id: "gpt-6.2-sol", api: .openAIResponses, provider: "openai", baseURL: "https://api.openai.com/v1"))
        OpenAIResponsesShared.appendReasoningRemovals([
            ["type": "reasoning", "id": "rs_edited", "action": "removed", "reason": "prefix_mismatch"],
            ["type": "reasoning", "id": "rs_old", "action": "removed", "reason": "turn_boundary"],
            ["type": "message", "id": "msg_1", "action": "removed", "reason": "prefix_mismatch"],
        ], to: &output)
        let diagnostic = try #require(output.diagnostics?.first)
        #expect(output.diagnostics?.count == 1)
        #expect(diagnostic.type == "openai_input_transformations")
        let removals = try #require(diagnostic.details?["transformations"]?.arrayValue)
        #expect(removals.map { $0["id"]?.stringValue } == ["rs_edited"])
        OpenAIResponsesShared.appendReasoningRemovals(.array([]), to: &output)
        #expect(output.diagnostics?.count == 1)
    }
}

@Suite struct PartialJSONTests {
    @Test func parsesCompleteJSON() {
        #expect(PartialJSON.parseObject(#"{"path": "a.txt", "line": 3}"#) == ["path": "a.txt", "line": 3])
    }

    @Test func keepsCompleteValuesOfTruncatedJSON() {
        #expect(PartialJSON.parseObject(#"{"path": "a.t"#) == ["path": "a.t"])
        #expect(PartialJSON.parseObject(#"{"path": "a.txt", "lin"#) == ["path": "a.txt"])
        #expect(PartialJSON.parseObject(#"{"items": [1, 2, {"x": tr"#) == ["items": [1, 2, [:]]])
        #expect(PartialJSON.parseObject(#"{"n": 12"#) == [:])
        #expect(PartialJSON.parseObject("") == [:])
    }

    @Test func repairsInvalidStrings() {
        #expect(PartialJSON.parseObject("{\"text\": \"a\nb\\q\"}") == ["text": "a\nb\\q"])
        #expect(PartialJSON.parseObject(#"{"emoji": "😀"}"#) == ["emoji": "😀"])
    }
}

@Suite struct ServerSentEventTests {
    struct Bytes: AsyncSequence {
        typealias Element = UInt8
        let data: [UInt8]

        struct AsyncIterator: AsyncIteratorProtocol {
            var remaining: ArraySlice<UInt8>
            mutating func next() async throws -> UInt8? { remaining.popFirst() }
        }

        func makeAsyncIterator() -> AsyncIterator { AsyncIterator(remaining: data[...]) }
    }

    func events(_ text: String) async throws -> [ServerSentEvent] {
        var result: [ServerSentEvent] = []
        for try await event in ServerSentEvents(base: Bytes(data: Array(text.utf8))) {
            result.append(event)
        }
        return result
    }

    @Test func splitsOnBlankLinesAcrossLineEndings() async throws {
        let parsed = try await events("event: a\ndata: 1\n\n: comment\r\ndata: x\r\ndata: y\r\n\r\ndata:z")
        #expect(parsed == [
            ServerSentEvent(event: "a", data: "1"),
            ServerSentEvent(event: nil, data: "x\ny"),
            ServerSentEvent(event: nil, data: "z"),
        ])
    }
}

@Suite struct TranscriptTests {
    let gpt = Model(id: "gpt-x", api: .openAIResponses, provider: "openai", baseURL: "https://api.openai.com/v1")
    let claude = Model(id: "claude-x", api: .anthropicMessages, provider: "anthropic", baseURL: "https://api.anthropic.com")

    func assistant(from model: Model, _ content: [AssistantContent], stop: StopReason = .stop) -> Message {
        var message = AssistantMessage(model: model)
        message.content = content
        message.stopReason = stop
        return .assistant(message)
    }

    @Test func foreignThinkingAndSignaturesDrop() {
        let history: [Message] = [
            .user(UserMessage("hi")),
            assistant(from: gpt, [
                .thinking(ThinkingContent("plan", thinkingSignature: "{}")),
                .thinking(ThinkingContent("secret", thinkingSignature: "opaque", redacted: true)),
                .text(TextContent("hello", textSignature: "msg_1")),
            ]),
        ]
        guard case .assistant(let replayed) = Transcript.transform(history, for: claude)[1] else {
            Issue.record("assistant message dropped")
            return
        }
        #expect(replayed.content == [.text(TextContent("hello"))])
    }

    @Test func orphanedToolCallsGetSyntheticResultsAndIdsAreRemapped() {
        let call = ToolCall(id: "call|fc_1", name: "read", arguments: [:])
        let history: [Message] = [
            assistant(from: gpt, [.toolCall(call)], stop: .toolUse),
            .system(SystemMessage("later")),
            .user(UserMessage("next")),
        ]
        let transformed = Transcript.transform(history, for: claude) { id, _ in Transcript.sanitizeIdentifier(id) }
        guard transformed.count == 4, case .toolResult(let result) = transformed[1], case .system = transformed[2] else {
            Issue.record("unexpected transcript: \(transformed)")
            return
        }
        #expect(result.toolCallId == "call_fc_1")
        #expect(result.isError)
    }

    @Test func erroredTurnsAreDropped() {
        let history: [Message] = [.user(UserMessage("hi")), assistant(from: gpt, [.text(TextContent("partial"))], stop: .error)]
        #expect(Transcript.transform(history, for: gpt).count == 1)
    }

    @Test func collapsesSystemMessages() {
        let tool = Tool(name: "a", description: "", parameters: ["type": "object"])
        let messages: [Message] = [
            .system(SystemMessage("base", sections: [.init(name: "s", text: "one")], toolsAdded: [tool])),
            .user(UserMessage("hi")),
            .system(SystemMessage("more", sections: [.init(name: "s", text: nil)], toolsRemoved: ["a"])),
        ]
        let collapsed = Transcript.collapseSystemMessages(messages)
        guard case .system(let head) = collapsed[0] else {
            Issue.record("missing head")
            return
        }
        #expect(collapsed.count == 2)
        #expect(head.promptText == "base\n\nmore")
        #expect(head.toolsAdded.isEmpty)
    }
}

@Suite struct StrictSchemaTests {
    @Test func makesOptionalPropertiesNullableAndRequired() throws {
        let schema: JSONValue = [
            "type": "object",
            "properties": ["path": ["type": "string"], "limit": ["type": "integer"]],
            "required": ["path"],
        ]
        let strict = try StrictSchema.strict(schema, keywordCheck: nil)
        #expect(strict["additionalProperties"] == false)
        #expect(strict["required"] == ["limit", "path"])
        #expect(strict["properties"]?["limit"] == ["anyOf": [["type": "integer"], ["type": "null"]]])
    }

    @Test func rejectsUnsupportedKeywords() {
        let tool = Tool(name: "t", description: "", parameters: ["type": "object", "oneOf": []], strict: .require)
        #expect(throws: LLMError.self) { try StrictSchema.resolve(tool, supportsStrictMode: true) }
    }
}

@Suite struct AnthropicMessageThreadTests {
    let model = Model(
        id: "claude-test",
        api: .anthropicMessages,
        provider: "anthropic",
        baseURL: "https://api.anthropic.com"
    )
    let betas = [AnthropicMessages.messageThreadsBeta]

    func body(messages: [JSONValue]) -> JSONObject {
        [
            "model": "claude-test",
            "messages": .array(messages),
            "system": [["type": "text", "text": "You are a test assistant."]],
            "tools": [["name": "lookup", "input_schema": ["type": "object"]]],
            "max_tokens": 100,
            "stream": true,
        ]
    }

    @Test func continuesWithOnlyNewMessagesAfterAnUnchangedPrefix() throws {
        LLM.cleanupSessionResources()
        let sessionId = "thread-\(UUID().uuidString)"
        let first = body(messages: [[
            "role": "user",
            "content": [["type": "text", "text": "first", "cache_control": ["type": "ephemeral"]]],
        ]])
        let created = try #require(AnthropicMessages.prepareThreadRequest(
            model: model,
            sessionId: sessionId,
            body: first,
            betas: betas
        ))
        #expect(!created.isContinuation)
        #expect(created.body["thread"] == ["type": "create"])

        var output = AssistantMessage(model: model)
        output.responseId = "msg_1"
        output.content = [.text(TextContent("answer"))]
        output.stopReason = .stop
        AnthropicMessages.recordThreadReply(
            model: model,
            thread: created,
            body: first,
            output: output,
            isOAuth: false,
            allowEmptySignature: false,
            betas: betas
        )

        // The cache marker moved from the first user message to the newest one; the server's
        // thread prefix is unchanged, so only the new message is sent.
        let next = body(messages: [
            ["role": "user", "content": [["type": "text", "text": "first"]]],
            ["role": "assistant", "content": [["type": "text", "text": "answer"]]],
            ["role": "user", "content": [["type": "text", "text": "second", "cache_control": ["type": "ephemeral"]]]],
        ])
        let continued = try #require(AnthropicMessages.prepareThreadRequest(
            model: model,
            sessionId: sessionId,
            body: next,
            betas: betas
        ))
        #expect(continued.isContinuation)
        #expect(continued.body["thread"] == ["type": "continue", "previous_message_id": "msg_1"])
        #expect(continued.body["system"] == nil)
        #expect(continued.body["tools"] == nil)
        #expect(continued.body["messages"]?.arrayValue == [
            ["role": "user", "content": [["type": "text", "text": "second", "cache_control": ["type": "ephemeral"]]]],
        ])
    }

    @Test func startsANewThreadWhenAnEarlierMessageChanges() throws {
        LLM.cleanupSessionResources()
        let sessionId = "thread-\(UUID().uuidString)"
        let first = body(messages: [["role": "user", "content": "first"]])
        let created = try #require(AnthropicMessages.prepareThreadRequest(
            model: model,
            sessionId: sessionId,
            body: first,
            betas: betas
        ))
        var output = AssistantMessage(model: model)
        output.responseId = "msg_1"
        output.content = [.text(TextContent("answer"))]
        output.stopReason = .stop
        AnthropicMessages.recordThreadReply(
            model: model,
            thread: created,
            body: first,
            output: output,
            isOAuth: false,
            allowEmptySignature: false,
            betas: betas
        )

        let rewritten = body(messages: [
            ["role": "user", "content": "edited"],
            ["role": "assistant", "content": [["type": "text", "text": "answer"]]],
            ["role": "user", "content": "second"],
        ])
        let prepared = try #require(AnthropicMessages.prepareThreadRequest(
            model: model,
            sessionId: sessionId,
            body: rewritten,
            betas: betas
        ))
        #expect(!prepared.isContinuation)
        #expect(prepared.body["thread"] == ["type": "create"])
        #expect(prepared.body["messages"]?.arrayValue?.count == 3)
    }
}

@Suite struct ModelTests {
    @Test func catalogDecodes() {
        let catalog = ModelCatalog.builtin
        #expect(catalog.models.count > 1000)
        #expect(catalog.providers.contains("anthropic"))
        #expect(Set(catalog.providerInfo.keys) == Set(catalog.providers))
        #expect(catalog.providerInfo["deepseek"] == ProviderInfo(name: "DeepSeek", logo: "deepseek"))
        #expect(catalog.version != nil)
        #expect(catalog.generatedAt != nil)
    }

    @Test func snapshotRoundTrips() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelCatalog.builtin.write(to: url)
        let loaded = try ModelCatalog(contentsOf: url)
        #expect(loaded.version == ModelCatalog.builtin.version)
        #expect(loaded.generatedAt == ModelCatalog.builtin.generatedAt)
        #expect(loaded.models == ModelCatalog.builtin.models)
        #expect(loaded.models.allSatisfy { [.openAICompletions, .openAIResponses, .openAICodexResponses, .anthropicMessages, .googleGenerativeAI].contains($0.api) })
    }

    @Test func clampsThinkingLevels() {
        let model = Model(
            id: "m", api: .openAICompletions, provider: "p", baseURL: "https://x", reasoning: true,
            thinkingLevelMap: ThinkingLevelMap([.minimal: .unsupported, .medium: .unsupported, .max: .value("max")])
        )
        #expect(model.supportedThinkingLevels == [.off, .low, .high, .max])
        #expect(model.clampThinkingLevel(.minimal) == .low)
        #expect(model.clampThinkingLevel(.medium) == .high)
        #expect(model.clampThinkingLevel(.xhigh) == .max)
    }

    @Test func pricesTiersAndLongCacheWrites() {
        let model = Model(
            id: "m", api: .anthropicMessages, provider: "p", baseURL: "https://x",
            cost: ModelCost(
                rates: CostRates(input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75),
                tiers: [.init(inputTokensAbove: 200_000, rates: CostRates(input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5))]
            )
        )
        var usage = Usage()
        usage.input = 1_000_000
        usage.cacheWrite = 1_000_000
        usage.cacheWrite1h = 500_000
        usage.applyCost(of: model)
        #expect(usage.cost.input == 6)
        #expect(usage.cost.cacheWrite == 7.5 * 0.5 + 6 * 2 * 0.5)
    }
}

@Suite struct RequestTests {
    let deepseek = Model(
        id: "deepseek-reasoner", api: .openAICompletions, provider: "deepseek", baseURL: "https://api.deepseek.com",
        reasoning: true, contextWindow: 128_000, maxTokens: 8_192
    )

    @Test func deepSeekRequestUsesDetectedCompat() throws {
        let messages = Transcript.normalize(Context(systemPrompt: "Be brief.", messages: [.user(UserMessage("hi"))]))
        let options = ResolvedOptions(model: deepseek, transcript: messages, options: StreamOptions(apiKey: "k", reasoning: .high), useEnvironment: false)
        let body = try OpenAICompletions.requestBody(model: deepseek, messages: messages, options: options, compat: .init(model: deepseek))
        #expect(body["max_tokens"] == 8192)
        #expect(body["max_completion_tokens"] == nil)
        #expect(body["thinking"] == ["type": "enabled"])
        #expect(body["reasoning_effort"] == "high")
        #expect(body["store"] == nil)
        #expect(body["messages"]?[0] == ["role": "system", "content": "Be brief."])
        #expect(body["messages"]?[1] == ["role": "user", "content": "hi"])
    }

    @Test func replaysReasoningContentForDeepSeek() {
        var reply = AssistantMessage(model: deepseek)
        reply.content = [.thinking(ThinkingContent("think", thinkingSignature: "reasoning_content")), .text(TextContent("answer"))]
        reply.stopReason = .stop
        let wire = OpenAICompletions.convertMessages(
            model: deepseek,
            messages: [.user(UserMessage("q")), .assistant(reply)],
            compat: .init(model: deepseek)
        )
        #expect(wire[1] == ["role": "assistant", "content": "answer", "reasoning_content": "think"])
    }
}

/// Ports pi-ai's `test/identity.test.ts` (pi-mono 7c22a93), minus the `Models`/wrapper identity merging
/// the port has no equivalent for.
@Suite struct ClientIdentityTests {
    let acme = ClientAttribution(name: "acme", url: "https://acme.example")

    func model(_ provider: String, _ baseURL: String) -> Model {
        Model(id: "test-model", api: .openAICompletions, provider: provider, baseURL: baseURL)
    }

    @Test func keepsTheVersionInSyncWithTheCatalog() throws {
        let version = try #require(ModelCatalog.builtin.version)
        #expect(ClientIdentity().formattedUserAgent.hasPrefix("pi/\(version) ("))
    }

    @Test func formatsUserAgents() {
        #expect(ClientIdentity(version: "9.9.9").formattedUserAgent.wholeMatch(of: /pi\/9\.9\.9 \([a-z]+; [a-z0-9]+\)/) != nil)
        #expect(ClientIdentity(userAgent: .short, version: "9.9.9").formattedUserAgent == "pi")
        #expect(ClientIdentity(userAgent: .custom("acme/1.0")).formattedUserAgent == "acme/1.0")
    }

    @Test func encodesUserAgentModesAsPiAIJSON() throws {
        let json = String(decoding: try JSONEncoder().encode([UserAgentMode.short, .detailed, .custom("acme")]), as: UTF8.self)
        #expect(json == #"["short","detailed",{"custom":"acme"}]"#)
        #expect(try JSONDecoder().decode([UserAgentMode].self, from: Data(json.utf8)) == [.short, .detailed, .custom("acme")])
    }

    @Test func sendsNoAttributionUnlessEnabled() {
        let openRouter = model("openrouter", "https://openrouter.ai/api/v1")
        #expect(ClientIdentity().attributionHeaders(for: openRouter).isEmpty)
        #expect(ClientIdentity().headers(for: openRouter) == ["User-Agent": ClientIdentity().formattedUserAgent])
    }

    @Test func attributesOpenRouterRequests() {
        let openRouter = model("openrouter", "https://openrouter.ai/api/v1")
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: openRouter) == [
            "HTTP-Referer": "https://pi.dev", "X-OpenRouter-Title": "pi", "X-OpenRouter-Categories": "cli-agent",
        ])
        var categorized = acme
        categorized.categories = ["cli-agent", "cloud-agent"]
        #expect(ClientIdentity(attribution: categorized).attributionHeaders(for: openRouter) == [
            "HTTP-Referer": "https://acme.example", "X-OpenRouter-Title": "acme", "X-OpenRouter-Categories": "cli-agent,cloud-agent",
        ])
        var uncategorized = acme
        uncategorized.categories = []
        #expect(ClientIdentity(attribution: uncategorized).attributionHeaders(for: openRouter) == [
            "HTTP-Referer": "https://acme.example", "X-OpenRouter-Title": "acme",
        ])
        // Custom providers routed through OpenRouter, including legacy base URLs that are not valid URLs.
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: model("custom", "not-a-url-openrouter.ai"))["X-OpenRouter-Title"] == "pi")
    }

    @Test func attributesNvidiaWithACapitalizedNameAndOpenCodeWithTheName() {
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: model("nvidia", "https://example.test/v1")) == ["X-BILLING-INVOKE-ORIGIN": "Pi"])
        #expect(ClientIdentity(attribution: acme).attributionHeaders(for: model("custom-nim", "https://integrate.api.nvidia.com/v1")) == ["X-BILLING-INVOKE-ORIGIN": "Acme"])
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: model("opencode", "https://opencode.ai/zen/v1")) == ["x-opencode-client": "pi"])
        #expect(ClientIdentity(attribution: acme).attributionHeaders(for: model("custom", "https://opencode.ai/zen/v1")) == ["x-opencode-client": "acme"])
    }

    @Test func doesNotAttributeNvidiaModelsRoutedThroughGateways() {
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: model("openrouter", "https://openrouter.ai/api/v1"))["X-BILLING-INVOKE-ORIGIN"] == nil)
        #expect(ClientIdentity(attribution: .pi).attributionHeaders(for: model("vercel-ai-gateway", "https://ai-gateway.vercel.sh/v1")).isEmpty)
    }
}

/// Answers every request with a minimal Chat Completions stream and records the request, like pi-ai's `capturingFetch`.
final class CapturingURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var captured: URLRequest?

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CapturingURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.captured = request
        let body = """
            data: {"id":"chatcmpl-identity","choices":[{"delta":{"content":"ok"},"finish_reason":null,"index":0}]}

            data: {"id":"chatcmpl-identity","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":1,"completion_tokens":1}}

            data: [DONE]


            """
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["content-type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite(.serialized) struct ClientIdentityRequestTests {
    let context = Context(messages: [.user(UserMessage("hello"))])

    func send(_ model: Model, _ options: StreamOptions) async throws -> URLRequest {
        var options = options
        options.apiKey = "key"
        options.session = CapturingURLProtocol.session()
        _ = try await LLM.complete(model: model, context: context, options: options)
        return try #require(CapturingURLProtocol.captured)
    }

    @Test func appliesTheIdentityBelowModelAndRequestHeaders() async throws {
        let model = Model(id: "test-model", api: .openAICompletions, provider: "openrouter", baseURL: "https://openrouter.ai/api/v1",
                          headers: ["X-OpenRouter-Title": "model-title"])
        let request = try await send(model, StreamOptions(headers: ["HTTP-Referer": nil], identity: ClientIdentity(userAgent: .short, attribution: .pi)))
        #expect(request.value(forHTTPHeaderField: "user-agent") == "pi")
        #expect(request.value(forHTTPHeaderField: "x-openrouter-title") == "model-title")
        #expect(request.value(forHTTPHeaderField: "http-referer") == nil)
    }

    @Test func keepsImpersonatingModelHeadersOverTheIdentityUserAgent() async throws {
        let model = Model(id: "test-model", api: .openAICompletions, provider: "github-copilot", baseURL: "https://api.githubcopilot.com",
                          headers: ["User-Agent": "GitHubCopilotChat/0.35.0"])
        let request = try await send(model, StreamOptions(identity: ClientIdentity(userAgent: .custom("acme"))))
        #expect(request.value(forHTTPHeaderField: "user-agent") == "GitHubCopilotChat/0.35.0")
    }

    @Test func sendsTheOpenCodeSessionHeaderToCustomProvidersPointingAtOpenCode() async throws {
        let model = Model(id: "test-model", api: .openAICompletions, provider: "custom-opencode", baseURL: "https://opencode.ai/zen/v1")
        #expect(try await send(model, StreamOptions(sessionId: "session-1")).value(forHTTPHeaderField: "x-opencode-session") == "session-1")
        let configured = try await send(model, StreamOptions(headers: ["X-OpenCode-Session": "configured"], sessionId: "session-1"))
        #expect(configured.value(forHTTPHeaderField: "x-opencode-session") == "configured")
    }
}

/// Answers every request with the given SSE payload and records it, so adapter tests can assert the
/// wire request (pi-ai's `capturingFetch`, which Swift's URLProtocol needs for streaming bodies).
final class StubSSEURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var payload = ""
    nonisolated(unsafe) static var captured: URLRequest?

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubSSEURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    /// The body of the captured request; URLSession hands URLProtocol a stream instead of data.
    static func capturedJSON() throws -> JSONValue {
        let request = try #require(captured)
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                let count = stream.read(&buffer, maxLength: buffer.count)
                guard count > 0 else { break }
                data.append(buffer, count: count)
            }
        }
        return try JSONValue(parsing: String(decoding: data, as: UTF8.self))
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.captured = request
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["content-type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(Self.payload.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Ports pi-ai 1.6.0's request-shape tests through the real adapters.
@Suite(.serialized) struct Upstream160RequestTests {
    let context = Context(systemPrompt: "Be terse.", messages: [.user(UserMessage("Say hello"))])

    func send(_ model: Model, context: Context, options: StreamOptions) async throws -> AssistantMessage {
        StubSSEURLProtocol.captured = nil
        var options = options
        options.apiKey = options.apiKey ?? "key"
        options.session = StubSSEURLProtocol.session()
        return try await LLM.complete(model: model, context: context, options: options)
    }

    /// A ChatGPT OAuth token whose claims carry the account id.
    var chatGPTToken: String {
        let claims = Data(#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct"}}"#.utf8)
            .base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        return "header.\(claims).signature"
    }

    @Test func codexSendsTheSystemPromptAsADeveloperItemAndLevelsInsideTheConversation() async throws {
        StubSSEURLProtocol.payload = #"""
            data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}

            """#
        var compat = ModelCompat()
        compat.supportsMidConvoEffort = true
        let model = Model(
            id: "gpt-6.2-sol", api: .openAICodexResponses, provider: "openai-codex",
            baseURL: "https://chatgpt.com/backend-api", reasoning: true,
            contextWindow: 400_000, maxTokens: 128_000, compat: compat)
        let message = try await send(
            model, context: context,
            options: StreamOptions(apiKey: chatGPTToken, maxTokens: 5_000, reasoning: .high))
        let request = try StubSSEURLProtocol.capturedJSON()
        #expect(request["instructions"] == nil)
        #expect(request["max_output_tokens"] == 5000)
        #expect(request["reasoning"] == ["effort": "medium", "summary": "auto"])
        #expect(request["input"]?.arrayValue?.first == ["role": "developer", "content": "Be terse."])
        #expect(request["input"]?.arrayValue?.last
            == ["type": "configuration_update", "reasoning": ["effort": "high"]])
        #expect(message.providerThinkingLevel == "high")
        #expect(message.stopReason == .stop)
    }

    @Test func responsesReplaysProviderThinkingLevelsAsConfigurationUpdates() async throws {
        StubSSEURLProtocol.payload = #"""
            data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}

            """#
        var compat = ModelCompat()
        compat.supportsMidConvoEffort = true
        let model = Model(
            id: "gpt-6.2-sol", api: .openAIResponses, provider: "openai",
            baseURL: "https://api.openai.com/v1", reasoning: true,
            contextWindow: 272_000, maxTokens: 128_000, compat: compat)
        var previous = AssistantMessage(model: model)
        previous.content = [.text(TextContent("answer"))]
        previous.stopReason = .stop
        previous.providerThinkingLevel = "low"
        let history = Context(systemPrompt: "Be terse.", messages: [
            .user(UserMessage("first")), .assistant(previous), .user(UserMessage("second")),
        ])
        let message = try await send(model, context: history, options: StreamOptions(apiKey: "sk-key", reasoning: .high))
        let request = try StubSSEURLProtocol.capturedJSON()
        #expect(request["reasoning"] == ["effort": "medium", "summary": "auto"])
        #expect(request["input"]?.arrayValue?.map { $0["role"]?.stringValue ?? $0["type"]?.stringValue }
            == ["developer", "user", "configuration_update", "assistant", "user", "configuration_update"])
        #expect(message.providerThinkingLevel == "high")
    }

    @Test func responsesKeepTheEffortInTheRequestWithoutMidConversationEffort() async throws {
        StubSSEURLProtocol.payload = #"""
            data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}

            """#
        let model = Model(
            id: "gpt-6.2-sol", api: .openAIResponses, provider: "openai",
            baseURL: "https://api.openai.com/v1", reasoning: true)
        let message = try await send(model, context: context, options: StreamOptions(apiKey: "sk-key", reasoning: .high))
        let request = try StubSSEURLProtocol.capturedJSON()
        #expect(request["reasoning"] == ["effort": "high", "summary": "auto"])
        #expect(request["input"]?.arrayValue?.allSatisfy { $0["type"] != "configuration_update" } == true)
        #expect(message.providerThinkingLevel == nil)
    }

    @Test func anthropicNonStrictToolSchemasKeepRootAdditionalProperties() async throws {
        StubSSEURLProtocol.payload = #"""
            event: message_start
            data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

            event: message_delta
            data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

            event: message_stop
            data: {"type":"message_stop"}

            """#
        let model = Model(
            id: "claude-test", api: .anthropicMessages, provider: "anthropic",
            baseURL: "https://api.anthropic.com", maxTokens: 100)
        let tool = Tool(name: "lookup", description: "", parameters: [
            "type": "object",
            "properties": ["q": ["type": "string"]],
            "required": ["q"],
            "additionalProperties": false,
        ])
        let message = try await send(
            model, context: Context(messages: [.user(UserMessage("hi"))], tools: [tool]),
            options: StreamOptions(apiKey: "key"))
        let request = try StubSSEURLProtocol.capturedJSON()
        #expect(request["tools"]?[0]?["input_schema"]?["additionalProperties"] == false)
        #expect(message.stopReason == .stop)
    }

    @Test func googleMapsContinuationToLength() async throws {
        StubSSEURLProtocol.payload = #"""
            data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"CONTINUATION"}]}

            """#
        let model = Model(
            id: "gemini-test", api: .googleGenerativeAI, provider: "google",
            baseURL: "https://generativelanguage.googleapis.com/v1beta")
        let message = try await send(
            model, context: Context(messages: [.user(UserMessage("hello"))]), options: StreamOptions(apiKey: "key"))
        #expect(message.stopReason == .length)
    }

    @Test func chatCompletionsReadsTopLevelReasoningTokens() async throws {
        StubSSEURLProtocol.payload = #"""
            data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"ok"},"finish_reason":null,"index":0}]}

            data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"reasoning_tokens":3}}

            data: [DONE]

            """#
        let model = Model(id: "m", api: .openAICompletions, provider: "openai", baseURL: "https://api.openai.com/v1")
        let message = try await send(
            model, context: Context(messages: [.user(UserMessage("hello"))]), options: StreamOptions(apiKey: "key"))
        #expect(message.usage.reasoning == 3)
        #expect(message.usage.output == 5)
    }
}
