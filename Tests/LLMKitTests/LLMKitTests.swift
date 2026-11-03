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
