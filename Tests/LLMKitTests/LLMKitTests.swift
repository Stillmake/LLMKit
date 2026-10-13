import Foundation
import Testing

@testable import LLMKit

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

    @Test func foreignThinkingBecomesTextAndSignaturesDrop() {
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
        #expect(replayed.content == [.text(TextContent("plan")), .text(TextContent("hello"))])
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
        #expect(catalog.models.allSatisfy { [.openAICompletions, .openAIResponses, .openAICodexResponses, .anthropicMessages, .googleGenerativeAI].contains($0.api) })
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
        let options = ResolvedOptions(model: deepseek, transcript: messages, options: StreamOptions(apiKey: "k", reasoning: .high))
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
