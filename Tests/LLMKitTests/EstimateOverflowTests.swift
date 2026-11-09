import Foundation
import Testing
@testable import LLMKit

@Suite struct EstimateOverflowTests {
    let model = Model(id: "test-model", api: .openAIResponses, provider: "openai", baseURL: "https://example.com")

    func assistant(_ timestamp: Double, _ tokens: Int) -> AssistantMessage {
        var result = AssistantMessage(model: model, timestamp: Date(timeIntervalSince1970: timestamp))
        result.content = [.text(TextContent("kept"))]
        result.usage.totalTokens = tokens
        result.stopReason = .stop
        return result
    }

    @Test func upstreamEstimateFixtures() {
        let first: [Message] = [.assistant(assistant(100, 2000)), .user(UserMessage(String(repeating: "x", count: 3500), timestamp: Date(timeIntervalSince1970: 200)))]
        let second = Context(systemPrompt: "system", messages: [
            .user(UserMessage("summary", timestamp: Date(timeIntervalSince1970: 200))),
            .assistant(assistant(100, 9500)),
            .user(UserMessage(String(repeating: "x", count: 4000), timestamp: Date(timeIntervalSince1970: 300)))])
        let third: [Message] = [
            .user(UserMessage("summary", timestamp: Date(timeIntervalSince1970: 200))),
            .assistant(assistant(100, 9500)),
            .user(UserMessage("new prompt", timestamp: Date(timeIntervalSince1970: 300))),
            .assistant(assistant(400, 2000)),
            .user(UserMessage("tail", timestamp: Date(timeIntervalSince1970: 500)))]
        let results = [LLM.estimateContextTokens(first), LLM.estimateContextTokens(second), LLM.estimateContextTokens(third)]
        #expect(results.map(\.tokens) == [3000, 1149, 2002])
        #expect(results.map(\.usageTokens) == [2000, 0, 2000])
        #expect(results.map(\.trailingTokens) == [1000, 1149, 2])
        #expect(results.map(\.lastUsageIndex) == [0, nil, 3])
    }

    @Test func messageEstimation() {
        let tool = Tool(name: "a", description: "b", parameters: [:])
        let system = Message.system(SystemMessage("abc", toolsAdded: [tool], toolsRemoved: ["x"]))
        // ceil(3/3.5) + ceil(48/3.5) + ceil(5/3.5)
        #expect(LLM.estimateMessageTokens(system) == 17)
        #expect(LLM.estimateMessageTokens(.user(UserMessage("😀😀"))) == 2)
        var usage = Usage(); usage.input = 3; usage.output = 4; usage.cacheRead = 5; usage.cacheWrite = 6
        #expect(LLM.calculateContextTokens(usage) == 18)
        usage.totalTokens = 42
        #expect(LLM.calculateContextTokens(usage) == 42)
    }

    @Test func upstreamOverflowPatterns() {
        // Examples from upstream overflow tests and the provider examples in overflow.ts.
        let samples = [
            "prompt is too long: 213462 tokens > 200000 maximum", "Prompt exceeds max length", "request_too_large",
            "input is too long for requested model", "Your input exceeds the context window of this model",
            "Requested token count exceeds the model's maximum context length of 131072 tokens", "Context Window exceeded for given call",
            "input token count (1196265) exceeds the maximum", "maximum prompt length is 131072", "reduce the length of the messages",
            "maximum context length is 100 tokens", "exceeds the maximum allowed input length of 131040 tokens",
            "The input (516368 tokens) is longer than the model's context length (262144 tokens)", "exceeds the limit of 100",
            "exceeds the available context size", "greater than the context length", "context window exceeds limit",
            "exceeded model token limit", "too large for model with 100 maximum context length",
            "Prompt has 256468 tokens, but the configured context size is 256000 tokens", "model_context_window_exceeded",
            "prompt too long; exceeded max context length", "Range of input length should be [1, 100]",
            "context_length_exceeded", "too many tokens", "token limit exceeded", "custom-provider API error (413): Payload Too Large"
        ]
        let patterns = LLM.getOverflowPatterns()
        #expect(patterns.count == samples.count)
        for (pattern, sample) in zip(patterns, samples) {
            #expect(pattern.firstMatch(in: sample, range: NSRange(sample.startIndex..., in: sample)) != nil)
            var message = assistant(0, 0); message.stopReason = .error; message.errorMessage = sample
            #expect(LLM.isContextOverflow(message))
        }
    }

    @Test func overflowAndLengthBoundaries() {
        var message = assistant(0, 0)
        message.stopReason = .error; message.errorMessage = "Throttling error: Too many tokens, please wait before trying again."
        #expect(!LLM.isContextOverflow(message))
        message.errorMessage = "400 status code (no body)"; message.provider = "cerebras"
        #expect(LLM.isContextOverflow(message))
        message.provider = "other"; #expect(!LLM.isContextOverflow(message))
        message.stopReason = .stop; message.usage.input = 100; message.usage.cacheRead = 1
        #expect(LLM.isContextOverflow(message, contextWindow: 100))
        message.stopReason = .length; message.usage.input = 98; message.usage.cacheRead = 1
        #expect(LLM.isContextOverflow(message, contextWindow: 100))
        #expect(LLM.isRecoverableLength(message, desiredMaxOutput: 10))
        message.usage.output = 10
        #expect(!LLM.isRecoverableLength(message, desiredMaxOutput: 10))
        #expect(!LLM.isContextOverflow(message, contextWindow: 100))
    }
}
