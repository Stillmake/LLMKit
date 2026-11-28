import Testing
@testable import LLMKit

@Suite struct RetryTests {
    let model = Model(id: "test-model", api: .openAIResponses, provider: "openai", baseURL: "https://example.com")

    @Test(arguments: [
        "525 SSL handshake failed",
        "Your request rate increased too quickly",
        "The provider is temporarily unavailable. Retry shortly.",
        "The provider ended the response before it was complete.",
        "Provider finish_reason: error",
        "OpenAI Responses stream ended before a terminal response event",
        "read ECONNRESET",
        "The network connection was lost.",
        #"{"type":"error","error":{"type":"api_error","message":"Unable to complete this request right now."}}"#,
        "subscription_sharing_usage_unavailable: Usage cannot be checked.",
    ])
    func identifiesTransientProviderAndTransportErrors(error: String) {
        var message = AssistantMessage(model: model)
        message.stopReason = .error
        message.errorMessage = error
        #expect(LLM.isRetryableAssistantError(message))
    }

    @Test(arguments: [
        "429 quota exceeded",
        "429 GoUsageLimitError: rate limit reached",
        "429 subscription_sharing_usage_limit_exceeded",
        "429 subscription_sharing_workplace_individual_usage_limit_exceeded",
        #"429 {"code":"1308","message":"Usage limit reached."}"#,
        #"{"type":"api_error","message":"Monthly quota exceeded"}"#,
        "The provider is unavailable in your region",
        "Provider finish_reason: content_filter",
        "401 Invalid API key (request req_011CU503xyz)",
        "400 image exceeds 5 MB maximum: 5242880 bytes",
        "Invalid key ERR_EPIPE_ID",
    ])
    func rejectsPersistentErrorsAndPartialCodeMatches(error: String) {
        var message = AssistantMessage(model: model)
        message.stopReason = .error
        message.errorMessage = error
        #expect(!LLM.isRetryableAssistantError(message))
    }

    @Test func recognizesNormalizedProviderErrors() {
        var message = AssistantMessage(model: model)
        message.stopReason = .error
        message.errorMessage = HTTP.errorMessage(
            from: #"{"type":"error","error":{"type":"api_error","message":"Unable to complete this request right now."}}"#)
        #expect(LLM.isRetryableAssistantError(message))

        let quota = HTTP.errorMessage(from: #"{"error":{"code":1308,"message":"Usage limit reached."}}"#)
        message.errorMessage = LLMError.http(status: 429, message: quota, headers: [:]).errorDescription
        #expect(message.errorMessage == "429 1308: Usage limit reached.")
        #expect(!LLM.isRetryableAssistantError(message))
    }

    @Test func classifiesOnlyErroredTurnsWithAMessage() {
        var message = AssistantMessage(model: model)
        message.stopReason = .aborted
        message.errorMessage = "The network connection was lost."
        #expect(!LLM.isRetryableAssistantError(message))
        message.stopReason = .stop
        #expect(!LLM.isRetryableAssistantError(message))
        message.stopReason = .error
        message.errorMessage = nil
        #expect(!LLM.isRetryableAssistantError(message))
    }
}
