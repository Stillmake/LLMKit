import Foundation

// Error classification from pi-ai’s utils/retry.ts.
private let nonRetryableProviderLimitErrorPattern = try! NSRegularExpression(
    pattern: [
        #"GoUsageLimitError"#,
        #"FreeUsageLimitError"#,
        #"Monthly usage limit reached"#,
        #"available balance"#,
        #"insufficient_quota"#,
        #"out of budget"#,
        #"quota exceeded"#,
        #"billing"#,
        #"subscription_sharing_usage_limit_exceeded"#,
        #"subscription_sharing_workplace_individual_usage_limit_exceeded"#,
        #""code"\s*:\s*"?1308(?![0-9])"#,
        // Foundation and LLMKit error descriptions.
        #"\A(?:\d{3} )?1308:"#,
    ].joined(separator: "|"),
    options: .caseInsensitive
)

private let retryableProviderErrorPattern = try! NSRegularExpression(
    pattern: [
        #"overloaded"#,
        #"server_busy"#,
        #"servers are currently busy"#,
        #"currently experiencing high demand"#,
        #"model is at capacity"#,
        #"admission ceiling"#,
        #"rate.?limit"#,
        #"too many requests"#,
        #"rate increased too quickly"#,
        #"(?<![a-z0-9])(?:429|500|502|503|504|520|524|525)(?![a-z0-9])"#,
        #"service.?(?:is.?)?(?:temporarily.?)?unavailable"#,
        #"provider.?(?:is.?)?temporarily.?unavailable"#,
        #"server.?error"#,
        #"internal.?error"#,
        #"provider.?returned.?(?:an.?)?error"#,
        #""type"\s*:\s*"api_error""#,
        #"exceeded request buffer limit while retrying upstream"#,
        #"unable to forward upstream"#,
        #"upstream.?request.?failed"#,
        #"upstream_stream_read_error"#,
        #"stream was interrupted"#,
        #"http/?2 stream failed"#,
        #"network.?error"#,
        #"connection.?error"#,
        #"connection.?refused"#,
        #"connection.?lost"#,
        #"connection.?closed"#,
        #"other side closed"#,
        #"fetch failed"#,
        #"request to \S+ failed, reason:"#,
        #"getaddrinfo"#,
        #"(?<![a-z0-9_])(?:ECONNRESET|ECONNABORTED|ECONNREFUSED|ETIMEDOUT|EPIPE)(?![a-z0-9_])"#,
        #"connection reset by peer"#,
        #"ENOTFOUND"#,
        #"EAI_AGAIN"#,
        #"upstream.?connect"#,
        #"reset before headers"#,
        #"socket hang up"#,
        #"socket connection was closed"#,
        #"timed? out"#,
        #"timeout"#,
        #"terminated"#,
        #"websocket.?closed"#,
        #"websocket.?error"#,
        #"websocket.?connection.?limit.?reached"#,
        #"ended without"#,
        #"stream ended before message_stop"#,
        #"stream ended before a terminal response event"#,
        #"http2 request did not get a response"#,
        #"provider finish_reason: error"#,
        #"provider stopped with: error"#,
        #"returned an empty response"#,
        #"stopped before completing the response"#,
        #"stream disconnected before completion"#,
        #"cut short after the reply had begun"#,
        #"ended the response before it was complete"#,
        #"connection to the provider was reset"#,
        #"reported a failure while generating the response"#,
        #"pending stream has been canceled"#,
        #"retry delay"#,
        #"you can retry your request"#,
        #"try your request again"#,
        #"please retry your request"#,
        #"repeat the request later"#,
        #"ResourceExhausted"#,
        #"subscription_sharing_usage_unavailable"#,
        #"subscription_sharing_user_unavailable"#,
        // Foundation and LLMKit error descriptions.
        #"\A(?:\d{3} )?api_error:"#,
        #"network connection was lost"#,
        #"could not connect to the server"#,
    ].joined(separator: "|"),
    options: .caseInsensitive
)

extension LLM {
    /// Whether an assistant error resembles a transient provider or transport failure.
    ///
    /// Check `isContextOverflow(_:contextWindow:)` first. Callers decide how to discard a
    /// partial turn, apply backoff, and restart the request.
    public static func isRetryableAssistantError(_ message: AssistantMessage) -> Bool {
        guard message.stopReason == .error, let error = message.errorMessage, !error.isEmpty else { return false }
        let range = NSRange(error.startIndex..., in: error)
        return nonRetryableProviderLimitErrorPattern.firstMatch(in: error, range: range) == nil
            && retryableProviderErrorPattern.firstMatch(in: error, range: range) != nil
    }
}
