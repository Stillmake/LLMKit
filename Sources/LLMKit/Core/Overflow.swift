import Foundation

private let overflowPatterns: [NSRegularExpression] = [
        #"prompt (?:is )?too long"#,
        #"prompt exceeds max length"#,
        #"request_too_large"#,
        #"input is too long for requested model"#,
        #"exceeds the context window"#,
        #"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))"#,
        #"context window exceeded"#,
        #"input token count.*exceeds the maximum"#,
        #"maximum prompt length is \d+"#,
        #"reduce the length of the messages"#,
        #"maximum context length is \d+ tokens"#,
        #"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?"#,
        #"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)"#,
        #"exceeds the limit of \d+"#,
        #"exceeds the available context size"#,
        #"greater than the context length"#,
        #"context window exceeds limit"#,
        #"exceeded model token limit"#,
        #"too large for model with \d+ maximum context length"#,
        #"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?"#,
        #"model_context_window_exceeded"#,
        #"prompt too long; exceeded (?:max )?context length"#,
        #"range of input length should be"#,
        #"context[_ ]length[_ ]exceeded"#,
        #"too many tokens"#,
        #"token limit exceeded"#,
        #"^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))"#,
].map { try! NSRegularExpression(pattern: $0, options: .caseInsensitive) }

private let nonOverflowPatterns: [NSRegularExpression] = [
        #"^(Throttling error|Service unavailable):"#,
        #"rate limit"#,
        #"too many requests"#,
].map { try! NSRegularExpression(pattern: $0, options: .caseInsensitive) }

private let cerebrasBodylessOverflowPattern = try! NSRegularExpression(
    pattern: #"^4(?:00|13)\s*(?:status code)?\s*\(no body\)"#, options: .caseInsensitive)

extension LLM {
    public static func getOverflowPatterns() -> [NSRegularExpression] { overflowPatterns }

    public static func isContextOverflow(_ message: AssistantMessage, contextWindow: Int? = nil) -> Bool {
        if message.stopReason == .error, let error = message.errorMessage, !error.isEmpty {
            let range = NSRange(error.startIndex..., in: error)
            if !nonOverflowPatterns.contains(where: { $0.firstMatch(in: error, range: range) != nil }) {
                if overflowPatterns.contains(where: { $0.firstMatch(in: error, range: range) != nil }) { return true }
                if message.provider == "cerebras", cerebrasBodylessOverflowPattern.firstMatch(in: error, range: range) != nil { return true }
            }
        }
        if let window = contextWindow, window != 0 {
            let input = message.usage.input + message.usage.cacheRead
            if message.stopReason == .stop && input > window { return true }
            if message.stopReason == .length && message.usage.output == 0 && Double(input) >= Double(window) * 0.99 { return true }
        }
        return false
    }

    public static func isRecoverableLength(_ message: AssistantMessage, desiredMaxOutput: Int) -> Bool {
        message.stopReason == .length && desiredMaxOutput > 0 && message.usage.output < desiredMaxOutput
    }
}
