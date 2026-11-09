import Foundation

public struct ContextUsageEstimate: Sendable, Hashable {
    public var tokens: Int
    public var usageTokens: Int
    public var trailingTokens: Int
    public var lastUsageIndex: Int?
}

extension LLM {
    public static func calculateContextTokens(_ usage: Usage) -> Int {
        usage.totalTokens != 0 ? usage.totalTokens : usage.input + usage.output + usage.cacheRead + usage.cacheWrite
    }

    public static func estimateMessageTokens(_ message: Message) -> Int {
        func estimate(_ chars: Int) -> Int { Int((Double(chars) / 3.5).rounded(.up)) }
        func contentChars(_ content: [UserContent]) -> Int {
            content.reduce(0) { count, block in
                switch block {
                case .text(let text): count + text.text.utf16.count
                case .image: count + 4800
                }
            }
        }
        switch message {
        case .system(let system):
            let tools: [JSONValue] = system.toolsAdded.map { tool in
                var value: JSONObject = ["name": .string(tool.name), "description": .string(tool.description), "parameters": tool.parameters]
                if let strict = tool.strict { value["strict"] = .string(strict.rawValue) }
                return .object(value)
            }
            return estimate(system.promptText.utf16.count)
                + (tools.isEmpty ? 0 : estimate(JSONValue.array(tools).jsonString().utf16.count))
                + (system.toolsRemoved.isEmpty ? 0 : estimate(JSONValue.array(system.toolsRemoved.map(JSONValue.string)).jsonString().utf16.count))
        case .user(let user): return estimate(contentChars(user.content))
        case .toolResult(let result): return estimate(contentChars(result.content))
        case .assistant(let assistant):
            return estimate(assistant.content.reduce(0) { count, block in
                switch block {
                case .text(let text): count + text.text.utf16.count
                case .thinking(let thinking): count + thinking.thinking.utf16.count
                case .toolCall(let call): count + call.name.utf16.count + JSONValue.object(call.arguments).jsonString().utf16.count
                }
            })
        }
    }

    public static func estimateContextTokens(_ context: Context) -> ContextUsageEstimate {
        estimateContextTokens(Transcript.normalize(context))
    }

    public static func estimateContextTokens(_ messages: [Message]) -> ContextUsageEstimate {
        var latestTimestamp = Date.distantPast
        var anchor: (index: Int, tokens: Int)?
        for (index, message) in messages.enumerated() {
            if case .assistant(let assistant) = message, assistant.timestamp >= latestTimestamp,
                assistant.stopReason != .error, assistant.stopReason != .aborted {
                let tokens = calculateContextTokens(assistant.usage)
                if tokens > 0 { anchor = (index, tokens) }
            }
            latestTimestamp = max(latestTimestamp, message.timestamp)
        }
        let trailing = messages.dropFirst((anchor?.index ?? -1) + 1).reduce(0) { $0 + estimateMessageTokens($1) }
        return ContextUsageEstimate(tokens: (anchor?.tokens ?? 0) + trailing, usageTokens: anchor?.tokens ?? 0,
                                    trailingTokens: trailing, lastUsageIndex: anchor?.index)
    }
}
