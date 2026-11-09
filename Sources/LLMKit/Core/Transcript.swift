import CryptoKit
import Foundation

/// Transcript utilities: system-message replay, cross-model message transformation, and
/// token estimation. Ports pi-ai's `transcript.ts`, `transform-messages.ts` and `estimate.ts`.
enum Transcript {
    /// Folds `Context.systemPrompt` and `Context.tools` into a leading system message.
    static func normalize(_ context: Context) -> [Message] {
        let hasPrompt = !(context.systemPrompt ?? "").isEmpty
        guard hasPrompt || !context.tools.isEmpty else { return context.messages }
        let head = SystemMessage(context.systemPrompt ?? "", toolsAdded: context.tools, timestamp: .distantPast)
        return [.system(head)] + context.messages
    }

    static func initialSystemMessage(_ messages: [Message]) -> SystemMessage? {
        guard case .system(let message)? = messages.first else { return nil }
        return message
    }

    /// The tools available after replaying every system message in order.
    static func currentTools(_ messages: [Message]) -> [Tool] {
        var order: [String] = []
        var tools: [String: Tool] = [:]
        for case .system(let message) in messages {
            for name in message.toolsRemoved {
                tools[name] = nil
                order.removeAll { $0 == name }
            }
            for tool in message.toolsAdded {
                if tools.updateValue(tool, forKey: tool.name) == nil {
                    order.append(tool.name)
                }
            }
        }
        return order.compactMap { tools[$0] }
    }

    /// Rebuilds the transcript for models without mid-conversation system messages: the replayed
    /// system message leads and every later system message is dropped.
    static func collapseSystemMessages(_ messages: [Message]) -> [Message] {
        var content: [String] = []
        var sections: [SystemMessage.Section] = []
        var timestamp: Date?
        for case .system(let message) in messages {
            timestamp = timestamp ?? message.timestamp
            if !message.content.isEmpty { content.append(message.content) }
            for section in message.sections {
                if let index = sections.firstIndex(where: { $0.name == section.name }) {
                    if section.text == nil { sections.remove(at: index) } else { sections[index] = section }
                } else if section.text != nil {
                    sections.append(section)
                }
            }
        }
        let tools = currentTools(messages)
        let rest = messages.filter { if case .system = $0 { false } else { true } }
        guard timestamp != nil || !tools.isEmpty else { return rest }
        let head = SystemMessage(
            content.joined(separator: "\n\n"),
            sections: sections,
            toolsAdded: tools,
            timestamp: timestamp ?? .distantPast
        )
        return [.system(head)] + rest
    }

    static func resolve(_ messages: [Message], supportsMidConvoSystemMessages: Bool) -> [Message] {
        supportsMidConvoSystemMessages ? messages : collapseSystemMessages(messages)
    }

    // MARK: Cross-model transformation

    /// Prepares stored history for `model`:
    /// - replaces images with placeholders when the model has no image input;
    /// - keeps provider-opaque data (signatures, redacted thinking) only for the model that produced
    ///   it, and drops thinking from other models;
    /// - rewrites tool-call ids via `normalizeToolCallId` for foreign assistant messages, and the
    ///   matching tool results;
    /// - drops errored or aborted assistant turns;
    /// - inserts synthetic error results for tool calls left unanswered, holding back system
    ///   messages that land between a call and its results.
    static func transform(
        _ messages: [Message],
        for model: Model,
        foreignReasoning: ForeignReasoning = .drop,
        normalizeToolCallId: ((String, AssistantMessage) -> String)? = nil
    ) -> [Message] {
        var idMap: [String: String] = [:]
        let transformed: [Message] = messages.map { message in
            switch message {
            case .system:
                return message
            case .user(var user):
                guard !model.supportsImages else { return message }
                user.content = replacingImages(user.content, placeholder: "(image omitted: model does not support images)")
                return .user(user)
            case .toolResult(var result):
                if !model.supportsImages {
                    result.content = replacingImages(result.content, placeholder: "(tool image omitted: model does not support images)")
                }
                if let mapped = idMap[result.toolCallId] { result.toolCallId = mapped }
                return .toolResult(result)
            case .assistant(var assistant):
                let isSameModel = assistant.provider == model.provider && assistant.api == model.api && assistant.model == model.id
                let source = assistant
                assistant.content = assistant.content.compactMap { block in
                    switch block {
                    case .thinking(let thinking):
                        if !isSameModel {
                            guard foreignReasoning == .text, !thinking.redacted, !thinking.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                            return .text(TextContent("<prior_reasoning model=\"\(source.provider)/\(source.model)\">\n\(thinking.thinking)\n</prior_reasoning>"))
                        }
                        if thinking.redacted || thinking.thinkingSignature != nil { return block }
                        return thinking.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : block
                    case .text(let text):
                        return isSameModel ? block : .text(TextContent(text.text))
                    case .toolCall(var call):
                        guard !isSameModel else { return block }
                        call.thoughtSignature = nil
                        if let normalizeToolCallId {
                            let normalized = normalizeToolCallId(call.id, source)
                            if normalized != call.id {
                                idMap[call.id] = normalized
                                call.id = normalized
                            }
                        }
                        return .toolCall(call)
                    }
                }
                return .assistant(assistant)
            }
        }

        var result: [Message] = []
        var pendingCalls: [ToolCall] = []
        var answeredIds: Set<String> = []
        var heldSystemMessages: [Message] = []
        func closePendingCalls() {
            for call in pendingCalls where !answeredIds.contains(call.id) {
                result.append(.toolResult(ToolResultMessage(
                    toolCallId: call.id,
                    toolName: call.name,
                    content: [.text(TextContent("No result provided"))],
                    isError: true
                )))
            }
            pendingCalls = []
            answeredIds = []
            result += heldSystemMessages
            heldSystemMessages = []
        }

        for message in transformed {
            switch message {
            case .assistant(let assistant):
                closePendingCalls()
                guard assistant.stopReason != .error, assistant.stopReason != .aborted else { continue }
                let calls = assistant.toolCalls
                if !calls.isEmpty { pendingCalls = calls }
                result.append(message)
            case .toolResult(let toolResult):
                answeredIds.insert(toolResult.toolCallId)
                result.append(message)
            case .system:
                if pendingCalls.isEmpty { result.append(message) } else { heldSystemMessages.append(message) }
            case .user:
                closePendingCalls()
                result.append(message)
            }
        }
        closePendingCalls()
        return result
    }

    static func replacingImages(_ content: [UserContent], placeholder: String) -> [UserContent] {
        var result: [UserContent] = []
        for block in content {
            switch block {
            case .image:
                if case .text(let previous)? = result.last, previous.text == placeholder { continue }
                result.append(.text(TextContent(placeholder)))
            case .text:
                result.append(block)
            }
        }
        return result
    }

    // MARK: Estimation

    /// Estimates the whole transcript and compares it with the latest valid usage anchor.
    static func estimatedTokens(_ messages: [Message], for model: Model) -> (tokens: Int, isAnchored: Bool) {
        func estimate(_ message: Message) -> Int {
            var parts: [String] = []
            var imageCharacters = 0
            switch message {
            case .system(let system):
                parts.append(system.promptText)
            case .user(let user):
                for block in user.content {
                    switch block {
                    case .text(let text): parts.append(text.text)
                    case .image: imageCharacters += 4800
                    }
                }
            case .toolResult(let result):
                for block in result.content {
                    switch block {
                    case .text(let text): parts.append(text.text)
                    case .image: imageCharacters += 4800
                    }
                }
            case .assistant(let assistant):
                for block in assistant.content {
                    switch block {
                    case .text(let text): parts.append(text.text)
                    case .thinking(let thinking): parts.append(thinking.thinking)
                    case .toolCall(let call):
                        parts.append(call.name)
                        parts.append(JSONValue.object(call.arguments).jsonString())
                    }
                }
            }
            var cjk = 0
            var other = imageCharacters
            for part in parts {
                var remaining = ""
                for scalar in part.unicodeScalars {
                    switch scalar.value {
                    case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0x20000...0x2EBEF, 0x30000...0x323AF,
                        0x3040...0x30FF, 0x31F0...0x31FF, 0x1B000...0x1B16F, 0xFF66...0xFF9F,
                        0x1100...0x11FF, 0x3130...0x318F, 0xA960...0xA97F, 0xAC00...0xD7FF:
                        cjk += 1
                    default:
                        remaining.unicodeScalars.append(scalar)
                    }
                }
                other += remaining.count
            }
            return cjk + Int((Double(other) / 3.5).rounded(.up))
        }

        var latestTimestamp = Date.distantPast
        var anchor: (index: Int, tokens: Int)?
        for (index, message) in messages.enumerated() {
            // A newer prefix message inserted after a response (a compaction summary) invalidates its usage.
            if case .assistant(let assistant) = message, assistant.provider == model.provider, assistant.api == model.api, assistant.model == model.id, assistant.timestamp >= latestTimestamp,
                assistant.stopReason != .error, assistant.stopReason != .aborted
            {
                let usage = assistant.usage
                let tokens = usage.totalTokens > 0 ? usage.totalTokens : usage.input + usage.output + usage.cacheRead + usage.cacheWrite
                if tokens > 0 { anchor = (index, tokens) }
            }
            latestTimestamp = max(latestTimestamp, message.timestamp)
        }
        let tools = currentTools(messages)
        let toolTokens = tools.isEmpty ? 0 : estimate(.system(SystemMessage(JSONValue.array(tools.map { tool in
            ["name": .string(tool.name), "description": .string(tool.description), "parameters": tool.parameters]
        }).jsonString())))
        let heuristic = messages.reduce(toolTokens) { $0 + estimate($1) }
        guard let anchor else { return (heuristic, false) }
        let anchored = anchor.tokens + messages[(anchor.index + 1)...].reduce(0) { $0 + estimate($1) }
        return anchored >= heuristic ? (anchored, true) : (heuristic, false)
    }

    /// GitHub Copilot wants to know who initiated a request and whether it carries images.
    static func copilotHeaders(_ messages: [Message]) -> [String: String?] {
        let hasImages = messages.contains { message in
            let content: [UserContent] = switch message {
            case .user(let user): user.content
            case .toolResult(let result): result.content
            default: []
            }
            return content.contains { if case .image = $0 { true } else { false } }
        }
        let initiator: String = if case .user? = messages.last { "user" } else { messages.isEmpty ? "user" : "agent" }
        return [
            "X-Initiator": initiator,
            "Openai-Intent": "conversation-edits",
            "Copilot-Vision-Request": hasImages ? "true" : nil,
        ]
    }

    // MARK: Identifiers

    /// Short stable hash, for deriving provider-safe identifiers from long ones.
    static func shortHash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// Replaces characters outside `[A-Za-z0-9_-]` with `_` and truncates.
    static func sanitizeIdentifier(_ id: String, maxLength: Int = 64) -> String {
        let sanitized = String(id.unicodeScalars.map { scalar in
            scalar.isASCII && (CharacterSet.alphanumerics.contains(scalar) || scalar == "_" || scalar == "-") ? Character(scalar) : "_"
        })
        guard sanitized.count > maxLength else { return sanitized }
        let hash = shortHash(id)
        assert(maxLength > hash.count, "Identifier length must leave room for a hash")
        guard maxLength > hash.count else { return String(hash.prefix(max(0, maxLength))) }
        return String(sanitized.prefix(maxLength - hash.count - 1)) + "_" + hash
    }
}
