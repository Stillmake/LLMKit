import Foundation

/// Accumulates one assistant response and emits its events.
///
/// Blocks are reference types so streamed text appends in place instead of copying the growing
/// message on every delta; the value-typed `AssistantMessage.content` is materialized once at the end.
final class ResponseBuilder {
    final class Block {
        enum Kind { case text, thinking, toolCall }

        let kind: Kind
        let index: Int
        /// Text, thinking, or the tool call's raw JSON arguments.
        var text: String
        var signature: String?
        var redacted = false
        var toolCallId = ""
        var toolName = ""
        var arguments: JSONObject = [:]
        var thoughtSignature: String?
        var namespace: String?
        var isClosed = false

        init(kind: Kind, index: Int, text: String) {
            self.kind = kind
            self.index = index
            self.text = text
        }

        var toolCall: ToolCall {
            ToolCall(id: toolCallId, name: toolName, arguments: arguments, thoughtSignature: thoughtSignature, namespace: namespace)
        }

        var content: AssistantContent {
            switch kind {
            case .text: .text(TextContent(text, textSignature: signature))
            case .thinking: .thinking(ThinkingContent(text, thinkingSignature: signature, redacted: redacted))
            case .toolCall: .toolCall(toolCall)
            }
        }
    }

    let model: Model
    var message: AssistantMessage
    private(set) var blocks: [Block] = []
    private let continuation: AsyncStream<AssistantMessageEvent>.Continuation
    private let startedAt = ContinuousClock.now
    private var started = false

    init(model: Model, continuation: AsyncStream<AssistantMessageEvent>.Continuation) {
        self.model = model
        self.message = AssistantMessage(model: model)
        self.continuation = continuation
    }

    var hasToolCalls: Bool { blocks.contains { $0.kind == .toolCall } }

    func start() {
        guard !started else { return }
        started = true
        continuation.yield(.start)
    }

    // MARK: Text and thinking

    @discardableResult
    func openText(_ initial: String = "", signature: String? = nil) -> Block {
        let block = append(Block(kind: .text, index: blocks.count, text: ""))
        block.signature = signature
        continuation.yield(.textStart(index: block.index))
        if !initial.isEmpty { appendText(initial, to: block) }
        return block
    }

    @discardableResult
    func openThinking(_ initial: String = "", signature: String? = nil, redacted: Bool = false) -> Block {
        let block = append(Block(kind: .thinking, index: blocks.count, text: ""))
        block.signature = signature
        block.redacted = redacted
        continuation.yield(.thinkingStart(index: block.index))
        if redacted {
            block.text = initial
        } else if !initial.isEmpty {
            appendText(initial, to: block)
        }
        return block
    }

    /// Appends to a text or thinking block.
    func appendText(_ delta: String, to block: Block) {
        assert(block.kind != .toolCall && !block.isClosed)
        guard !delta.isEmpty else { return }
        block.text += delta
        continuation.yield(block.kind == .text
            ? .textDelta(index: block.index, delta: delta)
            : .thinkingDelta(index: block.index, delta: delta))
    }

    // MARK: Tool calls

    @discardableResult
    func openToolCall(id: String, name: String, arguments: JSONObject = [:]) -> Block {
        let block = append(Block(kind: .toolCall, index: blocks.count, text: ""))
        block.toolCallId = id
        block.toolName = name
        block.arguments = arguments
        continuation.yield(.toolCallStart(index: block.index, id: id, name: name))
        return block
    }

    func appendToolArguments(_ delta: String, to block: Block) {
        assert(block.kind == .toolCall && !block.isClosed)
        guard !delta.isEmpty else { return }
        block.text += delta
        block.arguments = PartialJSON.parseObject(block.text)
        continuation.yield(.toolCallDelta(index: block.index, delta: delta, arguments: block.arguments))
    }

    /// Replaces the streamed arguments with the authoritative final JSON, emitting any unseen suffix.
    func setToolArguments(_ json: String, on block: Block) {
        if json.hasPrefix(block.text), json.count > block.text.count {
            appendToolArguments(String(json.dropFirst(block.text.count)), to: block)
        }
        block.text = json
        block.arguments = PartialJSON.parseObject(json)
    }

    // MARK: Closing

    /// Emits the authoritative `*End` event. Tool-call arguments are parsed from the streamed JSON
    /// when any arrived, otherwise the arguments given at open are kept.
    func close(_ block: Block) {
        guard !block.isClosed else { return }
        block.isClosed = true
        switch block.kind {
        case .text:
            continuation.yield(.textEnd(index: block.index, content: TextContent(block.text, textSignature: block.signature)))
        case .thinking:
            continuation.yield(.thinkingEnd(
                index: block.index,
                content: ThinkingContent(block.text, thinkingSignature: block.signature, redacted: block.redacted)
            ))
        case .toolCall:
            if !block.text.isEmpty { block.arguments = PartialJSON.parseObject(block.text) }
            continuation.yield(.toolCallEnd(index: block.index, toolCall: block.toolCall))
        }
    }

    // MARK: Termination

    /// Validates the stop reason and emits `done`.
    func complete() throws {
        try Task.checkCancellation()
        switch message.stopReason {
        case .pending:
            throw LLMError.protocolViolation("\(model.api) stream ended without a stop reason")
        case .error, .aborted:
            throw LLMError.provider(message.errorMessage ?? "Provider returned an error stop reason")
        case .stop, .length, .toolUse:
            break
        }
        blocks.forEach(close)
        message.content = blocks.map(\.content)
        message.duration = ContinuousClock.now - startedAt
        continuation.yield(.done(message))
    }

    /// Emits `error`, keeping whatever content arrived.
    func fail(_ error: Error) {
        let cancelled = Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled
        message.content = blocks.map(\.content)
        message.stopReason = cancelled ? .aborted : .error
        message.errorMessage = cancelled ? "Request was aborted" : error.localizedDescription
        message.duration = ContinuousClock.now - startedAt
        continuation.yield(.error(message))
    }

    private func append(_ block: Block) -> Block {
        blocks.append(block)
        return block
    }
}

/// The events of one assistant response. Single-consumer: iterate it, or await `result()`.
/// Cancelling the consuming task cancels the request.
public struct AssistantMessageStream: AsyncSequence, Sendable {
    public typealias Element = AssistantMessageEvent

    let events: AsyncStream<AssistantMessageEvent>

    public func makeAsyncIterator() -> AsyncStream<AssistantMessageEvent>.AsyncIterator {
        events.makeAsyncIterator()
    }

    /// The final message: the `done` message, or the `error` message with `stopReason` `.error` or `.aborted`.
    public func result() async throws -> AssistantMessage {
        for await event in events {
            switch event {
            case .done(let message), .error(let message): return message
            default: continue
            }
        }
        throw CancellationError()
    }
}
