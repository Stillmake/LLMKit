import Foundation

/// Sessions whose strict tools Claude rejected for exceeding its limits. Later requests send
/// `strict: "prefer"` tools non-strict right away; trying strict again would only repeat the
/// rejection and rewrite the tool definitions, missing the prompt cache.
enum StrictToolLimits {
    /// Claude rejects a request whose strict tools exceed its limits with one of these messages.
    /// Observed limits: 20 strict tools, and 16 optional or union-typed parameters across all
    /// strict tools.
    private static let limitMessages = [
        "The compiled grammar is too large",
        "Too many strict tools (",
        "Schemas contains too many parameters with union types (",
    ]

    private static let state = State()

    /// Returns true when a provider error says the request's strict tools exceed its limits.
    static func isLimitError(_ message: String) -> Bool {
        limitMessages.contains { message.contains($0) }
    }

    /// Records that a session's strict tools exceeded Claude's limits. Entries are capped so
    /// library users that never clean up a session do not grow memory without bound.
    static func markExceeded(_ sessionId: String?) {
        guard let sessionId else { return }
        state.mark(sessionId)
    }

    /// Returns true when an earlier request of this session had its strict tools rejected.
    static func hasExceeded(_ sessionId: String?) -> Bool {
        guard let sessionId else { return false }
        return state.contains(sessionId)
    }

    /// Frees one session, or all sessions when `sessionId` is nil.
    static func clear(_ sessionId: String?) {
        state.clear(sessionId)
    }

    private final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var sessions: Set<String> = []
        private var order: [String] = []
        private let maximumEntries = 10_000

        func mark(_ sessionId: String) {
            lock.lock()
            defer { lock.unlock() }
            if sessions.insert(sessionId).inserted {
                order.append(sessionId)
            }
            while order.count > maximumEntries {
                sessions.remove(order.removeFirst())
            }
        }

        func contains(_ sessionId: String) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return sessions.contains(sessionId)
        }

        func clear(_ sessionId: String?) {
            lock.lock()
            defer { lock.unlock() }
            guard let sessionId else {
                sessions.removeAll()
                order.removeAll()
                return
            }
            sessions.remove(sessionId)
            order.removeAll { $0 == sessionId }
        }
    }
}
