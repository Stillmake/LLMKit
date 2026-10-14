import CryptoKit
import Foundation
import Network
import os

/// HTTP, PKCE, device-code polling and the loopback redirect server shared by the OAuth flows.
enum OAuthHTTP {
    /// The session login and refresh requests go through; `Credentials` binds its own.
    @TaskLocal static var session: URLSession = .shared

    struct Response: Sendable {
        var status: Int
        var text: String
        var headers: HTTPURLResponse
        /// The body as a JSON object, or `[:]` when it is not one.
        var json: JSONObject

        var ok: Bool { (200..<300).contains(status) }
        func string(_ key: String) -> String? { json[key]?.stringValue.flatMap { $0.isEmpty ? nil : $0 } }
        func positive(_ key: String) -> Double? { json[key]?.doubleValue.flatMap { $0.isFinite && $0 > 0 ? $0 : nil } }
    }

    enum Body: Sendable {
        case form([String: String])
        case json(JSONValue)
    }

    static func send(
        _ url: String,
        method: String = "POST",
        body: Body? = nil,
        headers: [String: String] = [:],
        timeout: TimeInterval = 30
    ) async throws -> Response {
        guard let url = URL(string: url) else { throw LLMError.auth("Invalid OAuth URL: \(url)") }
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        switch body {
        case .form(let fields)?:
            request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(formEncode(fields).utf8)
        case .json(let value)?:
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = value.data()
        case nil:
            break
        }
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw LLMError.auth("Non-HTTP response from \(url)") }
        return Response(
            status: http.statusCode,
            text: String(decoding: data, as: UTF8.self),
            headers: http,
            json: (try? JSONValue(data: data))?.objectValue ?? [:]
        )
    }

    /// `application/x-www-form-urlencoded` with RFC 3986 unreserved characters left as-is.
    static func formEncode(_ fields: [String: String]) -> String {
        let unreserved = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        return fields.sorted { $0.key < $1.key }
            .map { "\($0.key.addingPercentEncoding(withAllowedCharacters: unreserved)!)=\($0.value.addingPercentEncoding(withAllowedCharacters: unreserved)!)" }
            .joined(separator: "&")
    }

    static func url(_ base: String, query: [(String, String)]) -> URL {
        var components = URLComponents(string: base)!
        components.percentEncodedQuery = query.map { formEncode([$0.0: $0.1]) }.joined(separator: "&")
        return components.url!
    }

    static func randomBytes(_ count: Int) -> [UInt8] {
        var generator = SystemRandomNumberGenerator()
        return (0..<count).map { _ in generator.next() }
    }

    static func base64URL<D: DataProtocol>(_ bytes: D) -> String {
        Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// PKCE S256: a 32-byte random verifier and its SHA-256 challenge, both base64url.
    static func pkce() -> (verifier: String, challenge: String) {
        let verifier = base64URL(randomBytes(32))
        return (verifier, base64URL(Array(SHA256.hash(data: Data(verifier.utf8)))))
    }

    /// Opened in the user's browser, so only http(s) URLs are trusted.
    static func trustedURL(_ value: String?, httpsOnly: Bool = false) -> URL? {
        guard let value, let url = URL(string: value), let scheme = url.scheme?.lowercased(), url.host != nil,
            scheme == "https" || (!httpsOnly && scheme == "http")
        else { return nil }
        return url
    }

    /// Parses a pasted authorization code: a redirect URL, `code#state`, a query string, or the bare code.
    static func parseAuthorizationInput(_ input: String) -> (code: String?, state: String?) {
        let value = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return (nil, nil) }
        if let components = URLComponents(string: value), components.scheme != nil {
            return (components.queryItems?.first { $0.name == "code" }?.value, components.queryItems?.first { $0.name == "state" }?.value)
        }
        if value.contains("#") {
            let parts = value.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)
            return (String(parts[0]), parts.count > 1 ? String(parts[1]) : nil)
        }
        if value.contains("code=") {
            let items = URLComponents(string: "?" + value)?.queryItems
            return (items?.first { $0.name == "code" }?.value, items?.first { $0.name == "state" }?.value)
        }
        return (value, nil)
    }

    /// The payload of a JWT, unverified: only for reading claims the issuer just handed us.
    static func jwtClaims(_ token: String) -> JSONValue? {
        let parts = token.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        let payload = parts[1].replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        return Data(base64Encoded: payload + String(repeating: "=", count: (4 - payload.count % 4) % 4)).flatMap { try? JSONValue(data: $0) }
    }

    /// `["account": <email>]` from a token response: Anthropic's `account.email_address`, else the
    /// OpenID `id_token`'s `email` claim. Empty when the response names no account.
    static func account(_ response: Response) -> JSONObject {
        let email = response.json["account"]?["email_address"]?.stringValue
            ?? response.string("id_token").flatMap { jwtClaims($0)?["email"]?.stringValue }
        return email.map { ["account": .string($0)] } ?? [:]
    }

    static func environment(_ name: String) -> String? {
        ProcessInfo.processInfo.environment[name].flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 }
    }

    /// The loopback address OAuth callback servers listen on.
    static var callbackHost: String { environment("PI_OAUTH_CALLBACK_HOST") ?? "127.0.0.1" }

    enum DevicePoll<T: Sendable>: Sendable {
        case pending
        /// The server's new required interval, if it sent one.
        case slowDown(TimeInterval?)
        case failed(String)
        case complete(T)
    }

    /// RFC 8628 polling: 5 s default interval, `slow_down` adopts the server's interval or adds 5 s.
    /// Cancelling the task cancels the flow.
    static func pollDeviceCode<T: Sendable>(
        interval: TimeInterval?,
        expiresIn: TimeInterval?,
        waitBeforeFirstPoll: Bool = false,
        poll: () async throws -> DevicePoll<T>
    ) async throws -> T {
        let deadline = expiresIn.map { Date.now.addingTimeInterval($0) } ?? .distantFuture
        var interval = max(1, interval ?? 5)
        var slowDowns = 0
        if waitBeforeFirstPoll {
            try await Task.sleep(for: .seconds(min(interval, max(0, deadline.timeIntervalSinceNow))))
        }
        while Date.now < deadline {
            try Task.checkCancellation()
            switch try await poll() {
            case .complete(let value):
                return value
            case .failed(let message):
                throw LLMError.auth(message)
            case .slowDown(let serverInterval):
                slowDowns += 1
                interval = serverInterval.flatMap { $0.isFinite && $0 > 0 ? max(1, $0) : nil } ?? interval + 5
            case .pending:
                break
            }
            let remaining = deadline.timeIntervalSinceNow
            if remaining <= 0 { break }
            try await Task.sleep(for: .seconds(min(interval, remaining)))
        }
        throw LLMError.auth(slowDowns > 0
            ? "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in a VM; sync the clock and try again."
            : "Device flow timed out")
    }

    enum CallbackOrManual<T: Sendable>: Sendable {
        case callback(T)
        case manual(String)
    }

    /// Waits for the browser callback, or for the user to paste the code or redirect URL when the
    /// browser cannot reach the loopback server. Without a server only the prompt is used.
    static func callbackOrManualInput<T: Sendable>(
        _ interaction: AuthInteraction,
        callback: OAuthCallbackServer<T>?,
        message: String,
        placeholder: String
    ) async throws -> CallbackOrManual<T> {
        try await withThrowingTaskGroup(of: CallbackOrManual<T>?.self) { group in
            group.addTask { .manual(try await interaction.prompt(.manualCode(message: message, placeholder: placeholder))) }
            if let callback {
                group.addTask { try await callback.wait().map(CallbackOrManual.callback) }
            }
            defer {
                callback?.cancel()
                group.cancelAll()
            }
            while let result = try await group.next() {
                if let result { return result }
            }
            throw CancellationError()
        }
    }
}

/// Moves the value out, leaving `nil`, so a continuation resumes exactly once.
private func take<V>(_ value: inout V?) -> V? {
    defer { value = nil }
    return value
}

/// One-shot loopback HTTP server for OAuth redirects, on Network.framework. Answers the browser
/// with a small page once `complete` has run, so exchange failures show up there too.
final class OAuthCallbackServer<T: Sendable>: @unchecked Sendable {
    private struct State {
        var claimed = false
        var result: Result<T?, any Error>?
        var waiter: CheckedContinuation<T?, any Error>?
    }

    let redirectURI: String
    private let listener: NWListener
    private let state = OSAllocatedUnfairLock<State>(initialState: State())

    /// Starts listening on `host:port` (`0` picks a free port). `state` is checked when the
    /// provider sends one; `complete` receives the callback's query items once `code` is present.
    static func start(
        providerName: String,
        host: String,
        port: UInt16,
        path: String,
        redirectHost: String? = nil,
        state expectedState: String? = nil,
        timeout: Duration? = nil,
        complete: @escaping @Sendable ([String: String]) async throws -> T
    ) async throws -> OAuthCallbackServer {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: port == 0 ? .any : NWEndpoint.Port(rawValue: port)!)
        let listener = try NWListener(using: parameters)
        let queue = DispatchQueue(label: "LLMKit.OAuthCallbackServer")
        let ready = OSAllocatedUnfairLock<CheckedContinuation<UInt16, any Error>?>(initialState: nil)
        let boundPort = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<UInt16, any Error>) in
            ready.withLock { $0 = continuation }
            listener.stateUpdateHandler = { update in
                switch update {
                case .ready: ready.withLock(take)?.resume(returning: listener.port?.rawValue ?? port)
                case .failed(let error):
                    listener.cancel()
                    ready.withLock(take)?.resume(throwing: error)
                default: break
                }
            }
            listener.newConnectionHandler = { _ in }
            listener.start(queue: queue)
        }
        // Connections are handled on tasks the listener spawns, outside the login's task locals.
        let session = OAuthHTTP.session
        let redirect = redirectHost ?? host
        let server = OAuthCallbackServer(
            redirectURI: "http://\(redirect.contains(":") ? "[\(redirect)]" : redirect):\(boundPort)\(path)",
            listener: listener
        )
        listener.stateUpdateHandler = { [weak server] update in
            if case .failed(let error) = update { server?.finish(.failure(error)) }
        }
        listener.newConnectionHandler = { [weak server] connection in
            guard let server else { return connection.cancel() }
            connection.start(queue: queue)
            server.receive(connection, buffer: Data()) { target in
                guard let components = URLComponents(string: "http://localhost" + target), components.path == path else {
                    return (404, page("Callback route not found."))
                }
                let query = Dictionary((components.queryItems ?? []).map { ($0.name, $0.value ?? "") }) { first, _ in first }
                if let expectedState, query["state"] != expectedState { return (400, page("State mismatch.")) }
                let claimed = server.state.withLock { state in
                    guard !state.claimed, state.result == nil else { return false }
                    state.claimed = true
                    return true
                }
                guard claimed else { return (409, page("This sign-in has already been handled.")) }
                if let error = query["error"] {
                    let description = query["error_description"] ?? error
                    server.finish(.failure(LLMError.auth("\(providerName) authorization failed: \(description)")))
                    return (400, page("\(providerName) authorization failed.", detail: description))
                }
                guard query["code"]?.isEmpty == false else {
                    server.state.withLock { $0.claimed = false }
                    return (400, page("Missing authorization code."))
                }
                do {
                    let value = try await OAuthHTTP.$session.withValue(session) { try await complete(query) }
                    server.finish(.success(value))
                    return (200, page("Signed in to \(providerName). You may now close this page."))
                } catch {
                    server.finish(.failure(error))
                    return (502, page("\(providerName) sign-in failed.", detail: error.localizedDescription))
                }
            }
        }
        if let timeout {
            Task { [weak server] in
                try? await Task.sleep(for: timeout)
                server?.finish(.failure(LLMError.auth("\(providerName) sign-in timed out")))
            }
        }
        return server
    }

    private init(redirectURI: String, listener: NWListener) {
        self.redirectURI = redirectURI
        self.listener = listener
    }

    deinit { listener.cancel() }

    /// The value `complete` returned, or `nil` after `cancel()`. Throws when the provider redirects
    /// with an error, `complete` fails, the timeout elapses, or the waiting task is cancelled.
    func wait() async throws -> T? {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let result = state.withLock { state -> Result<T?, any Error>? in
                    if state.result == nil { state.waiter = continuation }
                    return state.result
                }
                if let result { continuation.resume(with: result) }
            }
        } onCancel: {
            finish(.failure(CancellationError()))
        }
    }

    /// Stops waiting for the browser unless a callback is already being completed.
    func cancel() {
        if !state.withLock({ $0.claimed }) { finish(.success(nil)) }
    }

    func close() {
        finish(.failure(LLMError.auth("OAuth callback server closed")))
        listener.cancel()
    }

    private func finish(_ result: Result<T?, any Error>) {
        let waiter = state.withLock { state -> CheckedContinuation<T?, any Error>? in
            guard state.result == nil else { return nil }
            state.result = result
            return take(&state.waiter)
        }
        waiter?.resume(with: result)
    }

    /// Reads up to the end of the request head, answers with `handle(requestTarget)`, then closes.
    private func receive(_ connection: NWConnection, buffer: Data, handle: @escaping @Sendable (String) async -> (Int, String)) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) { [weak self] data, _, isComplete, error in
            var buffer = buffer
            if let data { buffer.append(data) }
            guard let self, error == nil, buffer.count < 65536 else { return connection.cancel() }
            guard let head = buffer.range(of: Data("\r\n\r\n".utf8)).map({ String(decoding: buffer[..<$0.lowerBound], as: UTF8.self) }) else {
                return isComplete ? connection.cancel() : self.receive(connection, buffer: buffer, handle: handle)
            }
            let requestLine = head.prefix { $0 != "\r" }.split(separator: " ")
            Task {
                let (status, html) = requestLine.count >= 2 && requestLine[0] == "GET"
                    ? await handle(String(requestLine[1]))
                    : (404, Self.page("Callback route not found."))
                let reason = [200: "OK", 400: "Bad Request", 404: "Not Found", 409: "Conflict"][status] ?? "Bad Gateway"
                let body = Data(html.utf8)
                let response = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\n"
                    + "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
                connection.send(content: Data(response.utf8) + body, completion: .contentProcessed { _ in connection.cancel() })
            }
        }
    }

    private static func page(_ message: String, detail: String? = nil) -> String {
        let escape = { (text: String) in
            text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
        }
        return """
            <!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\
            <title>Sign-in</title><style>body{font:16px -apple-system,system-ui,sans-serif;margin:15vh auto;max-width:32rem;padding:0 1rem;\
            text-align:center}p{color:#666}</style></head><body><h1>\(escape(message))</h1>\(detail.map { "<p>\(escape($0))</p>" } ?? "")</body></html>
            """
    }
}
