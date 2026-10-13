import Foundation

struct ServerSentEvent: Sendable, Equatable {
    var event: String?
    var data: String
}

/// Decodes a `text/event-stream` byte stream. Lines end at LF, CR or CRLF; a blank line
/// dispatches the event; `data` lines join with LF; comments are skipped.
struct ServerSentEvents<Base: AsyncSequence>: AsyncSequence where Base.Element == UInt8 {
    typealias Element = ServerSentEvent

    let base: Base

    struct AsyncIterator: AsyncIteratorProtocol {
        var bytes: Base.AsyncIterator
        var line: [UInt8] = []
        var event: String?
        var data: [String] = []
        var skipLineFeed = false
        var finished = false

        mutating func next() async throws -> ServerSentEvent? {
            while !finished {
                guard let byte = try await bytes.next() else {
                    finished = true
                    if !line.isEmpty {
                        let last = line
                        line = []
                        if let event = process(last) { return event }
                    }
                    return dispatch()
                }
                if skipLineFeed {
                    skipLineFeed = false
                    if byte == 0x0A { continue }
                }
                guard byte == 0x0A || byte == 0x0D else {
                    line.append(byte)
                    continue
                }
                skipLineFeed = byte == 0x0D
                let current = line
                line.removeAll(keepingCapacity: true)
                if let event = process(current) { return event }
            }
            return nil
        }

        private mutating func process(_ line: [UInt8]) -> ServerSentEvent? {
            guard !line.isEmpty else { return dispatch() }
            guard line[0] != UInt8(ascii: ":") else { return nil }
            let colon = line.firstIndex(of: UInt8(ascii: ":")) ?? line.endIndex
            var valueStart = colon < line.endIndex ? colon + 1 : colon
            if valueStart < line.endIndex, line[valueStart] == UInt8(ascii: " ") { valueStart += 1 }
            let value = String(decoding: line[valueStart...], as: UTF8.self)
            switch String(decoding: line[..<colon], as: UTF8.self) {
            case "event": event = value
            case "data": data.append(value)
            default: break
            }
            return nil
        }

        private mutating func dispatch() -> ServerSentEvent? {
            defer {
                event = nil
                data = []
            }
            guard event != nil || !data.isEmpty else { return nil }
            return ServerSentEvent(event: event, data: data.joined(separator: "\n"))
        }
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(bytes: base.makeAsyncIterator())
    }
}

enum HTTP {
    /// Joins a model base URL and a path, filling `{VARIABLE}` placeholders (Cloudflare account and
    /// gateway ids) from the environment.
    static func url(_ baseURL: String, path: String) throws -> URL {
        var base = baseURL
        while base.hasSuffix("/") { base.removeLast() }
        let environment = ProcessInfo.processInfo.environment
        let placeholder = /\{([A-Z0-9_]+)\}/
        if let missing = base.matches(of: placeholder).first(where: { environment[String($0.1)] == nil }) {
            throw LLMError.invalidRequest("Set \(missing.1) to use \(baseURL)")
        }
        base = base.replacing(placeholder) { environment[String($0.1)] ?? "" }
        guard let url = URL(string: base + path) else { throw LLMError.invalidRequest("Invalid URL: \(base + path)") }
        return url
    }

    /// Builds a JSON POST. Header precedence: LLMKit defaults, adapter defaults, model headers,
    /// caller headers (where `nil` removes); names compare case-insensitively.
    static func request(
        url: URL,
        body: JSONValue,
        headers: [String: String?],
        model: Model,
        options: ResolvedOptions
    ) -> URLRequest {
        var merged: [String: (name: String, value: String)] = [:]
        let layers: [[String: String?]] = [
            ["User-Agent": "LLMKit", "Content-Type": "application/json", "Accept": "text/event-stream"],
            headers,
            (model.headers ?? [:]).mapValues(Optional.some),
            options.raw.headers,
        ]
        for layer in layers {
            for (name, value) in layer {
                merged[name.lowercased()] = value.map { (name, $0) }
            }
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = body.data()
        request.timeoutInterval = Double(options.raw.timeout.components.seconds)
        for (name, value) in merged.values {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }

    /// Sends `request` and returns the streaming body of a 2xx response. Retries 408, 409, 429,
    /// 5xx and network failures with the OpenAI/Anthropic SDK policy (honoring `x-should-retry`,
    /// `retry-after-ms` and `retry-after`); server-requested delays above `maxRetryDelay` fail at once.
    static func send(
        _ request: URLRequest,
        model: Model,
        options: ResolvedOptions,
        isRetryable: (Int, String) -> Bool = { _, _ in true }
    ) async throws -> URLSession.AsyncBytes {
        var retriesLeft = options.raw.maxRetries
        while true {
            try Task.checkCancellation()
            let failure: LLMError
            var serverDelay: Duration?
            do {
                let (bytes, response) = try await options.raw.session.bytes(for: request)
                guard let response = response as? HTTPURLResponse else {
                    throw LLMError.protocolViolation("Non-HTTP response")
                }
                if (200..<300).contains(response.statusCode) {
                    options.raw.onResponse?(response, model)
                    return bytes
                }
                var body = Data()
                for try await byte in bytes {
                    body.append(byte)
                    if body.count >= 256 * 1024 { break }
                }
                let headers = Dictionary(
                    response.allHeaderFields.map { ("\($0.key)".lowercased(), "\($0.value)") },
                    uniquingKeysWith: { $1 }
                )
                let text = String(decoding: body, as: UTF8.self)
                failure = .http(status: response.statusCode, message: errorMessage(from: text), headers: headers)
                let shouldRetry: Bool = switch headers["x-should-retry"] {
                case "true": true
                case "false": false
                default: [408, 409, 429].contains(response.statusCode) || response.statusCode >= 500
                }
                guard retriesLeft > 0, shouldRetry, isRetryable(response.statusCode, text) else { throw failure }
                if let milliseconds = headers["retry-after-ms"].flatMap(Double.init) {
                    serverDelay = .milliseconds(milliseconds)
                } else if let retryAfter = headers["retry-after"] {
                    let formatter = DateFormatter()
                    formatter.locale = Locale(identifier: "en_US_POSIX")
                    formatter.timeZone = TimeZone(identifier: "GMT")
                    formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
                    serverDelay = Double(retryAfter).map { .seconds($0) }
                        ?? formatter.date(from: retryAfter).map { .seconds(max(0, $0.timeIntervalSinceNow)) }
                }
            } catch let error as URLError where error.code != .cancelled && retriesLeft > 0 {
                failure = .provider(error.localizedDescription)
            }

            let retryIndex = options.raw.maxRetries - retriesLeft
            retriesLeft -= 1
            let delay: Duration
            if let serverDelay {
                if options.raw.maxRetryDelay > .zero, serverDelay > options.raw.maxRetryDelay {
                    throw LLMError.provider(
                        "Server requested \(serverDelay.components.seconds)s retry delay (max: \(options.raw.maxRetryDelay.components.seconds)s). \(failure.localizedDescription)"
                    )
                }
                delay = serverDelay
            } else {
                delay = .milliseconds(min(0.5 * pow(2, Double(retryIndex)), 8) * 1000 * (1 - Double.random(in: 0..<0.25)))
            }
            try await Task.sleep(for: delay)
        }
    }

    /// Extracts a readable message from a provider error body.
    static func errorMessage(from body: String) -> String {
        guard let json = try? JSONValue(parsing: body) else {
            return body.isEmpty ? "Request failed" : body
        }
        let error = json["error"]
        if let message = error?["message"]?.stringValue {
            let code = error?["code"]?.stringValue ?? error?["type"]?.stringValue ?? error?["status"]?.stringValue
            let raw = error?["metadata"]?["raw"]?.stringValue
            return [code.map { "\($0): \(message)" } ?? message, raw].compactMap { $0 }.joined(separator: "\n")
        }
        return error?.stringValue ?? json["message"]?.stringValue ?? json["detail"]?.stringValue ?? body
    }

    /// Iterates SSE `data` payloads as JSON, skipping `[DONE]` and empty keep-alives.
    static func jsonEvents(_ bytes: URLSession.AsyncBytes) -> AsyncThrowingMapSequence<
        AsyncFilterSequence<ServerSentEvents<URLSession.AsyncBytes>>, (event: String?, json: JSONValue)
    > {
        ServerSentEvents(base: bytes)
            .filter { !$0.data.isEmpty && $0.data != "[DONE]" }
            .map { event in
                do {
                    return (event.event, try JSONValue(parsing: event.data))
                } catch {
                    throw LLMError.protocolViolation("Could not parse stream event \(event.event ?? "message"): \(event.data)")
                }
            }
    }
}
