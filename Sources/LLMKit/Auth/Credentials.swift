import Foundation
import os

/// Request auth derived from a credential. Anything that cannot be expressed as an API key,
/// headers or a base URL is provider configuration, not auth. Ports pi-ai's `ModelAuth`.
public struct ModelAuth: Sendable, Hashable {
    public var apiKey: String?
    /// Applied under the request's own `StreamOptions.headers`.
    public var headers: [String: String]
    /// Replaces the model's base URL (GitHub Copilot's per-account proxy endpoint).
    public var baseURL: String?

    public init(apiKey: String? = nil, headers: [String: String] = [:], baseURL: String? = nil) {
        self.apiKey = apiKey
        self.headers = headers
        self.baseURL = baseURL
    }
}

/// An OAuth token set. `extra` keeps provider-specific fields (`accountId`, `clientId`,
/// `enterpriseUrl`, `availableModelIds`, ...).
public struct OAuthCredential: Sendable, Hashable {
    public var access: String
    public var refresh: String
    /// When the access token must be refreshed; providers already subtract their safety margin.
    public var expires: Date
    public var extra: JSONObject

    public init(access: String, refresh: String, expires: Date, extra: JSONObject = [:]) {
        self.access = access
        self.refresh = refresh
        self.expires = expires
        self.extra = extra
    }

    /// The signed-in account's email, when the token response names it (Anthropic, OpenAI, xAI).
    public var account: String? {
        extra["account"]?.stringValue
    }

    /// Models the account may use, when the provider reports them (GitHub Copilot).
    public var availableModelIds: [String]? {
        extra["availableModelIds"]?.arrayValue?.compactMap(\.stringValue)
    }
}

/// One stored credential per provider. Encodes as pi's `auth.json` entries:
/// `{"type": "api_key", "key": ...}` or `{"type": "oauth", "access", "refresh", "expires" (ms), ...extra}`.
public enum Credential: Sendable, Hashable, Codable {
    case apiKey(String)
    case oauth(OAuthCredential)

    public init(from decoder: Decoder) throws {
        let value = try JSONValue(from: decoder)
        switch value["type"]?.stringValue {
        case "api_key":
            self = .apiKey(value["key"]?.stringValue ?? "")
        case "oauth":
            guard var object = value.objectValue,
                let access = object.removeValue(forKey: "access")?.stringValue,
                let refresh = object.removeValue(forKey: "refresh")?.stringValue,
                let expires = object.removeValue(forKey: "expires")?.doubleValue
            else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Malformed OAuth credential"))
            }
            object["type"] = nil
            self = .oauth(OAuthCredential(access: access, refresh: refresh, expires: Date(timeIntervalSince1970: expires / 1000), extra: object))
        default:
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Unknown credential type"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .apiKey(let key):
            try JSONValue.object(["type": "api_key", "key": .string(key)]).encode(to: encoder)
        case .oauth(let credential):
            var object = credential.extra
            object["type"] = "oauth"
            object["access"] = .string(credential.access)
            object["refresh"] = .string(credential.refresh)
            object["expires"] = .number((credential.expires.timeIntervalSince1970 * 1000).rounded())
            try JSONValue.object(object).encode(to: encoder)
        }
    }
}

/// A prompt shown to the user during login. `prompt` returns the entered text, or the chosen
/// option's id for `select`.
public enum AuthPrompt: Sendable, Hashable {
    public struct Option: Sendable, Hashable {
        public var id: String
        public var label: String
        public var description: String?

        public init(id: String, label: String, description: String? = nil) {
            self.id = id
            self.label = label
            self.description = description
        }
    }

    case text(message: String, placeholder: String?)
    case secret(message: String, placeholder: String?)
    case select(message: String, options: [Option])
    /// Paste an authorization code or redirect URL. Raced against the loopback callback: the
    /// prompt's task is cancelled when the browser completes the sign-in first.
    case manualCode(message: String, placeholder: String?)
}

public enum AuthEvent: Sendable, Hashable {
    /// Open this URL in a browser.
    case authURL(URL, instructions: String?)
    /// Show `userCode` and send the user to `verificationURL`.
    case deviceCode(userCode: String, verificationURL: URL, interval: TimeInterval?, expiresIn: TimeInterval?)
    case progress(String)
}

/// The app's login UI. `prompt` must throw when its task is cancelled (wrap the UI wait in
/// `withTaskCancellationHandler`); cancelling the login task cancels the whole flow.
public struct AuthInteraction: Sendable {
    public var prompt: @Sendable (AuthPrompt) async throws -> String
    public var notify: @Sendable (AuthEvent) -> Void

    public init(prompt: @escaping @Sendable (AuthPrompt) async throws -> String, notify: @escaping @Sendable (AuthEvent) -> Void) {
        self.prompt = prompt
        self.notify = notify
    }
}

/// App-supplied context for login flows that need it.
public struct LoginOptions: Sendable {
    /// Stable UUID of this installation, sent to OpenAI as its agent host id ("Sign in with ChatGPT").
    public var deviceId: String?
    /// The name this app introduces itself with: OpenAI's agent name hint and the Codex originator.
    public var agentName: String?

    public init(deviceId: String? = nil, agentName: String? = nil) {
        self.deviceId = deviceId
        self.agentName = agentName
    }
}

/// A provider's OAuth login. `refresh` produces a credential; `auth(for:)` derives request auth
/// from whichever credential ends up stored. Ports pi-ai's `OAuthAuth`.
public protocol OAuthProvider: Sendable {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    var name: String { get }
    /// Whether access is backed by a provider subscription.
    var isSubscription: Bool { get }
    /// Label for the sign-in button, e.g. "Sign in with SuperGrok or X Premium".
    var loginLabel: String? { get }
    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential
    /// Exchanges the refresh token. `Credentials` runs it under the store's lock.
    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential
    func auth(for credential: OAuthCredential) throws -> ModelAuth
}

public enum OAuthProviders {
    /// Keyed by catalog provider id.
    public static let builtin: [String: any OAuthProvider] = [
        "anthropic": AnthropicOAuth(),
        "openai": OpenAIChatGPTOAuth(),
        "openai-codex": OpenAICodexOAuth(),
        "github-copilot": GitHubCopilotOAuth(),
        "openrouter": OpenRouterOAuth(),
        "kimi-coding": KimiCodingOAuth(),
        "meta": MetaOAuth(),
        "xai": XAIOAuth(),
    ]
}

/// App-owned credential storage keyed by provider id. `modify` is the only write path and must
/// serialize per provider (across processes too, where the backing store can), so a refresh that
/// rotates the refresh token runs once. `body` returns the new credential, or `nil` to leave the
/// entry unchanged; `modify` returns the credential stored afterwards.
public protocol CredentialStore: Sendable {
    func read(_ provider: String) async throws -> Credential?
    func modify(_ provider: String, _ body: @escaping @Sendable (Credential?) async throws -> Credential?) async throws -> Credential?
    func delete(_ provider: String) async throws
    /// Every provider with a stored credential.
    func providers() async throws -> [String]
}

extension CredentialStore {
    /// Probes the built-in OAuth providers, for stores that cannot list their keys. Credentials
    /// stored under any other id are not found; override when the backing store can enumerate.
    public func providers() async throws -> [String] {
        var stored: [String] = []
        for provider in OAuthProviders.builtin.keys where try await read(provider) != nil {
            stored.append(provider)
        }
        return stored
    }
}

/// Stores each provider's credential as JSON under `"<keyPrefix><provider>"` in the caller's
/// `UserDefaults`, serializing writes per provider within this process. Use one store instance
/// per defaults suite so every refresh goes through the same lock.
public actor UserDefaultsCredentialStore: CredentialStore {
    private nonisolated(unsafe) let defaults: UserDefaults
    private let keyPrefix: String
    private var tails: [String: Task<Void, Never>] = [:]

    public init(defaults: UserDefaults, keyPrefix: String = "LLMKit.credential.") {
        self.defaults = defaults
        self.keyPrefix = keyPrefix
    }

    public func read(_ provider: String) throws -> Credential? {
        try defaults.data(forKey: keyPrefix + provider).map { try JSONDecoder().decode(Credential.self, from: $0) }
    }

    public func modify(_ provider: String, _ body: @escaping @Sendable (Credential?) async throws -> Credential?) async throws -> Credential? {
        try await enqueue(provider) {
            let current = try await self.read(provider)
            let next = try await body(current)
            if let next { try await self.store(next, for: provider) }
            return next ?? current
        }
    }

    public func delete(_ provider: String) async throws {
        try await enqueue(provider) { try await self.store(nil, for: provider) }
    }

    public func providers() -> [String] {
        defaults.dictionaryRepresentation().keys.filter { $0.hasPrefix(keyPrefix) }.map { String($0.dropFirst(keyPrefix.count)) }
    }

    private func store(_ credential: Credential?, for provider: String) throws {
        if let credential {
            defaults.set(try JSONEncoder().encode(credential), forKey: keyPrefix + provider)
        } else {
            defaults.removeObject(forKey: keyPrefix + provider)
        }
    }

    /// Runs `work` after every earlier task for `provider` settles. The work is unstructured, so a
    /// cancelled caller never abandons a refresh that already rotated the token.
    private func enqueue<T: Sendable>(_ provider: String, _ work: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tails[provider]
        let task = Task {
            await previous?.value
            return try await work()
        }
        tails[provider] = Task { _ = try? await task.value }
        return try await task.value
    }
}

/// Login, logout and request-auth resolution over a `CredentialStore`. Ports pi-ai's auth
/// resolution: a stored credential owns its provider, OAuth tokens within five minutes of expiry
/// are refreshed under the store's lock, and without a stored credential the adapters fall back
/// to the provider's environment variable.
///
/// ```swift
/// let credentials = Credentials(store: UserDefaultsCredentialStore(defaults: .standard))
/// try await credentials.login("xai", interaction: AuthInteraction(prompt: { … }, notify: { … }))
/// for await event in credentials.stream(model: model, context: context) { … }
/// ```
public struct Credentials: Sendable {
    public let store: any CredentialStore
    public var providers: [String: any OAuthProvider]
    /// Carries the built-in providers' login and token-refresh requests; custom `OAuthProvider`s
    /// make their own requests. Model requests use `StreamOptions.session`.
    public var session: URLSession

    public init(store: any CredentialStore, providers: [String: any OAuthProvider] = OAuthProviders.builtin, session: URLSession = .shared) {
        self.store = store
        self.providers = providers
        self.session = session
    }

    /// Runs the provider's OAuth login and stores the result.
    public func login(_ provider: String, interaction: AuthInteraction, options: LoginOptions = LoginOptions()) async throws {
        guard let oauth = providers[provider] else { throw LLMError.auth("No OAuth login for provider \(provider)") }
        let credential = try await OAuthHTTP.$session.withValue(session) { try await oauth.login(interaction, options: options) }
        _ = try await store.modify(provider) { _ in .oauth(credential) }
    }

    /// Runs the login as a stream of what the UI should show, so the app needs no `AuthInteraction`.
    /// The stream finishes once the credential is stored and throws when the login fails; ending
    /// iteration early (or cancelling the consuming task) cancels the login.
    ///
    /// ```swift
    /// for try await update in credentials.login("xai") {
    ///     switch update {
    ///     case .event(let event, _): show(event)
    ///     case .prompt(let prompt): ask(prompt)            // answer with prompt.answer(_:)
    ///     case .promptWithdrawn(let id): dismissPrompt(id)
    ///     }
    /// }
    /// ```
    public func login(_ provider: String, options: LoginOptions = LoginOptions()) -> AsyncThrowingStream<LoginUpdate, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await login(provider, interaction: AuthInteraction(
                        prompt: { prompt in
                            let pending = PendingPrompt(prompt)
                            return try await withTaskCancellationHandler {
                                try await withCheckedThrowingContinuation { reply in
                                    if pending.install(reply) { continuation.yield(.prompt(pending)) }
                                }
                            } onCancel: {
                                if pending.withdraw() { continuation.yield(.promptWithdrawn(pending.id)) }
                            }
                        },
                        notify: { continuation.yield(.event($0, receivedAt: .now)) }
                    ), options: options)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Providers holding an OAuth credential.
    public func signedInProviders() async throws -> Set<String> {
        var signedIn: Set<String> = []
        for provider in try await store.providers() {
            if case .oauth? = try await store.read(provider) { signedIn.insert(provider) }
        }
        return signedIn
    }

    public func setAPIKey(_ key: String, for provider: String) async throws {
        _ = try await store.modify(provider) { _ in .apiKey(key) }
    }

    public func logout(_ provider: String) async throws {
        try await store.delete(provider)
    }

    /// Request auth for `provider`, refreshing an OAuth token that expires within five minutes.
    /// `nil` only when nothing is stored: a stored credential owns the provider, so one that cannot
    /// be used throws instead of letting the request fall back to an environment key.
    public func auth(for provider: String) async throws -> ModelAuth? {
        guard let stored = try await store.read(provider) else { return nil }
        switch stored {
        case .apiKey(let key):
            return ModelAuth(apiKey: key)
        case .oauth(var credential):
            guard let oauth = providers[provider] else { throw LLMError.auth("No OAuth login for provider \(provider)") }
            let expiresSoon: @Sendable (OAuthCredential) -> Bool = { Date.now.addingTimeInterval(5 * 60) >= $0.expires }
            if expiresSoon(credential) {
                // The authoritative check runs under the lock, so concurrent callers refresh once.
                // The refresh is unstructured: once started, the provider may have rotated the refresh
                // token, so a cancelled caller must not discard the result.
                let store = store
                let session = session
                let post = try await Task {
                    try await store.modify(provider) { current in
                        guard case .oauth(let current)? = current, expiresSoon(current) else { return nil }
                        do {
                            return .oauth(try await withThrowingTaskGroup(of: OAuthCredential.self) { group in
                                group.addTask {
                                    var refreshed = try await OAuthHTTP.$session.withValue(session) { try await oauth.refresh(current) }
                                    // Refresh responses rarely carry an id_token; the account stays the same.
                                    refreshed.extra["account"] = refreshed.extra["account"] ?? current.extra["account"]
                                    return refreshed
                                }
                                group.addTask {
                                    try await Task.sleep(for: .seconds(15))
                                    throw LLMError.auth("timed out")
                                }
                                defer { group.cancelAll() }
                                return try await group.next()!
                            })
                        } catch let error as OAuthHTTP.Rejected {
                            // Only the provider refusing the token means signing in again; outages,
                            // rate limits, network failures and timeouts are worth retrying.
                            throw LLMError.signInRequired(provider: provider, message: "OAuth refresh failed for \(provider): \(error.localizedDescription)")
                        } catch LLMError.auth("timed out") {
                            throw LLMError.auth("OAuth refresh timed out for \(provider)")
                        } catch {
                            throw LLMError.auth("OAuth refresh failed for \(provider): \(error.localizedDescription)")
                        }
                    }
                }.value
                guard case .oauth(let refreshed)? = post else {
                    throw LLMError.signInRequired(provider: provider, message: "Signed out of \(provider) while refreshing")
                }
                credential = refreshed
            }
            return try oauth.auth(for: credential)
        }
    }

    /// `LLM.stream` with auth resolved from the store. An explicit `options.apiKey` wins.
    public func stream(model: Model, context: Context, options: StreamOptions = StreamOptions()) -> AssistantMessageStream {
        var options = options
        options.resolveAuth = { [self] model in try await auth(for: model.provider) }
        return LLM.stream(model: model, context: context, options: options)
    }

    public func complete(model: Model, context: Context, options: StreamOptions = StreamOptions()) async throws -> AssistantMessage {
        try await stream(model: model, context: context, options: options).result()
    }
}

/// What a login shows next; see `Credentials.login(_:options:)`.
public enum LoginUpdate: Sendable {
    /// `receivedAt` anchors relative lifetimes such as a device code's `expiresIn`.
    case event(AuthEvent, receivedAt: Date)
    case prompt(PendingPrompt)
    /// The flow stopped waiting for this prompt: the browser callback won the race with a
    /// `manualCode` prompt, or the login was cancelled.
    case promptWithdrawn(UUID)
}

/// A prompt waiting for the user. Answer it once; later answers and answers after withdrawal are ignored.
public final class PendingPrompt: Sendable, Identifiable {
    private enum State {
        case waiting
        case presented(CheckedContinuation<String, any Error>)
        case settled
    }

    public let id = UUID()
    public let prompt: AuthPrompt
    private let state = OSAllocatedUnfairLock(initialState: State.waiting)

    init(_ prompt: AuthPrompt) {
        self.prompt = prompt
    }

    public func answer(_ text: String) {
        settle()?.resume(returning: text)
    }

    /// Declines the prompt; the login fails with `CancellationError`, as when pi's prompt rejects.
    public func cancel() {
        settle()?.resume(throwing: CancellationError())
    }

    /// Whether the prompt should be presented; `false` when it was withdrawn first.
    func install(_ reply: CheckedContinuation<String, any Error>) -> Bool {
        let installed = state.withLock { state in
            guard case .waiting = state else { return false }
            state = .presented(reply)
            return true
        }
        if !installed { reply.resume(throwing: CancellationError()) }
        return installed
    }

    /// Whether the prompt had been presented and still needs dismissing.
    func withdraw() -> Bool {
        let reply = state.withLock { state -> CheckedContinuation<String, any Error>? in
            defer { state = .settled }
            guard case .presented(let reply) = state else { return nil }
            return reply
        }
        reply?.resume(throwing: CancellationError())
        return reply != nil
    }

    private func settle() -> CheckedContinuation<String, any Error>? {
        state.withLock { state in
            guard case .presented(let reply) = state else { return nil }
            state = .settled
            return reply
        }
    }
}
