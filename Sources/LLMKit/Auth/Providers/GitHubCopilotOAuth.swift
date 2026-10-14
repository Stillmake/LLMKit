import Foundation

/// GitHub Copilot device-code login (github.com or a GitHub Enterprise domain). The GitHub token is
/// stored as `refresh` and exchanged for a short-lived Copilot token (`access`), whose `proxy-ep`
/// names the account's API endpoint. Login also enables models whose policy is still unconfigured.
/// Ports pi-ai's `auth/oauth/github-copilot.ts`.
struct GitHubCopilotOAuth: OAuthProvider {
    let name = "GitHub Copilot"
    let isSubscription = true
    let loginLabel: String? = nil

    private static let clientId = String(decoding: Data(base64Encoded: "SXYxLmI1MDdhMDhjODdlY2ZlOTg=")!, as: UTF8.self)
    private static let userAgent = "GitHubCopilotChat/0.35.0"
    private static let headers = [
        "User-Agent": userAgent,
        "Editor-Version": "vscode/1.107.0",
        "Editor-Plugin-Version": "copilot-chat/0.35.0",
        "Copilot-Integration-Id": "vscode-chat",
    ]
    private static let individualBaseURL = "https://api.individual.githubcopilot.com"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let input = try await interaction.prompt(.text(message: "GitHub Enterprise URL/domain (blank for github.com)", placeholder: "company.ghe.com"))
        let enterpriseDomain = Self.normalizeDomain(input)
        if !input.trimmingCharacters(in: .whitespaces).isEmpty, enterpriseDomain == nil {
            throw LLMError.auth("Invalid GitHub Enterprise URL/domain")
        }
        let domain = enterpriseDomain ?? "github.com"

        let device = try await Self.json("https://\(domain)/login/device/code", body: .form(["client_id": Self.clientId, "scope": "read:user"]))
        guard let deviceCode = device.string("device_code"), let userCode = device.string("user_code"),
            let expiresIn = device.json["expires_in"]?.doubleValue,
            device.json["interval"] == nil || device.json["interval"]?.doubleValue != nil
        else { throw LLMError.auth("Invalid device code response fields") }
        guard let verification = OAuthHTTP.trustedURL(device.json["verification_uri"]?.stringValue) else {
            throw LLMError.auth("Untrusted verification_uri in device code response")
        }
        let interval = device.json["interval"]?.doubleValue
        interaction.notify(.deviceCode(userCode: userCode, verificationURL: verification, interval: interval, expiresIn: expiresIn))

        let githubToken: String = try await OAuthHTTP.pollDeviceCode(interval: interval, expiresIn: expiresIn, waitBeforeFirstPoll: true) {
            let response = try await Self.json("https://\(domain)/login/oauth/access_token", body: .form([
                "client_id": Self.clientId,
                "device_code": deviceCode,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            ]))
            if let token = response.json["access_token"]?.stringValue { return .complete(token) }
            guard let error = response.json["error"]?.stringValue else { return .failed("Invalid device token response") }
            switch error {
            case "authorization_pending": return .pending
            case "slow_down": return .slowDown(response.json["interval"]?.doubleValue)
            default: return .failed("Device flow failed: \(error)\(response.string("error_description").map { ": \($0)" } ?? "")")
            }
        }

        var credential = try await Self.copilotToken(githubToken, enterpriseDomain: enterpriseDomain)
        let models = try await Self.models(credential.access, enterpriseDomain: enterpriseDomain, retry: (2, .seconds(5)))
        var enabled: [String] = []
        if !models.policy.isEmpty {
            interaction.notify(.progress("Enabling models..."))
            // Some models (Claude, Grok, ...) need their policy enabled before use. Best effort:
            // a failed model is skipped, and exhausted rate limiting stops the batch.
            for modelId in models.policy {
                let response: OAuthHTTP.Response
                do {
                    response = try await Self.sendRetryingRateLimit(
                        "\(Self.baseURL(token: credential.access, enterpriseDomain: enterpriseDomain))/models/\(modelId)/policy",
                        method: "POST",
                        body: .json(["state": "enabled"]),
                        headers: Self.headers.merging([
                            "Authorization": "Bearer \(credential.access)", "openai-intent": "chat-policy", "x-interaction-type": "chat-policy",
                        ]) { $1 },
                        retry: (2, .seconds(5))
                    )
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    continue
                }
                if response.status == 429 { break }
                if response.ok { enabled.append(modelId) }
            }
        }
        var seen = Set<String>()
        credential.extra["availableModelIds"] = .array((models.available + enabled).filter { seen.insert($0).inserted }.map(JSONValue.string))
        return credential
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        let enterpriseDomain = Self.enterpriseDomain(credential)
        var refreshed = try await Self.copilotToken(credential.refresh, enterpriseDomain: enterpriseDomain)
        let models = try await Self.models(refreshed.access, enterpriseDomain: enterpriseDomain, retry: (0, .zero))
        refreshed.extra["availableModelIds"] = .array(models.available.map(JSONValue.string))
        return refreshed
    }

    /// Each credential carries its own proxy endpoint.
    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access, baseURL: Self.baseURL(token: credential.access, enterpriseDomain: Self.enterpriseDomain(credential)))
    }

    private static func normalizeDomain(_ input: String) -> String? {
        let trimmed = input.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        return URL(string: trimmed.contains("://") ? trimmed : "https://\(trimmed)")?.host
    }

    private static func enterpriseDomain(_ credential: OAuthCredential) -> String? {
        credential.extra["enterpriseUrl"]?.stringValue.flatMap(normalizeDomain)
    }

    /// The token's `proxy-ep=proxy.xxx` becomes `https://api.xxx`.
    private static func baseURL(token: String, enterpriseDomain: String?) -> String {
        if let range = token.range(of: #"proxy-ep=[^;]+"#, options: .regularExpression) {
            let host = token[range].dropFirst("proxy-ep=".count)
            return "https://" + (host.hasPrefix("proxy.") ? "api." + host.dropFirst("proxy.".count) : host)
        }
        return enterpriseDomain.map { "https://copilot-api.\($0)" } ?? individualBaseURL
    }

    private static func json(_ url: String, body: OAuthHTTP.Body) async throws -> OAuthHTTP.Response {
        let response = try await OAuthHTTP.send(url, body: body, headers: ["User-Agent": userAgent])
        guard response.ok else { throw LLMError.auth("\(response.status) \(HTTPURLResponse.localizedString(forStatusCode: response.status)): \(response.text)") }
        return response
    }

    private static func copilotToken(_ githubToken: String, enterpriseDomain: String?) async throws -> OAuthCredential {
        let response = try await OAuthHTTP.send(
            "https://api.\(enterpriseDomain ?? "github.com")/copilot_internal/v2/token",
            method: "GET",
            headers: headers.merging(["Authorization": "Bearer \(githubToken)"]) { $1 }
        )
        guard response.ok else {
            throw OAuthHTTP.failure("\(response.status) \(HTTPURLResponse.localizedString(forStatusCode: response.status)): \(response.text)", response)
        }
        guard let token = response.json["token"]?.stringValue, let expiresAt = response.json["expires_at"]?.doubleValue else {
            throw LLMError.auth("Invalid Copilot token response fields")
        }
        return OAuthCredential(
            access: token,
            refresh: githubToken,
            expires: Date(timeIntervalSince1970: expiresAt - 5 * 60),
            extra: enterpriseDomain.map { ["enterpriseUrl": .string($0)] } ?? [:]
        )
    }

    /// Retries 429 with `retry-after` or exponential backoff, within `retry.budget` overall.
    private static func sendRetryingRateLimit(
        _ url: String,
        method: String,
        body: OAuthHTTP.Body?,
        headers: [String: String],
        retry: (count: Int, budget: Duration)
    ) async throws -> OAuthHTTP.Response {
        let deadline = Date.now.addingTimeInterval(TimeInterval(retry.budget.components.seconds))
        for attempt in 0... {
            let response = try await OAuthHTTP.send(url, method: method, body: body, headers: headers, timeout: 5)
            guard response.status == 429, attempt < retry.count else { return response }
            var delay = 0.5 * Double(1 << attempt)
            if let retryAfter = response.headers.value(forHTTPHeaderField: "retry-after") {
                let formatter = DateFormatter()
                formatter.locale = Locale(identifier: "en_US_POSIX")
                formatter.timeZone = TimeZone(identifier: "GMT")
                formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
                guard let seconds = Double(retryAfter) ?? formatter.date(from: retryAfter)?.timeIntervalSinceNow else { return response }
                delay = max(0, seconds)
            }
            if delay >= deadline.timeIntervalSinceNow { return response }
            try await Task.sleep(for: .seconds(delay))
        }
        fatalError("unreachable")
    }

    /// The account's picker-enabled models, and catalog models whose policy is still unconfigured.
    private static func models(_ token: String, enterpriseDomain: String?, retry: (Int, Duration)) async throws -> (available: [String], policy: [String]) {
        let endpoint = baseURL(token: token, enterpriseDomain: enterpriseDomain)
        let response = try await sendRetryingRateLimit(
            "\(endpoint)/models",
            method: "GET",
            body: nil,
            headers: headers.merging(["Authorization": "Bearer \(token)", "X-GitHub-Api-Version": "2026-06-01"]) { $1 },
            retry: retry
        )
        guard response.ok else { throw LLMError.auth("\(response.status) \(HTTPURLResponse.localizedString(forStatusCode: response.status)): \(response.text)") }
        guard let data = response.json["data"]?.arrayValue else { throw LLMError.auth("Invalid Copilot models response") }
        let accountModels = data.compactMap { item -> (id: String, picker: Bool, policy: String?)? in
            guard let id = item["id"]?.stringValue, item["capabilities"]?["supports"]?["tool_calls"]?.boolValue != false else { return nil }
            return (id, item["model_picker_enabled"]?.boolValue == true, item["policy"]?["state"]?.stringValue)
        }
        let picker = accountModels.filter { $0.picker && $0.policy != "disabled" }.map(\.id)
        // Some Individual accounts report every picker flag false despite enabled policies; only
        // that endpoint falls back to the policy state.
        let policyFallback = endpoint == individualBaseURL && picker.isEmpty
        let available = policyFallback ? accountModels.filter { $0.policy == "enabled" }.map(\.id) : picker
        let catalog = Set(ModelCatalog.builtin.models(provider: "github-copilot").map(\.id))
        let policy = accountModels.filter { $0.policy == "unconfigured" && catalog.contains($0.id) && ($0.picker || policyFallback) }.map(\.id)
        return (available, policy)
    }
}
