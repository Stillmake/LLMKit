import Foundation

/// OpenAI Codex (ChatGPT Plus/Pro) login for the `openai-codex` provider: browser PKCE on the
/// Codex CLI's registered `localhost:1455` redirect, or a device code for headless sessions.
/// Ports pi-ai's `auth/oauth/openai-codex.ts`.
struct OpenAICodexOAuth: OAuthProvider {
    let name = "OpenAI (ChatGPT Plus/Pro)"
    let isSubscription = true
    let loginLabel: String? = nil

    private static let clientId = "app_EMoamEEZ73f0CkXaXp7hrann"
    private static let tokenURL = "https://auth.openai.com/oauth/token"
    private static let redirectURI = "http://localhost:1455/auth/callback"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let method = try await interaction.prompt(.select(message: "Select OpenAI Codex login method:", options: [
            .init(id: "browser", label: "Browser login (default)"),
            .init(id: "device_code", label: "Device code login (headless)"),
        ]))
        switch method {
        case "browser":
            let (verifier, challenge) = OAuthHTTP.pkce()
            let state = OAuthHTTP.randomBytes(16).map { String(format: "%02x", $0) }.joined()
            // Port 1455 is shared with the Codex CLI; when it is taken, the pasted redirect URL finishes login.
            let callback = try? await OAuthCallbackServer<String>.start(
                providerName: "OpenAI",
                host: OAuthHTTP.callbackHost,
                port: 1455,
                path: "/auth/callback",
                state: state
            ) { query in query["code"]! }
            defer { callback?.close() }
            interaction.notify(.authURL(
                OAuthHTTP.url("https://auth.openai.com/oauth/authorize", query: [
                    ("response_type", "code"), ("client_id", Self.clientId), ("redirect_uri", Self.redirectURI),
                    ("scope", "openid profile email offline_access"), ("code_challenge", challenge), ("code_challenge_method", "S256"),
                    ("state", state), ("id_token_add_organizations", "true"), ("codex_cli_simplified_flow", "true"),
                    ("originator", options.agentName ?? "pi"),
                ]),
                instructions: "A browser window should open. Complete login to finish."
            ))
            let code: String
            switch try await OAuthHTTP.callbackOrManualInput(
                interaction,
                callback: callback,
                message: "Complete login in your browser, or paste the authorization code / redirect URL here:",
                placeholder: Self.redirectURI
            ) {
            case .callback(let value):
                code = value
            case .manual(let input):
                let parsed = OAuthHTTP.parseAuthorizationInput(input)
                if let parsedState = parsed.state, parsedState != state { throw LLMError.auth("State mismatch") }
                guard let value = parsed.code else { throw LLMError.auth("Missing authorization code") }
                code = value
            }
            return try await Self.token([
                "grant_type": "authorization_code", "client_id": Self.clientId, "code": code,
                "code_verifier": verifier, "redirect_uri": Self.redirectURI,
            ], operation: "exchange")

        case "device_code":
            let device = try await OAuthHTTP.send(
                "https://auth.openai.com/api/accounts/deviceauth/usercode",
                body: .json(["client_id": .string(Self.clientId)])
            )
            guard device.ok else {
                if device.status == 404 {
                    throw LLMError.auth("OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.")
                }
                throw LLMError.auth("OpenAI Codex device code request failed with status \(device.status)\(device.text.isEmpty ? "" : ": \(device.text)")")
            }
            let interval = device.json["interval"]?.doubleValue
                ?? device.json["interval"]?.stringValue.flatMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard let deviceAuthId = device.string("device_auth_id"), let userCode = device.string("user_code"),
                let interval, interval.isFinite, interval >= 0
            else { throw LLMError.auth("Invalid OpenAI Codex device code response: \(device.text)") }
            let expiresIn: TimeInterval = 15 * 60
            interaction.notify(.deviceCode(
                userCode: userCode,
                verificationURL: URL(string: "https://auth.openai.com/codex/device")!,
                interval: interval,
                expiresIn: expiresIn
            ))
            let (code, verifier): (String, String) = try await OAuthHTTP.pollDeviceCode(interval: interval, expiresIn: expiresIn) {
                let response = try await OAuthHTTP.send(
                    "https://auth.openai.com/api/accounts/deviceauth/token",
                    body: .json(["device_auth_id": .string(deviceAuthId), "user_code": .string(userCode)])
                )
                if response.ok {
                    guard let code = response.string("authorization_code"), let verifier = response.string("code_verifier") else {
                        return .failed("Invalid OpenAI Codex device auth token response: \(response.text)")
                    }
                    return .complete((code, verifier))
                }
                if response.status == 403 || response.status == 404 { return .pending }
                switch response.json["error"]?.stringValue ?? response.json["error"]?["code"]?.stringValue {
                case "deviceauth_authorization_pending": return .pending
                case "slow_down": return .slowDown(nil)
                default: return .failed("OpenAI Codex device auth failed with status \(response.status)\(response.text.isEmpty ? "" : ": \(response.text)")")
                }
            }
            return try await Self.token([
                "grant_type": "authorization_code", "client_id": Self.clientId, "code": code,
                "code_verifier": verifier, "redirect_uri": "https://auth.openai.com/deviceauth/callback",
            ], operation: "exchange")

        default:
            throw LLMError.auth("Unknown OpenAI Codex login method: \(method)")
        }
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        try await Self.token(["grant_type": "refresh_token", "refresh_token": credential.refresh, "client_id": Self.clientId], operation: "refresh")
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    /// Exchanges or refreshes, and keeps the ChatGPT account id from the access token's claims.
    private static func token(_ fields: [String: String], operation: String) async throws -> OAuthCredential {
        let response = try await OAuthHTTP.send(tokenURL, body: .form(fields))
        guard response.ok else { throw OAuthHTTP.failure("OpenAI Codex token \(operation) failed (\(response.status)): \(response.text)", response) }
        guard let access = response.string("access_token"), let refresh = response.string("refresh_token"),
            let expiresIn = response.json["expires_in"]?.doubleValue
        else { throw LLMError.auth("OpenAI Codex token \(operation) response missing fields: \(response.text)") }
        guard let accountId = OAuthHTTP.jwtClaims(access)?["https://api.openai.com/auth"]?["chatgpt_account_id"]?.stringValue,
            !accountId.isEmpty
        else { throw LLMError.auth("Failed to extract accountId from token") }
        return OAuthCredential(access: access, refresh: refresh, expires: .now.addingTimeInterval(expiresIn), extra: OAuthHTTP.account(response).merging(["accountId": .string(accountId)]) { $1 })
    }
}
