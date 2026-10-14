import Foundation

/// Anthropic (Claude Pro/Max) PKCE login, by loopback browser callback or by pasting the code
/// Anthropic shows. Requests with the resulting `sk-ant-oat` token take the Claude Code identity
/// in `AnthropicMessages`. Ports pi-ai's `auth/oauth/anthropic.ts`.
struct AnthropicOAuth: OAuthProvider {
    let name = "Anthropic (Claude Pro/Max)"
    let isSubscription = true
    let loginLabel: String? = nil

    private static let clientId = String(decoding: Data(base64Encoded: "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")!, as: UTF8.self)
    private static let tokenURL = "https://platform.claude.com/v1/oauth/token"
    private static let scopes = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let method = try await interaction.prompt(.select(message: "Select Anthropic login method:", options: [
            .init(id: "browser", label: "Browser login (default)"),
            .init(id: "copy_code", label: "Copy code login (headless)"),
        ]))
        let (verifier, challenge) = OAuthHTTP.pkce()
        let authorizeURL = { (redirectURI: String) in
            OAuthHTTP.url("https://claude.ai/oauth/authorize", query: [
                ("code", "true"), ("client_id", Self.clientId), ("response_type", "code"), ("redirect_uri", redirectURI),
                ("scope", Self.scopes), ("code_challenge", challenge), ("code_challenge_method", "S256"), ("state", verifier),
            ])
        }
        let parse = { (input: String) throws -> (code: String, state: String) in
            let parsed = OAuthHTTP.parseAuthorizationInput(input)
            if let state = parsed.state, state != verifier { throw LLMError.auth("OAuth state mismatch") }
            guard let code = parsed.code else { throw LLMError.auth("Missing authorization code") }
            return (code, parsed.state ?? verifier)
        }

        switch method {
        case "copy_code":
            let redirectURI = "https://platform.claude.com/oauth/code/callback"
            interaction.notify(.authURL(
                authorizeURL(redirectURI),
                instructions: "Complete login in your browser, then copy the code Anthropic shows and paste it here."
            ))
            let (code, state) = try parse(try await interaction.prompt(.manualCode(message: "Paste the code Anthropic shows after you sign in:", placeholder: "code#state")))
            interaction.notify(.progress("Exchanging authorization code for tokens..."))
            return try await Self.exchange(code: code, state: state, verifier: verifier, redirectURI: redirectURI)

        case "browser":
            // Port 53692 can be forwarded into containers or over SSH; Anthropic accepts any loopback
            // port, so a busy one falls back to a free port, and then to the pasted redirect URL.
            let start = { (port: UInt16) in
                try await OAuthCallbackServer<String>.start(
                    providerName: "Anthropic",
                    host: OAuthHTTP.callbackHost,
                    port: port,
                    path: "/callback",
                    redirectHost: "localhost",
                    state: verifier
                ) { query in query["code"]! }
            }
            var callback: OAuthCallbackServer<String>?
            do { callback = try await start(53692) } catch { callback = try? await start(0) }
            defer { callback?.close() }
            let redirectURI = callback?.redirectURI ?? "http://localhost:53692/callback"
            interaction.notify(.authURL(
                authorizeURL(redirectURI),
                instructions: "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
            ))
            let (code, state): (String, String) = switch try await OAuthHTTP.callbackOrManualInput(
                interaction,
                callback: callback,
                message: "Complete login in your browser, or paste the authorization code / redirect URL here:",
                placeholder: redirectURI
            ) {
            case .callback(let value): (value, verifier)
            case .manual(let input): try parse(input)
            }
            interaction.notify(.progress("Exchanging authorization code for tokens..."))
            return try await Self.exchange(code: code, state: state, verifier: verifier, redirectURI: redirectURI)

        default:
            throw LLMError.auth("Unknown Anthropic login method: \(method)")
        }
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        try await Self.token(["grant_type": "refresh_token", "client_id": .string(Self.clientId), "refresh_token": .string(credential.refresh)], action: "token refresh")
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    private static func exchange(code: String, state: String, verifier: String, redirectURI: String) async throws -> OAuthCredential {
        try await token([
            "grant_type": "authorization_code",
            "client_id": .string(clientId),
            "code": .string(code),
            "state": .string(state),
            "redirect_uri": .string(redirectURI),
            "code_verifier": .string(verifier),
        ], action: "token exchange")
    }

    private static func token(_ body: JSONObject, action: String) async throws -> OAuthCredential {
        let response = try await OAuthHTTP.send(tokenURL, body: .json(.object(body)))
        guard response.ok else { throw LLMError.auth("Anthropic \(action) failed. status=\(response.status); body=\(response.text)") }
        guard let access = response.string("access_token"), let refresh = response.string("refresh_token"),
            let expiresIn = response.positive("expires_in")
        else { throw LLMError.auth("Anthropic \(action) returned an invalid response: \(response.text)") }
        return OAuthCredential(access: access, refresh: refresh, expires: .now.addingTimeInterval(expiresIn - 5 * 60), extra: OAuthHTTP.account(response))
    }
}
