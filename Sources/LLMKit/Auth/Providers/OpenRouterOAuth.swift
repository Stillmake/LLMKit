import Foundation

/// OpenRouter PKCE login. The authorization code is exchanged for a permanent, user-controlled API
/// key, so there is nothing to refresh. The loopback callback is raced against a manual prompt for
/// sessions whose browser cannot reach it. Ports pi-ai's `auth/oauth/openrouter.ts`.
struct OpenRouterOAuth: OAuthProvider {
    let name = "OpenRouter OAuth"
    let isSubscription = false
    let loginLabel: String? = "Sign in with OpenRouter"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let (verifier, challenge) = OAuthHTTP.pkce()
        // OpenRouter sends no `state`; the random path keeps stray requests from completing the sign-in.
        let callback = try await OAuthCallbackServer<OAuthCredential>.start(
            providerName: "OpenRouter",
            host: OAuthHTTP.callbackHost,
            port: 0,
            path: "/oauth/callback/\(UUID().uuidString.lowercased())",
            timeout: .seconds(5 * 60)
        ) { query in try await Self.exchange(code: query["code"]!, verifier: verifier) }
        defer { callback.close() }

        interaction.notify(.progress("Listening for OpenRouter OAuth callback on \(callback.redirectURI)"))
        interaction.notify(.authURL(
            OAuthHTTP.url("https://openrouter.ai/auth", query: [
                ("callback_url", callback.redirectURI), ("code_challenge", challenge), ("code_challenge_method", "S256"),
            ]),
            instructions: "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."
        ))
        switch try await OAuthHTTP.callbackOrManualInput(
            interaction,
            callback: callback,
            message: "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            placeholder: callback.redirectURI
        ) {
        case .callback(let credential):
            return credential
        case .manual(let input):
            guard let code = OAuthHTTP.parseAuthorizationInput(input).code else { throw LLMError.auth("Missing authorization code") }
            interaction.notify(.progress("Exchanging authorization code for an API key..."))
            return try await Self.exchange(code: code, verifier: verifier)
        }
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        credential
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    private static func exchange(code: String, verifier: String) async throws -> OAuthCredential {
        let response = try await OAuthHTTP.send(
            "https://openrouter.ai/api/v1/auth/keys",
            body: .json(["code": .string(code), "code_verifier": .string(verifier), "code_challenge_method": "S256"])
        )
        guard response.ok else {
            let detail = response.string("error_description") ?? response.string("message") ?? response.string("error")
                ?? response.json["error"]?["message"]?.stringValue
            throw LLMError.auth("OpenRouter OAuth key exchange failed (HTTP \(response.status))\(detail.map { ": \($0)" } ?? "")")
        }
        guard let key = response.string("key") else { throw LLMError.auth("OpenRouter OAuth response carries no \"key\"") }
        return OAuthCredential(access: key, refresh: "", expires: .distantFuture)
    }
}
