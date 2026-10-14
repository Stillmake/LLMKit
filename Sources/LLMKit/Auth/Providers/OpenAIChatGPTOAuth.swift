import Foundation

/// "Sign in with ChatGPT" for the `openai` provider: the user's ChatGPT access token is sent
/// straight to `api.openai.com/v1`. Each login registers a new public client; OpenAI returns its
/// id in the callback, and refreshes must use it. Ports pi-ai's `auth/oauth/openai-chatgpt.ts`.
struct OpenAIChatGPTOAuth: OAuthProvider {
    let name = "OpenAI (ChatGPT subscription)"
    let isSubscription = true
    let loginLabel: String? = "Sign in with ChatGPT"

    private static let redirectURI = "http://127.0.0.1:1455/auth/callback"
    private static let resource = "https://api.openai.com/v1"
    private static let directTokenScope = "chatgpt.tokens.use.direct"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        // OpenAI identifies each installation ("agent host") by a stable `urn:uuid:` URI.
        guard let deviceId = options.deviceId, UUID(uuidString: deviceId) != nil else {
            throw LLMError.auth("Sign in with ChatGPT requires a device ID (UUID) for this installation")
        }
        let (verifier, challenge) = OAuthHTTP.pkce()
        let state = OAuthHTTP.base64URL(OAuthHTTP.randomBytes(32))
        // Without this server the browser's callback would reach whatever else holds the port, which
        // rejects it as a state mismatch; fail with a clear error instead.
        let callback: OAuthCallbackServer<(code: String, clientId: String)>
        do {
            callback = try await OAuthCallbackServer.start(
                providerName: "ChatGPT",
                host: OAuthHTTP.callbackHost,
                port: 1455,
                path: "/auth/callback",
                redirectHost: "127.0.0.1",
                state: state
            ) { query in
                guard let clientId = query["client_id"]?.trimmingCharacters(in: .whitespaces), !clientId.isEmpty else {
                    throw LLMError.auth("OpenAI OAuth registration callback did not contain an issued client ID")
                }
                return (query["code"]!, clientId)
            }
        } catch {
            throw LLMError.auth("Port 1455 is in use, probably by an unfinished login or by the Codex CLI. Cancel that login and try again.")
        }
        defer { callback.close() }

        interaction.notify(.authURL(
            OAuthHTTP.url("https://auth.openai.com/api/accounts/authorize", query: [
                ("client_id", "dynamic_agent_client"), ("agent_name_hint", options.agentName ?? "Pi"),
                ("ext_agent_host_id", "urn:uuid:\(deviceId.lowercased())"), ("response_type", "code"),
                ("redirect_uri", Self.redirectURI), ("resource", Self.resource),
                ("scope", "openid profile email offline_access resource.invoke \(Self.directTokenScope)"), ("state", state),
                ("code_challenge", challenge), ("code_challenge_method", "S256"),
                ("nonce", OAuthHTTP.base64URL(OAuthHTTP.randomBytes(32))),
            ]),
            instructions: "Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here."
        ))
        let code: String
        let clientId: String
        switch try await OAuthHTTP.callbackOrManualInput(
            interaction,
            callback: callback,
            message: "Complete login in your browser, or paste the final redirect URL here:",
            placeholder: Self.redirectURI
        ) {
        case .callback(let result):
            code = result.code
            clientId = result.clientId
        case .manual(let input):
            guard let components = URLComponents(string: input.trimmingCharacters(in: .whitespacesAndNewlines)), components.scheme != nil else {
                throw LLMError.auth("Paste the full callback URL from the browser")
            }
            guard components.scheme == "http", components.host == "127.0.0.1", components.port == 1455, components.path == "/auth/callback" else {
                throw LLMError.auth("The pasted callback URL must start with \(Self.redirectURI)")
            }
            let query = Dictionary((components.queryItems ?? []).map { ($0.name, $0.value ?? "") }) { first, _ in first }
            if let error = query["error"] { throw LLMError.auth("ChatGPT authorization failed: \(error)") }
            guard let value = query["code"], !value.isEmpty else { throw LLMError.auth("Missing authorization code") }
            guard let returnedState = query["state"], !returnedState.isEmpty else { throw LLMError.auth("Missing OAuth state") }
            guard returnedState == state else { throw LLMError.auth("OAuth state mismatch") }
            guard let issued = query["client_id"]?.trimmingCharacters(in: .whitespaces), !issued.isEmpty else {
                throw LLMError.auth("OpenAI OAuth registration callback did not contain an issued client ID")
            }
            code = value
            clientId = issued
        }

        interaction.notify(.progress("Exchanging authorization code for tokens..."))
        let response = try await Self.token([
            "grant_type": "authorization_code", "client_id": clientId, "code": code,
            "code_verifier": verifier, "redirect_uri": Self.redirectURI, "resource": Self.resource,
        ])
        // The ID token is not used, but its presence is part of the token-response contract.
        guard response.string("id_token") != nil else { throw LLMError.auth("OpenAI OAuth token response did not contain an ID token") }
        return try Self.credential(from: response, clientId: clientId)
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        guard let clientId = credential.extra["clientId"]?.stringValue?.trimmingCharacters(in: .whitespaces), !clientId.isEmpty else {
            throw LLMError.auth("Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT")
        }
        let response = try await Self.token([
            "grant_type": "refresh_token", "client_id": clientId, "refresh_token": credential.refresh, "resource": Self.resource,
        ])
        return try Self.credential(from: response, clientId: clientId)
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    private static func token(_ fields: [String: String]) async throws -> OAuthHTTP.Response {
        let response = try await OAuthHTTP.send("https://auth.openai.com/api/accounts/oauth/token", body: .form(fields))
        guard response.ok else { throw LLMError.auth("OpenAI OAuth token request failed (\(response.status)): \(response.text)") }
        return response
    }

    private static func credential(from response: OAuthHTTP.Response, clientId: String) throws -> OAuthCredential {
        for field in ["access_token", "refresh_token", "scope"] where response.json[field]?.stringValue?.trimmingCharacters(in: .whitespaces).isEmpty ?? true {
            throw LLMError.auth("OpenAI OAuth token response has invalid \(field)")
        }
        guard let expiresIn = response.positive("expires_in") else { throw LLMError.auth("OpenAI OAuth token response has invalid expires_in") }
        let scopes = response.json["scope"]!.stringValue!.split(whereSeparator: \.isWhitespace).map(String.init)
        guard scopes.contains(directTokenScope) else { throw LLMError.auth("OpenAI OAuth grant did not include \(directTokenScope)") }
        return OAuthCredential(
            access: response.json["access_token"]!.stringValue!,
            refresh: response.json["refresh_token"]!.stringValue!,
            // Three minutes early so a request never starts with a token about to expire.
            expires: .now.addingTimeInterval(expiresIn - 3 * 60),
            extra: OAuthHTTP.account(response).merging(["clientId": .string(clientId), "scopes": .array(scopes.map(JSONValue.string))]) { $1 }
        )
    }
}
