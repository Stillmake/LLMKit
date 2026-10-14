import Foundation

/// Meta Model API device-code login. Meta splits identity from API access: the identity token is
/// exchanged for a Model API key that lives about a day. The identity token is stored as `refresh`
/// and the key as `access`, so a refresh re-mints the key. The identity token itself is not
/// renewable, so a 401/403 from the mint means the user must sign in again.
/// Ports pi-ai's `auth/oauth/meta.ts`.
struct MetaOAuth: OAuthProvider {
    let name = "Meta (Muse subscription)"
    let isSubscription = true
    let loginLabel: String? = "Sign in with Meta"

    private static let clientId = "1031625952748946"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let device = try await OAuthHTTP.send("https://auth.meta.com/oidc/device/authorization/", body: .form(["client_id": Self.clientId]))
        guard device.ok else { throw LLMError.auth("Meta device authorization failed with status \(device.status)\(Self.detail(device))") }
        guard let deviceCode = device.string("device_code"), let userCode = device.string("user_code"),
            let verification = OAuthHTTP.trustedURL(device.json["verification_uri_complete"]?.stringValue)
                ?? OAuthHTTP.trustedURL(device.json["verification_uri"]?.stringValue)
        else { throw LLMError.auth("Invalid Meta device authorization response: \(device.text)") }
        let interval = device.positive("interval")
        let expiresIn = device.positive("expires_in")
        interaction.notify(.deviceCode(userCode: userCode, verificationURL: verification, interval: interval, expiresIn: expiresIn))

        let identityToken: String = try await OAuthHTTP.pollDeviceCode(interval: interval, expiresIn: expiresIn, waitBeforeFirstPoll: true) {
            let response = try await OAuthHTTP.send("https://auth.meta.com/oidc/device/token/", body: .form([
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": deviceCode,
                "client_id": Self.clientId,
            ]))
            if response.ok, let token = response.string("access_token") { return .complete(token) }
            switch response.json["error"]?.stringValue {
            case "authorization_pending": return .pending
            case "slow_down": return .slowDown(response.positive("interval"))
            case "access_denied": return .failed("Meta login was denied.")
            case "expired_token": return .failed("Meta device authorization expired. Please restart login.")
            default: return .failed("Meta device token request failed with status \(response.status)\(Self.detail(response))")
            }
        }
        interaction.notify(.progress("Enabling Meta Model API access..."))
        return try await Self.mintAPIKey(identityToken)
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        try await Self.mintAPIKey(credential.refresh)
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    private static func mintAPIKey(_ identityToken: String) async throws -> OAuthCredential {
        let response = try await OAuthHTTP.send(
            "https://api.meta.ai/muse-code/key",
            body: .json([:]),
            headers: ["Authorization": "Bearer \(identityToken)", "x-api-version": "1.0.0"]
        )
        if response.status == 401 || response.status == 403 {
            throw OAuthHTTP.Rejected(errorDescription: "Meta session expired (status \(response.status)). Sign in again.\(detail(response))")
        }
        guard response.ok else { throw LLMError.auth("Meta API key mint failed with status \(response.status)\(detail(response))") }
        guard let apiKey = response.string("api_key") else {
            let action = OAuthHTTP.trustedURL(response.json["action_url"]?.stringValue).map { " Complete setup at \($0.absoluteString)" } ?? ""
            throw LLMError.auth("Meta did not issue an API key.\(action)")
        }
        return OAuthCredential(access: apiKey, refresh: identityToken, expires: .now.addingTimeInterval(24 * 60 * 60))
    }

    private static func detail(_ response: OAuthHTTP.Response) -> String {
        ["error_description", "detail", "message", "error"].lazy
            .compactMap { response.json[$0]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
            .map { ": \($0)" } ?? ""
    }
}
