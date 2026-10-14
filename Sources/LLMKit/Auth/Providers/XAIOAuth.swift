import Foundation

/// xAI (SuperGrok / X Premium) device-code login. The access token is the request's API key.
/// Ports pi-ai's `auth/oauth/xai.ts`.
struct XAIOAuth: OAuthProvider {
    let name = "xAI (Grok/X subscription)"
    let isSubscription = true
    let loginLabel: String? = "Sign in with SuperGrok or X Premium"

    private static let clientId = "b1a00492-073a-47ea-816f-4c329264a828"
    private static let tokenURL = "https://auth.x.ai/oauth2/token"

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let device = try await Self.post(
            "https://auth.x.ai/oauth2/device/code",
            ["client_id": Self.clientId, "scope": "openid profile email offline_access grok-cli:access api:access", "referrer": "pi"],
            action: "device authorization"
        )
        guard let deviceCode = device.string("device_code"), let userCode = device.string("user_code"),
            let verification = OAuthHTTP.trustedURL(device.string("verification_uri"), httpsOnly: true),
            let expiresIn = device.positive("expires_in")
        else { throw LLMError.auth("Invalid xAI OAuth device authorization response: \(device.text)") }
        // RFC 8628 allows interval 0; the poller's default applies to non-positive values.
        let interval = device.positive("interval")
        let complete = try device.string("verification_uri_complete").map {
            guard let url = OAuthHTTP.trustedURL($0, httpsOnly: true) else { throw LLMError.auth("Untrusted verification URI in xAI OAuth response") }
            return url
        }
        interaction.notify(.deviceCode(userCode: userCode, verificationURL: complete ?? verification, interval: interval, expiresIn: expiresIn))

        return try await OAuthHTTP.pollDeviceCode(interval: interval, expiresIn: expiresIn, waitBeforeFirstPoll: true) {
            let response = try await OAuthHTTP.send(Self.tokenURL, body: .form([
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": Self.clientId,
                "device_code": deviceCode,
            ]))
            if response.ok { return .complete(try Self.credential(from: response, previousRefresh: nil)) }
            switch response.string("error") {
            case "authorization_pending": return .pending
            case "slow_down": return .slowDown(response.positive("interval"))
            case "access_denied", "authorization_denied": return .failed("xAI device authorization was denied")
            case "expired_token": return .failed("xAI device code expired")
            default: return .failed(Self.failure("device token polling", response))
            }
        }
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        let response = try await Self.post(
            Self.tokenURL,
            ["grant_type": "refresh_token", "client_id": Self.clientId, "refresh_token": credential.refresh],
            action: "token refresh"
        )
        return try Self.credential(from: response, previousRefresh: credential.refresh)
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(apiKey: credential.access)
    }

    private static func post(_ url: String, _ fields: [String: String], action: String) async throws -> OAuthHTTP.Response {
        let response = try await OAuthHTTP.send(url, body: .form(fields))
        guard response.ok else { throw LLMError.auth(failure(action, response)) }
        return response
    }

    private static func failure(_ action: String, _ response: OAuthHTTP.Response) -> String {
        let detail = [response.string("error"), response.string("error_description")].compactMap { $0 }.joined(separator: ": ")
        return "xAI OAuth \(action) failed (HTTP \(response.status))\(detail.isEmpty ? "" : ": \(detail)")"
    }

    private static func credential(from response: OAuthHTTP.Response, previousRefresh: String?) throws -> OAuthCredential {
        guard let access = response.string("access_token") else { throw LLMError.auth("Invalid xAI OAuth response field: access_token") }
        // xAI may omit refresh_token on refresh when the token is not rotated.
        guard let refresh = response.json["refresh_token"] == nil ? previousRefresh : response.string("refresh_token") else {
            throw LLMError.auth("Invalid xAI OAuth response field: refresh_token")
        }
        let expiresIn = response.json["expires_in"] == nil ? 3600 : response.positive("expires_in")
        guard let expiresIn else { throw LLMError.auth("Invalid xAI OAuth response field: expires_in") }
        // Refresh five minutes early so a token never dies mid-request.
        return OAuthCredential(access: access, refresh: refresh, expires: .now.addingTimeInterval(expiresIn - 5 * 60))
    }
}
