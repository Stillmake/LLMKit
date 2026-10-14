import Foundation

/// Kimi Code (subscription) device-code login against `auth.kimi.com`. The access token goes in
/// an `Authorization: Bearer` header to `api.kimi.com/coding`. Ports pi-ai's `auth/oauth/kimi-coding.ts`.
struct KimiCodingOAuth: OAuthProvider {
    let name = "Kimi Code (subscription)"
    let isSubscription = true
    let loginLabel: String? = "Sign in with Kimi Code"

    private static let clientId = "17e5f671-d194-4dfb-9706-5516cb48c098"

    private static var host: String {
        let host = OAuthHTTP.environment("KIMI_CODE_OAUTH_HOST") ?? OAuthHTTP.environment("KIMI_OAUTH_HOST") ?? "https://auth.kimi.com"
        return host.hasSuffix("/") ? String(host.reversed().drop { $0 == "/" }.reversed()) : host
    }

    func login(_ interaction: AuthInteraction, options: LoginOptions) async throws -> OAuthCredential {
        let host = Self.host
        let device = try await OAuthHTTP.send("\(host)/api/oauth/device_authorization", body: .form(["client_id": Self.clientId]))
        guard device.ok else {
            throw LLMError.auth("Kimi Code device authorization failed with status \(device.status)\(device.text.isEmpty ? "" : ": \(device.text)")")
        }
        guard let deviceCode = device.json["device_code"]?.stringValue, let userCode = device.json["user_code"]?.stringValue,
            OAuthHTTP.trustedURL(device.json["verification_uri"]?.stringValue) != nil,
            let verification = OAuthHTTP.trustedURL(device.json["verification_uri_complete"]?.stringValue)
        else { throw LLMError.auth("Invalid Kimi Code device authorization response: \(device.text)") }
        let interval = device.positive("interval") ?? 5
        let expiresIn = device.positive("expires_in") ?? 15 * 60
        interaction.notify(.deviceCode(userCode: userCode, verificationURL: verification, interval: interval, expiresIn: expiresIn))

        return try await OAuthHTTP.pollDeviceCode(interval: interval, expiresIn: expiresIn, waitBeforeFirstPoll: true) {
            let response = try await OAuthHTTP.send("\(host)/api/oauth/token", body: .form([
                "client_id": Self.clientId,
                "device_code": deviceCode,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            ]))
            if response.status >= 500 {
                return .failed("Kimi Code device token request failed with status \(response.status)\(response.text.isEmpty ? "" : ": \(response.text)")")
            }
            if response.ok, response.json["access_token"]?.stringValue != nil {
                do { return .complete(try Self.credential(from: response, operation: "poll")) } catch { return .failed(error.localizedDescription) }
            }
            let error = response.json["error"]?.stringValue
            switch error {
            case "authorization_pending": return .pending
            case "slow_down": return .slowDown(response.positive("interval"))
            case "expired_token": return .failed("Kimi Code device authorization expired. Please restart login.")
            case "access_denied": return .failed("Kimi Code login was denied.")
            default:
                let description = response.string("error_description").map { ": \($0)" } ?? ""
                return .failed("Kimi Code device token request failed (status \(response.status))\(error.map { ": \($0)\(description)" } ?? "")")
            }
        }
    }

    /// Retries network failures, 429 and 5xx up to three times with exponential backoff.
    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        let host = Self.host
        var lastError: (any Error)?
        for attempt in 0...3 {
            if attempt > 0 { try await Task.sleep(for: .seconds(1 << (attempt - 1))) }
            let response: OAuthHTTP.Response
            do {
                response = try await OAuthHTTP.send("\(host)/api/oauth/token", body: .form([
                    "client_id": Self.clientId,
                    "grant_type": "refresh_token",
                    "refresh_token": credential.refresh,
                ]))
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
                continue
            }
            if response.ok { return try Self.credential(from: response, operation: "refresh") }
            // The stored credential is dead; the user must sign in again.
            if response.status == 401 || response.status == 403 || response.json["error"]?.stringValue == "invalid_grant" {
                let description = response.string("error_description").map { ": \($0)" } ?? ""
                throw LLMError.auth("Kimi Code token refresh unauthorized (status \(response.status))\(description)")
            }
            if response.status == 429 || response.status >= 500, attempt < 3 {
                lastError = LLMError.auth("Kimi Code token refresh failed with status \(response.status)")
                continue
            }
            throw LLMError.auth("Kimi Code token refresh failed with status \(response.status): \(response.text)")
        }
        throw lastError ?? LLMError.auth("Kimi Code token refresh failed")
    }

    func auth(for credential: OAuthCredential) -> ModelAuth {
        ModelAuth(headers: ["Authorization": "Bearer \(credential.access)"])
    }

    private static func credential(from response: OAuthHTTP.Response, operation: String) throws -> OAuthCredential {
        guard let access = response.string("access_token"), let refresh = response.string("refresh_token"),
            let expiresIn = response.positive("expires_in")
        else { throw LLMError.auth("Kimi Code token \(operation) response missing fields: \(response.text)") }
        return OAuthCredential(access: access, refresh: refresh, expires: .now.addingTimeInterval(expiresIn))
    }
}
