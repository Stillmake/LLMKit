import Foundation

public enum LLMError: Error, LocalizedError, Sendable {
    case missingAPIKey(provider: String)
    case unsupportedAPI(API)
    /// The provider rejected the request.
    case http(status: Int, message: String, headers: [String: String])
    /// The stream violated the wire protocol or ended early.
    case protocolViolation(String)
    /// The provider reported a failure inside an otherwise successful stream.
    case provider(String)
    case invalidRequest(String)
    /// Login, credential storage or token refresh failed.
    case auth(String)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey(let provider): "No API key for provider: \(provider)"
        case .unsupportedAPI(let api): "No adapter for API: \(api)"
        case .http(let status, let message, _): "\(status) \(message)"
        case .protocolViolation(let message), .provider(let message), .invalidRequest(let message), .auth(let message): message
        }
    }
}

/// Environment-variable fallbacks for API keys, as in pi-ai's `env-api-keys.ts`.
public enum APIKeyEnvironment {
    public static let apiKeyVariables: [String: [String]] = [
        "anthropic": ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
        "github-copilot": ["COPILOT_GITHUB_TOKEN"],
        "ant-ling": ["ANT_LING_API_KEY"],
        "qwen-token-plan": ["QWEN_TOKEN_PLAN_API_KEY"],
        "qwen-token-plan-cn": ["QWEN_TOKEN_PLAN_CN_API_KEY"],
        "qwen-token-plan-individual": ["QWEN_TOKEN_PLAN_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY"],
        "nvidia": ["NVIDIA_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "google": ["GEMINI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"],
        "xai": ["XAI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "vercel-ai-gateway": ["AI_GATEWAY_API_KEY"],
        "zai": ["ZAI_API_KEY"],
        "zai-coding-cn": ["ZAI_CODING_CN_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
        "minimax-cn": ["MINIMAX_CN_API_KEY"],
        "moonshotai": ["MOONSHOT_API_KEY"],
        "moonshotai-cn": ["MOONSHOT_API_KEY"],
        "huggingface": ["HF_TOKEN"],
        "fireworks": ["FIREWORKS_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "baseten": ["BASETEN_API_KEY"],
        "opencode": ["OPENCODE_API_KEY"],
        "opencode-go": ["OPENCODE_API_KEY"],
        "kimi-coding": ["KIMI_API_KEY"],
        "meta": ["META_API_KEY"],
        "cloudflare-workers-ai": ["CLOUDFLARE_API_KEY"],
        "cloudflare-ai-gateway": ["CLOUDFLARE_API_KEY"],
        "xiaomi": ["XIAOMI_API_KEY"],
        "xiaomi-token-plan-cn": ["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
        "xiaomi-token-plan-ams": ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
        "xiaomi-token-plan-sgp": ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
    ]

    public static func apiKey(for provider: String) -> String? {
        let environment = ProcessInfo.processInfo.environment
        return apiKeyVariables[provider]?.lazy.compactMap { environment[$0] }.first { !$0.isEmpty }
    }
}
