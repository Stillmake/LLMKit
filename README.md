# LLMKit

One Swift API for streaming, tool calls, and reasoning across OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter and 30+ providers.

LLMKit is a Swift port of [pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai), the API layer of the pi coding agent. It has no dependencies (Foundation only). About forty providers reduce to five wire formats; each one has a single adapter, and per-provider differences are data (`ModelCompat` plus a bundled catalog of 1,235 models).

```swift
import LLMKit

let model = ModelCatalog.builtin.model(provider: "deepseek", id: "deepseek-v4-pro")!
let context = Context(
    systemPrompt: "You are concise.",
    messages: [.user(UserMessage("Why is the sky blue?"))]
)

for await event in LLM.stream(model: model, context: context, options: StreamOptions(reasoning: .high)) {
    switch event {
    case .thinkingDelta(_, let delta): print(delta, terminator: "")
    case .textDelta(_, let delta): print(delta, terminator: "")
    case .done(let message): print("\n\(message.usage.totalTokens) tokens, $\(message.usage.cost.total)")
    case .error(let message): print("\nFailed: \(message.errorMessage ?? "")")
    default: break
    }
}
```

## Wire formats

| `API` | Providers |
|---|---|
| `openai-completions` | DeepSeek, OpenRouter, Groq, Cerebras, xAI, Together, Fireworks, Moonshot, Qwen, Z.ai, MiniMax, NVIDIA, Hugging Face, Vercel AI Gateway, vLLM, llama.cpp, … |
| `openai-responses` | OpenAI, xAI, GitHub Copilot, OpenCode |
| `openai-codex-responses` | ChatGPT subscription (Codex backend), with a ChatGPT OAuth access token |
| `anthropic-messages` | Anthropic, Kimi Coding, MiniMax, OpenRouter, Fireworks, GitHub Copilot, OpenCode |
| `google-generative-ai` | Gemini |

## Design

- **One conversation format.** `Message` (`.system`, `.user`, `.assistant`, `.toolResult`) is `Codable` and is what you store.
- **Switch models mid-conversation.** Provider-opaque data (thinking signatures, encrypted reasoning) is replayed only to its source model; for others, thinking becomes text and tool calls are repaired.
- **One event stream.** `start`, then text/thinking/tool-call start-delta-end triples, then `done` or `error`.
- **Typed parameters.** `temperature`, `maxTokens`, `reasoning` (clamped to the model), `toolChoice`, `cacheRetention`; `extraBody` and `onPayload` reach the raw request.
- **Compatibility as data.** `ModelCompat` captures per-provider quirks, detected from provider and URL and overridden by the catalog.
- **`stream` never throws.** Failures end it with `.error`; cancelling the task cancels the request; pre-stream failures are retried, honoring `retry-after`.

Auth comes from, in order: an explicit `StreamOptions.apiKey` (an override, even when empty), the provider's stored credential (an API key or a sign-in, never both), or its environment variable. `ModelCatalog.builtin.providerInfo` gives display names and models.dev logo ids.

## Subscription sign-in

```swift
let credentials = Credentials(store: UserDefaultsCredentialStore(defaults: .standard))
for try await update in credentials.login("xai") {
    switch update {
    case .event(let event, _): ui.show(event)              // auth URL, device code, progress
    case .prompt(let prompt): ui.ask(prompt)               // reply with prompt.answer(_:) or prompt.cancel()
    case .promptWithdrawn(let id): ui.dismissPrompt(id)    // the browser finished first
    }
}
for await event in credentials.stream(model: model, context: context) { … }
```

- Tokens near expiry are refreshed before each request; a rejected refresh throws `LLMError.signInRequired` from `credentials.auth(for:)`.
- `signedInProviders()` lists who is signed in; `OAuthCredential.account` is the account's email when the provider returns it.
- Credentials are stored as pi's `auth.json` entries. Keychain storage is not supported, by design; implement `CredentialStore` for other backends.
- LLMKit never opens a browser or starts a login on its own. `Credentials.session` carries the built-in flows' requests.

| Provider id | Login | Flow |
|---|---|---|
| `anthropic` | Claude Pro/Max | Browser PKCE (loopback or pasted code) |
| `openai-codex` | ChatGPT Plus/Pro (Codex) | Browser PKCE on `localhost:1455`, or device code |
| `openai` | Sign in with ChatGPT | Browser PKCE on `127.0.0.1:1455`; needs `LoginOptions.deviceId` |
| `github-copilot` | GitHub Copilot | Device code (github.com or Enterprise) |
| `xai` | SuperGrok / X Premium | Device code |
| `kimi-coding` | Kimi Code | Device code |
| `meta` | Meta Muse | Device code, then a day-long API key |
| `openrouter` | OpenRouter | Browser PKCE, exchanged for a permanent API key |

## Scope relative to pi-ai

Ported: the five adapters above, `transform-messages`, transcript system-message replay, the compat schema and detection, thinking-level mapping and budgets, prompt caching, strict JSON-schema tools, usage and cost (including tiers and service tiers), retries, and the model catalog.

Not ported yet: Bedrock, Vertex, Azure, Mistral's native API, Codex WebSocket transport, OpenAI grammar (custom) tools, mid-conversation tool additions and effort messages, Anthropic server-side fallbacks, the Radius gateway login, and image generation and classifiers.

## Updating the catalog at runtime

`ModelCatalog.builtin` is a bundled snapshot. Newer ones come from pi-ai's data on jsDelivr, when the app decides:

```swift
let latest = try await ModelCatalog.latestVersion()          // one small request
if latest != catalog.version {
    catalog = try await ModelCatalog.download(version: latest) // immutable, CDN-cached files
    try catalog.write(to: cacheURL)
}
// Next launch: (try? ModelCatalog(contentsOf: cacheURL)) ?? .builtin
```

## Syncing with pi-ai

Adapters ported from pi-mono commit `7d75a0e3eb273dff753d49c066ea29fbe7b3b6f6`; OAuth login flows (`Sources/LLMKit/Auth/`) and the Claude OAuth request handling from `7efebff189ddc4077b332e1552fff899ed387d28`. The catalog comes from `@earendil-works/pi-ai` 1.2.0. To refresh it:

```sh
Scripts/sync-catalog.py <path to pi-ai>/dist/providers/data
```

Provider names and logos come from models.dev; add missing providers to the script's `PROVIDER_OVERRIDES`. Upstream changes port file-for-file: `packages/ai/src/api/` → `Sources/LLMKit/API/`, `packages/ai/src/auth/oauth/` → `Sources/LLMKit/Auth/Providers/`.

## License

MIT. Portions derived from pi-ai, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
