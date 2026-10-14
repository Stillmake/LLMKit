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

- **One conversation format.** `Message` is `.system`, `.user`, `.assistant` or `.toolResult`. Assistant content is text, thinking and tool calls. It is `Codable` and is what you store.
- **Switching models mid-conversation.** Provider-opaque data (Anthropic thinking signatures, OpenAI encrypted reasoning items, Gemini thought signatures, OpenRouter reasoning details) stays attached to the message and its source model. It is replayed only to that model. For any other model, thinking becomes plain text, tool-call ids are rewritten to the target's rules, errored turns are dropped, and orphaned tool calls get synthetic results.
- **One event stream.** Every adapter emits `start`, `text*/thinking*/toolCall*` start-delta-end triples, then `done` or `error`. Deltas carry only new text. Tool-call deltas also carry a best-effort parse of the arguments so far.
- **Parameters.** Common settings are typed: `temperature`, `maxTokens`, `reasoning`, `toolChoice`, `cacheRetention`. Adapters translate them into each format's fields. Thinking levels are clamped to what the model supports (`Model.supportedThinkingLevels`). `samplingParams` merges into OpenAI-style bodies over the model's defaults. `extraBody` deep-merges into every adapter's body, and `onPayload` can rewrite the final request.
- **Compatibility as data.** `ModelCompat` covers the developer vs. system role, the max-tokens field name, reasoning-format variants, usage in streams, strict tools, cache-control conventions, session-affinity headers, and more. Defaults are detected from the provider and URL, and the catalog overrides them.
- **Errors never throw from `stream`.** A failure ends the stream with `.error(message)`, where `stopReason` is `.error` or `.aborted`. Cancelling the consuming task cancels the request. Requests that fail before streaming starts are retried (408, 409, 429, 5xx, network errors), honoring `retry-after`.

`ModelCatalog.builtin.providerInfo` gives each provider's display name and models.dev logo id (`https://models.dev/logos/<logo>.svg`).

API keys come from `StreamOptions.apiKey`, from a stored credential (see below), or from the provider's environment variable (`APIKeyEnvironment.apiKeyVariables`).

## Subscription sign-in

`Credentials` runs a provider's OAuth login, stores the result in a `CredentialStore`, and resolves request auth, refreshing tokens that expire within five minutes under the store's per-provider lock. `UserDefaultsCredentialStore` keeps each credential as JSON in the `UserDefaults` you pass, in the same shape as pi's `auth.json`.

```swift
let credentials = Credentials(store: UserDefaultsCredentialStore(defaults: .standard))
try await credentials.login("xai", interaction: AuthInteraction(
    prompt: { prompt in try await ui.ask(prompt) },   // text, secret, select, or paste-a-code
    notify: { event in ui.show(event) }               // auth URL, device code, progress
))
for await event in credentials.stream(model: model, context: context) { … }
```

| Provider id | Login | Flow |
|---|---|---|
| `anthropic` | Claude Pro/Max | Browser PKCE (loopback or pasted code); requests take the Claude Code identity |
| `openai-codex` | ChatGPT Plus/Pro (Codex) | Browser PKCE on `localhost:1455`, or device code |
| `openai` | Sign in with ChatGPT | Browser PKCE on `127.0.0.1:1455`; needs `LoginOptions.deviceId` |
| `github-copilot` | GitHub Copilot | Device code (github.com or Enterprise), then a Copilot token with its own endpoint |
| `xai` | SuperGrok / X Premium | Device code |
| `kimi-coding` | Kimi Code | Device code |
| `meta` | Meta Muse | Device code, then a day-long API key |
| `openrouter` | OpenRouter | Browser PKCE, exchanged for a permanent API key |

Browser flows listen on a loopback port (Network.framework) and also accept a pasted redirect URL, so `prompt` must throw when its task is cancelled: the paste prompt is cancelled when the browser callback wins. Cancelling the login task cancels the flow. LLMKit never logs in, opens a browser, or stores anything on its own.

## Scope relative to pi-ai

Ported: the five adapters above, `transform-messages`, transcript system-message replay, the compat schema and detection, thinking-level mapping and budgets, prompt caching, strict JSON-schema tools, usage and cost (including tiers and service tiers), retries, and the model catalog.

Not ported yet: Bedrock, Vertex, Azure, Mistral's native API, Codex WebSocket transport, OpenAI grammar (custom) tools, mid-conversation tool additions and effort messages, Anthropic server-side fallbacks, the Radius gateway login, and image generation and classifiers.

## Updating the catalog at runtime

`ModelCatalog.builtin` is a bundled snapshot so the package works offline. Apps fetch newer ones from pi-ai's published data on jsDelivr; LLMKit never refreshes by itself, so the app chooses when:

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

The script also reads https://models.dev/api.json once for provider names and logos; a provider models.dev lacks must be added to its `PROVIDER_OVERRIDES`.

When a provider's behavior changes upstream, review `packages/ai/src/api/` in pi-mono and port the change to the matching file under `Sources/LLMKit/API/`. Login flows live in `packages/ai/src/auth/oauth/` and map to `Sources/LLMKit/Auth/Providers/`.

## License

MIT. Portions derived from pi-ai, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
