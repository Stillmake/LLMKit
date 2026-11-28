<p align="center">
  <img src="assets/banner.png" alt="LLMKit: one Swift API for 40 LLM providers" width="1280">
</p>

# LLMKit

One Swift API for streaming, tool calls, and reasoning across OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter and 30+ providers.

LLMKit is a Swift port of [pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai), the API layer of the pi coding agent. It has no dependencies (Foundation only). It uses one adapter per wire format.

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
- **Switch models mid-conversation.** Provider-opaque data (signatures, encrypted reasoning) and thinking are replayed only to their source model; other models get text and tool calls, with tool-call ids and pairing repaired.
- **One event stream.** `start`, then text/thinking/tool-call start-delta-end triples, then `done` or `error`.
- **Typed parameters.** `temperature`, `maxTokens`, `reasoning` (clamped to the model), `toolChoice`, `cacheRetention`; `extraBody` and `onPayload` reach the raw request.
- **Compatibility as data.** `ModelCompat` captures per-provider quirks, detected from provider and URL and overridden by the catalog.
- **Tool changes in Anthropic Messages.** Supported models keep the initial tools in the request and replay later additions, redefinitions, and removals in the conversation. Set the initial tools in `Context.tools` and update them through `SystemMessage.toolsAdded` and `toolsRemoved`.
- **`stream` never throws.** Failures end it with `.error`. Cancelling the task cancels the request. Pre-stream failures are retried, honoring `retry-after`. Callers can check `LLM.isContextOverflow` and then use `LLM.isRetryableAssistantError` to identify transient failures before restarting a turn.
- **Session state is bounded.** Anthropic message threads reuse a reply when the preceding conversation is unchanged; call `LLM.cleanupSessionResources(sessionId)` when a conversation ends.

Auth comes from, in order: an explicit `StreamOptions.apiKey`, the provider's stored credential (an API key or a sign-in, never both), or its environment variable. A stored sign-in owns its provider; an empty key, passed or stored, reads the environment, as in pi-ai. `ModelCatalog.builtin.providerInfo` gives display names and models.dev logo ids.

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

Ported: the five adapters above, `transform-messages`, transcript system-message replay, context token estimation and overflow utilities, the compat schema and detection, thinking-level mapping and budgets, prompt caching, Anthropic message threads, strict JSON-schema tools, usage and cost (including tiers and service tiers), retries, and the model catalog.

Not ported yet: Bedrock, Vertex, Azure, Mistral's native API, Codex WebSocket transport, OpenAI grammar (custom) tools, native tool updates in OpenAI Responses and Kimi Completions, Anthropic effort messages and server-side fallbacks, the Radius gateway login, image generation, classifiers, and assistant-call retries after streaming starts.

## Updating the catalog at runtime

`ModelCatalog.builtin` is a compressed snapshot available offline. Its `version` may lag behind `ModelCatalog.pinnedVersion`. Use `try await ModelCatalog.downloadPinned()` to fetch LLMKit's selected pi-ai catalog with SHA-256 verification of the manifest and every provider file. Apps can also follow the newest published catalog on jsDelivr:

```swift
let latest = try await ModelCatalog.latestVersion()          // one small request
if latest != catalog.version {
    catalog = try await ModelCatalog.download(version: latest) // immutable, CDN-cached files
    try catalog.write(to: cacheURL)
}
// Next launch: (try? ModelCatalog(contentsOf: cacheURL)) ?? .builtin
```

## Syncing with pi-ai

This port tracks the supported chat APIs and OAuth login flows from pi-mono tag `v1.7.0`. The selected catalog version is recorded in `Sources/LLMKit/Resources/catalog-pin.json`. LLMKit syncs periodically and can skip intermediate releases.

LLMKit release tags are independent of pi-ai versions. Starting with this release, tags use the release date as `YYYY.M.D` with no zero padding (`2026.11.15`); an LLMKit tag is not a claim of parity with a pi-ai release.

Keep the upstream sources in the git-ignored `upstream/` folder so they are fetched once and updated in place:

```sh
git clone https://github.com/badlogic/pi-mono upstream/pi-mono   # later: git -C upstream/pi-mono pull --tags
git -C upstream/pi-mono diff <ported tag> HEAD -- packages/ai/src

mkdir -p upstream/pi-ai && (cd upstream/pi-ai && npm pack @earendil-works/pi-ai@<version> && tar xzf *.tgz)
Scripts/sync-catalog.py upstream/pi-ai/package/dist/providers/data
```

The script verifies the upstream provider files, updates the pin and provider metadata, and reports changes relative to the bundled snapshot. Pass `--refresh-builtin` to also regenerate `models.json.deflate`. Refresh the bundled snapshot when an adapter fix relies on new `compat` or `thinkingLevelMap` values, when new Copilot models need login-time enablement, or when shipping important model changes for offline use. Provider names and logos come from models.dev. Add missing providers to the script's `PROVIDER_OVERRIDES`.

Update the default pi-ai version in `ClientIdentity` when porting adapter code. Upstream changes port file-for-file from `packages/ai/src/api/` to `Sources/LLMKit/API/` and from `packages/ai/src/auth/oauth/` to `Sources/LLMKit/Auth/Providers/`. Run `swift test` before committing a sync.

## License

MIT. Portions derived from pi-ai, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
