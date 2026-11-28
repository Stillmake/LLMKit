#!/usr/bin/env python3
"""Pin a published pi-ai catalog and report differences against the offline snapshot.

Pass --refresh-builtin to also regenerate the SwiftPM resource. The resource
uses RFC 1951 raw DEFLATE, as expected by NSData.CompressionAlgorithm.zlib.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request
import zlib

SUPPORTED_APIS = {
    "openai-completions",
    "openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "google-generative-ai",
}

OFFLINE_FIELDS = (
    "api", "baseUrl", "compat", "contextWindow", "headers", "input", "maxTokens", "reasoning",
    "samplingParams", "samplingParamsByThinkingLevel", "thinkingLevelMap",
)

# pi-ai providers that models.dev lists under another id, or not at all: (name, models.dev logo id).
PROVIDER_OVERRIDES = {
    "ant-ling": ("Ant Ling", "bailing"),
    "fireworks": ("Fireworks AI", "fireworks-ai"),
    "kimi-coding": ("Kimi For Coding", "kimi-code-plan-global"),
    "openai-codex": ("OpenAI Codex", "openai"),
    "qwen-token-plan": ("Qwen Token Plan Team", "alibaba-token-plan"),
    "qwen-token-plan-cn": ("Qwen Token Plan Team (China)", "alibaba-token-plan"),
    "qwen-token-plan-individual": ("Qwen Token Plan (Individual)", "alibaba-token-plan"),
    "qwen-token-plan-individual-cn": ("Qwen Token Plan (Individual, China)", "alibaba-token-plan"),
    "together": ("Together AI", "togetherai"),
    "vercel-ai-gateway": ("Vercel AI Gateway", "vercel"),
    "zai-coding-cn": ("Z.AI Coding Plan (China)", "zhipuai-coding-plan"),
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=pathlib.Path, help="pi-ai package's dist/providers/data directory")
parser.add_argument("--refresh-builtin", action="store_true", help="replace the offline snapshot")
args = parser.parse_args()
source = args.source.resolve()
repository = pathlib.Path(__file__).resolve().parent.parent
resources = repository / "Sources/LLMKit/Resources"
snapshot_path = resources / "models.json.deflate"
if not snapshot_path.exists() and not args.refresh_builtin:
    sys.exit("The offline snapshot is missing. Run with --refresh-builtin to create it.")

manifest_bytes = (source / ".manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
version = json.loads((source / "../../../package.json").resolve().read_bytes())["version"]
models = []
for name, digest in manifest["files"].items():
    data = (source / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        sys.exit(f"Catalog checksum mismatch for {name} (pi-ai {version})")
    models.extend(
        model
        for api, group in json.loads(data).items()
        if api in SUPPORTED_APIS
        for model in group.values()
        if model.get("type", "chat") == "chat"
    )
models.sort(key=lambda model: (model["provider"], model["id"]))
candidate = {"version": version, "generatedAt": manifest["generatedAt"], "models": models}

if snapshot_path.exists():
    builtin = json.loads(zlib.decompress(snapshot_path.read_bytes(), wbits=-15))
else:
    builtin = {"version": None, "models": []}
before = {(model["provider"], model["id"]): model for model in builtin["models"]}
after = {(model["provider"], model["id"]): model for model in models}
added = after.keys() - before.keys()
removed = before.keys() - after.keys()
changed = {key for key in before.keys() & after.keys() if before[key] != after[key]}
print(f"Offline snapshot {builtin['version']} → pi-ai {version}: {len(added)} added, {len(removed)} removed, {len(changed)} changed")
for provider, model_id in sorted(changed):
    previous = before[(provider, model_id)]
    current = after[(provider, model_id)]
    fields = [field for field in OFFLINE_FIELDS if previous.get(field) != current.get(field)]
    if fields:
        print(f"  {provider}/{model_id}: {', '.join(fields)}")
for status, keys in (("added", added), ("removed", removed)):
    for provider, model_id in sorted(keys):
        if provider == "github-copilot":
            print(f"  {provider}/{model_id}: {status}")

# models.dev answers 403 to urllib's default User-Agent.
with urllib.request.urlopen(urllib.request.Request("https://models.dev/api.json", headers={"User-Agent": "curl/8"})) as response:
    models_dev = json.load(response)
providers_path = resources / "providers.json"
previous_providers = json.loads(providers_path.read_bytes())
pinned_providers = {model["provider"] for model in models}
bundled_providers = pinned_providers if args.refresh_builtin else {provider for provider, _ in before}
providers = {}
for provider in sorted(pinned_providers | bundled_providers):
    if provider in PROVIDER_OVERRIDES:
        name, logo = PROVIDER_OVERRIDES[provider]
    elif provider in models_dev:
        name, logo = models_dev[provider]["name"], provider
    elif provider not in pinned_providers and provider in previous_providers:
        providers[provider] = previous_providers[provider]
        continue
    else:
        sys.exit(f"No models.dev entry for provider {provider}; add it to PROVIDER_OVERRIDES")
    providers[provider] = {"name": name, "logo": logo}

outputs = {}
if args.refresh_builtin:
    data = json.dumps(candidate, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    encoder = zlib.compressobj(level=9, wbits=-15)
    outputs[snapshot_path] = encoder.compress(data) + encoder.flush()
outputs[providers_path] = (json.dumps(providers, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
pin = {"version": version, "manifestSHA256": hashlib.sha256(manifest_bytes).hexdigest()}
outputs[resources / "catalog-pin.json"] = (json.dumps(pin, indent=2) + "\n").encode("utf-8")
for path, data in outputs.items():
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
        print(f"{len(data):,} bytes -> {path.relative_to(repository)}")
