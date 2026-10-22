# AGENTS.md

- When syncing with pi-ai, record the pi-mono commit hash you ported from in `README.md` ("Syncing with pi-ai"), so the next sync can diff from that commit to find what changed.
- Run `swift build` once a change is complete, not after every intermediate edit.
- Keep the port's structure parallel to pi-ai: adapters under `Sources/LLMKit/API/` mirror `packages/ai/src/api/`, and login flows under `Sources/LLMKit/Auth/Providers/` mirror `packages/ai/src/auth/oauth/`, one file per upstream file, keeping upstream's constants and error messages.
- Regenerate the bundled catalog with `Scripts/sync-catalog.py` instead of editing `Resources/models.json` by hand.
- Each sync, port the most important offline upstream test from `packages/ai/test/` covering what the sync ported, as a suite in `Tests/LLMKitTests/` that runs under `swift test`, noting the source file and pi-mono commit in its doc comment.
