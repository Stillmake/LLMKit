# AGENTS.md

- When syncing with pi-ai, record the pi-mono release tag you ported from in `README.md` ("Syncing with pi-ai"), so the next sync can diff from that tag to find what changed.
- LLMKit release tags are independent of pi-mono versions. Use the release date as `YYYY.M.D` with no zero padding (for example `2026.11.15`); never reuse a pi-ai version number as an LLMKit tag.
- Run `swift build` once a change is complete, not after every intermediate edit.
- Keep the port's structure parallel to pi-ai: adapters under `Sources/LLMKit/API/` mirror `packages/ai/src/api/`, and login flows under `Sources/LLMKit/Auth/Providers/` mirror `packages/ai/src/auth/oauth/`, one file per upstream file, keeping upstream's constants and error messages.
- Use `Scripts/sync-catalog.py` to update the catalog pin and provider metadata. Pass `--refresh-builtin` to regenerate the bundled snapshot when offline behavior depends on catalog changes.
