# Serving this package to DSH

Two endpoints, both backed by the same Splash server:

| port | what | sampling |
|---|---|---|
| 8027 | the Splash engine itself | greedy — its `/v1` default (`temperature 0.0, top_p 1.0, top_k 0`) |
| **8000** | `converter/sampling_proxy.py` → 8027 | **`temperature 1.0, top_p 0.95, top_k 20`** — Swift's recommendation, which the package cannot carry and DSH cannot send |

**Why 8000.** `splash hermes|claude|codex|opencode` probe `http://127.0.0.1:8000`
(`launcher.py: PORT = 8000`) and accept whatever answers `/status` and returns a single
`owned_by: splash` model from `/v1/models` — they do not read the serve lock. The proxy
relays both transparently, so those clients find it and get the right sampling with no
client-side configuration. DSH points at the same port.

The trade-off: this occupies the port `splash serve` wants. Stop the proxy first if you
ever want to run the launcher's own server.

```sh
scripts/splash-serve.sh start      # brings up both
scripts/splash-serve.sh status
scripts/splash-serve.sh stop
```

## The API key, which is the non-obvious part

`pi-ai`'s `openai-completions` adapter refuses to build a client without one:

```js
function getClientApiKey(provider, apiKey, headers) {
    if (apiKey) return apiKey;
    if (hasHeader(headers, "authorization") || ...) return "unused";
    throw new Error(`No API key for provider: ${provider}`);
}
```

The local Splash server has no authentication and ignores whatever is sent (verified:
HTTP 200 with a bogus bearer token), but the field must still be present.

**A literal `apiKey:` in `settings.yaml` does not work** — DSH resolves keys through its
credential seam (`resolveApiKey`), not from the provider block. `apiKeyEnv` names a key in
`~/.dsh/.credentials.yaml` under `refs:`, *not* an environment variable: `OMLX_API_KEY`
resolves that way while being absent from the process environment.

So, mirroring the omlx provider:

```yaml
# ~/.dsh/.credentials.yaml
refs:
  SPLASH_API_KEY: local-no-auth
```

```yaml
# ~/.dsh/settings.yaml — llm-pi-ai.providers
    Splash:
      displayName: Splashv2
      api: openai-completions
      baseURL: http://127.0.0.1:8027/v1
      apiKeyEnv: SPLASH_API_KEY
      defaultInput: [text, image]
      defaultContextWindow: 126000
      defaultMaxTokens: 126000
      models:
        - id: local/Swift-Qwen3.8-27B-Splash
          name: Swift-Qwen3.8-27B (Splash, greedy)

    SplashSampled:          # identical, but baseURL http://127.0.0.1:8028/v1
      ...
```

Changes to either file need a fresh DSH session to take effect.

## Notes

- `defaultContextWindow: 126000` is conservative — Splash reports **256K**. Compaction
  thresholds are computed against whatever the provider declares, so raising it lets
  sessions run longer before compacting.
- DSH omits `temperature` when unset (`temperature === void 0 ? {} : {...}`), which is why
  the 8027 route is greedy and the 8028 route exists.
