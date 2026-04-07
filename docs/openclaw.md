# OpenClaw Manual Configuration

If the quick setup from the [README](../README.md#deploy-with-openclaw) does not work as expected, you can manually edit `~/.openclaw/openclaw.json`.

## 1. Check the Model ID

```bash
curl http://localhost:8000/v1/models
```

Note the model `id` from the response — it must be used exactly in the config below.

## 2. Add a Custom Provider

Add the following under `models.providers` in `~/.openclaw/openclaw.json`:

```jsonc
{
  "models": {
    "mode": "merge",
    "providers": {
      "triattention": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "local",
        "auth": "api-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "<model_id>",
            "name": "TriAttention",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

> **Note**: The `id` field must exactly match the model ID returned by `GET /v1/models`.

## 3. (Optional) Create an Agent

To bind a workspace to the provider, add an agent entry:

```jsonc
{
  "agents": {
    "list": [
      {
        "id": "my-agent",
        "name": "my-agent",
        "workspace": "<path_to_workspace>",
        "model": "triattention/<model_id>"
      }
    ]
  }
}
```

## 4. Verify

```bash
openclaw agents list
openclaw agent --agent my-agent -m "Hello, world!"
```

## Remote Server

If the vLLM server is on a remote machine, set up an SSH tunnel and keep `baseUrl` as `http://localhost:8000/v1`:

```bash
ssh -L 8000:127.0.0.1:8000 <remote-host>
```
