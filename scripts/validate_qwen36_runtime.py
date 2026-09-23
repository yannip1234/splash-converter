"""Smoke-test a running Splash server with a converted Qwen3.6 package.

The server must already be started with the converted target, tokenizer and
compatible draft/vision assets. This script records outputs for human review;
coherence and fine-tune identity cannot be established by keyword tests.
"""

import argparse
import json
import urllib.request
from pathlib import Path


def request(base, path, payload=None, *, api_key=None):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mlx-answers", type=Path,
                   help="optional JSON mapping case IDs to original MLX responses")
    args = p.parse_args(argv)
    cases = [
        {"id": "short", "messages": [{"role": "user", "content": "Explain why the sky is blue in one sentence."}]},
        {"id": "system", "messages": [
            {"role": "system", "content": "Answer concisely in plain English."},
            {"role": "user", "content": "Describe a practical way to organize a busy day."}]},
        {"id": "tool_call", "messages": [
            {"role": "user", "content": "Look up the current scope for project example using the provided tool."}],
         "tools": [{"type": "function", "function": {
             "name": "lookup_scope", "description": "Return authorized project scope",
             "parameters": {"type": "object", "properties": {"project": {"type": "string"}},
                            "required": ["project"]}}}],
         "tool_choice": "required"},
    ]
    report = {"status": request(args.base_url, "/status", api_key=args.api_key),
              "cases": {}}
    if args.mlx_answers:
        report["mlx_answers"] = json.loads(args.mlx_answers.read_text(encoding="utf-8"))
    for case in cases:
        payload = {"messages": case["messages"], "max_tokens": 128, "temperature": 0,
                   "stream": False}
        if "tools" in case:
            payload.update(tools=case["tools"], tool_choice=case["tool_choice"])
        response = request(args.base_url, "/v1/chat/completions", payload,
                           api_key=args.api_key)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise ValueError(f"{case['id']}: no model message")
        message = choices[0]["message"]
        if not message.get("content") and not message.get("tool_calls"):
            raise ValueError(f"{case['id']}: empty response")
        report["cases"][case["id"]] = {"request": payload, "response": response}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Runtime responses recorded: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
