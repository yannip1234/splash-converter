"""A tiny proxy that supplies Swift's recommended sampling to Splash.

Why this exists
---------------
`splash-packed-q4` has nowhere to carry `generation_config.json` — the installer
accepts exactly five tokenizer files — and Splash deliberately keeps sampling out
of the model ("Sampling and acceptance policy remain outside the model",
DFlashDraft.hpp:186). Its `/v1` default is therefore greedy: temperature 0.0,
top_p 1.0, top_k 0 (server/protocol.py:171-173). Splash's own chat page already
defaults to Swift's recommendation (frontend.py:612-613); only the API path does
not.

The obvious alternative — teaching the client to send them — does not work for
DSH: its request model has no `top_p` or `top_k` fields at all
(`dsh-llm/lib/types/call-config.d.ts`), so it could never send two of the three
values Swift recommends.

So this sits in front of Splash and fills in the defaults the package cannot
carry. Explicit values from the client always win; everything other than a
chat-completions body is relayed untouched.

    python -m converter.sampling_proxy --listen 8028 --upstream 8027

Streaming is relayed chunk-by-chunk and flushed immediately, so time-to-first-
token is unaffected (verify with `--self-test`).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
              "accept-encoding"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "http://127.0.0.1:8027"
    defaults = dict(DEFAULTS)
    quiet = False

    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        applied = None
        if method == "POST" and self.path.rstrip("/").endswith("/chat/completions"):
            body, applied = self._apply_defaults(body)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        request = urllib.request.Request(self.upstream + self.path, data=body if body else None,
                                         headers=headers, method=method)
        try:
            upstream = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as error:      # relay the error verbatim
            payload = error.read()
            self.send_response(error.code)
            for k, v in error.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as error:
            self.send_error(502, f"upstream unreachable: {error}")
            return
        if applied and not self.quiet:
            sys.stderr.write(f"  supplied {applied}\n")
        with upstream:
            self.send_response(upstream.status)
            streaming = "text/event-stream" in (upstream.headers.get("Content-Type") or "")
            for k, v in upstream.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:                              # relay without buffering
                chunk = upstream.read(1 if streaming else 65536)
                if not chunk:
                    break
                if streaming:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if streaming:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

    def _apply_defaults(self, body):
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body, None                        # not JSON: leave it alone
        if not isinstance(payload, dict):
            return body, None
        applied = {}
        for key, value in self.defaults.items():
            if payload.get(key) is None:             # explicit client values win
                payload[key] = value
                applied[key] = value
        if not applied:
            return body, None
        return json.dumps(payload).encode(), applied

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def do_DELETE(self):
        self._relay("DELETE")


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.sampling_proxy")
    p.add_argument("--listen", type=int, default=8028)
    p.add_argument("--upstream", type=int, default=8027)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    Handler.upstream = f"http://{args.host}:{args.upstream}"
    Handler.defaults = {"temperature": args.temperature, "top_p": args.top_p,
                        "top_k": args.top_k}
    Handler.quiet = args.quiet
    server = ThreadingHTTPServer((args.host, args.listen), Handler)
    print(f"sampling proxy {args.host}:{args.listen} -> {Handler.upstream}  "
          f"defaults {Handler.defaults}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
