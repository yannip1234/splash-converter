#!/bin/sh
# Serve the converted Swift package on the port the DSH "Splash" provider expects.
#
#   scripts/splash-serve.sh start|stop|status|log
#
# Starts two endpoints:
#   :8027  Splash directly — greedy (its /v1 default is temperature 0)
#   :8028  via converter/sampling_proxy.py — Swift's recommended temperature 1.0,
#          top_p 0.95, top_k 20, which the package cannot carry and DSH cannot send.
#          Deliberately NOT 8000: the splash launcher probes 8000 and would find
#          the proxy automatically, but 8000 is heavily contended by other dev
#          servers. Point clients at 8028 explicitly instead.
#
# DSH provider config lives in ~/.dsh/settings.yaml under llm-pi-ai.providers.Splash
# and points at http://127.0.0.1:8027/v1 with model id local/Swift-Qwen3.8-27B-Splash.
# The engine binary and server are the stock Splash install; only the package is ours.
set -eu

PORT="${SPLASH_PORT:-8027}"
PROXY_PORT="${SPLASH_PROXY_PORT:-8028}"   # sampling proxy (8000 avoided: too contended)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="${SPLASH_PACKAGE:-$ROOT/output/swift-splash}"
LIB=/opt/homebrew/opt/splash/libexec
MODEL=local/Swift-Qwen3.8-27B-Splash
LOG="$ROOT/output/reports/serve-dsh.log"

case "${1:-start}" in
  start)
    # The engine and the proxy are started independently, so `start` repairs a
    # half-up state instead of returning early when only one of them is running.
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
      echo "engine already serving on $PORT"
    else
      [ -f "$PKG/manifest.json" ] || { echo "no package at $PKG" >&2; exit 1; }
      mkdir -p "$(dirname "$LOG")"
      nohup "$LIB/python/bin/python3" -u "$LIB/server/server.py" \
        "$PKG/target" "$PKG/draft" --tokenizer "$PKG/tokenizer" \
        --model "$MODEL" --binary "$LIB/engine/splash" \
        --max-memory auto --max-context auto --port "$PORT" > "$LOG" 2>&1 &
      printf 'starting engine'
      while ! nc -z 127.0.0.1 "$PORT" 2>/dev/null; do printf '.'; sleep 2; done
      echo
      tail -1 "$LOG"
    fi
    if ! nc -z 127.0.0.1 "$PROXY_PORT" 2>/dev/null; then
      nohup "$ROOT/.venv/bin/python" -m converter.sampling_proxy \
        --listen "$PROXY_PORT" --upstream "$PORT" --quiet \
        > "$ROOT/output/reports/sampling-proxy.log" 2>&1 &
      while ! nc -z 127.0.0.1 "$PROXY_PORT" 2>/dev/null; do sleep 1; done
      echo "sampling proxy on $PROXY_PORT (temperature 1.0, top_p 0.95, top_k 20)"
    fi
    ;;
  stop)
    pkill -f "converter.sampling_proxy" 2>/dev/null && echo "proxy stopped" || true
    pkill -f "server.py.*$PKG/target" 2>/dev/null && echo stopped || echo "not running"
    ;;
  status)
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
      curl -s "http://127.0.0.1:$PORT/status" | head -c 200; echo
    else
      echo "not serving on $PORT"
    fi
    ;;
  log) tail -f "$LOG" ;;
  *) echo "usage: $0 start|stop|status|log" >&2; exit 2 ;;
esac
