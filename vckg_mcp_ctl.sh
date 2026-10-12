#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${VCKG_PYTHON:-$HERE/.venv/bin/python}"
LOG_DIR="${VCKG_LOG_DIR:-/CPG/logs}"
LOG="$LOG_DIR/vckg-mcp.log"
PID_FILE="$LOG_DIR/vckg-mcp.pid"
HOST="${VCKG_HOST:-127.0.0.1}"
PORT="${VCKG_PORT:-4250}"
HEALTH="http://$HOST:$PORT/health"

mkdir -p "$LOG_DIR"
[ -x "$PY" ] || { echo "Python venv not found: $PY" >&2; exit 1; }

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
healthy() { curl -fsS --max-time 3 "$HEALTH" >/dev/null 2>&1; }

start() {
  if running && healthy; then
    echo "already running (pid $(cat "$PID_FILE"), $HEALTH ok)"
    return
  fi
  if running; then stop; fi
  echo "starting VCKG MCP on $HOST:$PORT (log: $LOG)"
  cd "$HERE"
  nohup setsid "$PY" -m vckg_mcp.server >>"$LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 60); do
    healthy && break
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || {
      echo "VCKG MCP died during startup" >&2
      tail -40 "$LOG" >&2
      exit 1
    }
    sleep 1
  done
  healthy || { echo "VCKG MCP health check timed out" >&2; tail -40 "$LOG" >&2; exit 1; }
  status
}

stop() {
  if running; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    for _ in $(seq 1 15); do running || break; sleep 1; done
    running && kill -9 "$(cat "$PID_FILE")" 2>/dev/null || true
    echo "stopped"
  else
    echo "not running"
  fi
  rm -f "$PID_FILE"
}

status() {
  if running; then echo "pid:     $(cat "$PID_FILE")"; else echo "pid:     not running"; fi
  if healthy; then
    echo "health:  ok ($HEALTH)"
    curl -fsS --max-time 3 "$HEALTH"
    echo
  else
    echo "health:  DOWN ($HEALTH)"
  fi
  echo "log:     $LOG"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) tail -n "${2:-60}" -f "$LOG" ;;
  *) echo "usage: $0 start|stop|restart|status|logs [N]" >&2; exit 1 ;;
esac
