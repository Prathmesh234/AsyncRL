#!/bin/bash
# DisGenerator - Stop All Servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

echo "============================================"
echo "  Stopping DisGenerator Servers"
echo "============================================"

# Stop Prefill Server (kill entire process group)
if [ -f "$LOG_DIR/prefill.pid" ]; then
    PREFILL_PID=$(cat "$LOG_DIR/prefill.pid")
    if kill -0 "$PREFILL_PID" 2>/dev/null; then
        echo "Stopping Prefill Server (PGID: $PREFILL_PID)..."
        # Use negative PID to kill the process group
        kill -TERM -- -"$PREFILL_PID" 2>/dev/null || kill "$PREFILL_PID" 2>/dev/null || true
        rm "$LOG_DIR/prefill.pid"
    else
        echo "Prefill Server not running"
        rm -f "$LOG_DIR/prefill.pid"
    fi
fi

# Stop Decode Server (kill entire process group)
if [ -f "$LOG_DIR/decode.pid" ]; then
    DECODE_PID=$(cat "$LOG_DIR/decode.pid")
    if kill -0 "$DECODE_PID" 2>/dev/null; then
        echo "Stopping Decode Server (PGID: $DECODE_PID)..."
        # Use negative PID to kill the process group
        kill -TERM -- -"$DECODE_PID" 2>/dev/null || kill "$DECODE_PID" 2>/dev/null || true
        rm "$LOG_DIR/decode.pid"
    else
        echo "Decode Server not running"
        rm -f "$LOG_DIR/decode.pid"
    fi
fi

# Kill any remaining vLLM processes on target ports
echo "Cleaning up any remaining processes..."
fuser -k 8100/tcp 2>/dev/null || true
fuser -k 8200/tcp 2>/dev/null || true

echo "============================================"
echo "  All Servers Stopped"
echo "============================================"
