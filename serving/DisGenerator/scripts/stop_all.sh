#!/bin/bash
# =============================================================================
# DisGenerator - Stop All Components
# =============================================================================
# This script gracefully stops all DisGenerator components:
#   - Proxy server
#   - Prefill servers (vLLM instances)
#   - Decode servers (vLLM instances)
#
# Usage:
#   ./stop_all.sh
# =============================================================================

echo "=============================================="
echo "Stopping DisGenerator Services"
echo "=============================================="

# Stop proxy server
echo "Stopping proxy server..."
pkill -f "disagg_proxy.py" 2>/dev/null && echo "  ✅ Proxy stopped" || echo "  ⚪ Proxy not running"

# Stop all vLLM servers
echo "Stopping vLLM servers..."
pkill -f "vllm serve" 2>/dev/null && echo "  ✅ vLLM servers stopped" || echo "  ⚪ vLLM servers not running"

# Additional cleanup - kill any orphaned processes
pkill -f "P2pNcclConnector" 2>/dev/null || true

# Clean up any stale NCCL sockets (optional)
# rm -f /tmp/nccl-* 2>/dev/null || true

echo ""
echo "=============================================="
echo "All services stopped."
echo "=============================================="

# Show any remaining processes
REMAINING=$(pgrep -f "vllm|disagg_proxy" || true)
if [ -n "$REMAINING" ]; then
    echo ""
    echo "Warning: Some processes may still be running:"
    ps aux | grep -E "vllm|disagg_proxy" | grep -v grep
    echo ""
    echo "To force kill: kill -9 $REMAINING"
fi
