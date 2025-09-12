#!/usr/bin/env sh
# Stream the primary process stdout (uvicorn) inside the container.
# Useful when stdout is not captured to a file.

set -e

if [ -r /proc/1/fd/1 ]; then
  echo "Following /proc/1/fd/1 (app stdout). Press Ctrl+C to stop."
  exec tail -f -n +1 /proc/1/fd/1
else
  echo "Cannot access /proc/1/fd/1. Fallback to 'ps' and 'tail -f /app/logs/web_tool.log' if available."
  ps aux
fi

