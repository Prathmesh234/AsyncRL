#!/usr/bin/env python3
import os
import time
import json
import urllib.request
from urllib.error import URLError, HTTPError


URL = os.environ.get("READ_COMMAND_URL", "http://localhost:8000/read-command")
INTERVAL = float(os.environ.get("READ_COMMAND_INTERVAL", "2"))


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))


def send_reward(*args, **kwargs):
    # Disabled: web tool results are published by the server; do not echo commands to reward queue.
    return {"disabled": True}


def main():
    last = None
    print(f"Polling {URL} every {INTERVAL}s. Press Ctrl+C to stop.")
    while True:
        try:
            payload = fetch(URL)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(INTERVAL)
            continue

        if payload != last:
            print(json.dumps(payload, indent=2))
            last = payload

            # Decide whether to send a reward
            def is_actionable(data) -> bool:
                if not isinstance(data, dict):
                    return False
                # Skip idle/error or when receiver not started
                if data.get("status") in {"idle", "error"}:
                    return False
                if data.get("message") == "Receiver not started":
                    return False
                # Actionable if we have q/k (data-only shape)
                if ("q" in data and data.get("q")) or ("k" in data and data.get("k") is not None):
                    return True
                return False

            # Do not send rewards from this script. The server runs the WebTool and publishes results.
            if is_actionable(payload):
                print("Actionable command observed; server will process via WebTool and publish reward.")
            else:
                print("No actionable command (idle or missing q/k).")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
