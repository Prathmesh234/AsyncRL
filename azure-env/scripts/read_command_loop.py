#!/usr/bin/env python3
"""Polls the azure-env read-command endpoint for live telemetry, matching the web container."""
import json
import os
import time
import urllib.request
from urllib.error import URLError, HTTPError

URL = os.environ.get("READ_COMMAND_URL", "http://localhost:8001/read-command")
INTERVAL = float(os.environ.get("READ_COMMAND_INTERVAL", "2"))


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))


def send_reward(*_, **__):
    # Disabled: Azure results are published by the server worker; do not echo rewards from the CLI.
    return {"disabled": True}


def main():
    last = None
    print(f"Polling {URL} every {INTERVAL}s. Press Ctrl+C to stop.")
    while True:
        try:
            payload = fetch(URL)
        except (HTTPError, URLError, TimeoutError) as e:
            print(f"Error: {e}")
            time.sleep(INTERVAL)
            continue
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            time.sleep(INTERVAL)
            continue

        if payload != last:
            print(json.dumps(payload, indent=2))
            last = payload

            def is_actionable(data) -> bool:
                if not isinstance(data, dict):
                    return False
                if data.get("status") in {"idle", "error", "ignored"}:
                    return False
                if data.get("message") == "Receiver not started":
                    return False

                payload_type = (data.get("type") or data.get("tool_type") or "").strip().lower()

                if payload_type and payload_type not in {"azure"}:
                    return False

                if data.get("azure_command"):
                    return True

                received = data.get("received_command")
                if isinstance(received, dict) and received.get("azure_command"):
                    return True

                return False

            if is_actionable(payload):
                print("Actionable Azure command observed; server worker will process and publish reward.")
            else:
                print("No actionable command (idle, non-azure, or missing azure_command).")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
