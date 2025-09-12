#!/usr/bin/env python3
import os
import time

LOG_PATH = os.environ.get("WEB_TOOL_LOG", "/app/logs/web_tool.log")


def follow(path: str):
    """Simple tail -f implementation that survives rotations and file creation delays."""
    last_size = 0
    while True:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                # Seek to end on first open if file already existed
                if last_size == 0:
                    f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.25)
                        # detect rotation or truncation
                        try:
                            st = os.stat(path)
                            if st.st_size < f.tell():
                                break  # file truncated/rotated; reopen
                        except FileNotFoundError:
                            break
                # loop to reopen file
        except FileNotFoundError:
            # Wait for file to be created
            print(f"Waiting for log file: {path}")
            time.sleep(1.0)


if __name__ == "__main__":
    print(f"Tailing {LOG_PATH} (Ctrl+C to stop)...")
    follow(LOG_PATH)

