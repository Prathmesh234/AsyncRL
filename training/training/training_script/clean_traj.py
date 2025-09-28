from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


MIN_OUTPUT_TOKENS = 7000


def _collect_output_text(body: dict) -> str:
    parts: list[str] = []

    text_section = body.get("text")
    if isinstance(text_section, dict):
        value = text_section.get("value")
        if isinstance(value, str):
            parts.append(value)

    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for chunk in item.get("content", []):
                if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)

    return "\n".join(parts)


def _count_tokens(entry: dict, body: dict) -> int | None:
    usage = body.get("usage")
    if isinstance(usage, dict):
        tokens = usage.get("output_tokens")
        if isinstance(tokens, int):
            return tokens

    text = _collect_output_text(body)
    if text:
        return len(text.split())
    return None


def _validate_entry(entry: dict, min_tokens: int) -> tuple[bool, str | None]:
    if entry.get("error"):
        return False, "top_level_error"

    response = entry.get("response")
    if not isinstance(response, dict):
        return False, "missing_response"

    if response.get("status_code") != 200:
        return False, "status_code"

    body = response.get("body")
    if not isinstance(body, dict):
        return False, "missing_body"

    if body.get("status") != "completed":
        return False, "status"

    if body.get("error"):
        return False, "body_error"

    if body.get("incomplete_details"):
        return False, "incomplete_details"

    if not body.get("output") and not body.get("text"):
        return False, "empty_output"

    tokens = _count_tokens(entry, body)
    if tokens is None:
        return False, "missing_tokens"
    if tokens < min_tokens:
        return False, "short_output"

    return True, None


def clean_trajectories(directory: str | Path | None = None, *, min_output_tokens: int = MIN_OUTPUT_TOKENS) -> None:
    """Filter out incomplete responses in-place for every JSONL file in *directory*."""

    if directory is None:
        directory = Path(__file__).resolve().parents[2] / "synthetic_trajectories" / "synthetic_traj_gpt"

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    jsonl_files: Iterable[Path] = sorted(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {directory}")

    overall = Counter()

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh if line.strip()]

        keep: list[str] = []
        file_counts = Counter()

        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                file_counts["invalid_json"] += 1
                continue

            valid, reason = _validate_entry(entry, min_output_tokens)
            if valid:
                keep.append(json.dumps(entry, separators=(",", ":")))
            else:
                file_counts[reason or "invalid"] += 1

        if file_counts:
            with path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(keep))
                if keep:
                    fh.write("\n")

        removed = sum(file_counts.values())
        total = removed + len(keep)
        overall.update(file_counts)
        overall["processed"] += total
        overall["kept"] += len(keep)

        detail = ", ".join(f"{key}={count}" for key, count in file_counts.items()) or "no removals"
        print(f"{path.name}: kept {len(keep)} of {total} | removed {removed} ({detail})")

    print("\nSummary:")
    print(f"Total processed: {overall['processed']}")
    print(f"Total kept: {overall['kept']}")
    print(f"Total removed: {overall['processed'] - overall['kept']}")
    for reason, count in overall.items():
        if reason in {"processed", "kept"}:
            continue
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    clean_trajectories()
