"""Batch generation script for curriculum tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from openai import OpenAI

from synthetic_trajectories.curriculum_task import CurriculumTask


prompt = ""

DEFAULT_MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5")
BATCH_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
OUTPUT_DIR = Path(os.getenv("BATCH_OUTPUT_DIR", "training/synthetic_trajectories/batches"))


def create_batch_jsonl(
    tasks: Iterable[str],
    prompt_prefix: str,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str = BATCH_ENDPOINT,
) -> Path:
    """Write the provided tasks to a JSONL file formatted for batch processing."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            user_prompt = f"{prompt_prefix}{task}"
            request = {
                "custom_id": f"{output_path.stem}-{index:04d}",
                "method": "POST",
                "url": endpoint,
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False))
            handle.write("\n")

    return output_path


def submit_batch_from_file(
    client: OpenAI,
    jsonl_path: Path,
    *,
    endpoint: str = BATCH_ENDPOINT,
    completion_window: str = COMPLETION_WINDOW,
    metadata: Mapping[str, str] | None = None,
):
    """Upload a JSONL file and create a batch for asynchronous processing."""

    with jsonl_path.open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=dict(metadata or {}),
    )

    return batch


def main() -> None:
    curriculum = CurriculumTask()
    client = OpenAI()

    task_groups = {
        "easy": curriculum.generate_easy_tasks(),
        "medium": curriculum.generate_medium_tasks(),
        "medium_hard": curriculum.generate_medium_hard_tasks(),
        "hard": curriculum.generate_hard_tasks(),
    }

    for label, tasks in task_groups.items():
        output_path = OUTPUT_DIR / f"{label}.jsonl"
        create_batch_jsonl(tasks, prompt, output_path)
        batch = submit_batch_from_file(
            client,
            output_path,
            metadata={"curriculum_level": label},
        )
        print(
            f"Submitted batch {batch.id} for {label} tasks using file {output_path}"
        )


if __name__ == "__main__":
    main()
