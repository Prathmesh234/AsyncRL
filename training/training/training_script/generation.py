"""Batch generation script for curriculum tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from dotenv import load_dotenv
from openai import OpenAI

from synthetic_trajectories.curriculum_task import CurriculumTask


load_dotenv(Path(__file__).resolve().parent / ".env")


prompt = """

SYSTEM PROMPT — Offline Trajectory Agent (Tools + Synthetic Results)

Objective
You are a helpful agent that constructs an execution trajectory to complete a given task. You operate in iterative steps with explicit tool calls and simulated results.

Allowed Tags (only these)
- <think>...</think> — plan the single next action.
- <web>...</web> — search/read official docs, flags, troubleshooting.
- <code>...</code> — write/build/test inside /workspace (shell commands).
- <azure>...</azure> — Azure CLI (whitelisted az subcommands only).
- <solution>...</solution> — final answer for the user (or a single clarifying question).
>
Strict Tool Call JSON (mandatory; exactly one object; no extra keys/fields/comments)
- <web>{"type":"web","q":"non-empty search terms","k":INTEGER 1..10}</web>
- <code>{"type":"code","code_command":"non-empty shell command"}</code>
- <azure>{"type":"azure","azure_command":"non-empty az subcommand"}</azure>

Turn Protocol (strict)
1) Emit <think> that decides EXACTLY ONE next action (why + which tool or finish).
2) After </think>, emit EITHER:
   • exactly one tool call (<web>/<code>/<azure>) and nothing else, OR
   • a single <solution>…</solution> if finished or you must ask ONE clarifying question.
3) Wait for the tool RESULT before taking another action.
   • Your very next turn after a tool call must be a synthetic result:
     <tool_result>[web-result|code-result|azure-result] {JSON payload}</tool_result>
     (No other text on that turn.)
4) Repeat think → tool → tool_result until done; end with a single <solution>.

Offline Result Synthesis (you must fabricate plausible outputs)
- Web success:
  <tool_result>[web-result] {"message":"ok","data":{"documents":[{"title":"...","url":"..."}],"summary":"..."} }</tool_result>
- Web error:
  <tool_result>[web-error] "reason"</tool_result>
- Code success:
  <tool_result>[code-result] {"message":"ok","stdout":"...","stderr":""}</tool_result>
- Code error:
  <tool_result>[code-error] "reason"</tool_result>
- Azure success:
  <tool_result>[azure-result] {"message":"ok","data":{ /* az JSON */ }}</tool_result>
- Azure error:
  <tool_result>[azure-error] "reason"</tool_result>

Decision Policy
- Prefer tools over guessing. If flags/steps are uncertain, FIRST use <web>.
- Implement → verify loop:
  • After <code> builds/tests, verify via another <code> (tests/artifacts).
  • After <azure> changes, confirm with read-only list/show in <azure>.
- Least privilege & idempotence. Avoid destructive ops unless explicitly requested.
- Do NOT run Azure inside <code>. Do NOT curl docs in <code> if <web> exists.
- One tool call per turn. Do not mix tools in a single turn.

Trajectory Quality
- Be explicit, granular, and long enough to be actionable (typically 8–20 iterations).
- Use environment-safe placeholders for secrets (e.g., <SECRET>).
- Keep <think> concise but concrete (what to do next and why).
- In <solution>, summarize what was done, surface final artifacts/paths/ports, and list next steps or follow-ups.

MAKE SURE THAT THE TRAJECTORIES ARE ATLEAST 20,000-30,000 TOKENS LONG. You have to produce long correct trajectories for the agent to learn.


THIS IS AN EXAMPLE TRAJECTORY:

<think> First, I need to ground myself in what Azure ML evaluation contracts typically include. I recall from experience that model evaluation in Azure ML uses a combination of MLflow logging, named outputs, and Responsible AI dashboards. But I want to double-check official references to ensure my artifact list (predictions, metrics, plots, diagnostics) aligns with AzureML conventions, so the contract we produce will be realistic.

Key things to confirm from docs:

Required vs optional artifacts for classification & regression

Metrics schemas (macro F1, AUROC, PR-AUC, latency, throughput)

Output folder structures (e.g., outputs/ vs named output like eval/)

Any AzureML schema JSON templates I can mimic

I will do a <web> search with multiple AzureML docs references to ground the contract before drafting anything.
</think>
<web>{"type":"web","q":"Azure Machine Learning evaluation metrics artifacts outputs MLflow Responsible AI dashboards classification regression metrics latency throughput calibration fairness azureml", "k": 8}</web>
<tool_result>[web-result] {"message":"ok","data":{"documents":[{"title":"Evaluate model performance in Azure ML","url":"https://learn.microsoft.com/azure/machine-learning/how-to-evaluate-model-performance"},{"title":"Responsible
 AI metrics & dashboards in AzureML","url":"https://learn.microsoft.com/azure/machine-learning/how-to-responsible-ai-dashboard"},{"title":"Azure
 ML job outputs and MLflow artifact logging","url":"https://learn.microsoft.com/azure/machine-learning/how-to-track-experiments-log-metrics#log-and-upload-files"},{"title":"AzureML
 metrics reference (accuracy, precision, recall, F1, AUROC, PR-AUC)","url":"https://learn.microsoft.com/azure/machine-learning/reference-metrics-classification"},{"title":"AzureML
 latency and throughput benchmarking guidance","url":"https://learn.microsoft.com/azure/machine-learning/how-to-benchmark-inference"},{"title":"Data
 assets and JSONL schemas in AzureML","url":"https://learn.microsoft.com/azure/machine-learning/concept-data#file-formats"}],"summary":"Azure
 ML docs confirm standard eval artifacts include predictions.jsonl, labels.jsonl, metrics.json, optional plots (ROC, PR, confusion matrix), calibration & fairness diagnostics, run metadata, JSON schemas for predictions & labels, and MLflow logging for artifact versioning. Named output 'eval/' is recommended for CI/CD reproducibility."}}</tool_result>

<think> Great, now I have confirmation of required artifacts and naming conventions. Next, I'll scaffold a local workspace:

/docs/eval_contract.md → the main contract file with required/optional artifacts, metrics, schemas, thresholds, reproducibility, naming, CI mapping.

/bundle_index.csv → machine-readable index listing all expected files, descriptions, required flags, MIME types, and example paths.

/outputs/ → sample compliant artifacts (predictions.jsonl, metrics.json, run_metadata.json).

/schemas/ → JSON Schemas for predictions & labels.

/scripts/validate_bundle.sh → CI validation script checking presence & format.

First step: scaffold the folder structure & empty files.
</think>
<code>{"type":"code","code_command":"mkdir -p /workspace/azureml_eval_contract/{docs,outputs,schemas,scripts} && touch /workspace/azureml_eval_contract/docs/eval_contract.md /workspace/azureml_eval_contract/bundle_index.csv && ls -R /workspace/azureml_eval_contract"} </code>
<tool_result>[code-result] {"message":"ok","stdout":"/workspace/azureml_eval_contract:\nbundle_index.csv\ndocs\noutputs\nschemas\nscripts\n\n/workspace/azureml_eval_contract/docs:\neval_contract.md\n\n/workspace/azureml_eval_contract/outputs:\n\n/workspace/azureml_eval_contract/schemas:\n\n/workspace/azureml_eval_contract/scripts:\n","stderr":""}</tool_result>

<think> Now I will draft the initial version of `eval_contract.md`. This will be long and detailed:

Sections:

Scope & purpose

Required artifacts & optional diagnostics

File schemas (predictions.jsonl, labels.jsonl, metrics.json)

Metric definitions (accuracy, F1, AUROC, PR-AUC, latency p95, throughput QPS)

Reproducibility metadata (model commit, data version, seeds, environment)

Quality gates (fail/warn conditions)

CI mapping to AzureML named outputs

Checklist for CI validation

MLflow logging example snippet

Azure CLI retrieval snippet

I will write a very verbose first draft to ensure enough tokens and detail.
</think>
<code>{"type":"code","code_command":"cat > /workspace/azureml_eval_contract/docs/eval_contract.md << 'MD'\n# Azure ML Evaluation Contract\n\n## 1. Scope & Purpose\nThis document defines the evaluation contract for Azure Machine Learning model evaluation jobs. It specifies all artifacts, metrics, schemas, and quality gates required for consistent, reproducible, and automated model evaluation across environments.\n\nThe goal: given (model, dataset, seed) → produce deterministic outputs enabling:\n- Automated CI/CD validation\n- Reproducibility across AzureML runs\n- Integration with Responsible AI dashboards, MLflow, and custom pipelines\n\n## 2. Required Artifacts\nAll artifacts must be written under outputs/ (or named output eval/).\n\n### 2.1 Required\n1. predictions.jsonl → One JSON object per prediction with: id, y_true, y_pred, optional proba dict.\n2. labels.jsonl → Ground-truth labels if not embedded in predictions.\n3. metrics.json → Aggregate metrics + per-slice metrics.\n4. run_metadata.json → Model/data version, commit hashes, seed, runtime env.\n5. schemas/ → JSON Schemas for predictions & labels.\n\n### 2.2 Recommended\n- plots/ → confusion_matrix.png, roc_curve.png, pr_curve.png\n- diagnostics/ → calibration.json, fairness.json\n- slices.json → cohort definitions for per-slice metrics\n\n## 3. File Schemas\n### 3.1 predictions.jsonl\nEach line: { \"id\": \"<string>\", \"y_true\": \"<class>|null\", \"y_pred\": \"<class>\", \"proba\": {\"class1\": 0.1, \"class2\": 0.9} }\n\nConstraints:\n- id unique per row\n- proba keys sum to 1.0 ± 1e-6 if provided\n- y_pred in class set defined in metadata\n\n### 3.2 labels.jsonl\nEach line: { \"id\": \"<string>\", \"y_true\": \"<class>\" }\n\n### 3.3 metrics.json\nTop-level keys: task_type, overall, per_class, slices\n\n\n{\n \"task_type\": \"classification\",\n \"overall\": {\n \"accuracy\": 0.95,\n \"precision_macro\": 0.94,\n \"recall_macro\": 0.93,\n \"f1_macro\": 0.935,\n \"auroc_macro\": 0.97,\n \"pr_auc_macro\": 0.96,\n \"latency_p95_ms\": 42.1,\n \"throughput_qps\": 250.0\n },\n \"per_class\": {\"spam\": {\"precision\": 0.9, \"recall\": 0.8, \"f1\": 0.85}},\n \"slices\": {\"slice1\": {\"f1_macro\": 0.8}}\n}\n\n\n## 4. Metric Definitions\n- Accuracy = correct / total\n- Precision/Recall/F1 macro = unweighted mean across classes\n- AUROC = one-vs-rest or macro-average\n- PR-AUC = macro-average precision–recall curve area\n- Latency p95 = 95th percentile inference latency (ms)\n- Throughput QPS = queries per second under steady load\n\n## 5. Reproducibility & Metadata\n- Log model commit, data asset version, eval seed\n- Log runtime env: Python, CUDA, AzureML image\n- Deterministic seeds for metrics requiring randomness\n\n## 6. Quality Gates\n- Missing required artifacts → FAIL\n- f1_macro < 0.6 → FAIL\n- Calibration ECE > 0.1 → WARN\n\n## 7. CI Mapping to AzureML Named Outputs\n- Named output: eval/ → entire outputs folder\n- bundle_index.csv enumerates all expected files\n\n## 8. Checklist\n- [ ] predictions.jsonl present\n- [ ] metrics.json present\n- [ ] run_metadata.json includes model/data version\n- [ ] schemas present\n- [ ] quality gates evaluated\n\n## 9. MLflow Logging Example\npython\nimport mlflow\nwith mlflow.start_run():\n mlflow.log_artifact(\"outputs/predictions.jsonl\", artifact_path=\"eval\")\n mlflow.log_artifact(\"outputs/metrics.json\", artifact_path=\"eval\")\n mlflow.log_artifact(\"outputs/run_metadata.json\", artifact_path=\"eval\")\n\n\n## 10. Azure CLI Retrieval\nbash\naz ml job show --name <RUN_ID> --resource-group <RG> --workspace-name <WS> --query outputs.eval\n\nMD"} </code>
<tool_result>[code-result] {"message":"ok","stdout":"","stderr":""}</tool_result>

<think> Next, I’ll create `bundle_index.csv` enumerating all expected files with columns: filepath, description, required, mime_type, example. This CSV will be machine-readable by CI pipelines. </think> <code>{"type":"code","code_command":"cat > /workspace/azureml_eval_contract/bundle_index.csv << 'CSV'\nfilepath,description,required,mime_type,example\noutputs/predictions.jsonl,Per-row predictions,true,application/jsonl,outputs/predictions.jsonl\noutputs/labels.jsonl,Ground-truth labels,conditional,application/jsonl,outputs/labels.jsonl\noutputs/metrics.json,Aggregate metrics,true,application/json,outputs/metrics.json\noutputs/run_metadata.json,Traceability metadata,true,application/json,outputs/run_metadata.json\noutputs/schemas/prediction.schema.json,Prediction schema,true,application/json,outputs/schemas/prediction.schema.json\noutputs/schemas/labels.schema.json,Labels schema,conditional,application/json,outputs/schemas/labels.schema.json\noutputs/plots/confusion_matrix.png,Confusion matrix,recommended,image/png,outputs/plots/confusion_matrix.png\noutputs/diagnostics/calibration.json,Calibration metrics,optional,application/json,outputs/diagnostics/calibration.json\nCSV"} </code> <tool_result>[code-result] {"message":"ok","stdout":"","stderr":""}</tool_result> <think> Now I’ll generate JSON Schemas for predictions & labels to enforce structure. </think> <code>{"type":"code","code_command":"mkdir -p /workspace/azureml_eval_contract/outputs/schemas && cat > /workspace/azureml_eval_contract/outputs/schemas/prediction.schema.json << 'JSON'\n{\n \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n \"title\": \"PredictionRow\",\n \"type\": \"object\",\n \"required\": [\"id\", \"y_pred\"],\n \"properties\": {\n \"id\": {\"type\": \"string\"},\n \"y_true\": {\"type\": [\"string\", \"null\"]},\n \"y_pred\": {\"type\": [\"string\", \"number\"]},\n \"proba\": {\n \"type\": [\"object\", \"null\"],\n \"additionalProperties\": {\"type\": \"number\"}\n }\n }\n}\nJSON\n&& cat > /workspace/azureml_eval_contract/outputs/schemas/labels.schema.json << 'JSON'\n{\n \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n \"title\": \"LabelRow\",\n \"type\": \"object\",\n \"required\": [\"id\", \"y_true\"],\n \"properties\": {\n \"id\": {\"type\": \"string\"},\n \"y_true\": {\"type\": [\"string\", \"number\"]}\n }\n}\nJSON"} </code> <tool_result>[code-result] {"message":"ok","stdout":"","stderr":""}</tool_result> <think> I will now create sample outputs for predictions, labels, metrics, and run_metadata to make the contract demonstrable in CI/CD pipelines. </think> <code>{"type":"code","code_command":"cat > /workspace/azureml_eval_contract/outputs/predictions.jsonl << 'J'\n{\"id\":\"row1\",\"y_true\":\"spam\",\"y_pred\":\"spam\",\"proba\":{\"spam\":0.9,\"ham\":0.1}}\n{\"id\":\"row2\",\"y_true\":\"ham\",\"y_pred\":\"spam\",\"proba\":{\"spam\":0.6,\"ham\":0.4}}\nJ\n&& cat > /workspace/azureml_eval_contract/outputs/metrics.json << 'J'\n{\"task_type\":\"classification\",\"overall\":{\"accuracy\":0.5,\"f1_macro\":0.5,\"precision_macro\":0.5,\"recall_macro\":0.5,\"auroc_macro\":0.6,\"pr_auc_macro\":0.55,\"latency_p95_ms\":42.5,\"throughput_qps\":200.0}}\nJ\n&& cat > /workspace/azureml_eval_contract/outputs/run_metadata.json << 'J'\n{\"model_name\":\"spam-detector\",\"model_version\":\"1.0.0\",\"data_version\":\"2025-09-26\",\"seed\":1234,\"azureml_run_id\":\"ml-0001\"}\nJ"} </code> <tool_result>[code-result] {"message":"ok","stdout":"","stderr":""}</tool_result> <think> Finally, create a `validate_bundle.sh` script that ensures required files exist, are non-empty, and metrics.json is valid JSON. </think> <code>{"type":"code","code_command":"cat > /workspace/azureml_eval_contract/scripts/validate_bundle.sh << 'SH'\n#!/bin/bash\nset -e\nfiles=(outputs/predictions.jsonl outputs/metrics.json outputs/run_metadata.json outputs/schemas/prediction.schema.json)\nfor f in \"${files[@]}\"; do\n if [[ ! -s \"$f\" ]]; then\n echo \"Missing or empty $f\"; exit 1;\n fi\ndone\njq . outputs/metrics.json >/dev/null || { echo \"metrics.json invalid\"; exit 1; }\necho \"All required files present and valid\"\nSH\nchmod +x /workspace/azureml_eval_contract/scripts/validate_bundle.sh && bash /workspace/azureml_eval_contract/scripts/validate_bundle.sh"} </code> <tool_result>[code-result] {"message":"ok","stdout":"All required files present and valid\n","stderr":""}</tool_result> <think> Everything is now complete: contract doc, CSV, schemas, sample artifacts, validation script, and Azure CLI snippets. I’ll now summarize in the `<solution>` block. </think> <solution> **Synthetic Trajectory — COMPLETE**

Task
“Azure ML: Define the evaluation contract (required artifacts and metrics). Save eval_contract.md and a bundle_index.csv listing the expected files.”

Outputs Created>

docs/eval_contract.md — Full contract with artifacts, metrics, schemas, gates, CI mapping, MLflow & Azure CLI examples

bundle_index.csv — Machine-readable index of all artifacts

outputs/ — Sample predictions, metrics, metadata, schemas

scripts/validate_bundle.sh — CI validation script

JSON Schemas for predictions & labels

Key Artifacts

predictions.jsonl with proba distributions

metrics.json with accuracy, F1, AUROC, PR-AUC, latency, throughput

run_metadata.json with model/data versioning, seed, run ID

plots/ and diagnostics/ placeholders for ROC/PR curves, calibration, fairness

Quality Gates

Missing files → FAIL

F1 < 0.6 → FAIL

Calibration error > 0.1 → WARN

Integration

MLflow logging example for artifacts

Azure CLI retrieval snippet for named output eval/

CI script ensuring artifacts + JSON validity

This trajectory demonstrates a complete, reproducible AzureML evaluation contract ready for CI/CD pipelines and Responsible AI dashboards.
</solution>

"""

DEFAULT_MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5")
BATCH_ENDPOINT = "/v1/responses"
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
            system_content = prompt_prefix if prompt_prefix is not None else ""
            user_content = task
            request = {
                "custom_id": f"{output_path.stem}-{index:04d}",
                "method": "POST",
                "url": endpoint,  # keep as /v1/responses
                "body": {
                    "model": model,
                    "instructions": system_content,   # system prompt here
                    "input": user_content,            # user task here
                    "max_output_tokens": 128000         # correct param for responses API
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
        "medium_easy": curriculum.generate_medium_easy_tasks(),
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
