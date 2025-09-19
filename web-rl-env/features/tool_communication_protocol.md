# Tool Communication Protocol

## Overview
The `serving/run_model.py` orchestrator streams model tokens and hands off tool execution to helper functions in `serving/ToolGRPOTrainer`. Each tool call is relayed through Azure Service Bus queues so that external executors can process the request and push a reward/response message back.

## Environment configuration
- `SERVICE_BUS_CONNECTION_STRING` must resolve to the Azure Service Bus namespace used for both command and reward queues.
- `QUEUE_NAME` selects the command queue (defaults to `commandqueue`). All web/code/azure requests are enqueued here with a type tag.
- `REWARD_QUEUE_NAME` selects the reward queue (defaults to `rewardqueue`) that the trainer or tool runner posts results to.
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `SYSTEM_PROMPT` drive the streaming loop but do not affect queue traffic directly.

## Request validation and shaping
`run_model.py` exposes three tool helpers:
- `run_web_tool(payload: str) -> str`
- `run_code_tool(payload: str) -> str`
- `run_azure_tool(payload: str) -> str`

For every invocation the raw payload emitted by the model is logged (`[TOOL][type] payload=...`) and parsed by the matching validator in `serving/validation.py`:
- Web: `ensure_web_payload` → `{ "q": str, "k": int }` with bounds on `k`.
- Code: `ensure_code_payload` → `{ "code_command": str }`.
- Azure: `ensure_azure_payload` → `{ "azure_command": str }`.

A malformed payload returns `[type-error] ...` to the model without touching Service Bus.

## Command dispatch flow
All three helpers delegate to `send_*_command` in `serving/ToolGRPOTrainer/`:
1. Generate a `request_id = uuid4()` for correlation.
2. Construct the command envelope and label it with `type`:
   - Web: `{ "type": "web", "q", "k", "request_id" }`.
   - Code: `{ "type": "code", "code_command", "request_id" }`.
   - Azure: `{ "type": "azure", "azure_command", "request_id" }`.
3. Open a `ServiceBusQueueWeb` context targeting the command queue (`QUEUE_NAME`).
4. Call `send_web_result(..., wrap=False)` so the Service Bus message body remains the original `{"type", "q", "k", "request_id"}` payload (top-level `type`/`q`/`k` keeps downstream readers simple).
5. Return early with `[type-error] ...` if connection details are missing, the payload is empty, or the send operation fails.

## Reward polling
After enqueueing the command, each sender awaits a response via `_wait_for_response()`:
- Poll interval: 1 second; maximum duration: `timeout_s` (default 10 seconds in senders, invoked with 15s by `run_model.py`).
- On each poll, instantiate `ServiceBusQueueWeb` for the reward queue (`REWARD_QUEUE_NAME`) and call `receive_web_reward_async()`.
- Skip placeholder messages (`"No rewards received"` or `"No messages received"`).
- Accept the first payload whose `request_id` (or nested `data.request_id`) matches the command. If the executor omits `request_id`, the first non-placeholder message is returned.
- When the timeout elapses, synthesize `{ "message": "No response within timeout", "request_id": ... }`.

`receive_web_reward_async()` unwraps Service Bus messages into dictionaries and acknowledges each message. Any JSON parsing errors are dead-lettered. The helper returns `{ "data": ..., "message_id": ... }` when available, otherwise a default status message.

## Returning results to the model
The sender serializes the reward payload with `json.dumps(...)` and prefixes it with `[type-result]`. For example:
- Web: `[web-result] { ... }`
- Code: `[code-result] { ... }`
- Azure: `[azure-result] { ... }`

`run_model.py` embeds that string inside `<tool_result>...</tool_result>` and appends it to the live conversation transcript. The streaming loop resumes token generation with the new context until a `<solution>` tag is produced or the turn limit is reached.

## Error handling behaviors
- Missing Service Bus connection details → immediate `[type-error] Missing SERVICE_BUS_CONNECTION_STRING env var`.
- Empty tool payloads → `[type-error] Empty ... command`.
- Exceptions while sending or polling → logged via `logging` and surfaced as `[type-error] {exception}`.
- Timeout while waiting on the reward queue → `[type-result] {"message": "No response within timeout", ...}` so the model can decide on a retry or alternate strategy.

## Implementation references
- Orchestrator loop and tool routers: `serving/run_model.py`.
- Payload validators: `serving/validation.py`.
- Command senders and reward polling: `serving/ToolGRPOTrainer/{command_sender,code_command_sender,azure_command_sender}.py`.
- Service Bus helper: `serving/servicebus_web.py`.
