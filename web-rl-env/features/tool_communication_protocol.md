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
- Web: `ensure_web_payload` → `{ "type": "web", "q": str, "k": int }` with bounds on `k`.
- Code: `ensure_code_payload` → `{ "type": "code", "code_command": str }`.
- Azure: `ensure_azure_payload` → `{ "type": "azure", "azure_command": str }`.

A malformed payload returns `[type-error] ...` to the model without touching Service Bus.

All validators reject unexpected fields so that every command body matches the strict JSON schemas:

```json
{ "type": "web", "q": "...", "k": 3 }
{ "type": "code", "code_command": "..." }
{ "type": "azure", "azure_command": "..." }
```

## Command dispatch flow
All three helpers delegate to `send_*_command` in `serving/ToolGRPOTrainer/`:
1. Construct the command envelope with only the mandatory keys (see schemas above).
2. Open a `ServiceBusQueueWeb` context targeting the command queue (`QUEUE_NAME`).
3. Call `send_web_result(..., wrap=False)` so the Service Bus message body remains the strict JSON command payload.
4. Return early with `[type-error] ...` if connection details are missing, the payload is empty, or the send operation fails.

## Reward polling
After enqueueing the command, each sender awaits a response via `_wait_for_response()`:
- Poll interval: 1 second; maximum duration: `timeout_s` (default 10 seconds in senders, invoked with 15s by `run_model.py`).
- On each poll, instantiate `ServiceBusQueueWeb` for the reward queue (`REWARD_QUEUE_NAME`) and call `receive_web_reward_async()`.
- Skip placeholder messages (`"No rewards received"` or `"No messages received"`).
- Accept the first non-placeholder payload. Queue order combined with the strict schema keeps the pairing unambiguous.
- When the timeout elapses, synthesize `{ "message": "No response within timeout" }`.

`receive_web_reward_async()` unwraps Service Bus messages into dictionaries and acknowledges each message. Any JSON parsing errors are dead-lettered. The helper returns `{ "data": ..., "message_id": ... }` when available, otherwise a default status message.

## Web executor behavior
- `CommandQueue` logs `request accepted by web` when a payload has `"type": "web"` and `request rejected by web` for every other message.
- The background worker in `web-rl-env` ignores rejected commands; only accepted web queries are executed.

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
