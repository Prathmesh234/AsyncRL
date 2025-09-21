# Tool Communication Protocol

## Overview
The `serving/run_model.py` orchestrator streams model tokens and hands off tool execution to helper functions in `serving/ToolGRPOTrainer`. Each tool call is relayed through Azure Service Bus topics so that external executors (subscribers) can process the request and publish a reward/response message back to a reward topic.

## Environment configuration
- `SERVICE_BUS_CONNECTION_STRING` must resolve to the Azure Service Bus namespace used for both command and reward topics.
- `COMMAND_TOPIC_NAME` selects the command topic (defaults to `commandtopic`). All tool requests are published here; executors share the subscription `rlcommandbustopic` (fan-out not used yet).
- `REWARD_TOPIC_NAME` selects the reward topic (defaults to `rewardtopic`) that the executors publish results to (trainer may have its own subscription).
- `WEB_SUBSCRIPTION_NAME`, `AZURE_SUBSCRIPTION_NAME`, etc. identify the subscription names for each environment.
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `SYSTEM_PROMPT` drive the streaming loop but do not affect topic traffic directly.

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

## Command dispatch flow (topic-based)
All three helpers delegate to `send_*_command` in `serving/ToolGRPOTrainer/`:
1. Construct the command envelope with only the mandatory keys (see schemas above).
2. Open a `ServiceBusTopicWeb` (topic-based) context targeting the command topic (`COMMAND_TOPIC_NAME`).
3. Call `send_web_result(..., wrap=False)` so the Service Bus message body remains the strict JSON command payload.
4. Return early with `[type-error] ...` if connection details are missing, the payload is empty, or the send operation fails.

## Reward polling
After publishing the command, each sender awaits a response via `_wait_for_response()`:
- Poll interval: 1 second; maximum duration: `timeout_s` (default 10 seconds in senders, invoked with 15s by `run_model.py`).
- On each poll, instantiate the topic helper for the reward topic (`REWARD_TOPIC_NAME`) and call `receive_web_reward_async()` via its subscription.
- Skip placeholder messages (`"No rewards received"` or `"No messages received"`).
- Accept the first non-placeholder payload. Topic ordering + strict schema keeps the pairing unambiguous for simple workloads.
- When the timeout elapses, synthesize `{ "message": "No response within timeout" }`.

`receive_web_reward_async()` unwraps Service Bus messages into dictionaries and acknowledges each message. Any JSON parsing errors are dead-lettered. The helper returns `{ "data": ..., "message_id": ... }` when available, otherwise a default status message.

## Web executor behavior
- `CommandQueue` now uses `get_subscription_receiver` and logs `request accepted by web` when a payload has `"type": "web"`; other types log `Not web command, ignored` and are completed (preventing backlog).

## Returning results to the model
Same pattern; the sender serializes the reward payload and prefixes it:
- Web: `[web-result] { ... }`
- Code: `[code-result] { ... }`
- Azure: `[azure-result] { ... }`

`run_model.py` embeds that string inside `<tool_result>...</tool_result>` and appends it to the live conversation transcript.

## Error handling behaviors
Unchanged from queue model, adapted to topics:
- Missing connection string → `[type-error] Missing SERVICE_BUS_CONNECTION_STRING env var`.
- Empty payload → `[type-error] Empty ... command`.
- Exceptions during publish or polling → logged and surfaced as `[type-error] {exception}`.
- Timeout → result wrapper with timeout message.

## Implementation references
- Orchestrator loop and tool routers: `serving/run_model.py`.
- Payload validators: `serving/validation.py`.
- Command senders and reward polling: `serving/ToolGRPOTrainer/{command_sender,code_command_sender,azure_command_sender}.py`.
- Service Bus helpers: `serving/servicebus_web.py`, `serving/servicebus_azure.py`.
