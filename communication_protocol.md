# Inter-Container Communication Protocol

This document describes the enforced format for all tool calls exchanged between the orchestrator, Service Bus queues, and tool executors.

## Tool call JSON format (mandatory)
Every tool invocation **must** be emitted as an exact JSON object with only the fields listed below and double-quoted keys/values:

```json
{ "type": "web", "q": "<non-empty string>", "k": <integer 1-10> }
{ "type": "code", "code_command": "<non-empty string>" }
{ "type": "azure", "azure_command": "<non-empty string>" }
```

No additional properties (for example `request_id` or comments) are permitted. Payloads that fail to match the schema are rejected before they reach any queue.

## Validation pipeline
- `serving/parser.py` only accepts tool tags that contain a strict JSON object. Anything else is marked invalid.
- `serving/validation.py` rejects unexpected keys and enforces value constraints prior to queue dispatch.
- `serving/servicebus_web.py` validates outbound command messages and refuses to enqueue malformed payloads.

## Service Bus queue behavior
- Command senders in `serving/ToolGRPOTrainer/*_command_sender.py` publish the exact JSON body shown above and attach an AMQP `message_id` for telemetry only. Responses are correlated by queue order; no metadata is embedded inside the payload.
- `web-rl-env/src/command_queue.py` logs `request accepted by web` for `"type": "web"` messages and `request rejected by web` otherwise. The background worker executes only accepted web searches.
- Reward queue handling is unchanged and continues to forward executor output verbatim.

## System prompt requirements
Both `serving/run_model.py` and `serving/ToolGRPOTrainer/run_custom_grpo.py` instruct the model to:
- Use only the `<think>`, `<web>`, `<code>`, `<azure>`, and `<solution>` tags.
- Wrap tool calls in the exact JSON objects above with no extra fields.
- Emit one tool call per turn and wait for `<tool_result>` before proceeding.

These rules ensure every component—from prompt to queue to executor—enforces the same strict interface.
