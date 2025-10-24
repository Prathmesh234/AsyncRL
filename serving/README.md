## ToolGRPOTrainer reward configuration

ToolGRPOTrainer now uses **only** the Service-Bus based grader reward. Ensure the following when running `serving/ToolGRPOTrainer/run_custom_grpo.py` or anything that imports `reward_fn/grader_reward.py`:

1. `SERVICE_BUS_CONNECTION_STRING` – required Azure Service Bus connection string.
2. `COMMAND_TOPIC_NAME` – topic where tool and grader commands are published (defaults to `commandtopic`).
3. `REWARD_TOPIC_NAME` – topic where rewards/grades are read (defaults to `rewardtopic`).
4. `GRADER_SUBSCRIPTION_NAME` – subscription on the command topic consumed by the Grader-2 worker (defaults to `gradersubscription`).
5. `GRADER_REWARD_SUBSCRIPTION_NAME` – subscription on the reward topic that the trainer polls for grader responses (defaults to `webrewardsubscription`).

`reward_fn/grader_reward.py` forwards each prompt/completion pair through `ToolGRPOTrainer/grader_command_sender.py`, waits for the numeric grade (1–5) emitted on the reward topic, and normalizes it to `[0.0, 1.0]` before GRPO aggregates it with the intrinsic rewards. No other external reward functions are present.

## ToolGRPOTrainer W&B logging

`run_custom_grpo.py` mirrors the base GRPO script’s Weights & Biases setup:

- Honors the standard `WANDB_*` variables used elsewhere in the repo (API key, project, run name, entity).
- Optional overrides via `TOOL_GRPO_WANDB_API_KEY`, `TOOL_GRPO_WANDB_PROJECT`, `TOOL_GRPO_WANDB_RUN_NAME`, and `TOOL_GRPO_WANDB_ENTITY` take precedence when provided and are injected back into the corresponding `WANDB_*` variables so TRL and wandb share the same context.
- Logging is active when `WANDB_DISABLED` is unset (or not in `{1,true,yes}`) and the `wandb` package is available; otherwise `report_to` falls back to `none` and the trainer prints a notice that telemetry is disabled.
