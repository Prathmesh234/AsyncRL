import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
import torch
from transformers import TextIteratorStreamer
from threading import Thread
from trl import GRPOTrainer
from parser import stream_parser
from validation import ensure_web_payload, ensure_code_payload, ensure_azure_payload
import logging, warnings
from transformers.utils import logging as hf_logging
# Replace relative import with absolute to support script execution
try:
    import wandb
except ImportError:
    wandb = None


def _log_tool_usage(tool_name: str) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.log({f"tools/{tool_name}_calls": 1}, commit=False)
    except Exception:
        pass

try:
    from command_sender import send_web_command
    from azure_command_sender import send_azure_command
    from code_command_sender import send_code_command
    from grader_command_sender import send_grader_command
except ImportError:
    # Fallback: attempt relative style if executed in package context
    from ToolGRPOTrainer.command_sender import send_web_command
    from ToolGRPOTrainer.azure_command_sender import send_azure_command
    from ToolGRPOTrainer.code_command_sender import send_code_command
    from ToolGRPOTrainer.grader_command_sender import send_grader_command

# Minimal logging switches (debug prints removed)
SHOW_DEBUG = False  # retained for future use if needed

# Suppress specific noisy warnings
hf_logging.get_logger("transformers.models.qwen3.modeling_qwen3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

# -------------------------------
# Tool functions
# When <web>...</web> is produced in a streaming chunk, we call send_web_command,
# which publishes to commandtopic and polls rewardtopic (via subscription) for the correlated result.
# The returned JSON is wrapped into <tool_result> so model can continue.
# -------------------------------

def run_web_tool(payload: str) -> str:
    print(f"[TOOL][web] payload={payload!r}")
    # Validate and normalize to the exact format the container expects
    try:
        web = ensure_web_payload(payload, default_k=3)  # => {"q": str, "k": int}
    except Exception as exc:
        return f"[web-error] {exc}"
    result = send_web_command(web, timeout_s=15)
    _log_tool_usage("web")
    return result

def run_code_tool(payload: str) -> str:
    print(f"[TOOL][code] payload={payload!r}")
    try:
        code_payload = ensure_code_payload(payload)
    except Exception as exc:
        return f"[code-error] {exc}"
    result = send_code_command(code_payload, timeout_s=15)
    _log_tool_usage("code")
    return result

def run_azure_tool(payload: str) -> str:
    print(f"[TOOL][azure] payload={payload!r}")
    try:
        azure_payload = ensure_azure_payload(payload)
    except Exception as exc:
        return f"[azure-error] {exc}"
    result = send_azure_command(azure_payload, timeout_s=15)
    _log_tool_usage("azure")
    return result

def send_to_grader(query: str, completion: str) -> str:
    """Send query and completion to Grader2 for evaluation.

    This function sends the prompt and completion to the remote Grader-2 service
    via Service Bus, which will return a numeric score (1-5).

    Args:
        query: The original user query/prompt
        completion: The model's generated completion

    Returns:
        String containing the grading score or error message
    """
    print(f"[GRADER] Sending to Grader-2: query={query[:50]}... completion={completion[:50]}...")
    result = send_grader_command(query, completion, timeout_s=30)
    print(f"[GRADER] Result: {result}")
    return result

# -------------------------------
# Custom Trainer with streaming rollouts
# -------------------------------

class ToolCallingGRPOTrainer(GRPOTrainer):
    """Extension of GRPOTrainer adding streaming + multi turn tool call during TRAINING.

    Flow per completion:
      * Stream tokens
      * Detect tool tag via stream_parser
      * Execute corresponding run_*_tool (web -> waits for rewardtopic)
      * Append <tool_result>...</tool_result> to conversation
      * Continue generation until <solution> or repetition guard triggers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, *args, **kwargs):
        return super()._step(*args, **kwargs)

    def _stream_generate_one(self, prompt_text: str, max_turns: int = 8, turn_max_new_tokens: int = 1536) -> str:
        """Run multi-turn streaming generation for a single prompt, with multi turn tool calling.
        Returns ONLY the completion portion (excluding the original prompt_text)."""
        print("[GEN] start")  # simple high-level generation start notice
        conversation = prompt_text
        full_trace = prompt_text
        turns = 0
        repeat_guard_prev_add = None
        repeat_guard_count = 0
        while turns < max_turns:
            buffer = ""
            inputs = self.processing_class(text=conversation, return_tensors="pt", add_special_tokens=False)
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(self.accelerator.device)
            streamer = TextIteratorStreamer(self.processing_class, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=turn_max_new_tokens,
                temperature=self.temperature or 0.7,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.processing_class.pad_token_id,
                use_cache=False,
            )
            if getattr(self.args, 'generation_kwargs', None):
                gen_kwargs.update(self.args.generation_kwargs)

            def _gen():
                with torch.no_grad():
                    autocast_enable = getattr(self.args, 'bf16', False) or getattr(self.args, 'fp16', False)
                    with torch.cuda.amp.autocast(enabled=autocast_enable):
                        self.model.generate(**gen_kwargs)

            thread = Thread(target=_gen)
            thread.start()
            tool_triggered = False
            for new_text in streamer:
                if new_text:
                    buffer += new_text
                    full_trace += new_text
                tool_call = stream_parser(buffer)
                if tool_call:
                    tool_type = tool_call.get("type")
                    content = tool_call.get("content")
                    if tool_type == "web":
                        # This will block until rewardtopic response (or timeout)
                        result = run_web_tool(content)
                    elif tool_type == "code":
                        result = run_code_tool(content)
                    elif tool_type == "azure":
                        result = run_azure_tool(content)
                    else:
                        result = "[error] unknown tool"
                    tool_result = f"<tool_result>{result}</tool_result>\n"
                    conversation += buffer + tool_result
                    full_trace += tool_result
                    buffer = ""
                    tool_triggered = True
                    break
            thread.join()
            if not tool_triggered:
                conversation += buffer
                if buffer == repeat_guard_prev_add:
                    repeat_guard_count += 1
                else:
                    repeat_guard_prev_add = buffer
                    repeat_guard_count = 0
                if repeat_guard_count >= 2:
                    break
            if "<solution>" in full_trace:
                break
            turns += 1
        completion_only = full_trace[len(prompt_text):]
        print(completion_only, flush=True)  # final completion print per sample
        return completion_only

    def _generate_and_score_completions(self, generation_batch):  # noqa: C901
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [ex["prompt"] for ex in generation_batch]
        prompts_text = prompts
        completions_text = [self._stream_generate_one(ptxt) for ptxt in prompts_text]
        # Tokenization
        prompt_enc = self.processing_class(text=prompts_text, padding=True, add_special_tokens=False, return_tensors="pt")
        comp_enc = self.processing_class(text=completions_text, padding=True, add_special_tokens=False, return_tensors="pt")
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_mask = prompt_enc["attention_mask"].to(device)
        completion_ids = comp_enc["input_ids"].to(device)
        completion_mask = comp_enc["attention_mask"].to(device)
        if self.max_completion_length is not None and completion_ids.size(1) > self.max_completion_length:
            completion_ids = completion_ids[:, : self.max_completion_length]
            completion_mask = completion_mask[:, : self.max_completion_length]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        any_eos = is_eos.any(dim=1)
        if any_eos.any():
            eos_idx[any_eos] = is_eos.int().argmax(dim=1)[any_eos]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        generate_every = self.args.steps_per_generation * self.num_iterations
        if mode == "train" and self.args.gradient_accumulation_steps % generate_every != 0:
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                batch_size=self.args.per_device_train_batch_size,
            )
        else:
            old_per_token_logps = None
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep,
                    batch_size=self.args.per_device_train_batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                        batch_size=self.args.per_device_train_batch_size,
                    )
        else:
            ref_per_token_logps = None
        completion_ids_list = [row[mask_row.bool()].tolist() for row, mask_row in zip(completion_ids, completion_mask)]
        rewards_per_func = self._calculate_rewards(generation_batch, prompts, completions_text, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(f"Invalid value for scale_rewards: {self.scale_rewards}")
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
        terminated_with_eos = self.accelerator.gather(any_eos)
        clipped_ratio = 1 - terminated_with_eos.float().mean().item()
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_ratio)
        for i, name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{name}/mean"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        if getattr(self.accelerator, "gather_object", None) is not None:
            gathered_prompts = self.accelerator.gather_object(prompts_text)
            gathered_completions = self.accelerator.gather_object(completions_text)
            if self.accelerator.is_main_process:
                self._logs["prompt"].extend(gathered_prompts)
                self._logs["completion"].extend(gathered_completions)
        else:
            if self.accelerator.is_main_process:
                self._logs["prompt"].extend(prompts_text)
                self._logs["completion"].extend(completions_text)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(advantages.tolist())
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def generate_completions(self, prompts, **gen_kwargs):
        return self._generate_completions(prompts, **gen_kwargs)

    def _generate_completions(self, prompts, **gen_kwargs):  # manual path
        completions = []
        for prompt in prompts:
            print("[GEN] start")
            conversation = str(prompt)
            full_trace = conversation
            done = False
            turns = 0
            repeat_guard_prev_add = None
            repeat_guard_count = 0
            while not done and turns < 8:
                buffer = ""
                inputs = self.processing_class(text=conversation, return_tensors="pt", add_special_tokens=False)
                for k in list(inputs.keys()):
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(self.accelerator.device)
                streamer = TextIteratorStreamer(self.processing_class, skip_prompt=True, skip_special_tokens=True)
                local_gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=1536,
                    temperature=0.7,
                    do_sample=True,
                    streamer=streamer,
                    pad_token_id=self.processing_class.pad_token_id,
                    use_cache=False,
                )
                local_gen_kwargs.update(gen_kwargs)

                def _gen():
                    with torch.no_grad():
                        autocast_enable = getattr(self.args, 'bf16', False) or getattr(self.args, 'fp16', False)
                        with torch.cuda.amp.autocast(enabled=autocast_enable):
                            self.model.generate(**local_gen_kwargs)

                thread = Thread(target=_gen)
                thread.start()
                tool_triggered = False
                for new_text in streamer:
                    if new_text:
                        buffer += new_text
                        full_trace += new_text
                    tool_call = stream_parser(buffer)
                    if tool_call:
                        tool_type = tool_call.get("type")
                        content = tool_call.get("content")
                        if tool_type == "web":
                            result = run_web_tool(content)
                        elif tool_type == "code":
                            result = run_code_tool(content)
                        elif tool_type == "azure":
                            result = run_azure_tool(content)
                        else:
                            result = "[error] unknown tool"
                        tool_result = f"<tool_result>{result}</tool_result>\n"
                        conversation += buffer + tool_result
                        full_trace += tool_result
                        buffer = ""
                        tool_triggered = True
                        break
                thread.join()
                if not tool_triggered:
                    conversation += buffer
                    if buffer == repeat_guard_prev_add:
                        repeat_guard_count += 1
                    else:
                        repeat_guard_prev_add = buffer
                        repeat_guard_count = 0
                    if repeat_guard_count >= 2:
                        done = True
                if "<solution>" in full_trace:
                    completions.append(full_trace)
                    done = True
                turns += 1
            if not done:
                completions.append(full_trace)
            completion_only = full_trace[len(prompt):]
            print(completion_only, flush=True)
        return completions
