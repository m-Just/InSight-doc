# Standalone Eval Model/Inference Configs

This note lists only model and inference-related fields for the cleaned
`--model-config` interface. Dataset, reward, export, resume, general runtime,
and agent behavior fields are intentionally omitted from model config.

## Model/Backend Schema

The cleaned standalone interface should use one model identifier and one backend
selector:

| Config | Used by | Default | Meaning |
|---|---|---:|---|
| `model` | both | required | Backend-specific model identifier. For `ray_vllm`, this is an HF/vLLM-loadable model id or directory. For `https_openai_chat`, this is the remote API model name. |
| `backend` | both | `ray_vllm` | `ray_vllm` or `https_openai_chat` |

Raw verl FSDP checkpoint directories are not directly loadable by the standalone
ray/vLLM path. They should first be converted or merged into a standalone
HF-compatible directory.

Model provenance such as `hf`, `verl_sft`, `verl_rl`, or `api` should not drive
eval behavior once the model is loadable by the selected backend. If provenance
is useful for reporting, put it under run metadata rather than model loading
config.

The cleaned schema should remove the older redundant fields:

| Removed config | Replacement |
|---|---|
| `model.type` | Optional metadata only |
| `model.name` | Use run metadata if a display name is needed |
| `model.path` | Use top-level `model` for `ray_vllm` |
| `backend.type` | Use top-level `backend` |
| `https.model` | Use top-level `model` for `https_openai_chat` |

## Shared Budget Configs

| Config | Used by | Default | Meaning |
|---|---|---:|---|
| `generation.max_tokens_after_initial_prompt` | both | `16384` | Token budget after the initial prompt. In tool-use eval this includes assistant tokens plus tool-observation tokens; in direct API/no-tool eval it maps to the API completion-token limit. |

`generation.prompt_length` should be removed from the cleaned standalone eval
interface. For `ray_vllm`, the effective context limit should be
`ray_vllm.max_model_len`. For `https_openai_chat`, follow the old
`close_source_test_0501.sh` behavior: do not pre-tokenize or pre-fit against a
local prompt token budget; send the request and retry context-overflow failures
by shrinking the image rescale ratio.

`generation.response_length` should be renamed to
`generation.max_tokens_after_initial_prompt`. The old verl val path uses this as
the budget for all tokens appended after the initial prompt: assistant-generated
tokens are mask `1`, tool-observation tokens are mask `0`, and both consume the
same trajectory budget. For direct API/no-tool eval there are no tool
observations, so this is equivalent to `max_completion_tokens`.

Downstream cleanup: API models do not require local tokenizer or processor
paths. The `https_openai_chat` path reads parquet rows directly, sends
OpenAI-compatible chat messages with image URLs, and follows provider-side
context handling. Tokenizer and processor loading is only used by local
`ray_vllm` models, where both default to the top-level `model`; separate
tokenizer/processor fields are omitted from the cleaned schema unless a future
special case proves necessary.

## Agent/Tool Schema Configs

These fields should stay under `--agent-config`, not `--model-config`.

Tool availability should be controlled only by the tool list:

| Config | Meaning |
|---|---|
| `agent.qwen_tool_list` | List of qwen_agent tool registry names to expose and execute. Empty list means no tools and no tool schema. |
| `agent.coerce_tool_role_to_user` | Optional message-format compatibility setting for providers that cannot accept `role=tool`; irrelevant for direct no-tool API eval. |

For Qwen3-VL, when `agent.qwen_tool_list` is non-empty, the resolved tool
schemas are passed to `processor.apply_chat_template(..., tools=...)`, and the
Qwen3-VL chat template renders those schemas inside the serialized system block.
When `agent.qwen_tool_list` is empty, tools should be passed as `None`, so no
tool schema is rendered or sent.

The cleaned schema should remove the older redundant flags:

| Removed config | Reason |
|---|---|
| `agent.no_tool_schema` / `--no-tool-schema` | Redundant with `agent.qwen_tool_list: []`; keeping both creates duplicated truth. |
| `https_openai_chat.send_tool_schema` | Tool schema sending/rendering should follow global tool availability, not an HTTPS-specific boolean. |

The only normal no-tool/no-system eval setting should be:

```yaml
agent:
  qwen_tool_list: []
  coerce_tool_role_to_user: false
```

A non-empty tool-use setting should be:

```yaml
agent:
  qwen_tool_list:
    - image_zoom_in_tool_qwen3vl
  coerce_tool_role_to_user: false
```

## `ray_vllm` Backend

Used for `hf`, `verl_sft`, and `verl_rl`.

These defaults match the old rl_ckpt700 fast val vLLM serving setup, except
that `generation.max_tokens_after_initial_prompt` is intentionally standardized
to `16384` for both backends.

| Config | Default | Meaning |
|---|---:|---|
| `ray_vllm.num_replicas` | `4` | Number of vLLM replicas |
| `ray_vllm.gpus_per_replica` | `1` | Tensor parallel size / GPUs per replica |
| `ray_vllm.max_model_len` | `262144` | vLLM engine max sequence length |
| `ray_vllm.max_num_seqs` | `1024` | Max concurrent sequences per replica |
| `ray_vllm.max_num_batched_tokens` | `32768` | vLLM batching budget |
| `ray_vllm.gpu_memory_utilization` | `0.9` | vLLM GPU memory fraction |
| `ray_vllm.enable_prefix_caching` | `true` | vLLM prefix cache |
| `ray_vllm.enable_chunked_prefill` | `true` | vLLM chunked prefill |
| `ray_vllm.enforce_eager` | `true` | Disable CUDA graph path if true |

The following fields should be removed completely from the cleaned
`eval_defaults.yaml` interface, not kept as advanced overrides:

| Removed config | Reason |
|---|---|
| `ray_vllm.dtype` | vLLM can infer this from the model/checkpoint in normal eval use. |
| `ray_vllm.load_format` | vLLM's default loading behavior is sufficient for the cleaned interface. |
| `ray_vllm.trust_remote_code` | Not a normal eval knob; avoid exposing it unless the model path itself requires a separate launch path. |
| `ray_vllm.enable_sleep_mode` | Not needed for standalone eval semantics. |
| `ray_vllm.scheduling_policy` | vLLM's official default is sufficient and usually should not be tuned. |

### `ray_vllm.sampling`

Sampling parameters should be backend-specific. For ray/vLLM:

| Config | Default | Meaning |
|---|---:|---|
| `ray_vllm.sampling.temperature` | `0.7` | Sampling temperature |
| `ray_vllm.sampling.top_p` | `0.8` | Nucleus sampling |
| `ray_vllm.sampling.top_k` | `20` | Top-k sampling |
| `ray_vllm.sampling.presence_penalty` | `1.5` | Presence penalty |
| `ray_vllm.sampling.repetition_penalty` | `1.0` | Repetition penalty |

`logprobs` is not part of the normal model/inference config. It can be added as
an advanced/debug option if token logprobs are explicitly needed, but current
standalone eval metrics and rewards do not depend on it.

## `https_openai_chat` Backend

Used for OpenAI-compatible HTTPS API models.

These defaults follow `close_source_test_0501.sh`, except that
`https_openai_chat.reasoning_effort` is intentionally set to `high`.

| Config | Default | Meaning |
|---|---:|---|
| `https_openai_chat.base_url` | `$OPENAI_BASE_URL` | OpenAI-compatible endpoint base URL |
| `https_openai_chat.api_key_env` | `OPENAI_API_KEY` | Env var containing API key |
| `https_openai_chat.timeout` | `180` | Request timeout in seconds |
| `https_openai_chat.max_retries` | `1` | API timeout retry count |
| `https_openai_chat.image_format` | `png` | Encoded image format |
| `https_openai_chat.image_detail` | `high` | OpenAI image detail |
| `https_openai_chat.reasoning_effort` | `high` | Reasoning effort for supported API models |

API sampling parameters are intentionally omitted from the cleaned schema. For
most HTTPS/API model evals we should use provider defaults, except for the
shared `generation.max_tokens_after_initial_prompt` budget. In direct API
single-turn eval this should map to `max_completion_tokens`, matching
`close_source_test_0501.sh`.

The HTTPS path should also keep the old close-source context overflow behavior:
retry provider context-length failures by halving the current image rescale
ratio up to a configured maximum number of attempts, instead of relying on a
local tokenizer/processor prompt fit.
