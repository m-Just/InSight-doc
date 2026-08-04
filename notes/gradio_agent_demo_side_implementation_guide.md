# Gradio Agent Demo Implementation Guide

Date: 2026-07-03

This guide is for the Hugging Face / Gradio demo developer. The demo only needs to load the model, implement a small Transformers generation backend, send user inputs to `AgentDemoRuntime`, and render stream events. File layout and UI architecture are up to the demo-side developer.

## Imports

```python
from agent_demo_runtime import (
    AgentDemoRequest,
    AgentStreamEvent,
    BackendGenerationEvent,
    TransformerGenerateBackend,
    create_agent_demo_runtime,
)
from PIL import Image
```

Image values passed into and returned from the runtime are `PIL.Image.Image`. Any component-specific conversion should happen outside this interface.

## Startup

Load model assets once, then create the runtime.

```python
MODEL_PATH = "..."

AGENT_CONFIG_PATH = (
    # Provided by the model/runtime package.
    "agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml"
)

DEFAULT_SAMPLING_PARAMS = {
    # The runtime maps this to HF model.generate(max_new_tokens=...).
    "max_tokens": 16384,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

DEFAULT_PROMPT_LENGTH = 262144

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

backend = TransformersGenerateBackend(model=model, tokenizer=tokenizer)

agent_runtime = create_agent_demo_runtime(
    tokenizer=tokenizer,
    processor=processor,
    generation_backend=backend,
    agent_config_path=AGENT_CONFIG_PATH,
    agent_config_overrides=None,
    default_sampling_params=DEFAULT_SAMPLING_PARAMS,
    prompt_length=DEFAULT_PROMPT_LENGTH,
    model_inputs_builder="qwen3_vl_transformers",
)
```

Sampling is configured through `default_sampling_params`; it is not passed per request. Use `prompt_length=262144` as the prompt token budget for this model family. This value is not passed to `TransformerGenerateBackend.generate_stream(...)` and does not configure HF `model.generate(...)`; it only controls prompt/image preparation before generation. If the initial prompt length after the requested `rescale_ratio` exceeds `prompt_length`, the runtime emits an `error` event before generation. `processor_concurrency` can usually stay at its default.

For Qwen3-VL Transformers demos, use `model_inputs_builder="qwen3_vl_transformers"`. The runtime then owns the Qwen3-VL-specific conversion from the shared agent scaffold into full HF `model.generate(...)` inputs. The backend should not rebuild prompts or process images again.

## Request Handler Pattern

```python
async def predict_stream(prompt: str, images: list[Image.Image], rescale_ratio: float):
    request = AgentDemoRequest(
        prompt=prompt,
        images=images,
        rescale_ratio=rescale_ratio,
    )

    state = DemoState.from_request(request)

    async for event in agent_runtime.stream(request):
        state.apply(event)
        yield render_outputs(state)
```

The same `agent_runtime` can serve different `rescale_ratio` values. Put the desired value into each `AgentDemoRequest`; it is applied per request.

`AgentDemoRequest` contains only:

```python
@dataclass
class AgentDemoRequest:
    prompt: str
    images: list[Image.Image]
    rescale_ratio: float
    conversation_id: str | None = None
```

## Transformers Backend

The backend receives prepared `model_inputs` and HF-compatible generation kwargs in `sampling_params`. It does not receive raw images, chat messages, tool schemas, or tool kwargs. The runtime maps `max_tokens` to `max_new_tokens`, implements `presence_penalty`, and, with `model_inputs_builder="qwen3_vl_transformers"`, prepares Qwen3-VL multimodal fields before calling the backend.

```python
class TransformersGenerateBackend(TransformerGenerateBackend):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def generate_stream(
        self,
        *,
        request_id: str,
        model_inputs: Mapping[str, Any],
        sampling_params: dict[str, Any],
    ) -> AsyncIterator[BackendGenerationEvent]:
        ...
```

Implementation sketch:

```python
def send_to_model_device(model_inputs, model):
    device = next(model.parameters()).device
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


async def generate_stream(self, *, request_id, model_inputs, sampling_params):
    streamer = AsyncTextIteratorStreamer(
        self.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **send_to_model_device(model_inputs, self.model),
        **sampling_params,
        "streamer": streamer,
    }

    task = asyncio.create_task(
        asyncio.to_thread(self.model.generate, **generation_kwargs)
    )

    async for text_delta in streamer:
        yield BackendGenerationEvent(kind="delta", text_delta=text_delta)

    output_ids = await task
    prompt_len = model_inputs["input_ids"].shape[-1]
    response_ids = output_ids[0, prompt_len:].tolist()

    yield BackendGenerationEvent(kind="done", token_ids=response_ids)
```

Backend event rules:

```python
BackendGenerationEvent(kind="delta", text_delta=str, token_ids=None)
BackendGenerationEvent(kind="done", text_delta=None, token_ids=list[int])
```

Return response-only token IDs. Do not return `prompt_ids + response_ids`.

## Stream Events

`agent_runtime.stream(request)` yields `AgentStreamEvent`:

```python
event.kind: str
event.request_id: str
event.turn: int | None
event.payload: dict[str, Any]
```

`request_id` is stable for one request. `turn` is 1-based for assistant/tool events and `None` for request-level events.
`tool_results_done` is emitted after a tool finishes and before the next `generation_start`; its `turn` matches the assistant turn that requested the tool.

Payload contract:

```python
preprocess_start: {}
preprocess_done: {}
generation_start: {}
generation_delta: {"text": str}
assistant_message_done: {"message": str}
tool_calls_detected: {
    "tool_calls": [
        {"name": str, "arguments": str},
    ],
}
tool_results_done: {
    "tool_results": [
        {
            "name": str | None,
            "text": str,
            "images": list[Image.Image],
        },
    ],
}
final: {"failure_reasons": list[str]}
error: {"error_type": str, "error_message": str}
```

All listed payload keys are required for that event kind. `failure_reasons` is an empty list on success.

## Event Handling Pattern

The demo can map events to its UI state however it prefers. A typical reducer looks like:

```python
if event.kind == "preprocess_start":
    state.status = "Preparing prompt and images..."

elif event.kind == "preprocess_done":
    state.status = "Generating..."

elif event.kind == "generation_start":
    state.start_assistant_turn(event.turn)

elif event.kind == "generation_delta":
    state.append_assistant_delta(event.payload["text"])

elif event.kind == "assistant_message_done":
    state.commit_assistant_message(event.payload["message"])

elif event.kind == "tool_calls_detected":
    state.show_tool_calls(event.payload["tool_calls"])

elif event.kind == "tool_results_done":
    state.add_tool_results(event.payload["tool_results"])

elif event.kind == "final":
    state.finish(failure_reasons=event.payload["failure_reasons"])

elif event.kind == "error":
    state.fail(event.payload["error_type"], event.payload["error_message"])
```

The runtime stream does not send full internal messages, export payloads, or metrics. The UI reconstructs display state from incremental events.
