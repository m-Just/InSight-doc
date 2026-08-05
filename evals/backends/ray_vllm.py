from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Awaitable, Callable

from transformers import AutoProcessor, AutoTokenizer

from insight_agent_core import CoreFunctionCall, InSightQwenAgentRunner, StandaloneInSightRuntime
from insight_agent_core.prompt_length import create_prompt_length_estimator
from evals.config.agent import (
    DEFAULT_PROCESSOR_CONCURRENCY,
    apply_agent_settings_to_args,
    build_core_config,
    build_dataset_config,
    build_sampling_params,
    build_prompt_signature,
    build_tool_schemas,
    describe_processor,
    infer_processor_image_patch_size,
    load_agent_settings,
)
from evals.backends.base import RolloutJob
from evals.backends.ray_vllm_endpoint import RayVLLMEndpointPool
from evals.backends.ray_vllm_servers import connect_ray_vllm_servers
from evals.core.export import build_ray_sample_record, make_export_id
from evals.core.resume import build_row_provenance
from evals.core.tool_parser import ToolParser
from evals.core.utils import json_safe, parse_list_arg
from verl.utils.dataset.rl_dataset import RLHFDataset


def tokenizer_processor_kwargs(model_path: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if model_path and "glm" in str(model_path).lower():
        kwargs["fix_mistral_regex"] = True
    return kwargs


def load_ray_server_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not args.ray_server_manifest:
        raise ValueError("--ray-server-manifest is required for ray_vllm standalone eval")
    manifest_path = Path(args.ray_server_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend") != "ray_vllm":
        raise ValueError(f"ray server manifest backend must be ray_vllm: {manifest_path}")
    expected_hash = getattr(args, "_model_config_sha256", None)
    actual_hash = (manifest.get("model_config") or {}).get("sha256")
    if expected_hash != actual_hash:
        raise ValueError(
            "ray server manifest model_config hash does not match this eval's --model-config: "
            f"server={actual_hash} eval={expected_hash}"
        )
    ray_info = manifest.get("ray") or {}
    actor_names = list(ray_info.get("actor_names") or manifest.get("actor_names") or [])
    namespace = ray_info.get("namespace") or manifest.get("ray_namespace")
    address = ray_info.get("address") or manifest.get("ray_address") or "auto"
    if not actor_names or not namespace:
        raise ValueError(f"ray server manifest is missing actor names or namespace: {manifest_path}")
    args._ray_actor_names = actor_names
    args._ray_namespace = namespace
    args._ray_address = address
    args._ray_server_manifest_data = manifest
    return manifest, list(manifest.get("server_metadata") or [])


def start_ray_server_heartbeat(manifest: dict[str, Any], *, interval_seconds: float = 30.0):
    heartbeat_path = manifest.get("heartbeat_path")
    if not heartbeat_path:
        return lambda: None
    stop_event = threading.Event()
    path = Path(str(heartbeat_path))
    path.touch()

    def heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                path.touch()
            except Exception as exc:
                print(f"warning: failed to update Ray server heartbeat {path}: {exc}", flush=True)

    thread = threading.Thread(target=heartbeat_loop, name="ray-server-heartbeat", daemon=True)
    thread.start()

    def stop() -> None:
        stop_event.set()
        try:
            path.touch()
        except Exception:
            pass

    return stop


def compute_worker_concurrency(args: argparse.Namespace, n_workers: int | None = None) -> int:
    return int(args.worker_concurrency)


def resolve_dataset_image_patch_size(args: argparse.Namespace, processor: Any) -> int:
    configured = getattr(args, "ray_processor_image_patch_size", None)
    if configured is not None:
        return int(configured)
    return infer_processor_image_patch_size(processor)


async def build_process_agent_runner_components(
    args_dict: dict[str, Any],
) -> tuple[argparse.Namespace, InSightQwenAgentRunner, Any, dict[str, Any]]:
    args = argparse.Namespace(**args_dict)
    loader_kwargs = tokenizer_processor_kwargs(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **loader_kwargs)
    runtime_processor = AutoProcessor.from_pretrained(args.model_path, **loader_kwargs)
    if args.custom_chat_template_file:
        template = Path(args.custom_chat_template_file).read_text(encoding="utf-8")
        tokenizer.chat_template = template
        runtime_processor.chat_template = template

    agent_settings, args.agent_name = load_agent_settings(args.agent_config, args.agent_config_override)
    apply_agent_settings_to_args(args, agent_settings)
    tool_schemas = build_tool_schemas(parse_list_arg(args.qwen_tool_list))
    core_config = build_core_config(args, agent_settings, tool_schemas)
    prompt_length_estimator = create_prompt_length_estimator(
        estimator_name=getattr(args, "ray_prompt_length_estimator", "tokenized"),
        model_path=args.model_path,
        require_supported=bool(getattr(args, "ray_require_prompt_length_estimator", False)),
    )
    args._core_config_for_export = dict(core_config.__dict__)
    args._prompt_signature_for_export = build_prompt_signature(
        args,
        agent_settings,
        tool_schemas,
        processor_metadata=describe_processor(runtime_processor),
    )
    sampling_params = build_sampling_params(args)

    if args.generation_backend == "ray_vllm":
        endpoint_pool = RayVLLMEndpointPool(connect_ray_vllm_servers(args))
    else:
        raise ValueError(f"token-level worker only supports ray_vllm, got: {args.generation_backend}")

    tool_parser = ToolParser.get_tool_parser(args.tool_parser, tokenizer)

    async def extract_tool_calls(response_ids: list[int]) -> list[CoreFunctionCall]:
        _, calls = await tool_parser.extract_tool_calls(response_ids)
        return [CoreFunctionCall(name=call.name, arguments=call.arguments) for call in calls]

    runtime = StandaloneInSightRuntime(
        tokenizer=tokenizer,
        processor=runtime_processor,
        endpoint_pool=endpoint_pool,
        tool_call_extractor=extract_tool_calls,
        apply_chat_template_kwargs={"max_tool_calls": args.max_user_turns},
        processor_concurrency=DEFAULT_PROCESSOR_CONCURRENCY,
        image_patch_size=resolve_dataset_image_patch_size(args, runtime_processor),
        prompt_length_estimator=prompt_length_estimator,
    )
    runner = InSightQwenAgentRunner(core_config, runtime)
    return args, runner, tokenizer, sampling_params


def run_agent_global_queue_worker_process(
    *,
    args_dict: dict[str, Any],
    job_queue: Any,
    result_queue: Any,
    export_dir: str,
    worker_idx: int,
) -> dict[str, Any]:
    return asyncio.run(
        run_agent_global_queue_worker_process_async(
            args_dict=args_dict,
            job_queue=job_queue,
            result_queue=result_queue,
            export_dir=export_dir,
            worker_idx=worker_idx,
        )
    )


async def run_agent_global_queue_worker_process_async(
    *,
    args_dict: dict[str, Any],
    job_queue: Any,
    result_queue: Any,
    export_dir: str,
    worker_idx: int,
) -> dict[str, Any]:
    args, runner, tokenizer, sampling_params = await build_process_agent_runner_components(args_dict)
    worker_concurrency = int(getattr(args, "_resolved_worker_concurrency", 0) or compute_worker_concurrency(args))
    print(
        f"agent worker {worker_idx}: global_queue local_concurrency={worker_concurrency} "
        f"processor_concurrency={DEFAULT_PROCESSOR_CONCURRENCY}",
        flush=True,
    )
    completed = 0
    completed_lock = asyncio.Lock()

    async def consume(slot_idx: int) -> None:
        nonlocal completed
        while True:
            item = await asyncio.to_thread(job_queue.get)
            if item is None:
                return
            output_index, job_key_value, sample_idx, trial_idx, resume_val_file, resume_file_row_idx, row = item
            extra_info = dict(row.get("extra_info") or {})
            extra_info.setdefault("question_id", f"sample-{sample_idx}")
            conversation_export_id = make_export_id({**row, "extra_info": extra_info}, sample_idx, trial_idx)
            started = time.perf_counter()
            result = await runner.run(
                dict(sampling_params),
                raw_prompt=row["raw_prompt"],
                extra_info=extra_info,
                tools_kwargs=row.get("tools_kwargs", {}),
                validate=True,
                conversation_export_id=conversation_export_id,
            )
            sample = build_ray_sample_record(
                result=result,
                row=row,
                args=args,
                sampling_params=sampling_params,
                export_dir=Path(export_dir),
                sample_index=sample_idx,
                trial_idx=trial_idx,
                tokenizer=tokenizer,
                started=started,
            )
            await asyncio.to_thread(result_queue.put, ("sample", worker_idx, output_index, sample))
            async with completed_lock:
                completed += 1

    await asyncio.gather(*(consume(slot_idx) for slot_idx in range(worker_concurrency)))
    print(f"worker {worker_idx} global_queue complete: {completed} samples", flush=True)
    return {"worker_idx": worker_idx, "completed": completed}


class RayVLLMBackend:
    backend_name = "ray_vllm_global_queue"

    def __init__(self, args: argparse.Namespace, agent_settings: dict[str, Any], export_dir: Path):
        self.args = args
        self.agent_settings = agent_settings
        self.export_dir = export_dir
        self.tokenizer = None
        self.dataset_processor = None
        self.dataset_processor_metadata_before_dataset: dict[str, Any] = {}
        self.dataset_processor_metadata_after_dataset: dict[str, Any] = {}
        self.core_config = None
        self.sampling_params: dict[str, Any] = {}
        self.prompt_signature: dict[str, Any] = {}
        self.ray_server_manifest: dict[str, Any] = {}
        self.server_metadata: list[dict[str, Any]] = []
        self.stop_ray_heartbeat = lambda: None
        self.parallelism: dict[str, Any] = {}

    async def prepare(self) -> None:
        args = self.args
        loader_kwargs = tokenizer_processor_kwargs(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, **loader_kwargs)
        self.dataset_processor = AutoProcessor.from_pretrained(args.model_path, **loader_kwargs)
        if args.custom_chat_template_file:
            template = Path(args.custom_chat_template_file).read_text(encoding="utf-8")
            self.tokenizer.chat_template = template
            self.dataset_processor.chat_template = template

        self.dataset_processor_metadata_before_dataset = describe_processor(self.dataset_processor)
        print(
            "standalone dataset processor before dataset: "
            f"{json.dumps(json_safe(self.dataset_processor_metadata_before_dataset), ensure_ascii=False)}"
        )
        prompt_length_estimator = create_prompt_length_estimator(
            estimator_name=getattr(args, "ray_prompt_length_estimator", "tokenized"),
            model_path=args.model_path,
            require_supported=bool(getattr(args, "ray_require_prompt_length_estimator", False)),
        )
        print(
            "standalone prompt length estimator: "
            f"name={prompt_length_estimator.name} "
            f"required={bool(getattr(args, 'ray_require_prompt_length_estimator', False))} "
            f"safety_margin={int(getattr(args, 'ray_prompt_length_safety_margin', 0) or 0)}",
            flush=True,
        )

        tool_schemas = build_tool_schemas(parse_list_arg(args.qwen_tool_list))
        self.core_config = build_core_config(args, self.agent_settings, tool_schemas)
        args._core_config_for_export = dict(self.core_config.__dict__)
        self.prompt_signature = build_prompt_signature(
            args,
            self.agent_settings,
            tool_schemas,
            processor_metadata=self.dataset_processor_metadata_before_dataset,
        )
        args._prompt_signature_for_export = self.prompt_signature
        self.sampling_params = build_sampling_params(args)

        if args.generation_backend != "ray_vllm":
            raise ValueError(f"unsupported generation backend: {args.generation_backend}")
        self.ray_server_manifest, self.server_metadata = load_ray_server_manifest(args)
        self.stop_ray_heartbeat = start_ray_server_heartbeat(self.ray_server_manifest)

    async def load_rows(self, val_files: list[str], max_samples: int) -> list[dict[str, Any]]:
        provenance = build_row_provenance(val_files, -1)
        image_patch_size = resolve_dataset_image_patch_size(self.args, self.dataset_processor)
        print(f"standalone dataset image_patch_size: {image_patch_size}", flush=True)
        dataset = RLHFDataset(
            data_files=val_files,
            tokenizer=self.tokenizer,
            processor=self.dataset_processor,
            config=build_dataset_config(
                self.args,
                image_patch_size=image_patch_size,
            ),
            max_samples=-1,
        )
        if len(dataset) != len(provenance):
            raise RuntimeError(
                "RLHFDataset row count does not match parquet provenance count: "
                f"dataset={len(dataset)} provenance={len(provenance)}"
            )
        self.dataset_processor_metadata_after_dataset = describe_processor(self.dataset_processor)
        print(
            "standalone dataset processor after dataset: "
            f"{json.dumps(json_safe(self.dataset_processor_metadata_after_dataset), ensure_ascii=False)}"
        )

        rows = []
        for idx in range(len(dataset)):
            row = dataset[idx]
            row_agent_name = row.get("agent_name")
            if row_agent_name and str(row_agent_name) != str(self.args.agent_name):
                raise ValueError(
                    "standalone eval agent config does not match parquet row agent_name: "
                    f"row={row_agent_name!r} config={self.args.agent_name!r} "
                    f"file={provenance[idx][0]} row_idx={provenance[idx][1]}"
                )
            resume_val_file, resume_file_row_idx = provenance[idx]
            row["resume_val_file"] = resume_val_file
            row["resume_file_row_idx"] = int(resume_file_row_idx)
            if "uid" not in row:
                row["uid"] = str((row.get("extra_info") or {}).get("question_id") or idx)
            rows.append(row)
        return rows

    def basic_config_extra(self) -> dict[str, Any]:
        return {
            "prompt_signature": self.prompt_signature,
            "ray_server_manifest": {
                "model_config_sha256": (self.ray_server_manifest.get("model_config") or {}).get("sha256"),
            }
        }

    async def generate_many(
        self,
        jobs: list[RolloutJob],
        on_sample: Callable[[RolloutJob, dict[str, Any]], Awaitable[None]],
    ) -> None:
        if not jobs:
            return
        resolved_agent_worker_processes = min(max(1, self.args.agent_worker_processes), len(jobs))
        resolved_worker_concurrency = compute_worker_concurrency(self.args, resolved_agent_worker_processes)
        worker_args = vars(self.args).copy()
        worker_args["_resolved_worker_concurrency"] = resolved_worker_concurrency
        self.parallelism = {
            "agent_worker_processes": self.args.agent_worker_processes,
            "resolved_agent_worker_processes": resolved_agent_worker_processes,
            "worker_concurrency": self.args.worker_concurrency,
            "resolved_worker_concurrency": resolved_worker_concurrency,
            "total_concurrency": resolved_agent_worker_processes * resolved_worker_concurrency,
            "queue_policy": "global_queue",
            "processor_concurrency_per_process": DEFAULT_PROCESSOR_CONCURRENCY,
        }
        print(
            f"using {resolved_agent_worker_processes} global-queue agent worker processes for {len(jobs)} jobs "
            f"(total_concurrency={resolved_agent_worker_processes * resolved_worker_concurrency}, "
            f"worker_concurrency={self.args.worker_concurrency})",
            flush=True,
        )
        process_pool_kwargs: dict[str, Any] = {
            "max_workers": resolved_agent_worker_processes,
            "mp_context": mp.get_context("spawn"),
        }
        manager_context = mp.get_context("spawn")
        jobs_by_output = {job.output_index: job for job in jobs}
        with manager_context.Manager() as manager:
            job_queue = manager.Queue()
            result_queue = manager.Queue()
            for job in jobs:
                job_queue.put(
                    (
                        job.output_index,
                        job.job_key,
                        job.sample_index,
                        job.trial_idx,
                        job.resume_val_file,
                        job.resume_file_row_idx,
                        job.row,
                    )
                )
            for _ in range(resolved_agent_worker_processes * resolved_worker_concurrency):
                job_queue.put(None)
            with ProcessPoolExecutor(**process_pool_kwargs) as executor:
                futures = [
                    executor.submit(
                        run_agent_global_queue_worker_process,
                        args_dict=worker_args,
                        job_queue=job_queue,
                        result_queue=result_queue,
                        export_dir=str(self.export_dir),
                        worker_idx=worker_idx,
                    )
                    for worker_idx in range(resolved_agent_worker_processes)
                ]
                received = 0
                while received < len(jobs):
                    try:
                        item = result_queue.get(timeout=0.5)
                    except queue.Empty:
                        for future in futures:
                            if future.done():
                                exc = future.exception()
                                if exc is not None:
                                    raise exc
                        if all(future.done() for future in futures):
                            raise RuntimeError(f"all agent workers exited after {received}/{len(jobs)} queued samples")
                        continue
                    kind, _worker_idx, output_index, sample = item
                    if kind != "sample":
                        raise RuntimeError(f"unknown result queue item kind: {kind!r}")
                    await on_sample(jobs_by_output[int(output_index)], sample)
                    received += 1
                for future in as_completed(futures):
                    future.result()

    async def close(self) -> None:
        self.stop_ray_heartbeat()

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "core_config": self.core_config.__dict__ if self.core_config is not None else None,
            "prompt_signature": self.prompt_signature,
            "sampling_params": self.sampling_params,
            "ray_server_manifest": {
                "path": str(Path(self.args.ray_server_manifest).resolve()),
                "data": getattr(self.args, "_ray_server_manifest_data", None),
            },
            "processor_metadata": {
                "dataset_before_dataset": self.dataset_processor_metadata_before_dataset,
                "dataset_after_dataset": self.dataset_processor_metadata_after_dataset,
                "runtime_processor_concurrency": DEFAULT_PROCESSOR_CONCURRENCY,
            },
        }
