from __future__ import annotations

import asyncio
import uuid
from typing import Any

from omegaconf import OmegaConf


def ray_namespace(args: Any) -> str:
    namespace = getattr(args, "_ray_namespace", None) or getattr(args, "ray_namespace", None)
    if namespace:
        return str(namespace)
    return f"evals_{uuid.uuid4().hex[:12]}"


def ray_rollout_config(args: Any) -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "async",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "dtype": args.ray_dtype,
            "gpu_memory_utilization": args.ray_gpu_memory_utilization,
            "ignore_eos": False,
            "enforce_eager": args.ray_enforce_eager,
            "free_cache_engine": False,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "tensor_model_parallel_size": args.ray_gpus_per_replica,
            "pipeline_model_parallel_size": 1,
            "max_num_batched_tokens": args.ray_max_num_batched_tokens,
            "scheduling_policy": args.ray_scheduling_policy,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.ray_max_num_seqs,
            "disable_log_stats": args.ray_disable_log_stats,
            "enable_chunked_prefill": args.ray_enable_chunked_prefill,
            "enable_prefix_caching": args.ray_enable_prefix_caching,
            "load_format": args.ray_load_format,
            "enable_sleep_mode": args.ray_enable_sleep_mode,
            "engine_kwargs": {"vllm": {"max_model_len": args.max_model_len}},
        }
    )


def ray_model_config(args: Any) -> Any:
    return OmegaConf.create(
        {
            "path": args.model_path,
            "trust_remote_code": args.ray_trust_remote_code,
            "load_tokenizer": True,
            "use_shm": False,
            "lora_rank": 0,
            "override_config": {},
        }
    )


async def launch_ray_vllm_servers(args: Any) -> tuple[list[Any], list[str], list[dict[str, Any]], str]:
    import ray
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    namespace = ray_namespace(args)
    ray_init_kwargs: dict[str, Any] = {
        "namespace": namespace,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": True,
    }
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    else:
        ray_init_kwargs["num_gpus"] = args.ray_num_replicas * args.ray_gpus_per_replica
        if args.ray_num_cpus:
            ray_init_kwargs["num_cpus"] = args.ray_num_cpus
        ray_init_kwargs["_temp_dir"] = args.ray_temp_dir or f"/tmp/vray_{uuid.uuid4().hex[:6]}"
    ray_context = ray.init(**ray_init_kwargs)

    actor_cls = ray.remote(vLLMHttpServer)
    rollout_config = ray_rollout_config(args)
    model_config = ray_model_config(args)
    handles: list[Any] = []
    actor_names: list[str] = []
    metadata: list[dict[str, Any]] = []

    for replica_rank in range(args.ray_num_replicas):
        actor_name = f"standalone_vllm_{replica_rank}_{uuid.uuid4().hex[:8]}"
        server = actor_cls.options(
            num_gpus=args.ray_gpus_per_replica,
            num_cpus=args.ray_cpus_per_server,
            name=actor_name,
            namespace=namespace,
        ).remote(
            config=rollout_config,
            model_config=model_config,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=replica_rank,
            node_rank=0,
            gpus_per_node=args.ray_gpus_per_replica,
            nnodes=1,
        )
        handles.append(server)
        actor_names.append(actor_name)

    master_addresses = await asyncio.gather(*[server.get_master_address.remote() for server in handles])
    for replica_rank, (master_address, master_port) in enumerate(master_addresses):
        print(
            f"launching Ray vLLM server replica={replica_rank} "
            f"actor={actor_names[replica_rank]} master={master_address}:{master_port}",
            flush=True,
        )

    await asyncio.gather(
        *[
            server.launch_server.remote(master_address=master_address, master_port=master_port)
            for server, (master_address, master_port) in zip(handles, master_addresses, strict=True)
        ]
    )

    server_addresses = await asyncio.gather(*[server.get_server_address.remote() for server in handles])
    for replica_rank, (server_address, server_port) in enumerate(server_addresses):
        metadata.append(
            {
                "endpoint_type": "verl_ray_vllm",
                "actor_name": actor_names[replica_rank],
                "server_address": f"{server_address}:{server_port}",
                "replica_rank": replica_rank,
                "model": args.model_path,
                "max_model_len": args.max_model_len,
                "ray_namespace": namespace,
            }
        )
        print(
            f"ready Ray vLLM server replica={replica_rank} "
            f"actor={actor_names[replica_rank]} address={server_address}:{server_port}",
            flush=True,
        )

    args._ray_namespace = namespace
    args._ray_actor_names = actor_names
    address_info = getattr(ray_context, "address_info", {}) or {}
    args._ray_address = args.ray_address or address_info.get("address") or address_info.get("gcs_address") or "auto"
    return handles, actor_names, metadata, namespace


def connect_ray_vllm_servers(args: Any) -> list[Any]:
    import ray

    namespace = getattr(args, "_ray_namespace", None) or getattr(args, "ray_namespace", None)
    actor_names = list(getattr(args, "_ray_actor_names", None) or [])
    if not namespace or not actor_names:
        raise ValueError("Ray server actor names/namespace are missing; pass --ray-server-manifest")
    if not ray.is_initialized():
        ray.init(address=getattr(args, "_ray_address", None) or getattr(args, "ray_address", None) or "auto", namespace=namespace)
    return [ray.get_actor(name, namespace=namespace) for name in actor_names]


def cleanup_ray_vllm_servers(handles: list[Any]) -> None:
    if not handles:
        return
    try:
        import ray

        for handle in handles:
            ray.kill(handle, no_restart=True)
        ray.shutdown()
    except Exception as exc:
        print(f"warning: failed to clean up Ray vLLM servers: {exc}", flush=True)
