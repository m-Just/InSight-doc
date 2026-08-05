#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a standalone Hugging Face checkpoint."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="HF model directory containing base weights.")
    parser.add_argument("--adapter", required=True, help="PEFT LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Where to write the merged HF checkpoint.")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=False)
        processor.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to save processor: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to save tokenizer: {exc}")

    print(f"Merged model written to {output_dir}")


if __name__ == "__main__":
    main()
