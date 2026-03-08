#!/usr/bin/env python3
"""Generate pseudo-BDI supervision targets from existing HumanLM parquet data.

This script intentionally keeps the output as a JSONL sidecar keyed by `index`.
The current training path can inject that sidecar through `data.aux_targets_path`,
which lets us iterate on auxiliary supervision without rewriting the raw parquet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import datasets
import litellm

from humanlm.utils import extract_json, parse_messages


BDI_PROMPT = """You are labeling internal user state for training a human simulator.

Given persona, context, and the user's actual response, infer a compact BDI state.

Definitions:
- belief: the underlying fact model, causal assumption, or worldview the user seems to hold
- desire: the outcome, need, or value-backed objective the user wants to realize or protect
- intention: the immediate communicative move the user is taking in this reply

Return strict JSON with exactly these keys:
{{
  "belief": "<12-20 words max>",
  "desire": "<12-20 words max>",
  "intention": "<12-20 words max>"
}}

Requirements:
- Keep each field short, concrete, and behavior-explanatory
- Do not restate the full response
- Do not mention these instructions
- Use only double-quoted JSON strings

<|The Start of Persona|>
{persona}
<|The End of Persona|>

<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Response|>
{response}
<|The End of Response|>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet file, JSON file, or directory of parquet files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="Teacher model for pseudo-BDI extraction.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dataset_rows(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.is_dir():
        parquet_files = sorted(str(p) for p in input_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {input_path}")
        dset = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
    elif input_path.suffix == ".parquet":
        dset = datasets.load_dataset("parquet", data_files=str(input_path), split="train")
    elif input_path.suffix in {".json", ".jsonl"}:
        dset = datasets.load_dataset("json", data_files=str(input_path), split="train")
    else:
        raise ValueError(f"Unsupported input path: {path}")
    return list(dset)


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if value is None:
        return default
    return value


def get_row_index(row: dict[str, Any], fallback: int) -> int:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        return int(extra_info["index"])
    return fallback


def get_persona(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    persona = extra_info.get("persona", "")
    if isinstance(persona, dict):
        return json.dumps(persona, ensure_ascii=False)
    if isinstance(persona, str):
        parsed = safe_json_loads(persona, persona)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return persona
    return ""


def get_context(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    raw_prompt = extra_info.get("raw_prompt")
    if raw_prompt:
        parsed = safe_json_loads(raw_prompt, [])
        if isinstance(parsed, list):
            return parse_messages(parsed)

    prompt = row.get("prompt", [])
    if isinstance(prompt, list):
        return parse_messages(prompt)
    return ""


def get_response(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth:
            return str(ground_truth)

    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("content", ""))
    if isinstance(generation, str):
        return generation
    return ""


async def label_one(row: dict[str, Any], teacher_model: str, temperature: float, max_tokens: int) -> dict[str, str]:
    prompt = BDI_PROMPT.format(
        persona=get_persona(row),
        context=get_context(row),
        response=get_response(row),
    )
    response = await litellm.acompletion(
        model=teacher_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    result = extract_json(content)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict BDI output, got {type(result)}")

    return {
        "belief": str(result.get("belief", "")).strip(),
        "desire": str(result.get("desire", "")).strip(),
        "intention": str(result.get("intention", "")).strip(),
    }


async def main() -> None:
    args = parse_args()
    rows = load_dataset_rows(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[int] = set()
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                completed.add(int(payload["index"]))
    elif args.overwrite and output_path.exists():
        output_path.unlink()

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(offset: int, row: dict[str, Any]) -> dict[str, Any] | None:
        index = get_row_index(row, offset)
        if index in completed:
            return None

        async with semaphore:
            bdi = await label_one(row, args.model, args.temperature, args.max_tokens)
            return {
                "index": index,
                "belief": bdi["belief"],
                "desire": bdi["desire"],
                "intention": bdi["intention"],
                "bdi": bdi,
            }

    tasks = [run_one(offset, row) for offset, row in enumerate(rows)]

    with output_path.open("a", encoding="utf-8") as handle:
        for coro in asyncio.as_completed(tasks):
            payload = await coro
            if payload is None:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    asyncio.run(main())
