#!/usr/bin/env python3
"""Build an SFT dataset for BDI auxiliary supervision from RL parquet + pseudo-BDI sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SLOT_PROMPTS = {
    "belief": ROOT / "system_prompts" / "bdi_belief.txt",
    "desire": ROOT / "system_prompts" / "bdi_desire.txt",
    "intention": ROOT / "system_prompts" / "bdi_intention.txt",
}
SFT_TEMPLATE = {
    "prompt": [
        {"role": "system", "content": "", "name": ""},
        {"role": "user", "content": "", "name": ""},
    ],
    "generation": {"role": "user", "content": "", "name": ""},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing train/val/test parquet files.")
    parser.add_argument("--targets-dir", required=True, help="Directory containing split-matched pseudo-BDI JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write SFT parquet files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process. Only splits with both parquet and target JSONL are used.",
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        default=["belief", "desire", "intention"],
        choices=["belief", "desire", "intention"],
        help="BDI slots to materialize as separate supervision examples.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for quick experiments.")
    return parser.parse_args()


def load_targets(path: Path) -> dict[int, dict[str, str]]:
    by_index: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            index = payload.get("index")
            if index is None:
                raise ValueError(f"Missing index on line {line_no} in {path}")
            bdi = payload.get("bdi", {})
            by_index[int(index)] = {
                "belief": str(payload.get("belief", bdi.get("belief", ""))).strip(),
                "desire": str(payload.get("desire", bdi.get("desire", ""))).strip(),
                "intention": str(payload.get("intention", bdi.get("intention", ""))).strip(),
            }
    return by_index


def read_prompt_template(slot: str) -> str:
    prompt_path = SLOT_PROMPTS[slot]
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt template for slot '{slot}': {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def format_slot_generation(slot: str, value: str, speaker_name: str) -> dict[str, str]:
    return {
        "role": "user",
        "name": speaker_name,
        "content": f"<{slot}>\n{value}\n</{slot}>",
    }


def ensure_extra_info(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return dict(extra_info)
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def ensure_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    if isinstance(prompt, str):
        parsed = json.loads(prompt)
        if isinstance(parsed, list):
            return [dict(message) for message in parsed]
    raise ValueError(f"Unsupported prompt payload: {type(prompt)}")


def replace_system_prompt(messages: list[dict[str, Any]], new_system_prompt: str) -> list[dict[str, Any]]:
    output = [dict(message) for message in messages]
    if not output:
        return [{"role": "system", "name": "", "content": new_system_prompt}]
    if output[0].get("role") == "system":
        output[0]["content"] = new_system_prompt
        output[0]["name"] = output[0].get("name", "")
    else:
        output.insert(0, {"role": "system", "name": "", "content": new_system_prompt})
    return output


def format_persona_from_extra(extra_info: dict[str, Any]) -> str:
    persona = extra_info.get("persona", "")
    if isinstance(persona, str):
        try:
            persona = json.loads(persona)
        except json.JSONDecodeError:
            return persona
    if isinstance(persona, dict):
        return render_persona(persona)
    return str(persona)


def render_persona(persona: dict[str, Any]) -> str:
    if not isinstance(persona, dict):
        return str(persona)

    lines: list[str] = []
    demographics = persona.get("demographics", {}) or {}
    if demographics:
        lines.append("Demographics:")
        for key, value in demographics.items():
            if value and str(value).strip() != "NA":
                lines.append(f"  {key}: {value}")
    else:
        lines.append("Demographics: Missing")

    for aspect in ["interests", "values", "communication", "statistics"]:
        values = persona.get(aspect, []) or []
        if values:
            lines.append(f"{aspect.capitalize()}:")
            for item in values:
                lines.append(f"  {item}")
        else:
            lines.append(f"{aspect.capitalize()}: Missing")
    return "\n".join(lines)


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    message_struct = pa.struct(
        [
            ("role", pa.string()),
            ("content", pa.string()),
            ("name", pa.string()),
        ]
    )
    sft_schema = pa.schema(
        [
            ("prompt", pa.list_(message_struct)),
            ("generation", message_struct),
        ]
    )

    if rows:
        table = pa.Table.from_pylist(rows, schema=sft_schema)
    else:
        table = pa.Table.from_pylist([], schema=sft_schema)
    pq.write_table(table, path)


def build_rows(
    rows: list[dict[str, Any]],
    targets: dict[int, dict[str, str]],
    slots: list[str],
) -> list[dict[str, Any]]:
    templates = {slot: read_prompt_template(slot) for slot in slots}
    output_rows: list[dict[str, Any]] = []

    for fallback_index, row in enumerate(rows):
        extra_info = ensure_extra_info(row.get("extra_info"))
        index = int(extra_info.get("index", fallback_index))
        slot_targets = targets.get(index)
        if slot_targets is None:
            continue

        prompt = ensure_prompt(row.get("prompt"))
        persona = format_persona_from_extra(extra_info)
        speaker_name = str(extra_info.get("name") or "HUMAN")

        for slot in slots:
            target = slot_targets.get(slot, "").strip()
            if not target:
                continue

            system_prompt = templates[slot].format(persona=persona)
            slot_prompt = replace_system_prompt(prompt, system_prompt)
            output_rows.append(
                {
                    "prompt": slot_prompt,
                    "generation": format_slot_generation(slot, target, speaker_name),
                }
            )

    return output_rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    targets_dir = Path(args.targets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        parquet_path = input_dir / f"{split}.parquet"
        target_path = targets_dir / f"{split}.jsonl"
        if not parquet_path.exists() or not target_path.exists():
            print(f"Skipping {split}: missing parquet or target file")
            continue

        rows = load_parquet_rows(parquet_path)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        targets = load_targets(target_path)
        built_rows = build_rows(rows, targets, args.slots)

        out_parquet = output_dir / f"{split}.parquet"
        out_preview = output_dir / f"{split}.example.json"
        write_parquet_rows(out_parquet, built_rows)
        if built_rows:
            with out_preview.open("w", encoding="utf-8") as handle:
                json.dump(built_rows[:10], handle, ensure_ascii=False, indent=2)
        print(f"Wrote {split}: {len(built_rows)} rows -> {out_parquet}")


if __name__ == "__main__":
    main()
