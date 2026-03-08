import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path("/blue/buyuheng/chengzhi.ucsb/code/toby/humanlm/humanlm_train/verl-recipe-humanlm")
BUILD_SCRIPT = REPO_ROOT / "humanlm" / "build_bdi_aux_dataset.py"
BOOTSTRAP_SCRIPT = REPO_ROOT / "humanlm" / "bootstrap_bdi_targets.py"


def run_script(script: Path, *args: str, timeout: int = 5) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def sample_persona() -> str:
    return json.dumps(
        {
            "demographics": {"age": "30s", "location": "CA"},
            "interests": ["hiking", "reading"],
            "values": ["kindness", "honesty"],
            "communication": ["direct", "calm"],
            "statistics": ["posts daily", "replies thoughtfully"],
        }
    )


def sample_rl_row(index: int) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "old prompt", "name": ""},
            {"role": "user", "content": "How should I respond?", "name": "POSTER"},
        ],
        "generation": {"role": "user", "content": "I agree with you.", "name": "HUMAN"},
        "extra_info": {"index": index, "persona": sample_persona(), "name": "HUMAN"},
    }


def sample_targets(index: int) -> dict:
    bdi = {
        "belief": "People deserve honest answers.",
        "desire": "Keep the conversation respectful.",
        "intention": "Agree while sounding measured.",
    }
    return {"index": index, **bdi, "bdi": bdi}


def test_bootstrap_help_returns_quickly():
    result = run_script(BOOTSTRAP_SCRIPT, "--help")
    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout


def test_build_help_returns_quickly():
    result = run_script(BUILD_SCRIPT, "--help")
    assert result.returncode == 0, result.stderr
    assert "--input-dir" in result.stdout


def test_build_bdi_aux_dataset_from_local_parquet(tmp_path: Path):
    input_dir = tmp_path / "input"
    targets_dir = tmp_path / "targets"
    output_dir = tmp_path / "output"

    write_parquet(input_dir / "train.parquet", [sample_rl_row(0)])
    write_parquet(input_dir / "val.parquet", [sample_rl_row(1)])
    write_jsonl(targets_dir / "train.jsonl", [sample_targets(0)])
    write_jsonl(targets_dir / "val.jsonl", [sample_targets(1)])

    result = run_script(
        BUILD_SCRIPT,
        "--input-dir",
        str(input_dir),
        "--targets-dir",
        str(targets_dir),
        "--output-dir",
        str(output_dir),
        "--splits",
        "train",
        "val",
        timeout=30,
    )

    assert result.returncode == 0, result.stderr

    train_rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    val_rows = pq.read_table(output_dir / "val.parquet").to_pylist()

    assert len(train_rows) == 3
    assert len(val_rows) == 3
    assert "<belief>" in train_rows[0]["generation"]["content"]
    assert "Your persona" in train_rows[0]["prompt"][0]["content"]
