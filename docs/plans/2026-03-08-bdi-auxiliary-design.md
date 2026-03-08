# BDI Auxiliary Training Design

## Context

This branch changes HumanLM in one specific direction:

- simplify the original state space into `belief / desire / intention / response`
- add an auxiliary supervision path so BDI is not only an RL-time generation format
- keep the current RL recipe intact enough that we can still warm-start from auxiliary SFT and continue with GRPO

The main reason for this design is that the original HumanLM path depends heavily on online LLM-as-judge reward. That is useful as a bootstrap signal, but it is expensive, high-variance, and structurally weak for learning a stable latent representation of user intent. The BDI path is meant to create a more compact and more reusable internal state interface before we attempt more aggressive judge-free training.

## Design Goal

The design goal is not to fully remove judge-based reward in one step.

The goal is to build an intermediate training path:

1. extract pseudo-BDI supervision offline from `persona + context + ground-truth response`
2. train the model to predict BDI slots with regular SFT
3. use that checkpoint as the warm start for BDI RL
4. inject the same pseudo-BDI sidecar back into the RL dataset so later losses or metrics can use it

This gives us a staged path from response imitation to structured latent alignment, without breaking the current HumanLM training stack.

## Why BDI

`belief / desire / intention` is the smallest state decomposition here that still has causal structure:

- `belief`: what the user seems to treat as true or likely true
- `desire`: what outcome or value the user wants to protect
- `intention`: what communicative move the user is taking right now

Compared with the original wider state set, this version is easier to supervise, easier to explain, and closer to the later latent-consistency direction we discussed. It also matches the SIM-CoT-inspired intuition that latent reasoning needs intermediate structure, not only final-output reward.

## Core Principle

The current implementation follows one principle:

`judge-free training should first become judge-light training, not supervision-free training`

That means:

- we do not trust pure self-play latent reward yet
- we first add structured slot supervision with pseudo labels
- we keep RL as the second stage, not the only stage

This is the practical bridge between the current HumanLM recipe and the longer-term latent reward idea.

## Implemented Architecture

### 1. Offline pseudo-BDI extraction

File: `humanlm/bootstrap_bdi_targets.py`

This script reads existing RL-format parquet/json input and writes a JSONL sidecar keyed by example `index`.

Each output row contains:

- `belief`
- `desire`
- `intention`
- nested `bdi`

Important design choices:

- output is a sidecar, not an in-place rewrite of the RL parquet
- extraction is defined against the ground-truth response, not the model rollout
- teacher dependence is isolated to an offline step

This keeps the expensive teacher call out of the online RL loop.

### 2. Slot-level auxiliary SFT dataset

File: `humanlm/build_bdi_aux_dataset.py`

This script converts RL parquet plus pseudo-BDI sidecars into an SFT dataset where one original sample becomes up to three supervision samples:

- belief sample
- desire sample
- intention sample

Each sample:

- reuses the original prompt structure
- replaces the system prompt with the slot-specific BDI prompt
- sets the target generation to the corresponding XML-tagged slot output

This is the SIM-CoT-inspired part of the design: supervision is applied at slot level, not only at final response level.

### 3. Auxiliary decoder training

File: `humanlm/train_sft_bdi_aux.sh`

This is a standard SFT entrypoint for the BDI auxiliary dataset. It does not change the underlying trainer. It only provides a stable training path for:

- BDI slot prediction
- multirole chat template usage
- checkpoint production for RL warm start

The reason to keep this as plain SFT is to reduce moving parts. We want one clean stage where the model learns BDI structure before RL noise is introduced.

### 4. End-to-end auxiliary pipeline

File: `humanlm/train_bdi_aux_pipeline.sh`

This script bundles the previous two stages:

1. bootstrap pseudo-BDI targets
2. build slot-level auxiliary SFT parquet
3. launch auxiliary SFT

This is the shortest path to produce a BDI-aware checkpoint from existing HumanLM RL data.

### 5. RL warm start from auxiliary checkpoint

Files:

- `humanlm/train_bdi_rl_from_aux.sh`
- `humanlm/train_rl_humanlm.sh`

`train_bdi_rl_from_aux.sh` exists so the user does not need to manually wire:

- model checkpoint override
- auxiliary sidecar path
- `train_bdi_humanlm` mode

`train_rl_humanlm.sh` was updated so RL can use `MODEL_PATH_OVERRIDE` and keep the existing cluster config intact.

This preserves backward compatibility for existing HumanLM training while adding a clear BDI-specific path.

### 6. Split-aware auxiliary target injection

File: `humanlm/state_dataset.py`

The RL dataset class now supports `aux_targets_path` as either:

- a single JSONL file
- a directory containing split-matched sidecars such as `train.jsonl`, `val.jsonl`, `test.jsonl`

The dataset injects:

- `pseudo_belief`
- `pseudo_desire`
- `pseudo_intention`

into `extra_info`.

This was added so later RL losses, metrics, or auxiliary heads can reuse the same targets without changing the underlying parquet schema again.

## Why Sidecars Instead of Rewriting Parquet

This was an intentional design choice.

Benefits:

- no destructive rewrite of existing processed data
- teacher targets can be regenerated independently
- RL and SFT can share the same source data with different sidecars
- easier ablation across different teacher models or prompt styles

This makes experimentation cheaper and reduces data-pipeline coupling.

## Why Auxiliary SFT Before RL

There are three reasons.

### 1. Stability

If BDI only appears during RL generation, the model can learn output formatting before it learns state semantics.

### 2. Better initialization

Warm-starting RL from a BDI-aware checkpoint should reduce exploration waste. The policy starts from a model that already knows how to express BDI slots.

### 3. Cleaner future transition to latent reward

Later, if we add latent consistency losses or a BDI projector, we want the model to already have a structured internal interface. Auxiliary SFT is the simplest first step toward that.

## What This Design Does Not Do Yet

This branch does not yet implement:

- a learned BDI encoder or projector
- latent-space self-reward
- cycle-consistency between `response -> BDI -> response`
- joint multi-task training where auxiliary loss is applied inside RL updates
- removal of online judge reward from the RL stage

Those are later stages. This branch is only the bootstrap layer that makes those stages feasible.

## Expected Experimental Use

The intended training order is:

1. prepare or reuse HumanLM RL parquet
2. run `train_bdi_aux_pipeline.sh`
3. obtain an auxiliary SFT checkpoint
4. run `train_bdi_rl_from_aux.sh`
5. compare against the original HumanLM RL path

The first comparison to care about is not final benchmark gain. It is whether the BDI path produces:

- cleaner slot generations
- better formatting validity
- more stable RL startup
- less dependence on online judge signal early in training

## Main Tradeoff

The design trades one thing for another:

- it adds an offline teacher stage
- in exchange, it removes pressure from the online RL loop

I think this is the right tradeoff for now. Pure judge-free latent reward is still too unconstrained. This branch intentionally chooses a conservative bridge: structured pseudo supervision first, latent reward later.

## Summary

The implemented design is a three-stage bridge:

1. teacher extracts pseudo BDI
2. auxiliary SFT teaches slot semantics
3. RL continues from that checkpoint with BDI-aware state expansion

The point is not that BDI is the final answer. The point is that BDI gives us a smaller and more usable interface for the next generation of HumanLM experiments, especially if we later move reward from external judge signals toward latent consistency.
