# CTPO

Code for **Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective**.

CTPO (Cumulative Token Policy Optimization) studies the importance-sampling ratio used in LLM policy optimization. Instead of using only the current-token ratio as in PPO/GRPO, or a full sequence ratio with high variance, CTPO uses the cumulative token ratio up to each position. The method also applies position-adaptive log-space clipping, scaling the clip range with token position to better match the variance growth of the cumulative log-ratio.

This repository contains a VERL-based implementation of CTPO for tool-integrated reasoning (TIR), including multi-turn rollout with a local Python sandbox.

## Paper

- Paper: [Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective](https://arxiv.org/abs/2605.07331)
- Method: CTPO / CTPO-LA
- Main training entry: `ctpo.sh`

## Built On VERL

This codebase is built on [VERL](https://github.com/verl-project/verl), a reinforcement learning framework for LLM post-training. We keep VERL's training, rollout, FSDP, vLLM/SGLang, logging, and configuration infrastructure, and add the CTPO policy-loss implementation and TIR training script.

The main CTPO loss is implemented as `cum-token-cumprod-la` in:

- `verl/trainer/ppo/core_algos.py`
- `verl/workers/utils/losses.py`
- `verl/workers/config/actor.py`

## Setup

Install the base dependencies following VERL's installation instructions for your CUDA/PyTorch environment. This repository also uses math verification utilities and a local FastAPI sandbox; `ctpo.sh` installs the small Python-side dependencies it needs at runtime.

## Data

The default script expects parquet data under:

```text
dataset/deepscaler/train.parquet
dataset/aime/aime25.parquet
dataset/competition_benchmarks/aime26.parquet
```

You can override these paths without editing the script:

```bash
export TRAIN_FILES='["/path/to/train.parquet"]'
export VAL_FILES='["/path/to/val.parquet"]'
```

## Run

Launch the default CTPO-LA training job:

```bash
bash ctpo.sh
```

Common overrides:

```bash
MODEL_PATH=Qwen/Qwen3-14B \
TRAIN_BATCH_SIZE=512 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=8 \
NGPUS=8 \
bash ctpo.sh
```

The script starts a local sandbox server, runs a smoke test, and then launches `verl.trainer.main_ppo` with the CTPO policy loss.

## Citation

```bibtex
@article{zhang2026rethinking,
  title={Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective},
  author={Zhang, Yuheng and Ye, Chenlu and Jin, Shuowei and Yu, Changlong and Xiong, Wei and Sahu, Saurabh and Jiang, Nan},
  journal={arXiv preprint arXiv:2605.07331},
  year={2026}
}
```

## Acknowledgements

This implementation is built on VERL. Please also cite and follow the license of the upstream VERL project when using this code.
