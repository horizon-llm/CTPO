# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding

_LAST_DUMPED_PG_IS_STEP: dict[int, int] = {}

# Policy-loss PG IS dump: same cumulative ratio as core_algos CTPO / CTPO-LA / CISPO-LA (pre–length-adaptive clip).
_PG_DUMP_CUM_TOKEN_GEOMEAN_MODES = frozenset({
    "cum-token-geomean",
    "cum-token-geomean-la",
})
_PG_DUMP_CUM_TOKEN_CUMPROD_MODES = frozenset({
    "cum-token-cumprod",
    "cum-token-cumprod-la",
})


def _get_scalar_non_tensor(data: TensorDict, key: str, default: Any):
    value = tu.get_non_tensor_data(data=data, key=key, default=default)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item() if value.dtype != object else value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else default
    return value


def _maybe_dump_pg_update_is_weights(
    *,
    data: TensorDict,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mode: str,
) -> None:
    """Dump policy-loss IS ratios used by PG update for selected loss modes."""
    dump_dir = _get_scalar_non_tensor(data, "pg_is_dump_dir", None)
    if not dump_dir:
        return
    global_step = int(_get_scalar_non_tensor(data, "pg_is_dump_global_step", -1))
    dump_interval = int(_get_scalar_non_tensor(data, "pg_is_dump_interval", 10))
    if global_step < 0 or dump_interval <= 0 or (global_step % dump_interval != 0):
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    if _LAST_DUMPED_PG_IS_STEP.get(rank) == global_step:
        return

    mask = response_mask.to(dtype=log_prob.dtype)
    negative_approx_kl = log_prob - old_log_prob

    # Unclipped PG importance ratio (same construction as core_algos cumulative CTPO / CTPO-LA / CISPO-LA).
    if loss_mode in _PG_DUMP_CUM_TOKEN_GEOMEAN_MODES | _PG_DUMP_CUM_TOKEN_CUMPROD_MODES:
        # Same as compute_policy_loss_cum_token() / _compute_policy_loss_cum_token_la_impl (ratio before LA clip).
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        cumulative_sum = torch.cumsum(negative_approx_kl * mask, dim=-1)
        if loss_mode in _PG_DUMP_CUM_TOKEN_GEOMEAN_MODES:
            cumulative_count = torch.cumsum(mask, dim=-1).clamp(min=1.0)
            kl_values = torch.where(mask > 0, cumulative_sum / cumulative_count, torch.zeros_like(cumulative_sum))
        else:
            kl_values = torch.where(mask > 0, cumulative_sum, torch.zeros_like(cumulative_sum))
        log_importance_ratio = kl_values.detach() + log_prob - log_prob.detach()
        log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)
        ratio = torch.where(mask > 0, torch.exp(log_importance_ratio), torch.zeros_like(log_importance_ratio))
    elif loss_mode == "gspo":
        # Same as compute_policy_loss_gspo()
        seq_lengths = torch.sum(mask, dim=-1).clamp(min=1.0)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * mask, dim=-1) / seq_lengths
        log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
        ratio = torch.where(mask > 0, torch.exp(log_seq_importance_ratio), torch.zeros_like(log_seq_importance_ratio))
    elif loss_mode == "vanilla":
        # Same as compute_policy_loss_vanilla()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.where(mask > 0, torch.exp(negative_approx_kl), torch.zeros_like(negative_approx_kl))
    else:
        return

    os.makedirs(dump_dir, exist_ok=True)
    filename = os.path.join(dump_dir, f"step_{global_step:07d}_rank_{rank:04d}.npz")
    np.savez_compressed(
        filename,
        pg_update_is_weights=ratio.detach().to(torch.float32).cpu().numpy(),
        response_mask=response_mask.detach().to(torch.int8).cpu().numpy(),
        global_step=np.array([global_step], dtype=np.int64),
        rank=np.array([rank], dtype=np.int64),
    )
    _LAST_DUMPED_PG_IS_STEP[rank] = global_step


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: model output tensor of shape [bsz, 1]
        data: TensorDict with "prompt_ids", "response_ids", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = response_ids.offsets().max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    if loss_mode in {
        "cum-token-geomean",
        "cum-token-cumprod",
        "cum-token-geomean-la",
        "cum-token-cumprod-la",
        "vanilla",
        "gspo",
    }:
        _maybe_dump_pg_update_is_weights(
            data=data,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            response_mask=response_mask,
            loss_mode=loss_mode,
        )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
