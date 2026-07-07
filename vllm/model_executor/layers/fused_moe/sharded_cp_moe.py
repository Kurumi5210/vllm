# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-CP MoE token-row layout helpers.

These helpers describe the token-row communication around MoE layers without
depending on a specific expert kernel.  They are small enough to unit-test on
CPU and are reused by the DeepSeek Sharded-CP path when it enters the MoE.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed.sharded_cp_utils import (
    ShardedCPTokenRange,
    all_gather_token_rows,
    assemble_token_all_gather_chunks,
    reduce_scatter_token_rows,
)


@dataclass(frozen=True)
class ShardedCPMoEInputs:
    """Global MoE inputs assembled from CP-local token rows."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    activation_scales: torch.Tensor | None = None


@dataclass(frozen=True)
class ShardedCPMoERoutingMetadata:
    """Global top-k routing metadata assembled in token-row order."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


def _validate_local_token_rows(
    name: str,
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> None:
    if x.shape[0] != token_range.num_tokens:
        raise ValueError(
            f"{name} row count must match token_range.num_tokens: "
            f"got {x.shape[0]}, expected {token_range.num_tokens}."
        )


def _validate_global_token_rows(
    name: str,
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> None:
    if x.shape[0] != token_range.total_tokens:
        raise ValueError(
            f"{name} row count must match token_range.total_tokens: "
            f"got {x.shape[0]}, expected {token_range.total_tokens}."
        )


def _validate_same_local_rows(
    lhs_name: str,
    lhs: torch.Tensor,
    rhs_name: str,
    rhs: torch.Tensor,
) -> None:
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(
            f"{lhs_name} and {rhs_name} must have the same row count: "
            f"got {lhs.shape[0]} and {rhs.shape[0]}."
        )


def assemble_sharded_cp_moe_input_chunks(
    gathered_hidden_states: list[torch.Tensor],
    gathered_router_logits: list[torch.Tensor],
    token_range: ShardedCPTokenRange,
    gathered_activation_scales: list[torch.Tensor] | None = None,
) -> ShardedCPMoEInputs:
    """Assemble gathered MoE input chunks into global token-row order."""
    hidden_states = assemble_token_all_gather_chunks(
        gathered_hidden_states, token_range
    )
    router_logits = assemble_token_all_gather_chunks(
        gathered_router_logits, token_range
    )
    if hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError(
            "assembled hidden states and router logits must have the same "
            f"row count: got {hidden_states.shape[0]} and "
            f"{router_logits.shape[0]}."
        )

    activation_scales = None
    if gathered_activation_scales is not None:
        activation_scales = assemble_token_all_gather_chunks(
            gathered_activation_scales, token_range
        )
        if activation_scales.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "assembled activation scales and hidden states must have the "
                f"same row count: got {activation_scales.shape[0]} and "
                f"{hidden_states.shape[0]}."
            )

    return ShardedCPMoEInputs(
        hidden_states=hidden_states,
        router_logits=router_logits,
        activation_scales=activation_scales,
    )


def all_gather_sharded_cp_moe_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    activation_scales: torch.Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> ShardedCPMoEInputs:
    """All-gather local MoE activations and router logits for expert dispatch."""
    _validate_local_token_rows("hidden_states", hidden_states, token_range)
    _validate_local_token_rows("router_logits", router_logits, token_range)
    _validate_same_local_rows(
        "hidden_states", hidden_states, "router_logits", router_logits
    )
    if hidden_states.device != router_logits.device:
        raise ValueError("hidden_states and router_logits must be on the same device.")

    global_hidden_states = all_gather_token_rows(
        hidden_states,
        token_range,
        group=group,
    )
    global_router_logits = all_gather_token_rows(
        router_logits,
        token_range,
        group=group,
    )
    global_activation_scales = None
    if activation_scales is not None:
        _validate_local_token_rows(
            "activation_scales", activation_scales, token_range
        )
        _validate_same_local_rows(
            "hidden_states", hidden_states, "activation_scales", activation_scales
        )
        if activation_scales.device != hidden_states.device:
            raise ValueError(
                "activation_scales and hidden_states must be on the same device."
            )
        global_activation_scales = all_gather_token_rows(
            activation_scales,
            token_range,
            group=group,
        )

    return ShardedCPMoEInputs(
        hidden_states=global_hidden_states,
        router_logits=global_router_logits,
        activation_scales=global_activation_scales,
    )


def assemble_sharded_cp_moe_routing_chunks(
    gathered_topk_weights: list[torch.Tensor],
    gathered_topk_ids: list[torch.Tensor],
    token_range: ShardedCPTokenRange,
) -> ShardedCPMoERoutingMetadata:
    """Assemble gathered top-k routing chunks into global token-row order."""
    topk_weights = assemble_token_all_gather_chunks(
        gathered_topk_weights, token_range
    )
    topk_ids = assemble_token_all_gather_chunks(gathered_topk_ids, token_range)
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "assembled topk_weights and topk_ids must have the same shape: "
            f"got {tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}."
        )
    return ShardedCPMoERoutingMetadata(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def all_gather_sharded_cp_moe_routing_metadata(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
) -> ShardedCPMoERoutingMetadata:
    """All-gather top-k routing metadata while preserving global token order."""
    _validate_local_token_rows("topk_weights", topk_weights, token_range)
    _validate_local_token_rows("topk_ids", topk_ids, token_range)
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape: "
            f"got {tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}."
        )
    if topk_weights.device != topk_ids.device:
        raise ValueError("topk_weights and topk_ids must be on the same device.")

    global_topk_weights = all_gather_token_rows(
        topk_weights,
        token_range,
        group=group,
    )
    global_topk_ids = all_gather_token_rows(
        topk_ids,
        token_range,
        group=group,
        pad_value=-1.0,
    )
    return ShardedCPMoERoutingMetadata(
        topk_weights=global_topk_weights,
        topk_ids=global_topk_ids,
    )


def reduce_scatter_sharded_cp_moe_output(
    expert_output: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Reduce-scatter global MoE output contributions back to CP-local rows."""
    return reduce_scatter_token_rows(
        expert_output,
        token_range,
        group=group,
    )


def slice_sharded_cp_moe_output(
    expert_output: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> torch.Tensor:
    """Return this rank's reference MoE output rows from a global tensor."""
    _validate_global_token_rows("expert_output", expert_output, token_range)
    return expert_output[token_range.start : token_range.end]


def combine_sharded_cp_moe_residual(
    attention_output_local: torch.Tensor,
    moe_output_local: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> torch.Tensor:
    """Add residual/MoE outputs after checking both tensors own the same rows."""
    _validate_local_token_rows(
        "attention_output_local", attention_output_local, token_range
    )
    _validate_local_token_rows("moe_output_local", moe_output_local, token_range)
    if attention_output_local.shape != moe_output_local.shape:
        raise ValueError(
            "attention_output_local and moe_output_local must have the same "
            f"shape: got {tuple(attention_output_local.shape)} and "
            f"{tuple(moe_output_local.shape)}."
        )
    return attention_output_local + moe_output_local
