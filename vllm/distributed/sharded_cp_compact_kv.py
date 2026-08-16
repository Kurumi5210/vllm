# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-CP compact KV communication helpers.

These helpers keep the compact KV payload layout explicit and unit-testable
before the sparse MLA kernel path is enabled end to end.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed.sharded_cp_utils import (
    ShardedCPTokenRange,
    TokenRowsAllGather,
    all_gather_token_rows_async,
)


@dataclass(frozen=True)
class ShardedCPCompactKVLayout:
    """Column layout for the compact Sharded-CP KV all-gather payload."""

    kv_lora_rank: int
    qk_rope_head_dim: int
    indexer_head_dim: int

    @property
    def kv_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def total_dim(self) -> int:
        return self.kv_dim + self.indexer_head_dim


def _validate_positive_dim(name: str, dim: int) -> None:
    if dim <= 0:
        raise ValueError(f"{name} must be > 0, got {dim}.")


def _flatten_k_pe(k_pe: torch.Tensor) -> torch.Tensor:
    if k_pe.dim() == 2:
        return k_pe
    if k_pe.dim() == 3 and k_pe.shape[1] == 1:
        return k_pe.squeeze(1)
    raise ValueError(
        "k_pe must have shape [num_tokens, rope_dim] or [num_tokens, 1, rope_dim]."
    )


def pack_sharded_cp_compact_kv(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    indexer_k: torch.Tensor,
) -> torch.Tensor:
    """Pack MLA compact KV and Indexer-K into one token-row payload.

    The packed row layout is ``[kv_c_normed || k_pe || indexer_k]``.
    """
    k_pe_flat = _flatten_k_pe(k_pe)
    if kv_c_normed.dim() != 2:
        raise ValueError("kv_c_normed must be a 2D tensor.")
    if indexer_k.dim() != 2:
        raise ValueError("indexer_k must be a 2D tensor.")
    if not (kv_c_normed.shape[0] == k_pe_flat.shape[0] == indexer_k.shape[0]):
        raise ValueError("all compact KV tensors must have the same row count.")
    if not (
        kv_c_normed.device == k_pe_flat.device == indexer_k.device
        and kv_c_normed.dtype == k_pe_flat.dtype == indexer_k.dtype
    ):
        raise ValueError("all compact KV tensors must have the same device and dtype.")
    return torch.cat((kv_c_normed, k_pe_flat, indexer_k), dim=-1)


def split_sharded_cp_compact_kv(
    compact_kv: torch.Tensor,
    layout: ShardedCPCompactKVLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[kv_c_normed || k_pe || indexer_k]`` back into tensors."""
    _validate_positive_dim("kv_lora_rank", layout.kv_lora_rank)
    _validate_positive_dim("qk_rope_head_dim", layout.qk_rope_head_dim)
    # indexer_head_dim may be 0: shared top-k (IndexCache) layers have no
    # indexer of their own and gather MLA KV only.
    if layout.indexer_head_dim < 0:
        raise ValueError(
            f"indexer_head_dim must be >= 0, got {layout.indexer_head_dim}."
        )
    if compact_kv.dim() != 2:
        raise ValueError("compact_kv must be a 2D tensor.")
    if compact_kv.shape[1] != layout.total_dim:
        raise ValueError(
            "compact_kv last dimension does not match layout: "
            f"got {compact_kv.shape[1]}, expected {layout.total_dim}."
        )
    kv_c_normed, k_pe, indexer_k = compact_kv.split(
        [layout.kv_lora_rank, layout.qk_rope_head_dim, layout.indexer_head_dim],
        dim=-1,
    )
    return kv_c_normed, k_pe.unsqueeze(1), indexer_k


class ShardedCPCompactKVAllGather:
    """Deferred compact KV all-gather result."""

    def __init__(
        self,
        handle: TokenRowsAllGather,
        layout: ShardedCPCompactKVLayout,
    ) -> None:
        self._handle = handle
        self._layout = layout
        self._result: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def wait(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._result is None:
            self._result = split_sharded_cp_compact_kv(
                self._handle.wait(),
                self._layout,
            )
        return self._result

    def release(self) -> None:
        """Drain the collective so no in-flight work leaks on error paths."""
        self.wait()


def all_gather_sharded_cp_compact_kv_async(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    indexer_k: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
    pad_value: float = 0.0,
) -> ShardedCPCompactKVAllGather:
    """Start compact MLA KV and Indexer-K all-gather for Sharded-CP."""
    compact_kv = pack_sharded_cp_compact_kv(kv_c_normed, k_pe, indexer_k)
    layout = ShardedCPCompactKVLayout(
        kv_lora_rank=kv_c_normed.shape[1],
        qk_rope_head_dim=_flatten_k_pe(k_pe).shape[1],
        indexer_head_dim=indexer_k.shape[1],
    )
    return ShardedCPCompactKVAllGather(
        all_gather_token_rows_async(
            compact_kv,
            token_range,
            group=group,
            pad_value=pad_value,
        ),
        layout,
    )


def all_gather_sharded_cp_compact_kv(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    indexer_k: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-gather compact MLA KV and Indexer-K rows for Sharded-CP."""
    return all_gather_sharded_cp_compact_kv_async(
        kv_c_normed,
        k_pe,
        indexer_k,
        token_range,
        group=group,
        pad_value=pad_value,
    ).wait()


def sharded_cp_topk_prefix(
    topk_indices_buffer: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> torch.Tensor:
    """Return the top-k buffer prefix owned by the local CP token rows."""
    if topk_indices_buffer.dim() != 2:
        raise ValueError("topk_indices_buffer must be a 2D tensor.")
    if topk_indices_buffer.shape[0] < token_range.num_tokens:
        raise ValueError(
            "topk_indices_buffer does not have enough local rows: "
            f"got {topk_indices_buffer.shape[0]}, "
            f"expected at least {token_range.num_tokens}."
        )
    return topk_indices_buffer[: token_range.num_tokens]
