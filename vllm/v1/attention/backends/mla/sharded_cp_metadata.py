# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-CP attention metadata localization helpers.

Sharded-CP slices the batch's token rows across the CP (=TP) group. Instead
of hand-slicing every backend-specific metadata object, the runner localizes
``CommonAttentionMetadata`` to this rank's token range (reusing the ubatch
slicing machinery) and re-runs the registered attention metadata builders on
the localized view. Backend metadata therefore stays correct as builders
evolve. The only backend-specific step is the optional "global compact KV"
override for pure first-prefill batches.
"""

from bisect import bisect_right
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace

import torch

from vllm.distributed.sharded_cp_utils import (
    SHARDED_CP_GLOBAL_SLOT_MAPPING_KEY,
    SHARDED_CP_TOKEN_RANGE_KEY,
    SHARDED_CP_USE_GLOBAL_COMPACT_KV_KEY,
    ShardedCPTokenRange,
)
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, _make_metadata_with_slice


@dataclass(frozen=True)
class _LocalRequestFragment:
    request_index: int
    local_start: int
    local_end: int
    global_start: int
    global_end: int
    request_global_start: int
    request_global_end: int

    @property
    def num_tokens(self) -> int:
        return self.local_end - self.local_start

    @property
    def start_offset_in_request(self) -> int:
        return self.global_start - self.request_global_start

    @property
    def end_offset_in_request(self) -> int:
        return self.global_end - self.request_global_start


def _request_fragments_for_token_range(
    query_start_loc_cpu: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> tuple[_LocalRequestFragment, ...]:
    starts = [int(v) for v in query_start_loc_cpu.cpu().tolist()]
    if len(starts) < 1:
        raise RuntimeError("Sharded-CP metadata requires query_start_loc.")
    if starts[0] != 0:
        raise RuntimeError("Sharded-CP query_start_loc must start at 0.")
    if token_range.start < 0 or token_range.end < token_range.start:
        raise RuntimeError("Sharded-CP token range is invalid.")
    if token_range.num_tokens == 0:
        return ()
    if token_range.end > starts[-1]:
        raise RuntimeError(
            "Sharded-CP token range exceeds attention metadata token count."
        )

    req_idx = max(0, bisect_right(starts, token_range.start) - 1)
    fragments: list[_LocalRequestFragment] = []
    while req_idx < len(starts) - 1 and starts[req_idx] < token_range.end:
        req_global_start = starts[req_idx]
        req_global_end = starts[req_idx + 1]
        overlap_start = max(token_range.start, req_global_start)
        overlap_end = min(token_range.end, req_global_end)
        if overlap_end > overlap_start:
            fragments.append(
                _LocalRequestFragment(
                    request_index=req_idx,
                    local_start=overlap_start - token_range.start,
                    local_end=overlap_end - token_range.start,
                    global_start=overlap_start,
                    global_end=overlap_end,
                    request_global_start=req_global_start,
                    request_global_end=req_global_end,
                )
            )
        req_idx += 1
    return tuple(fragments)


def annotate_token_range_with_request_fragments(
    token_range: ShardedCPTokenRange,
    query_start_loc_cpu: torch.Tensor,
) -> ShardedCPTokenRange:
    """Attach this rank's request-fragment layout to the token range."""
    fragments = _request_fragments_for_token_range(query_start_loc_cpu, token_range)
    return replace(
        token_range,
        local_request_starts=tuple(f.local_start for f in fragments),
        local_request_ends=tuple(f.local_end for f in fragments),
        local_request_global_starts=tuple(f.request_global_start for f in fragments),
        local_request_indices=tuple(f.request_index for f in fragments),
    )


def localize_common_attention_metadata(
    common: CommonAttentionMetadata,
    token_range: ShardedCPTokenRange,
) -> CommonAttentionMetadata:
    """Build token-range local CommonAttentionMetadata for one CP rank."""
    fragments = _request_fragments_for_token_range(
        common.query_start_loc_cpu, token_range
    )
    if not fragments:
        return _empty_common_attention_metadata(common)

    request_slice = slice(
        fragments[0].request_index, fragments[-1].request_index + 1
    )
    token_slice = slice(token_range.start, token_range.end)
    local = _make_metadata_with_slice(
        UBatchSlice(request_slice, token_slice), common
    )

    # A first request continuing from a previous rank keeps its full seq_len
    # slice, so derive per-fragment computed-token counts from the localized
    # views instead of inheriting the global request values.
    seq_lens_cpu = local._seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = local.seq_lens_cpu_upper_bound
    num_computed_tokens_cpu = None
    if seq_lens_cpu is not None:
        query_lens_cpu = (
            local.query_start_loc_cpu[1:] - local.query_start_loc_cpu[:-1]
        )
        num_computed_tokens_cpu = seq_lens_cpu - query_lens_cpu

    positions = common.positions
    if positions is not None:
        positions = positions[token_slice]
    is_prefilling = common.is_prefilling
    if is_prefilling is not None:
        is_prefilling = is_prefilling[request_slice]

    return replace(
        local,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        causal=common.causal,
        positions=positions,
        is_prefilling=is_prefilling,
    )


def _empty_common_attention_metadata(
    common: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Zero-token metadata for a CP rank that owns no rows this step."""
    query_start_loc = common.query_start_loc[:1] * 0
    query_start_loc_cpu = common.query_start_loc_cpu[:1] * 0
    return replace(
        common,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=common.seq_lens[:0],
        _seq_lens_cpu=(
            common._seq_lens_cpu[:0] if common._seq_lens_cpu is not None else None
        ),
        _num_computed_tokens_cpu=(
            common._num_computed_tokens_cpu[:0]
            if common._num_computed_tokens_cpu is not None
            else None
        ),
        seq_lens_cpu_upper_bound=(
            common.seq_lens_cpu_upper_bound[:0]
            if common.seq_lens_cpu_upper_bound is not None
            else None
        ),
        num_reqs=0,
        num_actual_tokens=0,
        max_query_len=0,
        block_table_tensor=common.block_table_tensor[:0],
        slot_mapping=common.slot_mapping[:0],
        positions=(
            common.positions[:0] if common.positions is not None else None
        ),
        is_prefilling=(
            common.is_prefilling[:0] if common.is_prefilling is not None else None
        ),
    )


def use_global_compact_kv_for_batch(common: CommonAttentionMetadata) -> bool:
    """A pure first prefill can attend over all-gathered compact KV directly.

    True when every request starts at zero computed tokens, i.e. per-request
    seq_lens equal query lens; chunked-prefill continuations and decodes need
    the paged KV path.
    """
    seq_lens_cpu = common._seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = common.seq_lens_cpu_upper_bound
    if seq_lens_cpu is None:
        return False
    query_lens_cpu = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]
    return bool(torch.equal(seq_lens_cpu, query_lens_cpu))


def supports_global_compact_kv(metadata: AttentionMetadata) -> bool:
    """Whether this backend metadata understands global-compact-KV overrides.

    Unknown backend types silently keep per-request top-k semantics, so the
    caller must fall back to the paged-KV path when any metadata object is
    not recognized here.
    """
    return isinstance(metadata, (DeepseekV32IndexerMetadata, FlashMLASparseMetadata))


def apply_global_compact_kv_overrides(
    metadata: AttentionMetadata,
    token_range: ShardedCPTokenRange,
    local_common: CommonAttentionMetadata,
) -> AttentionMetadata:
    """Rewrite locally-built backend metadata for the global-compact-KV path.

    Top-k indices and indexer KV spans switch to global token-row
    coordinates addressing the all-gathered ``[T, d_kv + d_kI]`` payload.
    """
    if isinstance(metadata, DeepseekV32IndexerMetadata):
        if metadata.num_decodes != 0:
            raise RuntimeError(
                "Sharded-CP global compact KV requires a pure prefill batch."
            )
        if metadata.prefill is not None:
            ks_all, ke_all = _global_compact_kv_spans(token_range, local_common)
            for chunk in metadata.prefill.chunks:
                rows = slice(chunk.token_start, chunk.token_end)
                chunk.cu_seqlen_ks = ks_all[rows]
                chunk.cu_seqlen_ke = ke_all[rows]
        metadata.k_is_global_compact = True
        return metadata

    if isinstance(metadata, FlashMLASparseMetadata):
        metadata.topk_indices_are_global_compact_offsets = True
        metadata.fp8_extra_metadata = None
        metadata.fp8_use_mixed_batch = False
        return metadata

    return metadata


def _global_compact_kv_spans(
    token_range: ShardedCPTokenRange,
    local_common: CommonAttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-local-row global KV spans ``[ks, ke)`` for pure first prefill.

    With zero computed tokens, a request's global KV rows coincide with its
    global query rows, so ``ks`` is the request's global start row and ``ke``
    is the row's own global index + 1 (causal).
    """
    if not token_range.has_local_request_fragments:
        raise RuntimeError(
            "Sharded-CP global compact KV requires request-fragment "
            "annotations on the token range."
        )
    assert token_range.local_request_starts is not None
    assert token_range.local_request_ends is not None
    assert token_range.local_request_global_starts is not None
    device = local_common.query_start_loc.device
    frag_lens = torch.tensor(
        [
            end - start
            for start, end in zip(
                token_range.local_request_starts, token_range.local_request_ends
            )
        ],
        dtype=torch.int64,
    )
    frag_ks = torch.tensor(
        token_range.local_request_global_starts, dtype=torch.int32
    )
    ks_all = torch.repeat_interleave(frag_ks, frag_lens).to(device)
    ke_all = torch.arange(
        token_range.start + 1,
        token_range.end + 1,
        dtype=torch.int32,
        device=device,
    )
    return ks_all, ke_all


def get_sharded_cp_token_range_from_context(
    forward_context: ForwardContext,
) -> ShardedCPTokenRange | None:
    """Return the active Sharded-CP token range, if any."""
    return forward_context.additional_kwargs.get(SHARDED_CP_TOKEN_RANGE_KEY)


@contextmanager
def sharded_cp_forward_context(
    forward_context: ForwardContext,
    token_range: ShardedCPTokenRange | None,
    local_attn_metadata: dict[str, AttentionMetadata] | None,
    *,
    use_global_compact_kv: bool = False,
) -> Iterator[None]:
    """Run the model body under CP-local attention metadata.

    The global slot mapping stays reachable through ``additional_kwargs`` so
    the global-compact-KV path can persist all-gathered rows into the local
    caches with global slot IDs.
    """
    if token_range is None:
        with nullcontext():
            yield
        return

    if local_attn_metadata is None:
        local_attn_metadata = {}
    local_slot_mapping = {
        layer_name: metadata.slot_mapping
        for layer_name, metadata in local_attn_metadata.items()
        if hasattr(metadata, "slot_mapping")
    }
    local_context = replace(
        forward_context,
        attn_metadata=local_attn_metadata,
        slot_mapping=local_slot_mapping,
        additional_kwargs={
            **forward_context.additional_kwargs,
            SHARDED_CP_GLOBAL_SLOT_MAPPING_KEY: forward_context.slot_mapping,
            SHARDED_CP_TOKEN_RANGE_KEY: token_range,
            SHARDED_CP_USE_GLOBAL_COMPACT_KV_KEY: use_global_compact_kv,
        },
    )
    with override_forward_context(local_context):
        yield
