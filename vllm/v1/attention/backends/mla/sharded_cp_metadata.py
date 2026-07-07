# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-CP attention metadata localization helpers."""

from bisect import bisect_right
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    is_deep_gemm_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
    kv_spans_from_batches,
)
from vllm.distributed.sharded_cp_utils import (
    ShardedCPTokenRange,
    get_sharded_cp_token_range,
)


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


def _request_slice_for_token_range(
    query_start_loc_cpu: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> slice:
    starts = query_start_loc_cpu.cpu()
    start_matches = torch.nonzero(starts == token_range.start, as_tuple=False).flatten()
    end_matches = torch.nonzero(starts == token_range.end, as_tuple=False).flatten()
    if start_matches.numel() == 0 or end_matches.numel() == 0:
        raise RuntimeError(
            "Sharded-CP metadata requires token range boundaries to align with "
            "request boundaries."
        )
    req_start = int(start_matches[0].item())
    req_end = int(end_matches[0].item())
    if req_end < req_start:
        raise RuntimeError(
            "Sharded-CP metadata received a non-monotonic request slice."
        )
    return slice(req_start, req_end)


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
    if starts and token_range.end > starts[-1]:
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


def _local_query_start_loc_cpu_from_fragments(
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    starts = [0]
    starts.extend(fragment.local_end for fragment in fragments)
    return torch.tensor(starts, dtype=torch.int32)


def _chunk_query_start_loc_cpu_from_fragments(
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    starts = [0]
    chunk_start = fragments[0].local_start
    starts.extend(fragment.local_end - chunk_start for fragment in fragments)
    return torch.tensor(starts, dtype=torch.int32)


def _local_query_start_loc_like(
    query_start_loc: torch.Tensor,
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    starts_cpu = _local_query_start_loc_cpu_from_fragments(fragments)
    return starts_cpu.to(device=query_start_loc.device, dtype=query_start_loc.dtype)


def _token_range_with_request_fragments(
    token_range: ShardedCPTokenRange,
    fragments: tuple[_LocalRequestFragment, ...],
) -> ShardedCPTokenRange:
    return replace(
        token_range,
        local_request_starts=tuple(fragment.local_start for fragment in fragments),
        local_request_ends=tuple(fragment.local_end for fragment in fragments),
        local_request_global_starts=tuple(
            fragment.request_global_start for fragment in fragments
        ),
        local_request_indices=tuple(
            fragment.request_index for fragment in fragments
        ),
    )


def _slice_optional_fragments(
    x: Any,
    fragments: tuple[_LocalRequestFragment, ...],
) -> Any:
    if x is None:
        return None
    indices = [fragment.request_index for fragment in fragments]
    if isinstance(x, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        return x.index_select(0, index)
    if isinstance(x, list):
        return [x[index] for index in indices]
    if isinstance(x, tuple):
        return tuple(x[index] for index in indices)
    return x[indices]


def _slice_optional_req_tensor(x: Any, request_slice: slice) -> Any:
    if x is None:
        return None
    return x[request_slice]


def _seq_lens_cpu(metadata: CommonAttentionMetadata) -> torch.Tensor:
    if metadata._seq_lens_cpu is not None:
        return metadata._seq_lens_cpu
    return metadata.seq_lens.cpu()


def _num_computed_tokens_cpu(metadata: CommonAttentionMetadata) -> torch.Tensor:
    if metadata._num_computed_tokens_cpu is not None:
        return metadata._num_computed_tokens_cpu
    query_lens = metadata.query_start_loc_cpu[1:] - metadata.query_start_loc_cpu[:-1]
    return _seq_lens_cpu(metadata) - query_lens


def _fragment_seq_lens_cpu(
    global_seq_lens_cpu: torch.Tensor,
    global_num_computed_tokens_cpu: torch.Tensor,
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    if not fragments:
        return global_seq_lens_cpu.new_empty((0,))
    values = [
        int(global_num_computed_tokens_cpu[fragment.request_index].item())
        + fragment.end_offset_in_request
        for fragment in fragments
    ]
    return torch.tensor(values, dtype=global_seq_lens_cpu.dtype)


def _fragment_num_computed_tokens_cpu(
    global_num_computed_tokens_cpu: torch.Tensor,
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    if not fragments:
        return global_num_computed_tokens_cpu.new_empty((0,))
    values = [
        int(global_num_computed_tokens_cpu[fragment.request_index].item())
        + fragment.start_offset_in_request
        for fragment in fragments
    ]
    return torch.tensor(values, dtype=global_num_computed_tokens_cpu.dtype)


def _index_select_fragments(
    x: torch.Tensor,
    fragments: tuple[_LocalRequestFragment, ...],
) -> torch.Tensor:
    indices = [fragment.request_index for fragment in fragments]
    if not indices:
        return x[:0]
    index = torch.tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(0, index)


def get_sharded_cp_token_range_from_forward_context(
    forward_context: ForwardContext,
    rank: int,
    world_size: int,
) -> ShardedCPTokenRange:
    """Compute this rank's balanced token range from forward metadata."""
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        raise RuntimeError(
            "Sharded-CP requires per-layer attention metadata in the forward "
            "context."
        )
    if not attn_metadata:
        raise RuntimeError("Sharded-CP requires non-empty attention metadata.")

    first_metadata = next(iter(attn_metadata.values()))
    query_start_loc = getattr(first_metadata, "query_start_loc", None)
    if query_start_loc is None:
        raise RuntimeError(
            "Sharded-CP attention metadata requires query_start_loc."
        )
    query_start_loc_cpu = query_start_loc.cpu()
    total_tokens = int(query_start_loc_cpu[-1].item())
    token_range = get_sharded_cp_token_range(total_tokens, rank, world_size)
    fragments = _request_fragments_for_token_range(
        query_start_loc_cpu, token_range
    )
    return _token_range_with_request_fragments(token_range, fragments)


def localize_common_attention_metadata(
    common_attn_metadata: CommonAttentionMetadata,
    token_range: ShardedCPTokenRange,
) -> CommonAttentionMetadata:
    """Build token-range local CommonAttentionMetadata for one CP rank."""
    fragments = _request_fragments_for_token_range(
        common_attn_metadata.query_start_loc_cpu, token_range
    )
    query_start_loc_cpu = _local_query_start_loc_cpu_from_fragments(fragments)
    query_start_loc = query_start_loc_cpu.to(
        device=common_attn_metadata.query_start_loc.device,
        dtype=common_attn_metadata.query_start_loc.dtype,
    )
    global_seq_lens_cpu = _seq_lens_cpu(common_attn_metadata)
    global_num_computed_tokens_cpu = _num_computed_tokens_cpu(common_attn_metadata)
    seq_lens_cpu = _fragment_seq_lens_cpu(
        global_seq_lens_cpu,
        global_num_computed_tokens_cpu,
        fragments,
    )
    seq_lens = seq_lens_cpu.to(
        device=common_attn_metadata.seq_lens.device,
        dtype=common_attn_metadata.seq_lens.dtype,
    )
    num_reqs = len(fragments)
    num_actual_tokens = token_range.num_tokens
    if num_reqs == 0:
        max_query_len = 0
        max_seq_len = 0
        num_computed_tokens_cpu = seq_lens_cpu
    else:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = int(query_lens_cpu.max().item())
        max_seq_len = int(seq_lens_cpu.max().item())
        num_computed_tokens_cpu = _fragment_num_computed_tokens_cpu(
            global_num_computed_tokens_cpu,
            fragments,
        )

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=_index_select_fragments(
            common_attn_metadata.block_table_tensor,
            fragments,
        ),
        slot_mapping=common_attn_metadata.slot_mapping[
            token_range.start : token_range.end
        ],
        causal=common_attn_metadata.causal,
        logits_indices_padded=common_attn_metadata.logits_indices_padded,
        num_logits_indices=common_attn_metadata.num_logits_indices,
        encoder_seq_lens=_slice_optional_fragments(
            common_attn_metadata.encoder_seq_lens, fragments
        ),
        encoder_seq_lens_cpu=_slice_optional_fragments(
            common_attn_metadata.encoder_seq_lens_cpu, fragments
        ),
        dcp_local_seq_lens=_slice_optional_fragments(
            common_attn_metadata.dcp_local_seq_lens, fragments
        ),
        dcp_local_seq_lens_cpu=_slice_optional_fragments(
            common_attn_metadata.dcp_local_seq_lens_cpu, fragments
        ),
    )


def _find_request_index(query_start_loc_cpu: torch.Tensor, token: int) -> int:
    matches = torch.nonzero(query_start_loc_cpu == token, as_tuple=False).flatten()
    if matches.numel() == 0:
        raise RuntimeError(
            "Sharded-CP metadata requires prefill chunk boundaries to align "
            "with request boundaries."
        )
    return int(matches[0].item())


def _build_indexer_prefill_chunk(
    *,
    fragments: tuple[_LocalRequestFragment, ...],
    seq_lens_cpu: torch.Tensor,
    num_computed_tokens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    use_global_compact_kv: bool,
) -> DeepseekV32IndexerPrefillChunkMetadata:
    if not fragments:
        raise RuntimeError("Cannot build an empty Sharded-CP prefill chunk.")
    for prev, current in zip(fragments, fragments[1:]):
        if prev.local_end != current.local_start:
            raise RuntimeError(
                "Sharded-CP indexer prefill chunks require contiguous local "
                "token fragments."
            )

    local_query_start_loc_cpu = _chunk_query_start_loc_cpu_from_fragments(
        fragments
    )
    local_seq_lens_cpu = _fragment_seq_lens_cpu(
        seq_lens_cpu,
        num_computed_tokens_cpu,
        fragments,
    )
    if use_global_compact_kv:
        cu_seqlen_ks = torch.tensor(
            [
                fragment.request_global_start
                for fragment in fragments
                for _ in range(fragment.num_tokens)
            ],
            dtype=torch.int32,
            device=block_table.device,
        )
        cu_seqlen_ke = torch.tensor(
            [
                global_row + 1
                for fragment in fragments
                for global_row in range(fragment.global_start, fragment.global_end)
            ],
            dtype=torch.int32,
            device=block_table.device,
        )
    else:
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            local_query_start_loc_cpu,
            local_seq_lens_cpu,
            block_table.device,
        )
    total_seq_lens = local_seq_lens_cpu.sum()
    seq_idx = torch.arange(0, len(fragments), dtype=torch.int32)
    token_to_seq = torch.repeat_interleave(seq_idx, local_seq_lens_cpu).to(
        block_table.device
    )
    cu_seq_lens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            local_seq_lens_cpu.cumsum(dim=0).to(torch.int32),
        ]
    ).to(block_table.device)
    token_start = fragments[0].local_start
    token_end = fragments[-1].local_end
    if token_end - token_start != cu_seqlen_ks.numel():
        raise RuntimeError(
            "Sharded-CP indexer prefill chunk row mismatch: "
            f"token_start={token_start}, token_end={token_end}, "
            f"cu_rows={cu_seqlen_ks.numel()}."
        )
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=block_table,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        token_start=token_start,
        token_end=token_end,
        num_reqs=len(fragments),
    )


def _split_contiguous_fragments(
    fragments: tuple[_LocalRequestFragment, ...],
) -> tuple[tuple[_LocalRequestFragment, ...], ...]:
    if not fragments:
        return ()
    groups: list[tuple[_LocalRequestFragment, ...]] = []
    group_start = 0
    for index in range(1, len(fragments)):
        if fragments[index - 1].local_end != fragments[index].local_start:
            groups.append(fragments[group_start:index])
            group_start = index
    groups.append(fragments[group_start:])
    return tuple(groups)


def _localize_indexer_prefill_metadata(
    metadata: DeepseekV32IndexerPrefillMetadata | None,
    global_query_start_loc_cpu: torch.Tensor,
    global_seq_lens_cpu: torch.Tensor,
    global_num_computed_tokens_cpu: torch.Tensor,
    fragments: tuple[_LocalRequestFragment, ...],
    token_range: ShardedCPTokenRange,
    use_global_compact_kv: bool,
) -> DeepseekV32IndexerPrefillMetadata | None:
    if metadata is None:
        return None

    chunks: list[DeepseekV32IndexerPrefillChunkMetadata] = []
    for chunk in metadata.chunks:
        if chunk.token_end <= token_range.start or chunk.token_start >= token_range.end:
            continue
        chunk_req_start = _find_request_index(
            global_query_start_loc_cpu, chunk.token_start
        )
        chunk_req_end = _find_request_index(global_query_start_loc_cpu, chunk.token_end)
        chunk_fragments = tuple(
            fragment
            for fragment in fragments
            if (
                chunk_req_start <= fragment.request_index < chunk_req_end
                and fragment.global_end > chunk.token_start
                and fragment.global_start < chunk.token_end
            )
        )
        if not chunk_fragments:
            continue
        for contiguous_fragments in _split_contiguous_fragments(chunk_fragments):
            chunk_block_table = _index_select_fragments(
                chunk.block_table,
                tuple(
                    replace(
                        fragment,
                        request_index=fragment.request_index - chunk_req_start,
                    )
                    for fragment in contiguous_fragments
                ),
            )
            chunks.append(
                _build_indexer_prefill_chunk(
                    fragments=contiguous_fragments,
                    seq_lens_cpu=global_seq_lens_cpu,
                    num_computed_tokens_cpu=global_num_computed_tokens_cpu,
                    block_table=chunk_block_table,
                    use_global_compact_kv=use_global_compact_kv,
                )
            )
    return DeepseekV32IndexerPrefillMetadata(chunks=chunks)


def _slice_indexer_decode_metadata(
    metadata: DeepseekV32IndexerMetadata,
    fragments: tuple[_LocalRequestFragment, ...],
    global_num_computed_tokens_cpu: torch.Tensor,
    local_max_seq_len: int,
) -> DeepSeekV32IndexerDecodeMetadata | None:
    decode_fragments = tuple(
        fragment
        for fragment in fragments
        if fragment.request_index < metadata.num_decodes
    )
    if not decode_fragments:
        return None

    decode = metadata.decode
    if decode is None:
        raise RuntimeError(
            "Sharded-CP DeepSeek V3.2 indexer metadata is missing decode "
            "metadata for a decode batch."
        )
    decode_rows = decode.decode_lens.shape[0]

    block_tables = []
    seq_lens_values: list[int] = []
    for fragment in decode_fragments:
        context_len = int(
            global_num_computed_tokens_cpu[fragment.request_index].item()
        )
        for offset in range(
            fragment.start_offset_in_request,
            fragment.end_offset_in_request,
        ):
            if decode_rows == metadata.num_decodes:
                row = fragment.request_index
            elif decode_rows == metadata.num_decode_tokens:
                row = (
                    int(metadata.query_start_loc[fragment.request_index].item())
                    + offset
                )
            else:
                raise RuntimeError(
                    "Sharded-CP DeepSeek V3.2 indexer decode metadata has "
                    "unsupported row layout."
                )
            block_tables.append(decode.block_table[row : row + 1])
            seq_lens_values.append(context_len + offset + 1)

    if decode.offsets is not None:
        local_decode_lens = [
            fragment.num_tokens for fragment in decode_fragments
        ]
        if any(length != 1 for length in local_decode_lens):
            raise RuntimeError(
                "Sharded-CP does not support splitting native multi-token "
                "decode metadata with shared offsets."
            )

    if not block_tables:
        return None
    block_table = torch.cat(block_tables, dim=0)
    seq_lens_cpu = torch.tensor(seq_lens_values, dtype=decode.seq_lens.dtype)
    seq_lens = seq_lens_cpu.to(device=decode.seq_lens.device)
    decode_lens = torch.ones_like(seq_lens)
    batch_size = seq_lens.shape[0]
    use_large_context_topk = batch_size <= 128 and local_max_seq_len > 8192

    schedule_metadata = decode.schedule_metadata
    if (
        current_platform.is_cuda()
        and seq_lens.is_cuda
        and is_deep_gemm_supported()
        and batch_size > 0
    ):
        device_index = seq_lens.device.index
        if device_index is None:
            device_index = 0
        schedule_metadata = get_paged_mqa_logits_metadata(
            seq_lens,
            decode.block_size,
            num_compute_units(device_index),
        )

    return DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        decode_lens=decode_lens,
        requires_padding=decode.requires_padding,
        schedule_metadata=schedule_metadata,
        use_large_context_topk=use_large_context_topk,
        offsets=decode.offsets,
        block_size=decode.block_size,
    )


def localize_deepseek_v32_indexer_metadata(
    metadata: DeepseekV32IndexerMetadata,
    token_range: ShardedCPTokenRange,
    *,
    use_global_compact_kv: bool = True,
) -> DeepseekV32IndexerMetadata:
    """Localize DeepSeek V3.2 indexer metadata for a CP token range."""
    global_query_start_loc_cpu = metadata.query_start_loc.cpu()
    global_seq_lens_cpu = metadata.seq_lens.cpu()
    global_query_lens_cpu = (
        global_query_start_loc_cpu[1:] - global_query_start_loc_cpu[:-1]
    )
    global_num_computed_tokens_cpu = global_seq_lens_cpu - global_query_lens_cpu
    fragments = _request_fragments_for_token_range(
        global_query_start_loc_cpu, token_range
    )
    query_start_loc = _local_query_start_loc_like(
        metadata.query_start_loc,
        fragments,
    )
    seq_lens_cpu = _fragment_seq_lens_cpu(
        global_seq_lens_cpu,
        global_num_computed_tokens_cpu,
        fragments,
    )
    seq_lens = seq_lens_cpu.to(
        device=metadata.seq_lens.device,
        dtype=metadata.seq_lens.dtype,
    )
    num_reqs = len(fragments)
    if num_reqs == 0:
        max_query_len = 0
        max_seq_len = 0
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.cpu().max().item())
        max_seq_len = int(seq_lens_cpu.max().item())

    decode_fragments = tuple(
        fragment
        for fragment in fragments
        if fragment.request_index < metadata.num_decodes
    )
    prefill_fragments = tuple(
        fragment
        for fragment in fragments
        if fragment.request_index >= metadata.num_decodes
    )
    local_num_decode_tokens = sum(fragment.num_tokens for fragment in decode_fragments)
    # Decode metadata is flattened to one row per local decode token.
    local_num_decodes = local_num_decode_tokens
    local_num_prefills = len(prefill_fragments)
    local_num_prefill_tokens = sum(
        fragment.num_tokens for fragment in prefill_fragments
    )

    return replace(
        metadata,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        num_actual_tokens=token_range.num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=metadata.slot_mapping[token_range.start : token_range.end],
        num_decodes=local_num_decodes,
        num_decode_tokens=local_num_decode_tokens,
        num_prefills=local_num_prefills,
        num_prefill_tokens=local_num_prefill_tokens,
        k_is_global_compact=use_global_compact_kv,
        decode=_slice_indexer_decode_metadata(
            metadata,
            fragments,
            global_num_computed_tokens_cpu,
            max_seq_len,
        ),
        prefill=_localize_indexer_prefill_metadata(
            metadata.prefill,
            global_query_start_loc_cpu,
            global_seq_lens_cpu,
            global_num_computed_tokens_cpu,
            prefill_fragments,
            token_range,
            use_global_compact_kv,
        ),
    )


def localize_flashmla_sparse_metadata(
    metadata: FlashMLASparseMetadata,
    token_range: ShardedCPTokenRange,
    *,
    use_global_compact_kv: bool = True,
) -> FlashMLASparseMetadata:
    """Localize FlashMLA sparse metadata for a CP token range."""
    if metadata.num_actual_tokens != metadata.req_id_per_token.shape[0]:
        raise RuntimeError(
            "Sharded-CP metadata only supports unpadded sparse MLA metadata."
        )
    fragments = _request_fragments_for_token_range(
        metadata.query_start_loc.cpu(), token_range
    )
    query_start_loc = _local_query_start_loc_like(
        metadata.query_start_loc,
        fragments,
    )
    if token_range.num_tokens == 0:
        req_id_per_token = metadata.req_id_per_token[:0]
    else:
        req_id_per_token = torch.empty(
            token_range.num_tokens,
            dtype=metadata.req_id_per_token.dtype,
            device=metadata.req_id_per_token.device,
        )
        for local_req_idx, fragment in enumerate(fragments):
            req_id_per_token[
                fragment.local_start : fragment.local_end
            ] = local_req_idx
    num_reqs = len(fragments)
    if num_reqs > 0:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item())
    else:
        max_query_len = 0
    if not use_global_compact_kv and metadata.fp8_extra_metadata is not None:
        raise RuntimeError(
            "Sharded-CP paged FlashMLA sparse metadata does not support "
            "fp8_ds_mla KV cache yet. Use the default auto/bfloat16 KV cache "
            "or a pure first-prefill batch that can use global compact KV."
        )

    return replace(
        metadata,
        num_reqs=num_reqs,
        max_query_len=max_query_len,
        num_actual_tokens=token_range.num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=metadata.slot_mapping[token_range.start : token_range.end],
        block_table=_index_select_fragments(metadata.block_table, fragments),
        req_id_per_token=req_id_per_token,
        fp8_extra_metadata=None,
        fp8_use_mixed_batch=False,
        topk_indices_are_global_compact_offsets=use_global_compact_kv,
    )


def localize_sharded_cp_attention_metadata(
    attn_metadata: AttentionMetadata,
    token_range: ShardedCPTokenRange,
    *,
    use_global_compact_kv: bool = True,
) -> AttentionMetadata:
    """Localize one attention metadata object for Sharded-CP."""
    if isinstance(attn_metadata, CommonAttentionMetadata):
        return localize_common_attention_metadata(attn_metadata, token_range)

    if isinstance(attn_metadata, DeepseekV32IndexerMetadata):
        return localize_deepseek_v32_indexer_metadata(
            attn_metadata,
            token_range,
            use_global_compact_kv=use_global_compact_kv,
        )

    if isinstance(attn_metadata, FlashMLASparseMetadata):
        return localize_flashmla_sparse_metadata(
            attn_metadata,
            token_range,
            use_global_compact_kv=use_global_compact_kv,
        )

    raise TypeError(
        "Unsupported Sharded-CP attention metadata type: "
        f"{type(attn_metadata).__name__}."
    )


def build_sharded_cp_attention_metadata(
    attn_metadata: dict[str, AttentionMetadata],
    token_range: ShardedCPTokenRange,
    *,
    use_global_compact_kv: bool = True,
) -> dict[str, AttentionMetadata]:
    """Build a per-layer Sharded-CP local attention metadata dict."""
    return {
        layer_name: localize_sharded_cp_attention_metadata(
            metadata,
            token_range,
            use_global_compact_kv=use_global_compact_kv,
        )
        for layer_name, metadata in attn_metadata.items()
    }


def _use_global_compact_kv_for_sharded_cp(
    attn_metadata: dict[str, AttentionMetadata],
) -> bool:
    saw_global_compact_signal = False
    for metadata in attn_metadata.values():
        if isinstance(metadata, DeepseekV32IndexerMetadata):
            query_start_loc_cpu = metadata.query_start_loc.cpu()
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            seq_lens_cpu = metadata.seq_lens.cpu()
            saw_global_compact_signal = True
            if metadata.num_decodes != 0 or not bool(
                torch.equal(seq_lens_cpu, query_lens_cpu)
            ):
                return False
            continue
        fp8_extra_metadata = getattr(metadata, "fp8_extra_metadata", None)
        num_decodes = getattr(fp8_extra_metadata, "num_decodes", None)
        if num_decodes is not None:
            saw_global_compact_signal = True
            if num_decodes != 0:
                return False
    if saw_global_compact_signal:
        return True
    return True


@contextmanager
def sharded_cp_forward_context(
    forward_context: ForwardContext,
    token_range: ShardedCPTokenRange | None,
) -> Iterator[None]:
    """Override the current forward context with Sharded-CP local metadata."""
    if token_range is None:
        with nullcontext():
            yield
        return

    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        raise RuntimeError(
            "Sharded-CP requires per-layer attention metadata in the forward "
            "context."
        )
    first_metadata = next(iter(attn_metadata.values()), None)
    if first_metadata is not None:
        query_start_loc = getattr(first_metadata, "query_start_loc", None)
        if query_start_loc is not None:
            token_range = _token_range_with_request_fragments(
                token_range,
                _request_fragments_for_token_range(
                    query_start_loc.cpu(), token_range
                ),
            )
    use_global_compact_kv = _use_global_compact_kv_for_sharded_cp(attn_metadata)
    local_attn_metadata = build_sharded_cp_attention_metadata(
        attn_metadata,
        token_range,
        use_global_compact_kv=use_global_compact_kv,
    )
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
            "sharded_cp_global_slot_mapping": forward_context.slot_mapping,
            "sharded_cp_token_range": token_range,
            "sharded_cp_use_global_compact_kv": use_global_compact_kv,
        },
    )
    with override_forward_context(local_context):
        yield
