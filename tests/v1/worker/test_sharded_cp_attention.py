# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import forward_context as forward_context_module
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers import sparse_attn_indexer as sparse_indexer_module
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.sharded_cp_attention import (
    ShardedCPCompactKVLayout,
    all_gather_sharded_cp_compact_kv,
    all_gather_sharded_cp_compact_kv_async,
    assemble_sharded_cp_compact_kv_chunks,
    pack_sharded_cp_compact_kv,
    sharded_cp_topk_prefix,
    split_sharded_cp_compact_kv,
)
from vllm.distributed.sharded_cp_utils import (
    ShardedCPTokenRange,
    get_request_aligned_sharded_cp_token_range,
    get_request_aligned_sharded_cp_token_ranges,
    get_sharded_cp_token_range,
    make_reduce_scatter_token_chunks,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_compact_kv_pack_split_round_trip():
    kv_c = torch.arange(12, dtype=torch.float32).view(3, 4)
    k_pe = torch.arange(6, dtype=torch.float32).view(3, 1, 2) + 100
    indexer_k = torch.arange(9, dtype=torch.float32).view(3, 3) + 200
    layout = ShardedCPCompactKVLayout(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        indexer_head_dim=3,
    )

    packed = pack_sharded_cp_compact_kv(kv_c, k_pe, indexer_k)
    out_kv_c, out_k_pe, out_indexer_k = split_sharded_cp_compact_kv(
        packed, layout
    )

    assert packed.shape == (3, 9)
    assert torch.equal(out_kv_c, kv_c)
    assert torch.equal(out_k_pe, k_pe)
    assert torch.equal(out_indexer_k, indexer_k)


def test_compact_kv_assemble_request_aligned_chunks_round_trip():
    query_start_loc = torch.tensor([0, 100, 300, 350], dtype=torch.int32)
    token_range = get_request_aligned_sharded_cp_token_range(
        query_start_loc,
        rank=0,
        world_size=2,
    )
    layout = ShardedCPCompactKVLayout(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        indexer_head_dim=3,
    )
    kv_c = torch.arange(350 * 4, dtype=torch.float32).view(350, 4)
    k_pe = torch.arange(350 * 2, dtype=torch.float32).view(350, 1, 2) + 10_000
    indexer_k = torch.arange(350 * 3, dtype=torch.float32).view(350, 3) + 20_000
    packed = pack_sharded_cp_compact_kv(kv_c, k_pe, indexer_k)

    chunks = make_reduce_scatter_token_chunks(packed, token_range, pad_value=-1.0)
    out_kv_c, out_k_pe, out_indexer_k = assemble_sharded_cp_compact_kv_chunks(
        chunks,
        token_range,
        layout,
    )

    assert chunks[0].shape == chunks[1].shape == (300, layout.total_dim)
    assert chunks[1][50:].eq(-1.0).all()
    assert torch.equal(out_kv_c, kv_c)
    assert torch.equal(out_k_pe, k_pe)
    assert torch.equal(out_indexer_k, indexer_k)


def test_compact_kv_single_rank_all_gather_fast_path():
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 3], dtype=torch.int32),
        rank=0,
        world_size=1,
    )
    kv_c = torch.arange(12, dtype=torch.float32).view(3, 4)
    k_pe = torch.arange(6, dtype=torch.float32).view(3, 2) + 100
    indexer_k = torch.arange(9, dtype=torch.float32).view(3, 3) + 200

    out_kv_c, out_k_pe, out_indexer_k = all_gather_sharded_cp_compact_kv(
        kv_c,
        k_pe,
        indexer_k,
        token_range,
    )

    assert torch.equal(out_kv_c, kv_c)
    assert torch.equal(out_k_pe, k_pe.unsqueeze(1))
    assert torch.equal(out_indexer_k, indexer_k)


def test_compact_kv_async_single_rank_wait_is_lazy():
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    kv_c = torch.arange(12, dtype=torch.float32).view(3, 4)
    k_pe = torch.arange(6, dtype=torch.float32).view(3, 1, 2) + 100
    indexer_k = torch.arange(9, dtype=torch.float32).view(3, 3) + 200

    handle = all_gather_sharded_cp_compact_kv_async(
        kv_c,
        k_pe,
        indexer_k,
        token_range,
    )

    out_kv_c, out_k_pe, out_indexer_k = handle.wait()
    assert torch.equal(out_kv_c, kv_c)
    assert torch.equal(out_k_pe, k_pe)
    assert torch.equal(out_indexer_k, indexer_k)
    assert handle.wait()[0] is out_kv_c


def test_compact_kv_pack_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="same row count"):
        pack_sharded_cp_compact_kv(
            torch.zeros(2, 4),
            torch.zeros(3, 1, 2),
            torch.zeros(2, 3),
        )


def test_topk_prefix_uses_only_local_token_rows():
    ranges = get_request_aligned_sharded_cp_token_ranges(
        torch.tensor([0, 100, 300, 350], dtype=torch.int32),
        world_size=2,
    )
    topk = torch.arange(400 * 4, dtype=torch.int32).view(400, 4)

    prefix = sharded_cp_topk_prefix(topk, ranges[1])

    assert prefix.shape == (50, 4)
    assert torch.equal(prefix, topk[:50])


def _reference_sharded_cp_topk(
    q_fp8,
    indexer_k_global,
    weights,
    token_range,
    query_start_loc,
    topk_tokens,
):
    q = q_fp8.float()
    k = indexer_k_global.float()
    weights = weights.float()
    topk_indices = torch.full(
        (token_range.num_tokens, topk_tokens),
        -1,
        dtype=torch.int32,
        device=q_fp8.device,
    )
    query_starts = [int(v) for v in query_start_loc.cpu().tolist()]
    global_request_starts = token_range.local_request_global_starts
    if global_request_starts is None:
        global_request_starts = tuple(
            token_range.start + req_start for req_start in query_starts[:-1]
        )
    for req_idx, (req_start, req_end) in enumerate(
        zip(query_starts, query_starts[1:])
    ):
        global_start = global_request_starts[req_idx]
        for local_row in range(req_start, req_end):
            global_end = token_range.start + local_row + 1
            num_valid = min(topk_tokens, global_end - global_start)
            if num_valid <= 0:
                continue
            scores = torch.einsum(
                "hd,sd->hs",
                q[local_row],
                k[global_start:global_end],
            )
            row_logits = (scores.relu() * weights[local_row].unsqueeze(-1)).sum(
                dim=0
            )
            topk = row_logits.topk(num_valid, dim=-1).indices + global_start
            topk_indices[local_row, :num_valid] = topk.to(torch.int32)
    return topk_indices


def _make_global_compact_indexer_metadata(
    token_range: ShardedCPTokenRange,
    query_start_loc: torch.Tensor,
    *,
    topk_tokens: int,
    head_dim: int,
) -> DeepseekV32IndexerMetadata:
    cu_seqlen_ks: list[int] = []
    cu_seqlen_ke: list[int] = []
    query_starts = [int(v) for v in query_start_loc.tolist()]
    global_request_starts = token_range.local_request_global_starts
    if global_request_starts is None:
        global_request_starts = tuple(
            token_range.start + req_start for req_start in query_starts[:-1]
        )
    for req_idx, (req_start, req_end) in enumerate(
        zip(query_starts, query_starts[1:])
    ):
        global_start = global_request_starts[req_idx]
        cu_seqlen_ks.extend([global_start] * (req_end - req_start))
        cu_seqlen_ke.extend(
            token_range.start + local_row + 1
            for local_row in range(req_start, req_end)
        )
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.empty((len(query_starts) - 1, 0), dtype=torch.int32),
        cu_seqlen_ks=torch.tensor(cu_seqlen_ks, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor(cu_seqlen_ke, dtype=torch.int32),
        cu_seq_lens=torch.empty(0, dtype=torch.int32),
        token_to_seq=torch.empty(0, dtype=torch.int32),
        total_seq_lens=token_range.total_tokens,
        token_start=0,
        token_end=token_range.num_tokens,
        num_reqs=len(query_starts) - 1,
    )
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(len(query_starts) - 1, dtype=torch.int32),
        num_reqs=len(query_starts) - 1,
        max_query_len=max(
            (end - start for start, end in zip(query_starts, query_starts[1:])),
            default=0,
        ),
        max_seq_len=token_range.total_tokens,
        num_actual_tokens=token_range.num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=torch.arange(token_range.num_tokens, dtype=torch.int64),
        head_dim=head_dim,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(query_starts) - 1,
        num_prefill_tokens=token_range.num_tokens,
        prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
        k_is_global_compact=True,
    )


def _run_global_compact_topk(
    monkeypatch,
    q_fp8: torch.Tensor,
    indexer_k_global: torch.Tensor,
    weights: torch.Tensor,
    token_range: ShardedCPTokenRange,
    query_start_loc: torch.Tensor,
    *,
    topk_tokens: int,
) -> torch.Tensor:
    metadata = _make_global_compact_indexer_metadata(
        token_range,
        query_start_loc,
        topk_tokens=topk_tokens,
        head_dim=indexer_k_global.shape[1],
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer.indexer.k_cache": metadata},
        slot_mapping={},
        virtual_engine=0,
    )
    monkeypatch.setattr(forward_context_module, "_forward_context", context)

    def _identity_quant(x, *args, **kwargs):
        scale = torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)
        return x, scale

    original_logits_torch = sparse_indexer_module.fp8_mqa_logits_torch

    def _logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, *args, **kwargs):
        return original_logits_torch(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )

    def _topk_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
    ):
        del stride0, stride1
        for row in range(num_rows):
            start = int(row_starts[row].item())
            end = int(row_ends[row].item())
            width = min(top_k, end - start)
            indices[row].fill_(-1)
            if width > 0:
                indices[row, :width] = logits[row, start:end].topk(width).indices

    monkeypatch.setattr(
        sparse_indexer_module,
        "per_token_group_quant_fp8",
        _identity_quant,
    )
    monkeypatch.setattr(sparse_indexer_module, "is_deep_gemm_supported", lambda: False)
    monkeypatch.setattr(sparse_indexer_module, "fp8_mqa_logits_torch", _logits)
    monkeypatch.setattr(
        torch.ops._C,
        "top_k_per_row_prefill",
        _topk_prefill,
        raising=False,
    )

    topk_buffer = torch.empty(token_range.num_tokens, topk_tokens, dtype=torch.int32)
    result = sparse_indexer_module.sparse_attn_indexer(
        torch.empty(token_range.num_tokens, 1),
        "layer.indexer.k_cache",
        torch.empty(0),
        q_fp8,
        indexer_k_global,
        weights,
        indexer_k_global.shape[1],
        "ue8m0",
        topk_tokens,
        indexer_k_global.shape[1],
        8192,
        8192,
        topk_buffer,
    )
    return result[: token_range.num_tokens]


def test_sharded_cp_global_compact_topk_keeps_request_local_global_indices(
    monkeypatch,
):
    topk_tokens = 4
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 2, 6], dtype=torch.int32),
        rank=1,
        world_size=3,
    )
    q_fp8 = torch.ones(token_range.num_tokens, 1, 2, dtype=torch.float32)
    indexer_k_global = torch.tensor(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.ones(token_range.num_tokens, 1, dtype=torch.float32)

    topk = _run_global_compact_topk(
        monkeypatch,
        q_fp8,
        indexer_k_global,
        weights,
        token_range,
        torch.tensor([0, 4], dtype=torch.int32),
        topk_tokens=topk_tokens,
    )

    assert topk.tolist() == [
        [2, -1, -1, -1],
        [3, 2, -1, -1],
        [4, 3, 2, -1],
        [5, 4, 3, 2],
    ]


def test_sharded_cp_global_compact_topk_matches_reference_for_multi_request(
    monkeypatch,
):
    topk_tokens = 3
    token_range = ShardedCPTokenRange(
        rank=0,
        world_size=1,
        start=2,
        end=9,
        padded_end=9,
        total_tokens=9,
    )
    q_fp8 = torch.tensor(
        [
            [[1.0, 0.5], [0.2, 1.0]],
            [[0.8, 0.4], [0.1, 1.5]],
            [[1.2, 0.1], [0.3, 0.9]],
            [[0.4, 1.1], [1.0, 0.2]],
            [[0.6, 0.3], [0.7, 0.8]],
            [[1.0, 1.0], [0.5, 0.2]],
            [[0.3, 0.9], [1.1, 0.4]],
        ],
        dtype=torch.float32,
    )
    indexer_k_global = torch.tensor(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [0.5, 0.1],
            [0.0, 1.0],
            [0.7, 0.2],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.9, 0.4],
            [0.3, 1.2],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [
            [1.0, 0.5],
            [0.8, 1.2],
            [1.1, 0.7],
            [0.4, 1.5],
            [1.0, 1.0],
            [0.6, 1.3],
            [1.4, 0.9],
        ],
        dtype=torch.float32,
    )
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)

    topk = _run_global_compact_topk(
        monkeypatch,
        q_fp8,
        indexer_k_global,
        weights,
        token_range,
        query_start_loc,
        topk_tokens=topk_tokens,
    )
    expected = _reference_sharded_cp_topk(
        q_fp8,
        indexer_k_global,
        weights,
        token_range,
        query_start_loc,
        topk_tokens,
    )

    assert torch.equal(topk, expected)


def test_sharded_cp_global_compact_topk_split_request_keeps_global_prefix(
    monkeypatch,
):
    topk_tokens = 2
    token_range = ShardedCPTokenRange(
        rank=1,
        world_size=2,
        start=2,
        end=4,
        padded_end=4,
        total_tokens=4,
        local_request_starts=(0,),
        local_request_ends=(2,),
        local_request_global_starts=(0,),
        local_request_indices=(0,),
    )
    q_fp8 = torch.ones(2, 1, 2, dtype=torch.float32)
    indexer_k_global = torch.tensor(
        [
            [9.0, 9.0],
            [8.0, 8.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    weights = torch.ones(2, 1, dtype=torch.float32)

    topk = _run_global_compact_topk(
        monkeypatch,
        q_fp8,
        indexer_k_global,
        weights,
        token_range,
        torch.tensor([0, 2], dtype=torch.int32),
        topk_tokens=topk_tokens,
    )

    assert topk.tolist() == [[0, 1], [0, 1]]


def test_sparse_indexer_torch_fallback_chunks_prefill_topk(monkeypatch):
    q_fp8 = torch.ones(4, 2, 2, dtype=torch.float32)
    k_fp8 = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
            [0.6, 0.0],
        ],
        dtype=torch.float32,
    )
    k_scale = torch.ones(6, dtype=torch.float32)
    weights = torch.ones(4, 2, dtype=torch.float32)
    cu_seqlen_ks = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([2, 3, 4, 6], dtype=torch.int32)
    topk_indices = torch.full((4, 2), -1, dtype=torch.int32)
    call_rows = []
    original_logits_torch = sparse_indexer_module.fp8_mqa_logits_torch

    def _record_logits(q, *args, **kwargs):
        call_rows.append(q.shape[0])
        return original_logits_torch(q, *args, **kwargs)

    def _topk_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
    ):
        del stride0, stride1
        for row in range(num_rows):
            start = int(row_starts[row].item())
            end = int(row_ends[row].item())
            width = min(top_k, end - start)
            indices[row].fill_(-1)
            if width > 0:
                indices[row, :width] = logits[row, start:end].topk(width).indices

    monkeypatch.setattr(
        sparse_indexer_module,
        "_TORCH_FALLBACK_MAX_SCORE_BYTES",
        6 * 2 * torch.tensor([], dtype=torch.float32).element_size(),
    )
    monkeypatch.setattr(
        sparse_indexer_module,
        "fp8_mqa_logits_torch",
        _record_logits,
    )
    monkeypatch.setattr(
        torch.ops._C,
        "top_k_per_row_prefill",
        _topk_prefill,
        raising=False,
    )

    sparse_indexer_module._fill_prefill_topk_from_mqa_logits_torch(
        q_fp8,
        (k_fp8, k_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
        2,
        add_cu_seqlen_ks=False,
    )

    assert call_rows == [1, 1, 1, 1]
    assert topk_indices.tolist() == [[1, 0], [2, 1], [3, 2], [5, 4]]


def test_sparse_indexer_deep_gemm_chunks_prefill_topk(monkeypatch):
    q_fp8 = torch.ones(4, 2, 2, dtype=torch.float32)
    k_fp8 = torch.ones(6, 2, dtype=torch.float32)
    k_scale = torch.ones(6, dtype=torch.float32)
    weights = torch.ones(4, 2, dtype=torch.float32)
    cu_seqlen_ks = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([2, 3, 4, 6], dtype=torch.int32)
    topk_indices = torch.full((4, 2), -1, dtype=torch.int32)
    call_rows = []

    def _record_logits(q, kv, weights, row_starts, row_ends, *, clean_logits):
        del weights, row_starts, row_ends, clean_logits
        call_rows.append(q.shape[0])
        logits = torch.arange(kv[0].shape[0], dtype=torch.float32)
        return logits.unsqueeze(0).expand(q.shape[0], -1).clone()

    def _topk_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
    ):
        del stride0, stride1
        for row in range(num_rows):
            start = int(row_starts[row].item())
            end = int(row_ends[row].item())
            width = min(top_k, end - start)
            indices[row].fill_(-1)
            if width > 0:
                indices[row, :width] = logits[row, start:end].topk(width).indices

    monkeypatch.setattr(
        sparse_indexer_module,
        "_DEEP_GEMM_MAX_LOGITS_BYTES",
        6 * 2 * torch.tensor([], dtype=torch.float32).element_size(),
    )
    monkeypatch.setattr(
        sparse_indexer_module,
        "fp8_mqa_logits",
        _record_logits,
    )
    monkeypatch.setattr(
        torch.ops._C,
        "top_k_per_row_prefill",
        _topk_prefill,
        raising=False,
    )

    sparse_indexer_module._fill_prefill_topk_from_mqa_logits(
        q_fp8,
        (k_fp8, k_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
        2,
        add_cu_seqlen_ks=False,
        use_deep_gemm=True,
    )

    assert call_rows == [2, 2]
    assert topk_indices.tolist() == [[1, 0], [2, 1], [3, 2], [5, 4]]


class _IdentityRotary(nn.Module):
    def forward(self, positions, q_pe, k_pe):
        return q_pe, k_pe


class _OffsetRotary(nn.Module):
    def forward(self, positions, q_pe, k_pe):
        return q_pe + 1000, k_pe + 2000


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x


class _TupleLinear(nn.Module):
    def __init__(self, out, calls=None, name=None):
        super().__init__()
        self.out = out
        self.calls = calls
        self.name = name

    def forward(self, x):
        if self.calls is not None and self.name is not None:
            self.calls.append(self.name)
        return self.out[: x.shape[0]].to(device=x.device, dtype=x.dtype), None


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.use_direct_call = False
        self.layer_name = "layer.attn"
        self.kv_cache = [torch.full((8, 3), -1.0)]
        self.kv_cache_dtype = "auto"
        self._k_scale = torch.ones(1)
        self.impl = SimpleNamespace(do_kv_cache_update=self._do_kv_cache_update)
        self.kv_cache_updates = []

    def forward(self, q, kv_c_normed, k_pe, output_shape=None, use_global_kv=False):
        self.calls.append((q, kv_c_normed, k_pe, output_shape, use_global_kv))
        return torch.arange(q.shape[0] * 2, dtype=q.dtype).view(q.shape[0], 2)

    def update_kv_cache(self, kv_c_normed, k_pe, layer_slot_mapping=None):
        if layer_slot_mapping is None:
            layer_slot_mapping = (
                forward_context_module.get_forward_context().slot_mapping.get(
                    self.layer_name
                )
            )
        self._do_kv_cache_update(
            kv_c_normed,
            k_pe,
            self.kv_cache[0],
            layer_slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
        )

    def _do_kv_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ):
        self.kv_cache_updates.append(
            (kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        )


class _RecordingOProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return x + 10, None


class _RejectingModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("empty Sharded-CP rank must not run local kernels")


class _ProjectingIndexer(nn.Module):
    head_dim = 2

    def __init__(self):
        super().__init__()
        self.topk_tokens = 2
        self.topk_indices_buffer = torch.empty(3, 2, dtype=torch.int32)
        self.global_compact_args = None
        self.paged_args = None
        self.local_k_cache_updates = []

    def project_k(self, hidden_states, positions, rotary_emb):
        return hidden_states[:, :2] + 100

    def project_q(self, q_c, positions, rotary_emb):
        q_fp8 = torch.ones(q_c.shape[0], 1, 2, dtype=q_c.dtype)
        q_scale = torch.ones(q_c.shape[0], 1, 1, dtype=q_c.dtype)
        return q_fp8, q_scale

    def project_weights(self, hidden_states, q_scale):
        return torch.ones(hidden_states.shape[0], 1, dtype=hidden_states.dtype)

    def project(self, hidden_states, q_c, positions, rotary_emb):
        q_fp8 = torch.ones(hidden_states.shape[0], 1, 2, dtype=hidden_states.dtype)
        indexer_k = hidden_states[:, :2] + 100
        weights = torch.ones(hidden_states.shape[0], 1, dtype=hidden_states.dtype)
        return q_fp8, indexer_k, weights

    def forward_global_compact(
        self,
        hidden_states,
        q_fp8,
        indexer_k_global,
        weights,
    ):
        self.global_compact_args = (
            hidden_states,
            q_fp8,
            indexer_k_global,
            weights,
        )
        self.topk_indices_buffer[: hidden_states.shape[0]] = torch.tensor(
            [[0, -1], [1, 0], [2, 1]],
            dtype=torch.int32,
        )
        return self.topk_indices_buffer[: hidden_states.shape[0]]

    def update_local_k_cache(self, indexer_k, layer_slot_mapping=None):
        self.local_k_cache_updates.append((indexer_k, layer_slot_mapping))

    def forward(self, hidden_states, q_c, positions, rotary_emb):
        q_fp8, indexer_k, weights = self.project(
            hidden_states, q_c, positions, rotary_emb
        )
        self.paged_args = (hidden_states, q_fp8, indexer_k, weights)
        return self.topk_indices_buffer[: hidden_states.shape[0]]


def test_wrapper_sharded_cp_uses_global_compact_kv(monkeypatch):
    calls = []
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    positions = torch.arange(3, dtype=torch.int64)
    q_c = torch.arange(6, dtype=torch.float32).view(3, 2) + 10
    kv_lora = torch.arange(9, dtype=torch.float32).view(3, 3) + 20
    q = torch.arange(9, dtype=torch.float32).view(3, 3) + 30
    fused_qkv = torch.cat([q_c, kv_lora], dim=-1)
    recording_attn = _RecordingAttention()
    indexer = _ProjectingIndexer()
    o_proj = _RecordingOProj()
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=3,
        max_seq_len=3,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        block_table=torch.arange(1, dtype=torch.int32).view(1, 1),
        req_id_per_token=torch.zeros(3, dtype=torch.int32),
        block_size=16,
        topk_tokens=2,
        topk_indices_are_global_compact_offsets=True,
    )
    context = ForwardContext(
        no_compile_layers={"layer.attn": recording_attn},
        attn_metadata={"layer.attn": metadata},
        slot_mapping={"layer.attn": torch.tensor([5, 6, 7], dtype=torch.int64)},
        virtual_engine=0,
        additional_kwargs={
            "sharded_cp_global_slot_mapping": {
                "layer.attn": torch.tensor([50, 51, 52, 99], dtype=torch.int64),
                "layer.indexer.k_cache": torch.tensor(
                    [60, 61, 62, 99], dtype=torch.int64
                ),
            },
            "sharded_cp_token_range": token_range,
            "sharded_cp_use_global_compact_kv": True,
        },
    )
    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.get_sharded_cp_group",
        lambda: SimpleNamespace(device_group=None),
    )
    original_async_gather = all_gather_sharded_cp_compact_kv_async

    class _RecordingHandle:
        def __init__(self, handle):
            self.handle = handle

        def wait(self):
            calls.append("wait_kv_ag")
            return self.handle.wait()

    def _record_async_gather(*args, **kwargs):
        calls.append("begin_kv_ag")
        return _RecordingHandle(original_async_gather(*args, **kwargs))

    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.all_gather_sharded_cp_compact_kv_async",
        _record_async_gather,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.MLAAttention",
        lambda *args, **kwargs: recording_attn,
    )

    wrapper = MultiHeadLatentAttentionWrapper(
        hidden_size=4,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        mla_modules=MLAModules(
            kv_a_layernorm=_IdentityNorm(),
            kv_b_proj=nn.Identity(),
            rotary_emb=_IdentityRotary(),
            o_proj=o_proj,
            fused_qkv_a_proj=_TupleLinear(fused_qkv, calls, "fused_qkv_a_proj"),
            kv_a_proj_with_mqa=None,
            q_a_layernorm=_IdentityNorm(),
            q_b_proj=_TupleLinear(q, calls, "q_b_proj"),
            q_proj=None,
            indexer=indexer,
            is_sparse=True,
            topk_indices_buffer=indexer.topk_indices_buffer,
            indexer_rotary_emb=_IdentityRotary(),
            enable_sharded_context_parallel=True,
        ),
        prefix="layer",
    )

    out = wrapper(positions, hidden_states)

    assert torch.equal(out, o_proj.input + 10)
    assert len(recording_attn.calls) == 1
    call_q, call_kv_c, call_k_pe, output_shape, use_global_kv = recording_attn.calls[0]
    assert use_global_kv is True
    assert output_shape == (3, 2)
    assert torch.equal(call_q, q.view(3, 1, 3))
    assert torch.equal(call_kv_c, kv_lora[:, :2])
    assert torch.equal(call_k_pe, kv_lora[:, 2:].unsqueeze(1))
    assert indexer.global_compact_args is not None
    assert torch.equal(indexer.global_compact_args[0], hidden_states)
    assert torch.equal(indexer.global_compact_args[2], hidden_states[:, :2] + 100)
    assert len(recording_attn.kv_cache_updates) == 1
    update_kv_c, update_k_pe, _, update_slot_mapping, _, _ = (
        recording_attn.kv_cache_updates[0]
    )
    assert torch.equal(update_kv_c, kv_lora[:, :2])
    assert torch.equal(update_k_pe, kv_lora[:, 2:].unsqueeze(1))
    assert update_slot_mapping.tolist() == [50, 51, 52]
    assert len(indexer.local_k_cache_updates) == 1
    update_indexer_k, update_indexer_slot_mapping = indexer.local_k_cache_updates[0]
    assert torch.equal(update_indexer_k, hidden_states[:, :2] + 100)
    assert update_indexer_slot_mapping.tolist() == [60, 61, 62]
    assert recording_attn.use_direct_call is True
    assert calls.index("begin_kv_ag") < calls.index("q_b_proj")
    assert calls.index("q_b_proj") < calls.index("wait_kv_ag")


def test_wrapper_sharded_cp_decode_uses_paged_kv_path(monkeypatch):
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    positions = torch.arange(3, dtype=torch.int64)
    q_c = torch.arange(6, dtype=torch.float32).view(3, 2) + 10
    kv_lora = torch.arange(9, dtype=torch.float32).view(3, 3) + 20
    q = torch.arange(9, dtype=torch.float32).view(3, 3) + 30
    expected_q = q.view(3, 1, 3).clone()
    expected_q[..., 2:] += 1000
    fused_qkv = torch.cat([q_c, kv_lora], dim=-1)
    recording_attn = _RecordingAttention()
    indexer = _ProjectingIndexer()
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([1, 1, 1], dtype=torch.int32),
        num_reqs=3,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        head_dim=2,
        num_decodes=3,
        num_decode_tokens=3,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=torch.zeros(3, 1, dtype=torch.int32),
            seq_lens=torch.ones(3, dtype=torch.int32),
            decode_lens=torch.ones(3, dtype=torch.int32),
            requires_padding=False,
            schedule_metadata=torch.empty(0, dtype=torch.int32),
            use_large_context_topk=False,
            offsets=None,
        ),
        k_is_global_compact=False,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer.indexer.k_cache": metadata},
        slot_mapping={},
        virtual_engine=0,
        additional_kwargs={
            "sharded_cp_token_range": token_range,
            "sharded_cp_use_global_compact_kv": False,
        },
    )

    def _reject_async_gather(*args, **kwargs):
        raise AssertionError("decode Sharded-CP must not all-gather compact KV")

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.all_gather_sharded_cp_compact_kv_async",
        _reject_async_gather,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.MLAAttention",
        lambda *args, **kwargs: recording_attn,
    )

    wrapper = MultiHeadLatentAttentionWrapper(
        hidden_size=4,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        mla_modules=MLAModules(
            kv_a_layernorm=_IdentityNorm(),
            kv_b_proj=nn.Identity(),
            rotary_emb=_OffsetRotary(),
            o_proj=_RecordingOProj(),
            fused_qkv_a_proj=_TupleLinear(fused_qkv),
            kv_a_proj_with_mqa=None,
            q_a_layernorm=_IdentityNorm(),
            q_b_proj=_TupleLinear(q),
            q_proj=None,
            indexer=indexer,
            is_sparse=True,
            topk_indices_buffer=indexer.topk_indices_buffer,
            indexer_rotary_emb=_IdentityRotary(),
            enable_sharded_context_parallel=True,
        ),
        prefix="layer",
    )

    wrapper(positions, hidden_states)

    assert indexer.global_compact_args is None
    assert indexer.paged_args is not None
    assert torch.equal(indexer.paged_args[2], hidden_states[:, :2] + 100)
    assert len(recording_attn.calls) == 1
    call_q, _, call_k_pe, _, use_global_kv = recording_attn.calls[0]
    assert use_global_kv is False
    assert torch.equal(call_q, expected_q)
    assert torch.equal(call_k_pe, kv_lora[:, 2:].unsqueeze(1) + 2000)


def test_wrapper_sharded_cp_begins_compact_kv_after_k_rope(monkeypatch):
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    positions = torch.arange(3, dtype=torch.int64)
    q_c = torch.arange(6, dtype=torch.float32).view(3, 2) + 10
    kv_lora = torch.arange(9, dtype=torch.float32).view(3, 3) + 20
    q = torch.arange(9, dtype=torch.float32).view(3, 3) + 30
    expected_q_rope = q.view(3, 1, 3)[..., 2:].clone() + 1000
    fused_qkv = torch.cat([q_c, kv_lora], dim=-1)
    recording_attn = _RecordingAttention()
    indexer = _ProjectingIndexer()
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=3,
        max_seq_len=3,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        block_table=torch.arange(1, dtype=torch.int32).view(1, 1),
        req_id_per_token=torch.zeros(3, dtype=torch.int32),
        block_size=16,
        topk_tokens=2,
        topk_indices_are_global_compact_offsets=True,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer.attn": metadata},
        slot_mapping={},
        virtual_engine=0,
        additional_kwargs={
            "sharded_cp_token_range": token_range,
            "sharded_cp_use_global_compact_kv": True,
        },
    )
    gathered_k_pe = []

    def _record_async_gather(kv_c_normed, k_pe, indexer_k, cp_range, *, group):
        gathered_k_pe.append(k_pe.clone())
        return all_gather_sharded_cp_compact_kv_async(
            kv_c_normed,
            k_pe,
            indexer_k,
            cp_range,
            group=group,
        )

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.get_sharded_cp_group",
        lambda: SimpleNamespace(device_group=None),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.all_gather_sharded_cp_compact_kv_async",
        _record_async_gather,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.MLAAttention",
        lambda *args, **kwargs: recording_attn,
    )

    wrapper = MultiHeadLatentAttentionWrapper(
        hidden_size=4,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        mla_modules=MLAModules(
            kv_a_layernorm=_IdentityNorm(),
            kv_b_proj=nn.Identity(),
            rotary_emb=_OffsetRotary(),
            o_proj=_RecordingOProj(),
            fused_qkv_a_proj=_TupleLinear(fused_qkv),
            kv_a_proj_with_mqa=None,
            q_a_layernorm=_IdentityNorm(),
            q_b_proj=_TupleLinear(q),
            q_proj=None,
            indexer=indexer,
            is_sparse=True,
            topk_indices_buffer=indexer.topk_indices_buffer,
            indexer_rotary_emb=_IdentityRotary(),
            enable_sharded_context_parallel=True,
        ),
        prefix="layer",
    )

    wrapper(positions, hidden_states)

    assert len(gathered_k_pe) == 1
    assert torch.equal(gathered_k_pe[0], kv_lora[:, 2:].unsqueeze(1) + 2000)
    call_q = recording_attn.calls[0][0]
    assert torch.equal(call_q[..., 2:], expected_q_rope)


def test_wrapper_sharded_cp_empty_rank_keeps_collective_without_local_kernels(
    monkeypatch,
):
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 4], dtype=torch.int32),
        rank=1,
        world_size=2,
    )
    hidden_states = torch.empty(0, 4, dtype=torch.float32)
    positions = torch.empty(0, dtype=torch.int64)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        virtual_engine=0,
        additional_kwargs={
            "sharded_cp_token_range": token_range,
            "sharded_cp_use_global_compact_kv": True,
        },
    )
    recording_attn = _RecordingAttention()
    gather_calls = []

    class _RejectingIndexer(nn.Module):
        topk_tokens = 2
        head_dim = 2

        def __init__(self):
            super().__init__()
            self.topk_indices_buffer = torch.empty(0, 2, dtype=torch.int32)

        def project(self, *args, **kwargs):
            raise AssertionError("empty Sharded-CP rank must not project indexer")

        def forward_global_compact(self, *args, **kwargs):
            raise AssertionError("empty Sharded-CP rank must not compute top-k")

    def _record_gather(kv_c_normed, k_pe, indexer_k, cp_range, *, group):
        gather_calls.append((kv_c_normed, k_pe, indexer_k, cp_range, group))
        return (
            torch.empty(4, 2, dtype=kv_c_normed.dtype),
            torch.empty(4, 1, 1, dtype=k_pe.dtype),
            torch.empty(4, 2, dtype=indexer_k.dtype),
        )

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.get_sharded_cp_group",
        lambda: SimpleNamespace(device_group="cp-group"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.all_gather_sharded_cp_compact_kv",
        _record_gather,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mla.MLAAttention",
        lambda *args, **kwargs: recording_attn,
    )

    wrapper = MultiHeadLatentAttentionWrapper(
        hidden_size=4,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        mla_modules=MLAModules(
            kv_a_layernorm=_RejectingModule(),
            kv_b_proj=nn.Identity(),
            rotary_emb=_RejectingModule(),
            o_proj=_RejectingModule(),
            fused_qkv_a_proj=_RejectingModule(),
            kv_a_proj_with_mqa=None,
            q_a_layernorm=_RejectingModule(),
            q_b_proj=_RejectingModule(),
            q_proj=None,
            indexer=_RejectingIndexer(),
            is_sparse=True,
            topk_indices_buffer=torch.empty(0, 2, dtype=torch.int32),
            indexer_rotary_emb=_RejectingModule(),
            enable_sharded_context_parallel=True,
        ),
        prefix="layer",
    )

    out = wrapper(positions, hidden_states)

    assert out.shape == (0, 4)
    assert out.dtype == hidden_states.dtype
    assert recording_attn.calls == []
    assert len(gather_calls) == 1
    kv_c_normed, k_pe, indexer_k, cp_range, group = gather_calls[0]
    assert kv_c_normed.shape == (0, 2)
    assert k_pe.shape == (0, 1, 1)
    assert indexer_k.shape == (0, 2)
    assert cp_range is token_range
    assert group == "cp-group"


def test_mla_attention_global_kv_skips_cache_update(monkeypatch):
    calls = SimpleNamespace(cache_updates=0, attn_kv=None)

    class _Backend:
        accept_output_buffer = True

    class _Impl:
        def do_kv_cache_update(self, *args, **kwargs):
            calls.cache_updates += 1

    def _forward_impl(q, kv_c_normed, k_pe, kv_cache, attn_metadata, output=None):
        calls.attn_kv = kv_cache
        output.copy_(torch.arange(output.numel(), dtype=output.dtype).view_as(output))
        return output

    layer = SimpleNamespace(
        calculate_kv_scales=False,
        use_direct_call=True,
        kv_cache=[torch.full((99, 3), -1.0)],
        layer_name="layer.attn",
        impl=_Impl(),
        kv_cache_dtype="auto",
        _k_scale=torch.tensor(1.0),
        attn_backend=_Backend(),
        forward_impl=_forward_impl,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer.attn": SimpleNamespace(num_actual_tokens=2)},
        slot_mapping={"layer.attn": torch.arange(2)},
        virtual_engine=0,
    )
    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    q = torch.zeros(2, 1, 3)
    kv_c = torch.arange(4, dtype=torch.float32).view(2, 2)
    k_pe = torch.arange(2, dtype=torch.float32).view(2, 1, 1) + 10

    out = MLAAttention.forward(
        layer,
        q,
        kv_c,
        k_pe,
        output_shape=torch.Size((2, 2)),
        use_global_kv=True,
    )

    assert calls.cache_updates == 0
    assert torch.equal(calls.attn_kv, torch.cat((kv_c, k_pe.squeeze(1)), dim=-1))
    assert out.shape == (2, 2)


def test_mla_attention_global_kv_rejects_opaque_path():
    layer = SimpleNamespace(calculate_kv_scales=False, use_direct_call=False)

    with pytest.raises(RuntimeError, match="global compact KV"):
        MLAAttention.forward(
            layer,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 2),
            torch.zeros(1, 1, 1),
            use_global_kv=True,
        )


def test_flashmla_sparse_global_compact_offsets_ignore_fp8_cache_path():
    impl = object.__new__(FlashMLASparseImpl)
    impl.kv_cache_dtype = "fp8_ds_mla"
    impl.topk_indices_buffer = torch.arange(6, dtype=torch.int32).view(3, 2)
    calls = []

    def _bf16_path(q, kv_cache, topk_indices, attn_metadata):
        calls.append(("bf16", topk_indices, attn_metadata))
        return torch.full((q.shape[0], 1, 2), 7.0)

    def _fp8_path(*args, **kwargs):
        raise AssertionError("global compact KV must not use FP8 paged-cache path")

    impl._forward_bf16_kv = _bf16_path
    impl._forward_fp8_kv_mixed_batch = _fp8_path
    impl._forward_fp8_kv_separate_prefill_decode = _fp8_path
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=3,
        max_seq_len=3,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        req_id_per_token=torch.zeros(3, dtype=torch.int32),
        fp8_use_mixed_batch=True,
        fp8_extra_metadata=object(),
        topk_indices_are_global_compact_offsets=True,
    )

    out, lse = FlashMLASparseImpl.forward_mqa(
        impl,
        torch.zeros(3, 1, 2),
        torch.zeros(3, 3),
        metadata,
        SimpleNamespace(),
    )

    assert lse is None
    assert out.tolist() == [[[7.0, 7.0]], [[7.0, 7.0]], [[7.0, 7.0]]]
    assert len(calls) == 1
    assert torch.equal(calls[0][1], impl.topk_indices_buffer)
    assert calls[0][2] is metadata


class _FakeLinear(nn.Module):
    instances: list["_FakeLinear"] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.disable_tp = kwargs.get("disable_tp", False)
        self.reduce_results = kwargs.get("reduce_results", True)
        _FakeLinear.instances.append(self)


class _FakeWrapper(nn.Module):
    instances: list["_FakeWrapper"] = []

    def __init__(self, hidden_size, num_heads, *args, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.args = args
        self.kwargs = kwargs
        _FakeWrapper.instances.append(self)


class _FakeIndexer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.topk_tokens = 4
        self.topk_indices_buffer = None


class _FakeNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _config(*, enable_sharded_cp: bool):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_sharded_context_parallel=enable_sharded_cp
        )
    )


def _hf_config(*, is_v32: bool = True):
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default"},
        qk_rope_head_dim=2,
        indexer_rope_interleave=False,
    )
    if is_v32:
        config.index_topk = 4
        config.index_n_heads = 8
        config.index_head_dim = 4
    return config


def _build_mla(
    monkeypatch,
    *,
    enable_sharded_cp: bool,
    is_v32: bool = True,
    prefix: str = "model.layers.0.self_attn",
):
    _FakeLinear.instances = []
    _FakeWrapper.instances = []
    monkeypatch.setattr(deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        deepseek_v2,
        "DeepSeekV2FusedQkvAProjLinear",
        _FakeLinear,
    )
    monkeypatch.setattr(deepseek_v2, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(deepseek_v2, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(deepseek_v2, "RMSNorm", _FakeNorm)
    monkeypatch.setattr(deepseek_v2, "LayerNorm", _FakeNorm)
    monkeypatch.setattr(deepseek_v2, "Indexer", _FakeIndexer)
    monkeypatch.setattr(deepseek_v2, "MultiHeadLatentAttentionWrapper", _FakeWrapper)
    monkeypatch.setattr(deepseek_v2, "get_rope", lambda *args, **kwargs: nn.Identity())

    return DeepseekV2MLAAttention(
        vllm_config=_config(enable_sharded_cp=enable_sharded_cp),
        config=_hf_config(is_v32=is_v32),
        hidden_size=16,
        num_heads=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        cache_config=None,
        quant_config=None,
        prefix=prefix,
        topk_indices_buffer=None,
    )


def test_deepseek_v32_sharded_cp_uses_full_head_attention_modules(monkeypatch):
    attn = _build_mla(monkeypatch, enable_sharded_cp=True)

    assert attn.use_sharded_cp_full_attention is True
    assert attn.q_b_proj.disable_tp is True
    assert attn.kv_b_proj.disable_tp is True
    assert attn.o_proj.disable_tp is True
    assert attn.o_proj.kwargs["input_is_parallel"] is True
    assert attn.o_proj.reduce_results is False
    assert attn.mla_attn.num_heads == 8
    assert attn.sharded_cp_shard_linear is not None
    assert attn.sharded_cp_shard_linear.layer_id == 0
    assert [name for name, _ in attn.sharded_cp_shard_linear.modules] == [
        "q_up_proj",
        "kv_b_proj",
        "o_proj",
    ]


def test_deepseek_v32_flag_off_keeps_tp_head_attention_modules(monkeypatch):
    attn = _build_mla(monkeypatch, enable_sharded_cp=False)

    assert attn.use_sharded_cp_full_attention is False
    assert attn.q_b_proj.disable_tp is False
    assert attn.kv_b_proj.disable_tp is False
    assert attn.o_proj.disable_tp is False
    assert attn.o_proj.kwargs["input_is_parallel"] is True
    assert attn.o_proj.reduce_results is True
    assert attn.mla_attn.num_heads == 4
    assert attn.sharded_cp_shard_linear is None


def test_deepseek_non_v32_keeps_tp_head_attention_modules(monkeypatch):
    attn = _build_mla(monkeypatch, enable_sharded_cp=True, is_v32=False)

    assert attn.use_sharded_cp_full_attention is False
    assert attn.q_b_proj.disable_tp is False
    assert attn.kv_b_proj.disable_tp is False
    assert attn.o_proj.disable_tp is False
    assert attn.o_proj.kwargs["input_is_parallel"] is True
    assert attn.o_proj.reduce_results is True
    assert attn.mla_attn.num_heads == 4
    assert attn.sharded_cp_shard_linear is None


def test_deepseek_sharded_cp_rejects_prefix_without_layer_id(monkeypatch):
    with pytest.raises(ValueError, match="Cannot infer layer id"):
        _build_mla(
            monkeypatch,
            enable_sharded_cp=True,
            prefix="draft.self_attn",
        )
