# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.attention.backends.mla.sharded_cp_metadata import (
    build_sharded_cp_attention_metadata,
    get_sharded_cp_token_range_from_forward_context,
    localize_common_attention_metadata,
    localize_deepseek_v32_indexer_metadata,
    localize_flashmla_sparse_metadata,
    sharded_cp_forward_context,
)
from vllm.distributed.sharded_cp_utils import (
    ShardedCPTokenRange,
    assemble_token_all_gather_chunks,
    get_request_aligned_sharded_cp_token_range,
    get_request_aligned_sharded_cp_token_ranges,
    get_sharded_cp_token_range,
    make_reduce_scatter_token_chunks,
)

pytestmark = pytest.mark.skip_global_cleanup


def _common_metadata():
    return create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 200, 50], query_lens=[100, 200, 50]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )


def _forward_context(attn_metadata):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping={},
        virtual_engine=0,
    )


def test_request_aligned_ranges_keep_whole_requests():
    ranges = get_request_aligned_sharded_cp_token_ranges(
        torch.tensor([0, 100, 300, 350], dtype=torch.int32),
        world_size=2,
    )

    assert [(r.start, r.end, r.padded_num_tokens) for r in ranges] == [
        (0, 300, 300),
        (300, 350, 300),
    ]
    assert ranges[0].rank_starts == (0, 300)
    assert ranges[0].rank_ends == (300, 350)


def test_request_aligned_reduce_scatter_chunks_round_trip():
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 100, 300, 350], dtype=torch.int32),
        rank=0,
        world_size=2,
    )
    x = torch.arange(350, dtype=torch.float32).view(350, 1)

    chunks = make_reduce_scatter_token_chunks(x, token_range, pad_value=-1.0)
    gathered = assemble_token_all_gather_chunks(chunks, token_range)

    assert chunks[0].shape == chunks[1].shape == (300, 1)
    assert chunks[1][50:].eq(-1.0).all()
    assert torch.equal(gathered, x)


def test_localize_common_metadata_request_aligned_rank0():
    metadata = _common_metadata()
    token_range = get_request_aligned_sharded_cp_token_range(
        metadata.query_start_loc_cpu,
        rank=0,
        world_size=2,
    )

    local = localize_common_attention_metadata(metadata, token_range)

    assert local.num_actual_tokens == 300
    assert local.num_reqs == 2
    assert local.query_start_loc_cpu.tolist() == [0, 100, 300]
    assert local.seq_lens.tolist() == [100, 200]
    assert local.slot_mapping.tolist() == list(range(300))
    assert local.block_table_tensor.shape[0] == 2


def test_localize_common_metadata_request_aligned_rank1():
    metadata = _common_metadata()
    token_range = get_request_aligned_sharded_cp_token_range(
        metadata.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = localize_common_attention_metadata(metadata, token_range)

    assert local.num_actual_tokens == 50
    assert local.num_reqs == 1
    assert local.query_start_loc_cpu.tolist() == [0, 50]
    assert local.seq_lens.tolist() == [50]
    assert local.slot_mapping.tolist() == list(range(300, 350))
    assert local.block_table_tensor.shape[0] == 1


def test_localize_common_metadata_supports_split_request():
    metadata = _common_metadata()
    token_range = get_sharded_cp_token_range(
        num_tokens=metadata.num_actual_tokens,
        rank=0,
        world_size=2,
    )

    local = localize_common_attention_metadata(metadata, token_range)

    assert local.num_actual_tokens == 175
    assert local.num_reqs == 2
    assert local.query_start_loc_cpu.tolist() == [0, 100, 175]
    assert local.seq_lens.tolist() == [100, 75]
    assert local._num_computed_tokens_cpu is not None
    assert local._num_computed_tokens_cpu.tolist() == [0, 0]
    assert local.slot_mapping.tolist() == list(range(175))
    assert local.block_table_tensor.tolist() == (
        metadata.block_table_tensor[:2].tolist()
    )


def test_localize_common_metadata_supports_single_request_balanced_split():
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[8192], query_lens=[8192]),
        block_size=64,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    token_range = get_sharded_cp_token_range(
        num_tokens=metadata.num_actual_tokens,
        rank=2,
        world_size=4,
    )

    local = localize_common_attention_metadata(metadata, token_range)

    assert (token_range.start, token_range.end) == (4096, 6144)
    assert local.num_actual_tokens == 2048
    assert local.num_reqs == 1
    assert local.query_start_loc_cpu.tolist() == [0, 2048]
    assert local.seq_lens.tolist() == [6144]
    assert local._num_computed_tokens_cpu is not None
    assert local._num_computed_tokens_cpu.tolist() == [4096]
    assert local.max_query_len == 2048
    assert local.max_seq_len == 6144
    assert local.slot_mapping.tolist() == list(range(4096, 6144))
    assert local.block_table_tensor.shape[0] == 1


def _indexer_metadata(common):
    query_start_loc_cpu = common.query_start_loc_cpu
    seq_lens_cpu = common._seq_lens_cpu
    assert seq_lens_cpu is not None
    block_table = common.block_table_tensor
    cu_seq_lens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lens_cpu.cumsum(dim=0).to(torch.int32),
        ]
    )
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=block_table,
        cu_seqlen_ks=torch.arange(common.num_actual_tokens, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(common.num_actual_tokens, dtype=torch.int32) + 1,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=torch.repeat_interleave(
            torch.arange(common.num_reqs, dtype=torch.int32),
            seq_lens_cpu,
        ),
        total_seq_lens=seq_lens_cpu.sum(),
        token_start=int(query_start_loc_cpu[0].item()),
        token_end=int(query_start_loc_cpu[-1].item()),
        num_reqs=common.num_reqs,
    )
    return DeepseekV32IndexerMetadata(
        seq_lens=common.seq_lens,
        num_reqs=common.num_reqs,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        num_actual_tokens=common.num_actual_tokens,
        query_start_loc=common.query_start_loc,
        slot_mapping=common.slot_mapping,
        head_dim=128,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=common.num_reqs,
        num_prefill_tokens=common.num_actual_tokens,
        prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
    )


def _indexer_metadata_with_decode(
    common,
    *,
    num_decodes: int,
    flattened_decode: bool = False,
):
    query_lens = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]
    num_decode_tokens = int(common.query_start_loc_cpu[num_decodes].item())
    num_prefill_tokens = common.num_actual_tokens - num_decode_tokens
    decode_lens = query_lens[:num_decodes].to(torch.int32)
    decode_lens_for_metadata = decode_lens
    decode_seq_lens = common.seq_lens[:num_decodes]
    decode_block_table = common.block_table_tensor[:num_decodes]
    if flattened_decode and num_decode_tokens > 0:
        decode_lens_for_metadata = torch.ones(num_decode_tokens, dtype=torch.int32)
        decode_seq_lens = torch.repeat_interleave(
            common.seq_lens[:num_decodes],
            decode_lens,
        )
        decode_block_table = torch.repeat_interleave(
            common.block_table_tensor[:num_decodes],
            decode_lens,
            dim=0,
        )

    decode_metadata = None
    if num_decodes > 0:
        decode_metadata = DeepSeekV32IndexerDecodeMetadata(
            block_table=decode_block_table,
            seq_lens=decode_seq_lens,
            decode_lens=decode_lens_for_metadata,
            requires_padding=False,
            schedule_metadata=torch.empty(0, dtype=torch.int32),
            use_large_context_topk=False,
            offsets=None,
            block_size=16,
        )

    prefill_metadata = None
    num_prefills = common.num_reqs - num_decodes
    if num_prefills > 0:
        seq_lens_cpu = common._seq_lens_cpu
        assert seq_lens_cpu is not None
        block_table = common.block_table_tensor
        prefill_query_start_loc = (
            common.query_start_loc_cpu[num_decodes:] - num_decode_tokens
        )
        cu_seq_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                seq_lens_cpu[num_decodes:].cumsum(dim=0).to(torch.int32),
            ]
        )
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=block_table[num_decodes:],
            cu_seqlen_ks=torch.arange(num_prefill_tokens, dtype=torch.int32),
            cu_seqlen_ke=torch.arange(num_prefill_tokens, dtype=torch.int32) + 1,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=torch.repeat_interleave(
                torch.arange(num_prefills, dtype=torch.int32),
                seq_lens_cpu[num_decodes:],
            ),
            total_seq_lens=seq_lens_cpu[num_decodes:].sum(),
            token_start=int(common.query_start_loc_cpu[num_decodes].item()),
            token_end=int(common.query_start_loc_cpu[-1].item()),
            num_reqs=num_prefills,
        )
        assert prefill_query_start_loc[0] == 0
        prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks=[chunk])

    return DeepseekV32IndexerMetadata(
        seq_lens=common.seq_lens,
        num_reqs=common.num_reqs,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        num_actual_tokens=common.num_actual_tokens,
        query_start_loc=common.query_start_loc,
        slot_mapping=common.slot_mapping,
        head_dim=128,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        decode=decode_metadata,
        prefill=prefill_metadata,
    )


def test_localize_indexer_metadata_rebuilds_prefill_chunk_for_rank1():
    common = _common_metadata()
    metadata = _indexer_metadata(common)
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = localize_deepseek_v32_indexer_metadata(
        metadata,
        token_range,
    )

    assert local.num_actual_tokens == 50
    assert local.k_is_global_compact is True
    assert local.num_reqs == 1
    assert local.query_start_loc.tolist() == [0, 50]
    assert local.slot_mapping.tolist() == list(range(300, 350))
    assert local.prefill is not None
    assert len(local.prefill.chunks) == 1
    chunk = local.prefill.chunks[0]
    assert chunk.token_start == 0
    assert chunk.token_end == 50
    assert chunk.num_reqs == 1
    assert chunk.block_table.tolist() == common.block_table_tensor[2:3].tolist()


def test_localize_indexer_metadata_rebuilds_split_prefill_fragment():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8192], query_lens=[8192]),
        block_size=64,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _indexer_metadata(common)
    token_range = get_sharded_cp_token_range(
        num_tokens=common.num_actual_tokens,
        rank=2,
        world_size=4,
    )

    local = localize_deepseek_v32_indexer_metadata(metadata, token_range)

    assert local.num_actual_tokens == 2048
    assert local.k_is_global_compact is True
    assert local.num_reqs == 1
    assert local.query_start_loc.tolist() == [0, 2048]
    assert local.seq_lens.tolist() == [6144]
    assert local.num_decodes == 0
    assert local.num_decode_tokens == 0
    assert local.num_prefills == 1
    assert local.num_prefill_tokens == 2048
    assert local.slot_mapping.tolist() == list(range(4096, 6144))
    assert local.prefill is not None
    assert len(local.prefill.chunks) == 1
    chunk = local.prefill.chunks[0]
    assert chunk.token_start == 0
    assert chunk.token_end == 2048
    assert chunk.num_reqs == 1
    assert chunk.cu_seqlen_ks[0].item() == 0
    assert chunk.cu_seqlen_ks[-1].item() == 0
    assert chunk.cu_seqlen_ke[0].item() == 4097
    assert chunk.cu_seqlen_ke[-1].item() == 6144
    assert chunk.block_table.tolist() == common.block_table_tensor.tolist()


def test_localize_indexer_metadata_slices_decode_for_rank1():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 9000, 30], query_lens=[1, 1, 1, 1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _indexer_metadata_with_decode(common, num_decodes=common.num_reqs)
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = localize_deepseek_v32_indexer_metadata(metadata, token_range)

    assert local.num_actual_tokens == 2
    assert local.num_reqs == 2
    assert local.query_start_loc.tolist() == [0, 1, 2]
    assert local.num_decodes == 2
    assert local.num_decode_tokens == 2
    assert local.num_prefills == 0
    assert local.num_prefill_tokens == 0
    assert local.prefill is None
    assert local.decode is not None
    assert local.decode.decode_lens.tolist() == [1, 1]
    assert local.decode.seq_lens.tolist() == [9000, 30]
    assert local.decode.block_table.tolist() == common.block_table_tensor[2:4].tolist()
    assert local.decode.use_large_context_topk is True


def test_localize_indexer_metadata_preserves_mixed_decode_prefill():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 40, 30], query_lens=[1, 1, 4, 3]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _indexer_metadata_with_decode(common, num_decodes=2)
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=0,
        world_size=2,
    )

    local = localize_deepseek_v32_indexer_metadata(metadata, token_range)

    assert local.num_actual_tokens == 6
    assert local.num_reqs == 3
    assert local.query_start_loc.tolist() == [0, 1, 2, 6]
    assert local.num_decodes == 2
    assert local.num_decode_tokens == 2
    assert local.num_prefills == 1
    assert local.num_prefill_tokens == 4
    assert local.decode is not None
    assert local.decode.decode_lens.tolist() == [1, 1]
    assert local.decode.seq_lens.tolist() == [10, 20]
    assert local.decode.block_table.tolist() == common.block_table_tensor[:2].tolist()
    assert local.prefill is not None
    assert len(local.prefill.chunks) == 1
    chunk = local.prefill.chunks[0]
    assert chunk.token_start == 2
    assert chunk.token_end == 6
    assert chunk.num_reqs == 1
    assert chunk.block_table.tolist() == common.block_table_tensor[2:3].tolist()


def test_localize_indexer_metadata_slices_flattened_decode_rows():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 30], query_lens=[2, 3, 1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _indexer_metadata_with_decode(
        common,
        num_decodes=common.num_reqs,
        flattened_decode=True,
    )
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=3,
    )

    local = localize_deepseek_v32_indexer_metadata(metadata, token_range)

    assert local.num_actual_tokens == 3
    assert local.num_reqs == 1
    assert local.query_start_loc.tolist() == [0, 3]
    assert local.num_decodes == 3
    assert local.num_decode_tokens == 3
    assert local.decode is not None
    assert local.decode.decode_lens.tolist() == [1, 1, 1]
    assert local.decode.seq_lens.tolist() == [18, 19, 20]
    assert local.decode.block_table.tolist() == (
        common.block_table_tensor[1:2].repeat(3, 1).tolist()
    )


def _flashmla_metadata(common):
    seg_lengths = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]
    req_id_per_token = torch.repeat_interleave(
        torch.arange(common.num_reqs, dtype=torch.int32),
        seg_lengths,
    )
    return FlashMLASparseMetadata(
        num_reqs=common.num_reqs,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        num_actual_tokens=common.num_actual_tokens,
        query_start_loc=common.query_start_loc,
        slot_mapping=common.slot_mapping,
        block_table=common.block_table_tensor,
        req_id_per_token=req_id_per_token,
        block_size=16,
        topk_tokens=32,
    )


def test_localize_flashmla_sparse_metadata_slices_req_ids():
    common = _common_metadata()
    metadata = _flashmla_metadata(common)
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = localize_flashmla_sparse_metadata(metadata, token_range)

    assert local.num_actual_tokens == 50
    assert local.num_reqs == 1
    assert local.query_start_loc.tolist() == [0, 50]
    assert local.slot_mapping.tolist() == list(range(300, 350))
    assert local.block_table.tolist() == common.block_table_tensor[2:3].tolist()
    assert local.req_id_per_token.unique().tolist() == [0]
    assert local.topk_indices_are_global_compact_offsets is True


def test_localize_flashmla_sparse_metadata_rebuilds_split_request():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8192], query_lens=[8192]),
        block_size=64,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _flashmla_metadata(common)
    token_range = get_sharded_cp_token_range(
        num_tokens=common.num_actual_tokens,
        rank=2,
        world_size=4,
    )

    local = localize_flashmla_sparse_metadata(metadata, token_range)

    assert local.num_actual_tokens == 2048
    assert local.num_reqs == 1
    assert local.query_start_loc.tolist() == [0, 2048]
    assert local.slot_mapping.tolist() == list(range(4096, 6144))
    assert local.block_table.tolist() == common.block_table_tensor.tolist()
    assert local.req_id_per_token.unique().tolist() == [0]
    assert local.topk_indices_are_global_compact_offsets is True


def test_localize_flashmla_sparse_metadata_converts_fp8_to_global_compact():
    common = _common_metadata()
    metadata = _flashmla_metadata(common)
    metadata.fp8_extra_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
        scheduler_metadata=object(),
        dummy_block_table=torch.ones(1, 1, dtype=torch.int32),
        cache_lens=torch.ones(1, dtype=torch.int32),
    )
    metadata.fp8_use_mixed_batch = True
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = localize_flashmla_sparse_metadata(metadata, token_range)

    assert local.num_actual_tokens == 50
    assert local.query_start_loc.tolist() == [0, 50]
    assert local.fp8_extra_metadata is None
    assert local.fp8_use_mixed_batch is False
    assert local.topk_indices_are_global_compact_offsets is True


def test_build_sharded_cp_attention_metadata_keeps_layer_keys():
    common = _common_metadata()
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    local = build_sharded_cp_attention_metadata(
        {
            "model.layers.0.self_attn.attn": _flashmla_metadata(common),
            "model.layers.0.self_attn.indexer.k_cache": _indexer_metadata(common),
        },
        token_range,
    )

    assert set(local) == {
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
    }
    assert local["model.layers.0.self_attn.attn"].num_actual_tokens == 50
    assert (
        local["model.layers.0.self_attn.indexer.k_cache"].num_actual_tokens == 50
    )


def test_sharded_cp_forward_context_overrides_and_restores(monkeypatch):
    common = _common_metadata()
    context = _forward_context({"layer": _flashmla_metadata(common)})
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with sharded_cp_forward_context(context, token_range):
        active = get_forward_context()
        assert active is not context
        assert active.attn_metadata["layer"].num_actual_tokens == 50
        assert active.slot_mapping["layer"].tolist() == list(range(300, 350))
        active_range = active.additional_kwargs["sharded_cp_token_range"]
        assert (active_range.start, active_range.end) == (
            token_range.start,
            token_range.end,
        )
        assert active_range.local_request_global_starts == (300,)

    assert get_forward_context() is context


def test_sharded_cp_forward_context_marks_pure_prefill_global_compact(monkeypatch):
    common = _common_metadata()
    context = _forward_context(
        {
            "layer.indexer.k_cache": _indexer_metadata(common),
            "layer.attn": _flashmla_metadata(common),
        }
    )
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=1,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with sharded_cp_forward_context(context, token_range):
        active = get_forward_context()
        assert active.additional_kwargs[
            "sharded_cp_use_global_compact_kv"
        ] is True
        assert active.attn_metadata[
            "layer.indexer.k_cache"
        ].k_is_global_compact is True
        assert active.attn_metadata[
            "layer.attn"
        ].topk_indices_are_global_compact_offsets is True

    assert get_forward_context() is context


def test_sharded_cp_forward_context_keeps_decode_on_paged_kv(monkeypatch):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 40, 30], query_lens=[1, 1, 4, 3]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    flashmla_metadata = _flashmla_metadata(common)
    context = _forward_context(
        {
            "layer.indexer.k_cache": _indexer_metadata_with_decode(
                common,
                num_decodes=2,
            ),
            "layer.attn": flashmla_metadata,
        }
    )
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=0,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with sharded_cp_forward_context(context, token_range):
        active = get_forward_context()
        assert active.additional_kwargs[
            "sharded_cp_use_global_compact_kv"
        ] is False
        assert active.attn_metadata[
            "layer.indexer.k_cache"
        ].k_is_global_compact is False
        assert active.attn_metadata[
            "layer.attn"
        ].topk_indices_are_global_compact_offsets is False

    assert get_forward_context() is context


def test_sharded_cp_forward_context_keeps_decode_paged_when_attn_metadata_first(
    monkeypatch,
):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 40, 30], query_lens=[1, 1, 4, 3]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    context = _forward_context(
        {
            "layer.attn": _flashmla_metadata(common),
            "layer.indexer.k_cache": _indexer_metadata_with_decode(
                common,
                num_decodes=2,
            ),
        }
    )
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=0,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with sharded_cp_forward_context(context, token_range):
        active = get_forward_context()
        assert active.additional_kwargs[
            "sharded_cp_use_global_compact_kv"
        ] is False
        assert active.attn_metadata[
            "layer.indexer.k_cache"
        ].k_is_global_compact is False
        assert active.attn_metadata[
            "layer.attn"
        ].topk_indices_are_global_compact_offsets is False

    assert get_forward_context() is context


def test_sharded_cp_forward_context_rejects_paged_fp8_flashmla_metadata(
    monkeypatch,
):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[10, 20, 40, 30], query_lens=[1, 1, 4, 3]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    flashmla_metadata = _flashmla_metadata(common)
    flashmla_metadata.fp8_extra_metadata = (
        FlashMLASparseMetadata.FP8SeparatePrefillDecode(
            num_decodes=2,
            num_prefills=2,
            num_decode_tokens=2,
            num_prefill_tokens=7,
        )
    )
    context = _forward_context(
        {
            "layer.indexer.k_cache": _indexer_metadata_with_decode(
                common,
                num_decodes=2,
            ),
            "layer.attn": flashmla_metadata,
        }
    )
    token_range = get_request_aligned_sharded_cp_token_range(
        common.query_start_loc_cpu,
        rank=0,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with pytest.raises(RuntimeError, match="paged FlashMLA sparse metadata"):
        with sharded_cp_forward_context(context, token_range):
            pass


def test_sharded_cp_forward_context_keeps_extend_prefill_on_paged_kv(monkeypatch):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20], query_lens=[4]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    context = _forward_context(
        {
            "layer.indexer.k_cache": _indexer_metadata(common),
            "layer.attn": _flashmla_metadata(common),
        }
    )
    token_range = get_sharded_cp_token_range(
        num_tokens=common.num_actual_tokens,
        rank=0,
        world_size=2,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with sharded_cp_forward_context(context, token_range):
        active = get_forward_context()
        assert active.additional_kwargs[
            "sharded_cp_use_global_compact_kv"
        ] is False
        indexer_metadata = active.attn_metadata["layer.indexer.k_cache"]
        assert indexer_metadata.k_is_global_compact is False
        assert indexer_metadata.prefill is not None
        chunk = indexer_metadata.prefill.chunks[0]
        assert chunk.token_start == 0
        assert chunk.token_end == 2
        assert chunk.token_end - chunk.token_start == chunk.cu_seqlen_ks.numel()
        assert chunk.cu_seqlen_ks.tolist() == [0, 0]
        assert chunk.cu_seqlen_ke.tolist() == [17, 18]
        flashmla_metadata = active.attn_metadata["layer.attn"]
        assert flashmla_metadata.topk_indices_are_global_compact_offsets is False
        assert flashmla_metadata.fp8_extra_metadata is None

    assert get_forward_context() is context


def test_localize_indexer_paged_profile_chunks_keep_row_counts_aligned():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[128] * 8, query_lens=[16] * 8),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = _indexer_metadata(common)
    token_range = get_sharded_cp_token_range(
        num_tokens=common.num_actual_tokens,
        rank=1,
        world_size=4,
    )

    local = localize_deepseek_v32_indexer_metadata(
        metadata,
        token_range,
        use_global_compact_kv=False,
    )

    assert local.k_is_global_compact is False
    assert local.prefill is not None
    assert local.num_actual_tokens == 32
    for chunk in local.prefill.chunks:
        assert chunk.token_end > chunk.token_start
        assert chunk.token_end - chunk.token_start == chunk.cu_seqlen_ks.numel()
        assert chunk.cu_seqlen_ks.shape == chunk.cu_seqlen_ke.shape


def test_token_range_from_forward_context_uses_balanced_token_boundaries():
    common = _common_metadata()
    context = _forward_context({"layer": _flashmla_metadata(common)})

    token_range = get_sharded_cp_token_range_from_forward_context(
        context,
        rank=0,
        world_size=2,
    )

    assert (token_range.start, token_range.end) == (0, 175)
    assert token_range.local_request_starts == (0, 100)
    assert token_range.local_request_ends == (100, 175)
    assert token_range.local_request_global_starts == (0, 100)


def test_token_range_from_forward_context_splits_single_request():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8192], query_lens=[8192]),
        block_size=64,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    context = _forward_context({"layer": _flashmla_metadata(common)})

    token_range = get_sharded_cp_token_range_from_forward_context(
        context,
        rank=1,
        world_size=4,
    )

    assert (token_range.start, token_range.end) == (2048, 4096)
    assert token_range.local_request_starts == (0,)
    assert token_range.local_request_ends == (2048,)
    assert token_range.local_request_global_starts == (0,)


def test_token_range_from_forward_context_rejects_missing_metadata():
    context = _forward_context(None)

    with pytest.raises(RuntimeError, match="per-layer attention metadata"):
        get_sharded_cp_token_range_from_forward_context(
            context,
            rank=0,
            world_size=2,
        )


def test_token_range_from_forward_context_rejects_non_dict_metadata():
    context = _forward_context([])

    with pytest.raises(RuntimeError, match="per-layer attention metadata"):
        get_sharded_cp_token_range_from_forward_context(
            context,
            rank=0,
            world_size=2,
        )


def test_sharded_cp_forward_context_rejects_missing_metadata(
    monkeypatch,
):
    context = _forward_context(None)
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 4], dtype=torch.int32),
        rank=0,
        world_size=1,
    )

    from vllm import forward_context as forward_context_module

    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    with pytest.raises(RuntimeError, match="per-layer attention metadata"):
        with sharded_cp_forward_context(context, token_range):
            pass

    assert get_forward_context() is context
