# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU unit tests for Sharded-CP building blocks.

Launch with:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
        tools/sharded_cp/test_dist_sharded_cp.py

Covers, on real NCCL collectives:
  - balanced / request-aligned token-row all-gather round trips
  - token-row reduce-scatter against a locally computed reference
  - compact ``[kv_c || k_pe || indexer_k]`` all-gather round trip
  - MoE input/routing gathers and output reduce-scatter
  - the all-to-all that converts CP rows x all heads into all rows x local
    heads (so o_proj keeps its TP weight) and the reduce-scatter back
"""

import os
import traceback

import torch
import torch.distributed as dist

from vllm.distributed.sharded_cp_compact_kv import (
    all_gather_sharded_cp_compact_kv,
    all_gather_sharded_cp_compact_kv_async,
)
from vllm.distributed.sharded_cp_utils import (
    all_gather_token_rows,
    all_to_all_cp_rows_to_tp_heads,
    get_request_aligned_sharded_cp_token_range,
    get_sharded_cp_token_range,
    reduce_scatter_padded_token_rows,
    reduce_scatter_token_rows,
)
from vllm.model_executor.layers.fused_moe.sharded_cp_moe import (
    all_gather_sharded_cp_moe_inputs,
    all_gather_sharded_cp_moe_routing_metadata,
    reduce_scatter_sharded_cp_moe_output,
)

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
HIDDEN = 64


def _global_reference(num_tokens: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234 + num_tokens + dim)
    return torch.randn(num_tokens, dim, generator=g, dtype=torch.float32).to(DEVICE)


def test_balanced_all_gather_round_trip():
    for num_tokens in (0, 1, WORLD, 4 * WORLD - 1, 37):
        ref = _global_reference(num_tokens)
        token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
        local = ref[token_range.start : token_range.end]
        gathered = all_gather_token_rows(local, token_range)
        assert gathered.shape == ref.shape, (gathered.shape, ref.shape)
        torch.testing.assert_close(gathered, ref)


def test_request_aligned_all_gather_round_trip():
    cases = [
        [0, 5, 9, 10],  # 3 requests, uneven
        [0, 10],  # single request -> later ranks empty
        [0, 1, 2, 3, 4, 5, 6, 7],  # many small requests
    ]
    for starts in cases:
        query_start_loc = torch.tensor(starts, dtype=torch.int32)
        token_range = get_request_aligned_sharded_cp_token_range(
            query_start_loc, RANK, WORLD
        )
        ref = _global_reference(starts[-1])
        local = ref[token_range.start : token_range.end]
        gathered = all_gather_token_rows(local, token_range)
        torch.testing.assert_close(gathered, ref)


def test_reduce_scatter_round_trip():
    for num_tokens in (1, WORLD, 4 * WORLD - 1, 37):
        token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
        # Per-rank contribution seeded by rank; reference = sum over ranks.
        contribs = [
            torch.randn(
                num_tokens,
                HIDDEN,
                generator=torch.Generator(device="cpu").manual_seed(
                    100 * num_tokens + r
                ),
                dtype=torch.float32,
            ).to(DEVICE)
            for r in range(WORLD)
        ]
        expected = torch.stack(contribs).sum(dim=0)[
            token_range.start : token_range.end
        ]
        reduced = reduce_scatter_token_rows(contribs[RANK], token_range)
        torch.testing.assert_close(reduced, expected, rtol=1e-4, atol=1e-4)


def test_compact_kv_round_trip():
    num_tokens = 4 * WORLD + 3
    kv_ref = _global_reference(num_tokens, 512)
    pe_ref = _global_reference(num_tokens, 64)
    ik_ref = _global_reference(num_tokens, 128)
    token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
    s = slice(token_range.start, token_range.end)

    kv_c, k_pe, indexer_k = all_gather_sharded_cp_compact_kv(
        kv_ref[s], pe_ref[s].unsqueeze(1), ik_ref[s], token_range
    )
    torch.testing.assert_close(kv_c, kv_ref)
    torch.testing.assert_close(k_pe.squeeze(1), pe_ref)
    torch.testing.assert_close(indexer_k, ik_ref)

    handle = all_gather_sharded_cp_compact_kv_async(
        kv_ref[s], pe_ref[s].unsqueeze(1), ik_ref[s], token_range
    )
    kv_c2, _, _ = handle.wait()
    torch.testing.assert_close(kv_c2, kv_ref)


def test_compact_kv_round_trip_zero_width_indexer_k():
    """Shared top-k (IndexCache) layers gather MLA KV with no Indexer-K."""
    num_tokens = 3 * WORLD + 1
    kv_ref = _global_reference(num_tokens, 512)
    pe_ref = _global_reference(num_tokens, 64)
    token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
    s = slice(token_range.start, token_range.end)
    empty_ik = kv_ref.new_empty((token_range.num_tokens, 0))

    kv_c, k_pe, indexer_k = all_gather_sharded_cp_compact_kv(
        kv_ref[s], pe_ref[s].unsqueeze(1), empty_ik, token_range
    )
    torch.testing.assert_close(kv_c, kv_ref)
    torch.testing.assert_close(k_pe.squeeze(1), pe_ref)
    assert indexer_k.shape == (num_tokens, 0)


def test_moe_layout_round_trip():
    num_tokens = 3 * WORLD + 1
    n_experts, topk = 16, 4
    hidden_ref = _global_reference(num_tokens)
    logits_ref = _global_reference(num_tokens, n_experts)
    token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
    s = slice(token_range.start, token_range.end)

    inputs = all_gather_sharded_cp_moe_inputs(
        hidden_ref[s], logits_ref[s], token_range
    )
    torch.testing.assert_close(inputs.hidden_states, hidden_ref)
    torch.testing.assert_close(inputs.router_logits, logits_ref)

    weights_ref = _global_reference(num_tokens, topk)
    ids_ref = (weights_ref * 1000).abs().long() % n_experts
    routing = all_gather_sharded_cp_moe_routing_metadata(
        weights_ref[s], ids_ref[s].to(weights_ref.dtype), token_range
    )
    torch.testing.assert_close(routing.topk_weights, weights_ref)
    torch.testing.assert_close(routing.topk_ids, ids_ref.to(weights_ref.dtype))

    contribs = [
        torch.randn(
            num_tokens,
            HIDDEN,
            generator=torch.Generator(device="cpu").manual_seed(7 + r),
            dtype=torch.float32,
        ).to(DEVICE)
        for r in range(WORLD)
    ]
    expected = torch.stack(contribs).sum(dim=0)[s]
    out = reduce_scatter_sharded_cp_moe_output(contribs[RANK], token_range)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_all_to_all_cp_rows_to_tp_heads_round_trip():
    """CP rows x all heads -> all rows x local heads, in global token order."""
    num_heads, head_dim = 4 * WORLD, 8
    for num_tokens in (WORLD, 3 * WORLD + 1, 17):
        token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
        # Value encodes (global_row, head) so misrouted data is detectable.
        rows = torch.arange(num_tokens, device=DEVICE, dtype=torch.float32)
        heads = torch.arange(num_heads, device=DEVICE, dtype=torch.float32)
        ref = rows[:, None, None] * 1000 + heads[None, :, None]
        ref = ref.expand(num_tokens, num_heads, head_dim).contiguous()

        local = ref[token_range.start : token_range.end]
        out = all_to_all_cp_rows_to_tp_heads(local, token_range)

        chunk = token_range.padded_num_tokens
        assert out.shape == (WORLD * chunk, num_heads // WORLD, head_dim)
        local_heads = num_heads // WORLD
        head_slice = slice(RANK * local_heads, (RANK + 1) * local_heads)
        expected = torch.zeros_like(out)
        expected[:num_tokens] = ref[:, head_slice]
        torch.testing.assert_close(out[:num_tokens], expected[:num_tokens])


def test_reduce_scatter_padded_token_rows_matches_sum():
    """Partial TP sums over padded global rows reduce to local CP rows."""
    for num_tokens in (WORLD, 3 * WORLD + 2):
        token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
        chunk = token_range.padded_num_tokens
        contribs = [
            torch.randn(
                WORLD * chunk,
                HIDDEN,
                generator=torch.Generator(device="cpu").manual_seed(50 + r),
                dtype=torch.float32,
            ).to(DEVICE)
            for r in range(WORLD)
        ]
        total = torch.stack(contribs).sum(dim=0)
        expected = total[RANK * chunk : RANK * chunk + token_range.num_tokens]
        out = reduce_scatter_padded_token_rows(contribs[RANK], token_range)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def main():
    torch.cuda.set_device(DEVICE)
    dist.init_process_group("nccl")
    tests = [
        test_balanced_all_gather_round_trip,
        test_request_aligned_all_gather_round_trip,
        test_reduce_scatter_round_trip,
        test_compact_kv_round_trip,
        test_compact_kv_round_trip_zero_width_indexer_k,
        test_moe_layout_round_trip,
        test_all_to_all_cp_rows_to_tp_heads_round_trip,
        test_reduce_scatter_padded_token_rows_matches_sum,
    ]
    failed = []
    for test in tests:
        try:
            test()
            if RANK == 0:
                print(f"PASS {test.__name__}")
        except Exception:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__} (rank {RANK})")
            traceback.print_exc()
        # Barrier outside the try block: a rank that failed must still join,
        # otherwise the passing ranks hang waiting for it.
        dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit(f"rank {RANK}: {len(failed)} failed: {failed}")
    if RANK == 0:
        print(f"all {len(tests)} distributed tests passed on {WORLD} ranks")


if __name__ == "__main__":
    main()
