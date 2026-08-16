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
  - Shard Linear owner broadcast, release, and async prefetch parity
"""

import os
import traceback

import torch
import torch.distributed as dist
from torch import nn

from vllm.distributed.sharded_cp_compact_kv import (
    all_gather_sharded_cp_compact_kv,
    all_gather_sharded_cp_compact_kv_async,
)
from vllm.distributed.sharded_cp_utils import (
    all_gather_token_rows,
    get_request_aligned_sharded_cp_token_range,
    get_sharded_cp_token_range,
    reduce_scatter_token_rows,
)
from vllm.model_executor.layers.fused_moe.sharded_cp_moe import (
    all_gather_sharded_cp_moe_inputs,
    all_gather_sharded_cp_moe_routing_metadata,
    reduce_scatter_sharded_cp_moe_output,
)
from vllm.model_executor.layers.sharded_cp_shard_linear import (
    ShardedCPShardLinearLayer,
    get_sharded_cp_shard_linear_owner,
)

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
HIDDEN = 64


class FakeGroup:
    """Duck-typed stand-in for GroupCoordinator over the WORLD group."""

    def __init__(self):
        self.world_size = WORLD
        self.rank_in_group = RANK
        self.ranks = list(range(WORLD))
        self.device_group = dist.group.WORLD

    def broadcast(self, tensor, src=0):
        dist.broadcast(tensor, src=self.ranks[src])


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


def _make_shard_linear(layer_id: int, group: FakeGroup):
    torch.manual_seed(layer_id)  # same full weights on every rank
    q_up = nn.Linear(32, 48, bias=False).to(DEVICE)
    o_proj = nn.Linear(48, 32, bias=False).to(DEVICE)
    shard = ShardedCPShardLinearLayer(
        layer_id, (("q_up_proj", q_up), ("o_proj", o_proj))
    )
    # Record full metadata, then drop non-owner copies like
    # post_process_weights_after_loading does.
    shard.describe_parameters(group)
    reference = {n: p.detach().clone() for n, p in shard._iter_params()}
    shard.release_non_owner(group)
    return shard, reference


def test_shard_linear_materialize_and_prefetch():
    group = FakeGroup()
    for layer_id in range(WORLD + 1):
        owner = get_sharded_cp_shard_linear_owner(layer_id, WORLD)
        shard, reference = _make_shard_linear(layer_id, group)

        for name, param in shard._iter_params():
            if RANK == owner:
                assert param.numel() > 0, f"owner lost {name}"
            else:
                assert param.numel() == 0, f"non-owner kept {name}"

        # Synchronous materialize scope.
        with shard.materialized(group):
            for name, param in shard._iter_params():
                torch.testing.assert_close(param.data, reference[name])
        for _, param in shard._iter_params():
            assert (RANK == owner) == (param.numel() > 0)

        # Async prefetch scope.
        prefetch = shard.prefetch(group)
        with prefetch.materialized():
            for name, param in shard._iter_params():
                torch.testing.assert_close(param.data, reference[name])
        assert prefetch.released
        for _, param in shard._iter_params():
            assert (RANK == owner) == (param.numel() > 0)


def test_prefetch_overlaps_compact_kv_gather():
    """Prefetch broadcasts and compact-KV gathers run on different
    communicators; interleaving issues and waits must stay correct."""
    group = FakeGroup()
    shard0, ref0 = _make_shard_linear(10, group)
    shard1, ref1 = _make_shard_linear(11, group)

    num_tokens = 4 * WORLD + 1
    kv_ref = _global_reference(num_tokens, 512)
    pe_ref = _global_reference(num_tokens, 64)
    ik_ref = _global_reference(num_tokens, 128)
    token_range = get_sharded_cp_token_range(num_tokens, RANK, WORLD)
    s = slice(token_range.start, token_range.end)

    # Issue two layers' weight broadcasts first (prefetch communicator),
    # then the compact-KV gather (default communicator), and wait for the
    # gather BEFORE the prefetches — the pattern the model produces.
    prefetch0 = shard0.prefetch(group)
    prefetch1 = shard1.prefetch(group)
    handle = all_gather_sharded_cp_compact_kv_async(
        kv_ref[s], pe_ref[s].unsqueeze(1), ik_ref[s], token_range
    )
    kv_c, k_pe, indexer_k = handle.wait()
    torch.testing.assert_close(kv_c, kv_ref)
    torch.testing.assert_close(k_pe.squeeze(1), pe_ref)
    torch.testing.assert_close(indexer_k, ik_ref)

    for shard, prefetch, reference in (
        (shard0, prefetch0, ref0),
        (shard1, prefetch1, ref1),
    ):
        with prefetch.materialized():
            for name, param in shard._iter_params():
                torch.testing.assert_close(param.data, reference[name])


def test_prefetch_group_teardown_and_reinit():
    """destroy_sharded_cp_prefetch_group() must allow a clean re-create."""
    from vllm.model_executor.layers.sharded_cp_shard_linear import (
        destroy_sharded_cp_prefetch_group,
        ensure_sharded_cp_prefetch_group,
    )

    group = FakeGroup()
    assert ensure_sharded_cp_prefetch_group(group) is not None
    destroy_sharded_cp_prefetch_group()

    shard, reference = _make_shard_linear(21, group)
    prefetch = shard.prefetch(group)  # re-creates the communicator lazily
    with prefetch.materialized():
        for name, param in shard._iter_params():
            torch.testing.assert_close(param.data, reference[name])


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
        test_shard_linear_materialize_and_prefetch,
        test_prefetch_overlaps_compact_kv_gather,
        test_prefetch_group_teardown_and_reinit,
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
