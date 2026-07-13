# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.sharded_cp_shard_linear import (
    ShardedCPShardLinearLayer,
    ShardedCPShardLinearPrefetch,
    get_sharded_cp_shard_linear_owner,
)

pytestmark = pytest.mark.skip_global_cleanup


class _FakeGroup:
    def __init__(
        self,
        world_size: int,
        rank: int,
        owner_payloads=None,
        fail_after: int | None = None,
    ):
        self.world_size = world_size
        self.rank_in_group = rank
        self.owner_payloads = owner_payloads or {}
        self.fail_after = fail_after
        self.broadcasts: list[tuple[int, torch.Size]] = []

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        if self.fail_after is not None and len(self.broadcasts) >= self.fail_after:
            raise RuntimeError("broadcast failed")
        self.broadcasts.append((src, torch.Size(tensor.shape)))
        if self.rank_in_group != src:
            payload = self.owner_payloads[tuple(tensor.shape)].to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
            tensor.copy_(payload)
        return tensor


class _FakeHandle:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


def _linear_with_weight(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


def test_shard_linear_owner_uses_layer_mod_world_size():
    assert [get_sharded_cp_shard_linear_owner(i, 4) for i in range(8)] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("layer_id", "world_size", "match"),
    [(-1, 2, "layer_id"), (0, 0, "world_size")],
)
def test_shard_linear_owner_rejects_invalid_inputs(layer_id, world_size, match):
    with pytest.raises(ValueError, match=match):
        get_sharded_cp_shard_linear_owner(layer_id, world_size)


def test_shard_linear_single_rank_keeps_parameter_storage():
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    linear = _linear_with_weight(weight)
    shard_linear = ShardedCPShardLinearLayer(0, [("q_b_proj", linear)])
    group = SimpleNamespace(world_size=1, rank_in_group=0)

    with shard_linear.materialized(group):
        assert torch.equal(linear.weight, weight)

    assert torch.equal(linear.weight, weight)


def test_shard_linear_describes_registered_parameters():
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    linear = _linear_with_weight(weight)
    shard_linear = ShardedCPShardLinearLayer(5, [("o_proj", linear)])
    group = SimpleNamespace(world_size=4, rank_in_group=1)

    params = shard_linear.describe_parameters(group)

    assert len(params) == 1
    assert params[0].name == "o_proj.weight"
    assert params[0].shape == torch.Size((3, 4))
    assert params[0].dtype == torch.float32
    assert params[0].owner_rank == 1


def test_shard_linear_materializes_non_owner_from_broadcast_then_releases():
    owner_weight = torch.arange(12, dtype=torch.float32).view(3, 4) + 100
    local_placeholder = torch.empty(0, dtype=torch.float32)
    linear = nn.Linear(4, 3, bias=False)
    linear.weight._sharded_cp_full_shape = torch.Size((3, 4))
    linear.weight._sharded_cp_full_dtype = torch.float32
    linear.weight._sharded_cp_full_device = torch.device("cpu")
    linear.weight.data = local_placeholder
    group = _FakeGroup(
        world_size=4,
        rank=0,
        owner_payloads={tuple(owner_weight.shape): owner_weight},
    )
    shard_linear = ShardedCPShardLinearLayer(1, [("q_b_proj", linear)])

    with shard_linear.materialized(group):
        assert torch.equal(linear.weight, owner_weight)

    assert linear.weight.numel() == 0
    assert group.broadcasts == [(1, torch.Size((3, 4)))]


def test_shard_linear_owner_keeps_persistent_storage_after_scope():
    owner_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    linear = _linear_with_weight(owner_weight)
    group = _FakeGroup(world_size=3, rank=2)
    shard_linear = ShardedCPShardLinearLayer(5, [("o_proj", linear)])

    with shard_linear.materialized(group):
        assert torch.equal(linear.weight, owner_weight)

    assert torch.equal(linear.weight, owner_weight)
    assert group.broadcasts == [(2, torch.Size((3, 4)))]


def test_shard_linear_owner_rejects_released_storage():
    owner_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    linear = _linear_with_weight(owner_weight)
    shard_linear = ShardedCPShardLinearLayer(0, [("q_b_proj", linear)])
    group = _FakeGroup(world_size=2, rank=0)
    shard_linear.describe_parameters(group)
    linear.weight.data = torch.empty(0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Owner Sharded-CP"):
        shard_linear.materialize(group)


def test_shard_linear_prefetch_materializes_then_releases_non_owner():
    owner_weight = torch.arange(12, dtype=torch.float32).view(3, 4) + 100
    linear = nn.Linear(4, 3, bias=False)
    linear.weight._sharded_cp_full_shape = torch.Size((3, 4))
    linear.weight._sharded_cp_full_dtype = torch.float32
    linear.weight._sharded_cp_full_device = torch.device("cpu")
    linear.weight.data = torch.empty(0, dtype=torch.float32)
    group = _FakeGroup(
        world_size=4,
        rank=0,
        owner_payloads={tuple(owner_weight.shape): owner_weight},
    )
    shard_linear = ShardedCPShardLinearLayer(1, [("q_b_proj", linear)])

    prefetch = shard_linear.prefetch(group)
    assert torch.equal(linear.weight, owner_weight)

    with prefetch.materialized():
        assert torch.equal(linear.weight, owner_weight)

    assert linear.weight.numel() == 0
    assert group.broadcasts == [(1, torch.Size((3, 4)))]


def test_shard_linear_prefetch_waits_async_handles_once():
    handle = _FakeHandle()
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    shard_linear = ShardedCPShardLinearLayer(0, [])
    prefetch = ShardedCPShardLinearPrefetch(shard_linear, group, [handle], [])

    prefetch.wait()
    prefetch.wait()

    assert handle.wait_count == 1


def test_shard_linear_prefetch_rejects_use_after_release():
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    shard_linear = ShardedCPShardLinearLayer(0, [])
    prefetch = ShardedCPShardLinearPrefetch(shard_linear, group, [], [])

    prefetch.release()

    with pytest.raises(RuntimeError, match="released"):
        prefetch.wait()
    with pytest.raises(RuntimeError, match="released more than once"):
        prefetch.release()


def test_shard_linear_prefetch_releases_non_owner_on_broadcast_failure():
    first_weight = torch.arange(12, dtype=torch.float32).view(3, 4) + 100
    second_weight = torch.arange(20, dtype=torch.float32).view(4, 5) + 200
    first_linear = nn.Linear(4, 3, bias=False)
    second_linear = nn.Linear(5, 4, bias=False)
    for linear, weight in (
        (first_linear, first_weight),
        (second_linear, second_weight),
    ):
        linear.weight._sharded_cp_full_shape = torch.Size(weight.shape)
        linear.weight._sharded_cp_full_dtype = torch.float32
        linear.weight._sharded_cp_full_device = torch.device("cpu")
        linear.weight.data = torch.empty(0, dtype=torch.float32)
    group = _FakeGroup(
        world_size=4,
        rank=0,
        owner_payloads={
            tuple(first_weight.shape): first_weight,
            tuple(second_weight.shape): second_weight,
        },
        fail_after=1,
    )
    shard_linear = ShardedCPShardLinearLayer(
        1,
        [("q_b_proj", first_linear), ("kv_b_proj", second_linear)],
    )

    with pytest.raises(RuntimeError, match="broadcast failed"):
        shard_linear.prefetch(group)

    assert first_linear.weight.numel() == 0
    assert second_linear.weight.numel() == 0
    assert group.broadcasts == [(1, torch.Size((3, 4)))]
