# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.sharded_cp_utils import (
    all_gather_token_rows,
    all_gather_token_rows_async,
    get_sharded_cp_group,
    get_sharded_cp_token_range,
    pad_for_token_all_gather,
    reduce_scatter_token_rows,
    slice_for_token_reduce_scatter,
    trim_token_all_gather,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("num_tokens", "world_size", "expected"),
    [
        (0, 4, [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]),
        (1, 4, [(0, 1, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)]),
        (8, 4, [(0, 2, 2), (2, 4, 4), (4, 6, 6), (6, 8, 8)]),
        (10, 4, [(0, 3, 3), (3, 6, 6), (6, 9, 9), (9, 10, 12)]),
    ],
)
def test_get_sharded_cp_token_range(num_tokens, world_size, expected):
    ranges = [
        get_sharded_cp_token_range(num_tokens, rank, world_size)
        for rank in range(world_size)
    ]

    assert [(r.start, r.end, r.padded_end) for r in ranges] == expected
    assert [(r.rank, r.world_size) for r in ranges] == [
        (rank, world_size) for rank in range(world_size)
    ]
    assert all(r.total_tokens == num_tokens for r in ranges)


@pytest.mark.parametrize(
    ("num_tokens", "rank", "world_size"),
    [
        (-1, 0, 1),
        (1, -1, 1),
        (1, 1, 1),
        (1, 0, 0),
    ],
)
def test_get_sharded_cp_token_range_rejects_invalid_inputs(
    num_tokens, rank, world_size
):
    with pytest.raises(ValueError):
        get_sharded_cp_token_range(num_tokens, rank, world_size)


def test_pad_for_token_all_gather_pads_only_trailing_rows():
    token_range = get_sharded_cp_token_range(num_tokens=10, rank=3, world_size=4)
    x = torch.tensor([[9.0]])

    padded = pad_for_token_all_gather(x, token_range, pad_value=-1.0)

    assert padded.tolist() == [[9.0], [-1.0], [-1.0]]


def test_pad_for_token_all_gather_rejects_wrong_local_rows():
    token_range = get_sharded_cp_token_range(num_tokens=10, rank=3, world_size=4)
    x = torch.zeros(3, 1)

    with pytest.raises(ValueError, match="row count"):
        pad_for_token_all_gather(x, token_range)


def test_trim_token_all_gather_removes_global_padding():
    token_range = get_sharded_cp_token_range(num_tokens=10, rank=0, world_size=4)
    gathered = torch.arange(12).view(12, 1)

    trimmed = trim_token_all_gather(gathered, token_range)

    assert trimmed.squeeze(-1).tolist() == list(range(10))


def test_slice_for_token_reduce_scatter_returns_padded_local_chunk():
    token_range = get_sharded_cp_token_range(num_tokens=10, rank=3, world_size=4)
    x = torch.arange(20).view(10, 2)

    local = slice_for_token_reduce_scatter(x, token_range, pad_value=-1)

    assert local.tolist() == [[18, 19], [-1, -1], [-1, -1]]


def test_slice_for_token_reduce_scatter_rejects_wrong_global_rows():
    token_range = get_sharded_cp_token_range(num_tokens=10, rank=0, world_size=4)
    x = torch.zeros(9, 1)

    with pytest.raises(ValueError, match="global tensor row count"):
        slice_for_token_reduce_scatter(x, token_range)


def test_all_gather_token_rows_returns_single_rank_input_without_dist():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    x = torch.arange(4).view(2, 2)

    gathered = all_gather_token_rows(x, token_range)

    assert torch.equal(gathered, x)


def test_all_gather_token_rows_async_single_rank_wait_returns_input():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    x = torch.arange(4).view(2, 2)

    handle = all_gather_token_rows_async(x, token_range)

    assert handle.wait() is x
    assert handle.wait() is x


def test_all_gather_token_rows_single_rank_rejects_wrong_local_rows():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    x = torch.zeros(3, 2)

    with pytest.raises(ValueError, match="local tensor row count"):
        all_gather_token_rows(x, token_range)


def test_all_gather_token_rows_returns_empty_single_rank_input_without_dist():
    token_range = get_sharded_cp_token_range(num_tokens=0, rank=0, world_size=1)
    x = torch.empty(0, 2)

    gathered = all_gather_token_rows(x, token_range)

    assert gathered is x


def test_all_gather_token_rows_requires_dist_for_multi_rank():
    token_range = get_sharded_cp_token_range(num_tokens=4, rank=0, world_size=2)
    x = torch.arange(4).view(2, 2)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        all_gather_token_rows(x, token_range)


def test_all_gather_token_rows_rank0_requires_dist_when_multi_rank():
    token_range = get_sharded_cp_token_range(num_tokens=1, rank=0, world_size=4)
    x = torch.arange(2).view(1, 2)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        all_gather_token_rows(x, token_range)


def test_reduce_scatter_token_rows_returns_single_rank_input_without_dist():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    x = torch.arange(4).view(2, 2)

    reduced = reduce_scatter_token_rows(x, token_range)

    assert torch.equal(reduced, x)


def test_reduce_scatter_token_rows_single_rank_rejects_wrong_global_rows():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    x = torch.zeros(3, 2)

    with pytest.raises(ValueError, match="global tensor row count"):
        reduce_scatter_token_rows(x, token_range)


def test_reduce_scatter_token_rows_requires_dist_for_multi_rank():
    token_range = get_sharded_cp_token_range(num_tokens=4, rank=0, world_size=2)
    x = torch.arange(8).view(4, 2)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        reduce_scatter_token_rows(x, token_range)


def test_reduce_scatter_token_rows_rank0_requires_dist_when_rows_match():
    token_range = get_sharded_cp_token_range(num_tokens=1, rank=0, world_size=4)
    x = torch.arange(2).view(1, 2)

    with pytest.raises(RuntimeError, match="torch.distributed"):
        reduce_scatter_token_rows(x, token_range)


def test_get_sharded_cp_group_reuses_tp_group(monkeypatch):
    tp_group = object()

    monkeypatch.setattr(
        "vllm.distributed.sharded_cp_utils.get_tp_group",
        lambda: tp_group,
    )

    assert get_sharded_cp_group() is tp_group
