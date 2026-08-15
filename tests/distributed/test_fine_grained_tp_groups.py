# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank topology of the fine-grained (per-module) tensor parallel groups.

These groups are carved out of the DP axis, so a group is only usable if it is a
subset of a single DP group -- otherwise the o_proj / lm_head / embedding
collectives would span ranks that are not running the same step together. The
layout is pure index arithmetic, so it is checked here without spawning ranks.
"""

import pytest
import torch

from vllm.distributed.parallel_state import _build_fine_grained_tp_ranks


def rank_grid(external_dp: int, dp: int, pp: int, pcp: int, tp: int) -> torch.Tensor:
    """The `ExternalDP x DP x PP x PCP x TP` grid initialize_model_parallel builds."""
    return torch.arange(external_dp * dp * pp * pcp * tp).reshape(
        external_dp, dp, pp, pcp, tp
    )


def dp_group_ranks(all_ranks: torch.Tensor, dp: int) -> list[set[int]]:
    """The `_DP` group ranks, built exactly as parallel_state.py does."""
    return [
        set(x.tolist()) for x in all_ranks.transpose(1, 4).reshape(-1, dp).unbind(0)
    ]


@pytest.mark.parametrize(
    "external_dp,dp,pp,pcp,tp",
    [
        (1, 8, 1, 1, 1),  # plain DP, the common DP-attention deployment
        (1, 8, 1, 1, 2),  # DP x TP
        (1, 4, 2, 1, 2),  # DP x PP x TP
        (2, 4, 1, 2, 2),  # ExternalDP x DP x PCP x TP
        (1, 6, 1, 1, 3),  # non-power-of-two
    ],
)
def test_groups_partition_world_within_dp_groups(external_dp, dp, pp, pcp, tp):
    all_ranks = rank_grid(external_dp, dp, pp, pcp, tp)
    dp_groups = dp_group_ranks(all_ranks, dp)

    for group_size in (g for g in range(1, dp + 1) if dp % g == 0):
        groups = _build_fine_grained_tp_ranks(all_ranks, dp, group_size)

        # Every rank belongs to exactly one group.
        flat = [rank for group in groups for rank in group]
        assert sorted(flat) == list(range(all_ranks.numel()))
        assert all(len(group) == group_size for group in groups)

        # A group may never straddle two DP groups.
        for group in groups:
            assert any(set(group) <= dp_ranks for dp_ranks in dp_groups), (
                f"group {group} is not contained in a single DP group"
            )


def test_group_size_equal_to_dp_reproduces_dp_groups():
    """The widest fine-grained group is exactly the DP group."""
    all_ranks = rank_grid(1, 4, 2, 1, 2)
    groups = _build_fine_grained_tp_ranks(all_ranks, 4, 4)
    assert [set(g) for g in groups] == dp_group_ranks(all_ranks, 4)


def test_groups_do_not_straddle_pipeline_stages():
    """Regression guard for a DP/PP axis swap.

    Deriving the grid as `reshape(pp, dp, tp)` instead of using the real
    `ExternalDP x DP x PP x PCP x TP` layout yields [[0, 1, 2, 3], [4, 5, 6, 7]]
    here, which groups ranks *across* pipeline stages.
    """
    all_ranks = rank_grid(1, 4, 2, 1, 1)
    assert _build_fine_grained_tp_ranks(all_ranks, 4, 4) == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]


def test_group_size_must_divide_dp_size():
    all_ranks = rank_grid(1, 6, 1, 1, 1)
    with pytest.raises(ValueError, match="must divide data_parallel_size"):
        _build_fine_grained_tp_ranks(all_ranks, 6, 4)
