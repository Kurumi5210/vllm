# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm import forward_context as forward_context_module
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.distributed.sharded_cp_utils import (
    all_gather_token_rows,
    get_sharded_cp_token_range,
    shard_global_token_rows,
    slice_for_token_reduce_scatter,
    trim_token_all_gather,
)

pytestmark = pytest.mark.skip_global_cleanup


def _gather_local_rows(local_rows: list[torch.Tensor], total_tokens: int):
    gathered = torch.cat(local_rows, dim=0)
    token_range = get_sharded_cp_token_range(
        num_tokens=total_tokens,
        rank=0,
        world_size=len(local_rows),
    )
    return trim_token_all_gather(gathered, token_range)


def test_fake_layer_round_trip_preserves_token_order_and_logits_shape():
    total_tokens = 10
    hidden_size = 4
    vocab_size = 7
    world_size = 4
    hidden = torch.arange(total_tokens * hidden_size, dtype=torch.float32).view(
        total_tokens, hidden_size
    )
    lm_head = torch.arange(hidden_size * vocab_size, dtype=torch.float32).view(
        hidden_size, vocab_size
    )
    fake_layer = nn.Identity()

    local_outputs = []
    for rank in range(world_size):
        local_hidden, token_range = shard_global_token_rows(
            hidden,
            rank,
            world_size,
            pad_value=0.0,
        )
        local_hidden = local_hidden[: token_range.num_tokens]

        local_outputs.append(fake_layer(local_hidden))

    gathered_hidden = _gather_local_rows(local_outputs, total_tokens)
    logits = gathered_hidden @ lm_head

    assert torch.equal(gathered_hidden, hidden)
    assert logits.shape == (total_tokens, vocab_size)


def test_sharded_cp_hidden_matches_unsharded_after_gather_for_uneven_tokens():
    total_tokens = 11
    hidden_size = 3
    world_size = 4
    hidden = torch.arange(total_tokens * hidden_size, dtype=torch.float32).view(
        total_tokens, hidden_size
    )

    local_outputs = [
        slice_for_token_reduce_scatter(
            hidden,
            get_sharded_cp_token_range(total_tokens, rank, world_size),
        )[: get_sharded_cp_token_range(total_tokens, rank, world_size).num_tokens]
        for rank in range(world_size)
    ]

    assert torch.equal(_gather_local_rows(local_outputs, total_tokens), hidden)


def test_embedding_contributions_reduce_then_gather_matches_full_hidden():
    total_tokens = 7
    hidden_size = 2
    world_size = 3
    contributions = [
        torch.full((total_tokens, hidden_size), rank + 1, dtype=torch.float32)
        for rank in range(world_size)
    ]
    full_hidden = torch.stack(contributions).sum(dim=0)

    local_reduced = []
    for rank in range(world_size):
        token_range = get_sharded_cp_token_range(total_tokens, rank, world_size)
        local_chunk = sum(
            slice_for_token_reduce_scatter(contribution, token_range)
            for contribution in contributions
        )
        local_reduced.append(local_chunk[: token_range.num_tokens])

    assert torch.equal(_gather_local_rows(local_reduced, total_tokens), full_hidden)


def test_inputs_embeds_are_sliced_without_embedding_reduction():
    total_tokens = 7
    hidden_size = 2
    world_size = 3
    inputs_embeds = torch.arange(total_tokens * hidden_size, dtype=torch.float32).view(
        total_tokens, hidden_size
    )

    local_hidden = []
    for rank in range(world_size):
        token_range = get_sharded_cp_token_range(total_tokens, rank, world_size)
        local_hidden.append(
            slice_for_token_reduce_scatter(inputs_embeds, token_range)[
                : token_range.num_tokens
            ]
        )

    assert torch.equal(_gather_local_rows(local_hidden, total_tokens), inputs_embeds)


def test_lm_head_all_gather_single_rank_fast_path_before_logits():
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)

    gathered = all_gather_token_rows(hidden, token_range)

    assert gathered is hidden


def test_sharded_cp_real_transformer_layers_can_enter_layer_loop_after_stage6():
    model = object.__new__(DeepseekV2Model)
    model.enable_sharded_context_parallel = True
    model.start_layer = 0
    model.end_layer = 1

    with model._sharded_cp_forward_context():
        pass


def test_sharded_cp_boundary_only_context_still_available_after_stage6():
    model = object.__new__(DeepseekV2Model)
    model.enable_sharded_context_parallel = True
    model.start_layer = 0
    model.end_layer = 0

    with model._sharded_cp_forward_context():
        pass


def test_sharded_cp_boundary_only_context_optional_without_forward_context(
    monkeypatch,
):
    monkeypatch.setattr(forward_context_module, "_forward_context", None)
    model = object.__new__(DeepseekV2Model)
    model.enable_sharded_context_parallel = True
    model.sharded_cp_token_range = None
    model.start_layer = 0
    model.end_layer = 0

    with model._sharded_cp_forward_context():
        pass


def test_model_forward_prefetches_sharded_cp_attention_weights(monkeypatch):
    events = []

    class _FakePrefetch:
        def __init__(self, layer_idx):
            self.layer_id = layer_idx
            self.layer_idx = layer_idx
            self.released = False

        def materialized(self):
            events.append(("wait", self.layer_idx))
            prefetch = self

            class _Scope:
                def __enter__(self):
                    events.append(("enter", prefetch.layer_idx))

                def __exit__(self, exc_type, exc, tb):
                    prefetch.release()
                    events.append(("exit", prefetch.layer_idx))

            return _Scope()

        def release(self):
            if self.released:
                raise RuntimeError("double release")
            self.released = True
            events.append(("release", self.layer_idx))

    class _FakeLayer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

        def prefetch_sharded_cp_attention_weights(self, group):
            events.append(("prefetch", self.layer_idx, group.rank_in_group))
            return _FakePrefetch(self.layer_idx)

        def __call__(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
            *,
            sharded_cp_prefetch=None,
        ):
            assert sharded_cp_prefetch is not None
            with sharded_cp_prefetch.materialized():
                events.append(("layer", self.layer_idx))
            return hidden_states + 1, hidden_states

    monkeypatch.setattr(
        deepseek_v2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2, device_group=None),
    )
    model = object.__new__(DeepseekV2Model)
    nn.Module.__init__(model)
    model.enable_sharded_context_parallel = True
    model.sharded_cp_token_range = get_sharded_cp_token_range(
        num_tokens=2,
        rank=1,
        world_size=2,
    )
    model.config = SimpleNamespace(llama_4_scaling=None)
    model.start_layer = 0
    model.end_layer = 4
    model.layers = nn.ModuleList([_FakeLayer(i) for i in range(4)])
    model.aux_hidden_state_layers = tuple()
    model.norm = lambda hidden_states, residual: (hidden_states, residual)
    model._maybe_scatter_to_sharded_cp = (
        lambda hidden_states, positions, *, reduce_hidden_states: (
            hidden_states,
            positions,
        )
    )
    model._sharded_cp_forward_context = nullcontext

    output = model.forward(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(2, 3),
    )

    assert torch.equal(output, torch.full((2, 3), 4.0))
    assert events == [
        ("prefetch", 0, 1),
        ("prefetch", 1, 1),
        ("prefetch", 2, 1),
        ("wait", 0),
        ("enter", 0),
        ("layer", 0),
        ("release", 0),
        ("exit", 0),
        ("prefetch", 3, 1),
        ("wait", 1),
        ("enter", 1),
        ("layer", 1),
        ("release", 1),
        ("exit", 1),
        ("wait", 2),
        ("enter", 2),
        ("layer", 2),
        ("release", 2),
        ("exit", 2),
        ("wait", 3),
        ("enter", 3),
        ("layer", 3),
        ("release", 3),
        ("exit", 3),
    ]


def test_model_forward_releases_pending_prefetch_on_layer_error(monkeypatch):
    events = []

    class _FakePrefetch:
        layer_id = 0

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.released = False

        def release(self):
            if self.released:
                raise RuntimeError("double release")
            self.released = True
            events.append(("release", self.layer_idx))

    class _FailingLayer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

        def prefetch_sharded_cp_attention_weights(self, group):
            events.append(("prefetch", self.layer_idx))
            return _FakePrefetch(self.layer_idx)

        def __call__(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
            *,
            sharded_cp_prefetch=None,
        ):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        deepseek_v2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2, device_group=None),
    )
    model = object.__new__(DeepseekV2Model)
    nn.Module.__init__(model)
    model.enable_sharded_context_parallel = True
    model.sharded_cp_token_range = get_sharded_cp_token_range(
        num_tokens=2,
        rank=0,
        world_size=2,
    )
    model.config = SimpleNamespace(llama_4_scaling=None)
    model.start_layer = 0
    model.end_layer = 1
    model.layers = nn.ModuleList([_FailingLayer(0)])
    model.aux_hidden_state_layers = tuple()
    model._maybe_scatter_to_sharded_cp = (
        lambda hidden_states, positions, *, reduce_hidden_states: (
            hidden_states,
            positions,
        )
    )
    model._sharded_cp_forward_context = nullcontext

    with pytest.raises(RuntimeError, match="boom"):
        model.forward(
            input_ids=None,
            positions=torch.arange(2),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros(2, 3),
        )

    assert events == [("prefetch", 0), ("release", 0)]


def test_maybe_scatter_to_sharded_cp_uses_balanced_forward_context(
    monkeypatch,
):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 200, 50], query_lens=[100, 200, 50]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=common.num_reqs,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        num_actual_tokens=common.num_actual_tokens,
        query_start_loc=common.query_start_loc,
        slot_mapping=common.slot_mapping,
        block_table=common.block_table_tensor,
        req_id_per_token=torch.arange(common.num_actual_tokens, dtype=torch.int32),
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer": metadata},
        slot_mapping={},
        virtual_engine=0,
    )
    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2, device_group=None),
    )
    model = object.__new__(DeepseekV2Model)
    model.enable_sharded_context_parallel = True
    model.sharded_cp_token_range = None
    hidden = torch.arange(350 * 2, dtype=torch.float32).view(350, 2)
    positions = torch.arange(350, dtype=torch.int64)

    local_hidden, local_positions = model._maybe_scatter_to_sharded_cp(
        hidden,
        positions,
        reduce_hidden_states=False,
    )

    assert (model.sharded_cp_token_range.start, model.sharded_cp_token_range.end) == (
        175,
        350,
    )
    assert model.sharded_cp_token_range.local_request_global_starts == (100, 300)
    assert torch.equal(local_hidden, hidden[175:350])
    assert torch.equal(local_positions, positions[175:350])


def test_compute_logits_clears_sharded_cp_token_range_after_gather(monkeypatch):
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    causal_lm._sharded_cp_logits_hidden_states_prepared = False

    def logits_processor(_lm_head, x):
        return x + 1

    causal_lm.model = SimpleNamespace(
        enable_sharded_context_parallel=True,
        sharded_cp_token_range=token_range,
    )
    causal_lm.lm_head = object()
    causal_lm.logits_processor = logits_processor
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(device_group=None),
    )

    logits = causal_lm.compute_logits(hidden)

    assert torch.equal(logits, hidden + 1)
    assert causal_lm.model.sharded_cp_token_range is None


def test_prepare_hidden_states_for_logits_clears_sharded_cp_token_range(monkeypatch):
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    causal_lm._sharded_cp_logits_hidden_states_prepared = False
    causal_lm.model = SimpleNamespace(
        enable_sharded_context_parallel=True,
        sharded_cp_token_range=token_range,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(device_group=None),
    )

    prepared = causal_lm.prepare_hidden_states_for_logits(hidden)

    assert prepared is hidden
    assert causal_lm.model.sharded_cp_token_range is None
    assert causal_lm._sharded_cp_logits_hidden_states_prepared is True


def test_compute_logits_uses_prepared_sharded_cp_hidden_without_second_gather():
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    causal_lm._sharded_cp_logits_hidden_states_prepared = True

    def logits_processor(_lm_head, x):
        return x + 1

    causal_lm.model = SimpleNamespace(
        enable_sharded_context_parallel=True,
        sharded_cp_token_range=None,
    )
    causal_lm.lm_head = object()
    causal_lm.logits_processor = logits_processor

    logits = causal_lm.compute_logits(hidden)

    assert torch.equal(logits, hidden + 1)
    assert causal_lm.model.sharded_cp_token_range is None
    assert causal_lm._sharded_cp_logits_hidden_states_prepared is True


def test_compute_logits_uses_unsharded_hidden_when_no_cp_token_range():
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    causal_lm._sharded_cp_logits_hidden_states_prepared = False

    def logits_processor(_lm_head, x):
        return x + 1

    causal_lm.model = SimpleNamespace(
        enable_sharded_context_parallel=True,
        sharded_cp_token_range=None,
    )
    causal_lm.lm_head = object()
    causal_lm.logits_processor = logits_processor

    logits = causal_lm.compute_logits(hidden)

    assert torch.equal(logits, hidden + 1)


def test_load_weights_defers_sharded_cp_non_owner_release(monkeypatch):
    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.config = SimpleNamespace(
        n_routed_experts=0,
        n_shared_experts=None,
    )
    causal_lm.num_redundant_experts = 0
    causal_lm.use_mha = False
    released = []
    causal_lm.model = SimpleNamespace(
        named_parameters=lambda: iter(()),
        release_sharded_cp_non_owner_weights=lambda: released.append(True),
    )
    monkeypatch.setattr(
        deepseek_v2.SharedFusedMoE,
        "make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )

    loaded = DeepseekV2ForCausalLM.load_weights(causal_lm, [])

    assert loaded == set()
    assert released == []

    causal_lm.post_process_weights_after_loading()

    assert released == [True]


def test_process_weights_after_loading_runs_sharded_cp_post_hook():
    calls = []

    class FakeModel(nn.Module):
        def post_process_weights_after_loading(self):
            calls.append("post")

    process_weights_after_loading(
        FakeModel(),
        SimpleNamespace(dtype=torch.float32, quantization=None),
        torch.device("cpu"),
    )

    assert calls == ["post"]
