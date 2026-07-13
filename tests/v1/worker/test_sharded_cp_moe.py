# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import forward_context as forward_context_module
from vllm.forward_context import ForwardContext
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MoE,
)
from vllm.model_executor.layers.fused_moe.sharded_cp_moe import (
    all_gather_sharded_cp_moe_inputs,
    all_gather_sharded_cp_moe_routing_metadata,
    assemble_sharded_cp_moe_input_chunks,
    assemble_sharded_cp_moe_routing_chunks,
    combine_sharded_cp_moe_residual,
    reduce_scatter_sharded_cp_moe_output,
    slice_sharded_cp_moe_output,
)
from vllm.distributed.sharded_cp_utils import (
    assemble_token_all_gather_chunks,
    get_request_aligned_sharded_cp_token_range,
    get_sharded_cp_token_range,
    make_reduce_scatter_token_chunks,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_moe_input_all_gather_single_rank_preserves_shape():
    token_range = get_sharded_cp_token_range(num_tokens=3, rank=0, world_size=1)
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4)
    router_logits = torch.arange(15, dtype=torch.float32).view(3, 5) + 100
    scales = torch.arange(3, dtype=torch.float32).view(3, 1) + 200

    gathered = all_gather_sharded_cp_moe_inputs(
        hidden,
        router_logits,
        token_range,
        activation_scales=scales,
    )

    assert torch.equal(gathered.hidden_states, hidden)
    assert torch.equal(gathered.router_logits, router_logits)
    assert gathered.activation_scales is not None
    assert torch.equal(gathered.activation_scales, scales)


def test_moe_input_assemble_request_aligned_chunks_round_trip():
    query_start_loc = torch.tensor([0, 100, 300, 350], dtype=torch.int32)
    token_range = get_request_aligned_sharded_cp_token_range(
        query_start_loc,
        rank=0,
        world_size=2,
    )
    hidden = torch.arange(350 * 4, dtype=torch.float32).view(350, 4)
    router_logits = torch.arange(350 * 6, dtype=torch.float32).view(350, 6) + 10_000
    scales = torch.arange(350, dtype=torch.float32).view(350, 1) + 20_000

    hidden_chunks = make_reduce_scatter_token_chunks(
        hidden,
        token_range,
        pad_value=-1.0,
    )
    router_chunks = make_reduce_scatter_token_chunks(
        router_logits,
        token_range,
        pad_value=-1.0,
    )
    scale_chunks = make_reduce_scatter_token_chunks(
        scales,
        token_range,
        pad_value=-1.0,
    )

    gathered = assemble_sharded_cp_moe_input_chunks(
        hidden_chunks,
        router_chunks,
        token_range,
        gathered_activation_scales=scale_chunks,
    )

    assert hidden_chunks[0].shape == hidden_chunks[1].shape == (300, 4)
    assert hidden_chunks[1][50:].eq(-1.0).all()
    assert torch.equal(gathered.hidden_states, hidden)
    assert torch.equal(gathered.router_logits, router_logits)
    assert gathered.activation_scales is not None
    assert torch.equal(gathered.activation_scales, scales)


def test_moe_routing_metadata_all_gather_order_matches_global_baseline():
    query_start_loc = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    token_range = get_request_aligned_sharded_cp_token_range(
        query_start_loc,
        rank=0,
        world_size=2,
    )
    router_logits = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.0],
            [0.3, 0.2, 0.8, 0.1],
            [0.7, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.3, 0.6],
            [0.0, 0.5, 0.2, 0.4],
            [0.9, 0.1, 0.3, 0.2],
            [0.1, 0.8, 0.4, 0.3],
            [0.2, 0.7, 0.6, 0.1],
        ],
        dtype=torch.float32,
    )
    baseline_weights, baseline_ids = torch.topk(router_logits, k=2, dim=-1)
    weight_chunks = make_reduce_scatter_token_chunks(
        baseline_weights,
        token_range,
        pad_value=-1.0,
    )
    id_chunks = make_reduce_scatter_token_chunks(
        baseline_ids.to(torch.int32),
        token_range,
        pad_value=-1,
    )

    routing = assemble_sharded_cp_moe_routing_chunks(
        weight_chunks,
        id_chunks,
        token_range,
    )

    assert torch.equal(routing.topk_weights, baseline_weights)
    assert torch.equal(routing.topk_ids, baseline_ids.to(torch.int32))
    assert routing.topk_ids[:, 0].tolist() == [1, 2, 0, 3, 1, 0, 1, 1]


def test_moe_routing_metadata_single_rank_fast_path():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    topk_weights = torch.tensor([[0.7, 0.2], [0.5, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)

    routing = all_gather_sharded_cp_moe_routing_metadata(
        topk_weights,
        topk_ids,
        token_range,
    )

    assert torch.equal(routing.topk_weights, topk_weights)
    assert torch.equal(routing.topk_ids, topk_ids)


def test_ep_dispatch_uses_same_experts_as_baseline_after_routing_gather():
    token_range = get_sharded_cp_token_range(num_tokens=5, rank=0, world_size=2)
    router_logits = torch.tensor(
        [
            [0.1, 0.5, 0.4],
            [0.7, 0.2, 0.1],
            [0.0, 0.6, 0.4],
            [0.2, 0.1, 0.9],
            [0.3, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )
    hidden = torch.arange(5 * 2, dtype=torch.float32).view(5, 2)
    baseline_weights, baseline_ids = torch.topk(router_logits, k=1, dim=-1)
    weight_chunks = make_reduce_scatter_token_chunks(
        baseline_weights,
        token_range,
        pad_value=-1.0,
    )
    id_chunks = make_reduce_scatter_token_chunks(
        baseline_ids.to(torch.int32),
        token_range,
        pad_value=-1,
    )
    routing = assemble_sharded_cp_moe_routing_chunks(
        weight_chunks,
        id_chunks,
        token_range,
    )
    expert_bias = torch.tensor(
        [[10.0, 10.0], [100.0, 100.0], [1000.0, 1000.0]]
    )

    baseline_out = hidden + expert_bias[baseline_ids.squeeze(-1)]
    gathered_out = hidden + expert_bias[routing.topk_ids.squeeze(-1)]

    assert torch.equal(routing.topk_ids, baseline_ids.to(torch.int32))
    assert torch.equal(gathered_out, baseline_out)


def test_moe_routing_metadata_rejects_mismatched_shape():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)

    with pytest.raises(ValueError, match="same shape"):
        all_gather_sharded_cp_moe_routing_metadata(
            torch.zeros(2, 2),
            torch.zeros(2, 1, dtype=torch.int32),
            token_range,
        )


def _dense_mlp_reference(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = hidden @ gate_up_weight.T
    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down_weight.T


def test_dense_mlp_on_cp_hidden_matches_global_path_bitwise():
    world_size = 3
    hidden = torch.arange(10 * 4, dtype=torch.float64).view(10, 4) / 8
    gate_up_weight = torch.arange(12 * 4, dtype=torch.float64).view(12, 4) / 32
    down_weight = torch.arange(4 * 6, dtype=torch.float64).view(4, 6) / 16
    global_out = _dense_mlp_reference(hidden, gate_up_weight, down_weight)
    ranges = [
        get_sharded_cp_token_range(hidden.shape[0], rank, world_size)
        for rank in range(world_size)
    ]
    local_outputs = [
        _dense_mlp_reference(
            hidden[token_range.start : token_range.end],
            gate_up_weight,
            down_weight,
        )
        for token_range in ranges
    ]
    chunks = [
        make_reduce_scatter_token_chunks(global_out, ranges[rank])[rank]
        for rank in range(world_size)
    ]
    gathered = assemble_token_all_gather_chunks(chunks, ranges[0])

    assert torch.equal(torch.cat(local_outputs, dim=0), global_out)
    assert torch.equal(gathered, global_out)


def test_moe_output_slice_and_residual_pair_global_tokens():
    token_range = get_sharded_cp_token_range(num_tokens=7, rank=2, world_size=3)
    attention_output = torch.arange(7 * 4, dtype=torch.float32).view(7, 4)
    moe_output = torch.arange(7 * 4, dtype=torch.float32).view(7, 4) + 100
    local_attention = attention_output[token_range.start : token_range.end]

    local_moe = slice_sharded_cp_moe_output(moe_output, token_range)
    local_next = combine_sharded_cp_moe_residual(
        local_attention,
        local_moe,
        token_range,
    )

    assert local_moe.shape == (1, 4)
    assert torch.equal(local_moe, moe_output[6:7])
    assert torch.equal(local_next, attention_output[6:7] + moe_output[6:7])


def test_moe_output_reduce_scatter_single_rank_fast_path():
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    expert_output = torch.arange(6, dtype=torch.float32).view(2, 3)

    local = reduce_scatter_sharded_cp_moe_output(expert_output, token_range)

    assert torch.equal(local, expert_output)


class _RecordingGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.out_dtype = None

    def forward(self, hidden_states):
        self.input = hidden_states
        logits = torch.cat((hidden_states + 10, hidden_states + 20), dim=-1)
        return logits, None


class _RecordingExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_internal_router = False
        self.calls = []

    def forward(self, hidden_states, router_logits):
        self.calls.append((hidden_states, router_logits))
        return None, hidden_states + router_logits[:, : hidden_states.shape[1]]


def test_deepseek_moe_sharded_cp_gathers_then_scatters(monkeypatch):
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    hidden = torch.arange(6, dtype=torch.float32).view(2, 3)
    gate = _RecordingGate()
    experts = _RecordingExperts()
    moe = object.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.experts = experts
    moe.gate = gate
    moe.shared_experts = None
    moe.is_rocm_aiter_moe_enabled = False
    moe.routed_scaling_factor = 1.0
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(device_group=None),
    )

    out = moe._forward_sharded_cp(hidden, token_range)

    assert torch.equal(gate.input, hidden)
    assert len(experts.calls) == 1
    global_hidden, global_router = experts.calls[0]
    assert torch.equal(global_hidden, hidden)
    assert torch.equal(global_router, torch.cat((hidden + 10, hidden + 20), dim=-1))
    assert torch.equal(out, hidden + hidden + 10)


def test_deepseek_moe_sharded_cp_empty_rank_skips_local_gate(monkeypatch):
    token_range = get_request_aligned_sharded_cp_token_range(
        torch.tensor([0, 4], dtype=torch.int32),
        rank=1,
        world_size=2,
    )
    hidden = torch.empty(0, 3, dtype=torch.float32)
    experts = _RecordingExperts()
    moe = object.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.experts = experts
    moe.gate = SimpleNamespace(out_dtype=torch.float32)
    moe.shared_experts = None
    moe.is_rocm_aiter_moe_enabled = False
    moe.routed_scaling_factor = 1.0
    moe.n_routed_experts = 6
    gathered_hidden = torch.arange(12, dtype=torch.float32).view(4, 3)
    gathered_router = torch.arange(24, dtype=torch.float32).view(4, 6) + 10
    gather_calls = []
    scatter_calls = []

    def _fake_gather(hidden_states, router_logits, cp_range, *, group):
        gather_calls.append((hidden_states, router_logits, cp_range, group))
        return SimpleNamespace(
            hidden_states=gathered_hidden,
            router_logits=gathered_router,
        )

    def _fake_scatter(final_hidden_states, cp_range, *, group):
        scatter_calls.append((final_hidden_states, cp_range, group))
        return final_hidden_states[cp_range.start : cp_range.end]

    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(device_group="cp-group"),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "all_gather_sharded_cp_moe_inputs",
        _fake_gather,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "reduce_scatter_sharded_cp_moe_output",
        _fake_scatter,
    )

    out = moe._forward_sharded_cp(hidden, token_range)

    assert out.shape == (0, 3)
    assert len(gather_calls) == 1
    local_hidden, local_router, cp_range, group = gather_calls[0]
    assert local_hidden.shape == (0, 3)
    assert local_router.shape == (0, 6)
    assert local_router.dtype == torch.float32
    assert cp_range is token_range
    assert group == "cp-group"
    assert len(experts.calls) == 1
    assert torch.equal(experts.calls[0][0], gathered_hidden)
    assert torch.equal(experts.calls[0][1], gathered_router)
    assert len(scatter_calls) == 1
    assert scatter_calls[0][1] is token_range


def test_deepseek_moe_forward_reads_token_range_from_context(monkeypatch):
    token_range = get_sharded_cp_token_range(num_tokens=2, rank=0, world_size=1)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        virtual_engine=0,
        additional_kwargs={"sharded_cp_token_range": token_range},
    )
    monkeypatch.setattr(forward_context_module, "_forward_context", context)
    hidden = torch.arange(6, dtype=torch.float32).view(2, 3)
    moe = object.__new__(DeepseekV2MoE)
    moe.use_sharded_cp_token_parallel = True
    monkeypatch.setattr(
        moe,
        "_forward_sharded_cp",
        lambda hidden_states, cp_range: hidden_states + cp_range.rank,
    )

    out = DeepseekV2MoE.forward(moe, hidden)

    assert torch.equal(out, hidden)


class _FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _FakeMLP(nn.Module):
    instances: list["_FakeMLP"] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        _FakeMLP.instances.append(self)


def _decoder_layer_config():
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        model_type="deepseek",
        num_attention_heads=2,
        n_routed_experts=None,
    )


def _decoder_vllm_config(enable_sharded_cp: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=_decoder_layer_config(),
            use_mla=False,
        ),
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_sharded_context_parallel=enable_sharded_cp
        ),
    )


@pytest.mark.parametrize(
    ("enable_sharded_cp", "expected_sequence_parallel"),
    [(True, True), (False, False)],
)
def test_dense_decoder_mlp_uses_cp_local_rows_without_tp_collective(
    monkeypatch,
    enable_sharded_cp,
    expected_sequence_parallel,
):
    _FakeMLP.instances = []
    monkeypatch.setattr(deepseek_v2, "DeepseekAttention", _FakeAttention)
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", _FakeMLP)
    monkeypatch.setattr(deepseek_v2, "RMSNorm", _FakeNorm)

    DeepseekV2DecoderLayer(
        _decoder_vllm_config(enable_sharded_cp),
        prefix="model.layers.0",
    )

    assert len(_FakeMLP.instances) == 1
    assert (
        _FakeMLP.instances[0].kwargs["is_sequence_parallel"]
        is expected_sequence_parallel
    )


def test_decoder_forward_materializes_sharded_cp_attention_weights(monkeypatch):
    events = []

    class _RecordingScope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class _RecordingShardLinear:
        def materialized(self, group):
            events.append(("materialized", group.rank_in_group, group.world_size))
            return _RecordingScope()

    class _RecordingAttention(nn.Module):
        sharded_cp_shard_linear = _RecordingShardLinear()

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, **kwargs):
            events.append("attention")
            return kwargs["hidden_states"] + 1

    class _IdentityMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            events.append("mlp")
            return hidden_states

    monkeypatch.setattr(deepseek_v2, "DeepseekAttention", _RecordingAttention)
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", _IdentityMLP)
    monkeypatch.setattr(deepseek_v2, "RMSNorm", _FakeNorm)
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2),
    )
    layer = DeepseekV2DecoderLayer(
        _decoder_vllm_config(enable_sharded_cp=True),
        prefix="model.layers.0",
    )

    hidden, residual = layer(
        positions=torch.arange(2),
        hidden_states=torch.zeros(2, 8),
        residual=None,
    )

    assert hidden.shape == (2, 8)
    assert residual.shape == (2, 8)
    assert events == [("materialized", 1, 2), "enter", "attention", "exit", "mlp"]


def test_decoder_forward_rejects_wrong_sharded_cp_prefetch(monkeypatch):
    class _WrongPrefetch:
        layer_id = 99

        def materialized(self):
            raise AssertionError("wrong prefetch must be rejected before wait")

    class _ShardLinear:
        layer_id = 0

        def materialized(self, group):
            raise AssertionError("provided prefetch should be validated first")

    class _Attention(nn.Module):
        sharded_cp_shard_linear = _ShardLinear()

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, **kwargs):
            return kwargs["hidden_states"]

    monkeypatch.setattr(deepseek_v2, "DeepseekAttention", _Attention)
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", _FakeMLP)
    monkeypatch.setattr(deepseek_v2, "RMSNorm", _FakeNorm)
    layer = DeepseekV2DecoderLayer(
        _decoder_vllm_config(enable_sharded_cp=True),
        prefix="model.layers.0",
    )

    with pytest.raises(RuntimeError, match="wrong layer"):
        layer(
            positions=torch.arange(2),
            hidden_states=torch.zeros(2, 8),
            residual=None,
            sharded_cp_prefetch=_WrongPrefetch(),
        )


def test_decoder_forward_empty_sharded_cp_rank_still_enters_layers(monkeypatch):
    events = []

    class _RecordingScope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class _RecordingShardLinear:
        def materialized(self, group):
            events.append(("materialized", group.rank_in_group, group.world_size))
            return _RecordingScope()

    class _RecordingAttention(nn.Module):
        sharded_cp_shard_linear = _RecordingShardLinear()

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, **kwargs):
            events.append(("attention", kwargs["hidden_states"].shape[0]))
            return kwargs["hidden_states"]

    class _RecordingMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            events.append(("mlp", hidden_states.shape[0]))
            return hidden_states

    monkeypatch.setattr(deepseek_v2, "DeepseekAttention", _RecordingAttention)
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", _RecordingMLP)
    monkeypatch.setattr(deepseek_v2, "RMSNorm", _FakeNorm)
    monkeypatch.setattr(
        deepseek_v2,
        "get_sharded_cp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2),
    )
    layer = DeepseekV2DecoderLayer(
        _decoder_vllm_config(enable_sharded_cp=True),
        prefix="model.layers.0",
    )

    hidden, residual = layer(
        positions=torch.empty(0, dtype=torch.int64),
        hidden_states=torch.empty(0, 8),
        residual=None,
    )

    assert hidden.shape == (0, 8)
    assert residual is not None
    assert residual.shape == (0, 8)
    assert events == [
        ("materialized", 1, 2),
        "enter",
        ("attention", 0),
        "exit",
        ("mlp", 0),
    ]
