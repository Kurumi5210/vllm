# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.config.offload import OffloadConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skip_global_cleanup


class FakeModelConfig:
    def __init__(
        self,
        *,
        use_mla: bool = True,
        has_index_topk: bool = True,
        architectures: list[str] | None = None,
    ):
        self.use_mla = use_mla
        self.architectures = (
            architectures if architectures is not None else ["DeepseekV3ForCausalLM"]
        )
        self.hf_config = SimpleNamespace()
        if has_index_topk:
            self.hf_config.index_topk = 2048


def test_parallel_config_accepts_sharded_cp_minimum_topology():
    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_sharded_context_parallel=True,
    )

    assert config.enable_sharded_context_parallel is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tensor_parallel_size": 1}, "tensor_parallel_size > 1"),
        ({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}, "pipeline"),
        (
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2},
            "prefill_context_parallel_size",
        ),
        (
            {"tensor_parallel_size": 2, "decode_context_parallel_size": 2},
            "decode_context_parallel_size",
        ),
        ({"tensor_parallel_size": 2, "enable_dbo": True}, "DBO/ubatching"),
        ({"tensor_parallel_size": 2, "ubatch_size": 2}, "DBO/ubatching"),
        (
            {"tensor_parallel_size": 2, "data_parallel_size": 2},
            "data parallelism",
        ),
        (
            {"tensor_parallel_size": 2, "enable_expert_parallel": True},
            "expert parallelism",
        ),
    ],
)
def test_parallel_config_rejects_sharded_cp_incompatible_topology(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParallelConfig(enable_sharded_context_parallel=True, **kwargs)


def _make_sharded_cp_vllm_config_for_validation(
    *,
    model_config=None,
    speculative_config=None,
    cudagraph_mode=CUDAGraphMode.NONE,
    attention_backend=None,
    offload_config=None,
):
    config = object.__new__(VllmConfig)
    config.model_config = model_config
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        enable_sharded_context_parallel=True,
    )
    config.speculative_config = speculative_config
    config.compilation_config = CompilationConfig(cudagraph_mode=cudagraph_mode)
    config.attention_config = AttentionConfig(backend=attention_backend)
    config.cache_config = CacheConfig()
    config.offload_config = offload_config or OffloadConfig()
    config._validate_sharded_context_parallel_config()
    return config


def test_vllm_config_accepts_sharded_cp_dsa_mla_model():
    config = _make_sharded_cp_vllm_config_for_validation(model_config=FakeModelConfig())

    assert config.parallel_config.enable_sharded_context_parallel is True


def test_vllm_config_accepts_sharded_cp_explicit_sparse_mla_backend():
    config = _make_sharded_cp_vllm_config_for_validation(
        model_config=FakeModelConfig(),
        attention_backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    )

    assert config.attention_config.backend == AttentionBackendEnum.FLASHMLA_SPARSE


@pytest.mark.parametrize(
    ("model_config", "match"),
    [
        (None, "requires a model_config"),
        (SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)), "MLA model"),
        (FakeModelConfig(use_mla=False), "requires an MLA model"),
        (SimpleNamespace(use_mla=True), "requires a DSA sparse MLA model config"),
        (
            FakeModelConfig(has_index_topk=False),
            "requires a DSA sparse MLA model config with index_topk",
        ),
    ],
)
def test_vllm_config_rejects_sharded_cp_incompatible_model(model_config, match):
    with pytest.raises(ValueError, match=match):
        _make_sharded_cp_vllm_config_for_validation(model_config=model_config)


def test_vllm_config_rejects_sharded_cp_with_speculative_config():
    with pytest.raises(ValueError, match="speculative decoding"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            speculative_config=object(),
        )


def test_vllm_config_rejects_sharded_cp_with_weight_offloading():
    offload_config = OffloadConfig()
    offload_config.uva.cpu_offload_gb = 4
    with pytest.raises(ValueError, match="weight offloading"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            offload_config=offload_config,
        )


def test_vllm_config_rejects_sharded_cp_with_torch_compile():
    from vllm.config.compilation import CompilationMode

    compilation_config = CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)
    compilation_config.mode = CompilationMode.VLLM_COMPILE
    config = object.__new__(VllmConfig)
    config.model_config = FakeModelConfig()
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        enable_sharded_context_parallel=True,
    )
    config.speculative_config = None
    config.compilation_config = compilation_config
    config.attention_config = AttentionConfig(backend=None)
    config.cache_config = CacheConfig()
    config.offload_config = OffloadConfig()
    with pytest.raises(ValueError, match="compilation mode"):
        config._validate_sharded_context_parallel_config()


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_vllm_config_rejects_sharded_cp_with_cudagraphs(cudagraph_mode):
    with pytest.raises(ValueError, match="cudagraph_mode NONE"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            cudagraph_mode=cudagraph_mode,
        )


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.FLASHMLA,
        # Sparse backends without Sharded-CP metadata localization support
        # are rejected when explicitly requested.
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
        AttentionBackendEnum.XPU_MLA_SPARSE,
    ],
)
def test_vllm_config_rejects_sharded_cp_with_unsupported_backend(backend):
    with pytest.raises(ValueError, match="Supported backends"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            attention_backend=backend,
        )


@pytest.mark.parametrize(
    "arch",
    [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        # The registry name real DeepSeek-V3.2 checkpoints use.
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    ],
)
def test_vllm_config_accepts_sharded_cp_supported_architectures(arch):
    config = _make_sharded_cp_vllm_config_for_validation(
        model_config=FakeModelConfig(architectures=[arch]),
    )

    assert config.parallel_config.enable_sharded_context_parallel is True


def test_vllm_config_rejects_sharded_cp_unsupported_architecture():
    """DSA models whose forward lacks the CP token scatter are rejected."""
    with pytest.raises(ValueError, match="only implemented for"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(architectures=["Glm4MoeLiteForCausalLM"]),
        )
