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
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skip_global_cleanup


class FakeModelConfig:
    def __init__(
        self,
        *,
        use_mla: bool = True,
        has_index_topk: bool = True,
        is_moe: bool = True,
    ):
        self.use_mla = use_mla
        self.is_moe = is_moe
        self.hf_config = SimpleNamespace()
        if has_index_topk:
            self.hf_config.index_topk = 2048

    def verify_with_parallel_config(self, parallel_config):
        pass

    def verify_dual_chunk_attention_config(self, load_config):
        pass

    def is_nvfp4_quantized(self):
        return False


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
    cache_dtype="auto",
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
    config.cache_config = CacheConfig(cache_dtype=cache_dtype)
    config._validate_sharded_context_parallel_config()
    return config


def test_vllm_config_accepts_sharded_cp_dsa_mla_model():
    config = _make_sharded_cp_vllm_config_for_validation(
        model_config=FakeModelConfig()
    )

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


def test_vllm_config_rejects_sharded_cp_with_full_cudagraphs():
    with pytest.raises(ValueError, match="full CUDA graphs"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            cudagraph_mode=CUDAGraphMode.FULL,
        )


def test_vllm_config_rejects_sharded_cp_with_dense_mla_backend():
    with pytest.raises(ValueError, match="sparse MLA attention backend"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            attention_backend=AttentionBackendEnum.FLASHMLA,
        )


def test_vllm_config_rejects_sharded_cp_flashinfer_sparse_with_fp8_kv_cache():
    with pytest.raises(ValueError, match="FlashInfer sparse MLA with FP8"):
        _make_sharded_cp_vllm_config_for_validation(
            model_config=FakeModelConfig(),
            attention_backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
            cache_dtype="fp8",
        )
