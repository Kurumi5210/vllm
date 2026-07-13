# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.sharded_cp_attention import (
    all_gather_sharded_cp_compact_kv,
    all_gather_sharded_cp_compact_kv_async,
)
from vllm.distributed.sharded_cp_utils import get_sharded_cp_group


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    enable_sharded_context_parallel: bool = False


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.enable_sharded_context_parallel = (
            mla_modules.enable_sharded_context_parallel
        )

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
        )
        if self.enable_sharded_context_parallel and self.is_sparse:
            self.mla_attn.use_direct_call = True

        self.prefix = prefix

    def _sharded_cp_token_range(self):
        if not self.enable_sharded_context_parallel:
            return None
        if not self.is_sparse or self.indexer is None:
            return None
        if not is_forward_context_available():
            return None
        return get_forward_context().additional_kwargs.get("sharded_cp_token_range")

    def _use_global_compact_kv_for_sharded_cp(self) -> bool:
        if not is_forward_context_available():
            return False
        return bool(
            get_forward_context().additional_kwargs.get(
                "sharded_cp_use_global_compact_kv",
                False,
            )
        )

    def _sharded_cp_indexer_metadata(self):
        if not is_forward_context_available():
            return None
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        return attn_metadata.get(f"{self.prefix}.indexer.k_cache")

    def _sharded_cp_global_slot_mapping(
        self,
        layer_name: str,
    ) -> torch.Tensor | None:
        if not is_forward_context_available():
            return None
        forward_context = get_forward_context()
        global_slot_mapping = forward_context.additional_kwargs.get(
            "sharded_cp_global_slot_mapping"
        )
        if isinstance(global_slot_mapping, dict):
            return global_slot_mapping.get(layer_name)
        slot_mapping = forward_context.slot_mapping
        if isinstance(slot_mapping, dict):
            return slot_mapping.get(layer_name)
        return None

    def _update_local_kv_cache_for_global_compact(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        layer_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.mla_attn.update_kv_cache(kv_c_normed, k_pe, layer_slot_mapping)

    def _update_local_indexer_k_cache_for_global_compact(
        self,
        indexer_k: torch.Tensor,
    ) -> None:
        indexer_k_cache = getattr(self.indexer, "k_cache", None)
        indexer_k_cache_prefix = getattr(
            indexer_k_cache,
            "prefix",
            f"{self.prefix}.indexer.k_cache",
        )
        indexer_slot_mapping = self._sharded_cp_global_slot_mapping(
            indexer_k_cache_prefix
        )
        if indexer_slot_mapping is not None:
            indexer_slot_mapping = indexer_slot_mapping[: indexer_k.shape[0]]
        self.indexer.update_local_k_cache(
            indexer_k,
            layer_slot_mapping=indexer_slot_mapping,
        )

    def _forward_empty_sharded_cp(
        self,
        hidden_states: torch.Tensor,
        token_range,
        *,
        use_global_compact_kv: bool,
    ) -> torch.Tensor:
        assert self.indexer is not None
        if not use_global_compact_kv:
            return hidden_states.new_empty((0, self.hidden_size))
        kv_c_normed = hidden_states.new_empty((0, self.kv_lora_rank))
        k_pe = hidden_states.new_empty((0, 1, self.qk_rope_head_dim))
        indexer_k = hidden_states.new_empty((0, self.indexer.head_dim))
        cp_group = get_sharded_cp_group()
        all_gather_sharded_cp_compact_kv(
            kv_c_normed,
            k_pe,
            indexer_k,
            token_range,
            group=cp_group.device_group,
        )
        return hidden_states.new_empty((0, self.hidden_size))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        sharded_cp_token_range = self._sharded_cp_token_range()
        use_global_compact_kv = (
            sharded_cp_token_range is not None
            and self._use_global_compact_kv_for_sharded_cp()
        )
        if (
            sharded_cp_token_range is not None
            and sharded_cp_token_range.num_tokens == 0
        ):
            return self._forward_empty_sharded_cp(
                hidden_states,
                sharded_cp_token_range,
                use_global_compact_kv=use_global_compact_kv,
            )

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if sharded_cp_token_range is not None and self.rotary_emb is not None:
            dummy_q_pe = k_pe.new_zeros(
                (k_pe.shape[0], 1, self.qk_rope_head_dim)
            )
            _, k_pe = self.rotary_emb(positions, dummy_q_pe, k_pe)

        compact_kv_handle = None
        q_fp8 = None
        indexer_weights = None
        try:
            if use_global_compact_kv:
                if q_c is None:
                    raise RuntimeError(
                        "Sharded-CP sparse MLA requires q_lora_rank for the "
                        "Indexer."
                    )
                indexer_k = self.indexer.project_k(
                    hidden_states,
                    positions,
                    self.indexer_rope_emb,
                )
                cp_group = get_sharded_cp_group()
                compact_kv_handle = all_gather_sharded_cp_compact_kv_async(
                    kv_c_normed,
                    k_pe,
                    indexer_k,
                    sharded_cp_token_range,
                    group=cp_group.device_group,
                )

            if self.q_lora_rank is not None:
                assert q_c is not None
                q = self.q_b_proj(q_c)[0]
            else:
                q = self.q_proj(hidden_states)[0]

            q = q.view(-1, self.num_heads, self.qk_head_dim)

            if self.rotary_emb is not None:
                if sharded_cp_token_range is not None:
                    dummy_k_pe = q.new_zeros(
                        (q.shape[0], 1, self.qk_rope_head_dim)
                    )
                    q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
                        positions,
                        q[..., self.qk_nope_head_dim :],
                        dummy_k_pe,
                    )
                else:
                    q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                        positions, q[..., self.qk_nope_head_dim :], k_pe
                    )

            if use_global_compact_kv:
                assert compact_kv_handle is not None
                assert q_c is not None
                q_fp8, q_scale = self.indexer.project_q(
                    q_c,
                    positions,
                    self.indexer_rope_emb,
                )
                indexer_weights = self.indexer.project_weights(
                    hidden_states, q_scale
                )
                indexer_metadata = self._sharded_cp_indexer_metadata()
                if getattr(indexer_metadata, "num_decodes", 0) > 0:
                    self.indexer.forward_local_paged(
                        hidden_states,
                        q_fp8,
                        indexer_k,
                        indexer_weights,
                    )
                kv_c_normed, k_pe, indexer_k_global = compact_kv_handle.wait()
                attn_slot_mapping = self._sharded_cp_global_slot_mapping(
                    self.mla_attn.layer_name
                )
                if attn_slot_mapping is not None:
                    attn_slot_mapping = attn_slot_mapping[: kv_c_normed.shape[0]]
                self._update_local_kv_cache_for_global_compact(
                    kv_c_normed,
                    k_pe,
                    attn_slot_mapping,
                )
                self._update_local_indexer_k_cache_for_global_compact(
                    indexer_k_global
                )
                self.indexer.forward_global_compact(
                    hidden_states,
                    q_fp8,
                    indexer_k_global,
                    indexer_weights,
                )
        except Exception:
            if compact_kv_handle is not None:
                compact_kv_handle.release()
            raise

        if not use_global_compact_kv and self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            use_global_kv=use_global_compact_kv,
        )

        return self.o_proj(attn_out)[0]
