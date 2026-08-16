# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.sharded_cp_compact_kv import (
    all_gather_sharded_cp_compact_kv,
    all_gather_sharded_cp_compact_kv_async,
)
from vllm.distributed.sharded_cp_utils import (
    SHARDED_CP_GLOBAL_SLOT_MAPPING_KEY,
    SHARDED_CP_TOKEN_RANGE_KEY,
    SHARDED_CP_USE_GLOBAL_COMPACT_KV_KEY,
    ShardedCPTokenRange,
    all_to_all_cp_rows_to_tp_heads,
    get_sharded_cp_group,
    reduce_scatter_padded_token_rows,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig


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
            # Global-compact KV and out-of-band cache writes are not
            # expressible through the unified attention custom ops.
            self.mla_attn.use_direct_call = True

        self.prefix = prefix

    def _rope_scratch(self, like: torch.Tensor) -> torch.Tensor:
        """Scratch key tensor for a query-only rope call.

        DeepseekScalingRotaryEmbedding requires a query/key pair. Only one
        side is needed here, so pass a single-head scratch tensor: cos/sin
        broadcast over the head dimension, making this negligible.
        """
        return like.new_zeros((like.shape[0], 1, self.qk_rope_head_dim))

    def _sharded_cp_token_range(self) -> ShardedCPTokenRange | None:
        if not self.enable_sharded_context_parallel:
            return None
        if not self.is_sparse:
            return None
        if not is_forward_context_available():
            return None
        return get_forward_context().additional_kwargs.get(SHARDED_CP_TOKEN_RANGE_KEY)

    def _sharded_cp_use_global_compact_kv(self) -> bool:
        return bool(
            get_forward_context().additional_kwargs.get(
                SHARDED_CP_USE_GLOBAL_COMPACT_KV_KEY, False
            )
        )

    def _sharded_cp_global_slot_mapping(self, layer_name: str) -> torch.Tensor | None:
        global_slot_mapping = get_forward_context().additional_kwargs.get(
            SHARDED_CP_GLOBAL_SLOT_MAPPING_KEY
        )
        if isinstance(global_slot_mapping, dict):
            return global_slot_mapping.get(layer_name)
        return None

    def _write_sharded_cp_global_caches(
        self,
        kv_c_global: torch.Tensor,
        k_pe_global: torch.Tensor,
        indexer_k_global: torch.Tensor,
    ) -> None:
        """Persist all-gathered global compact rows with global slot IDs.

        Every CP rank writes the full batch so its paged MLA and indexer-K
        caches stay complete for later steps regardless of how requests are
        redistributed across ranks.
        """
        attn_slot_mapping = self._sharded_cp_global_slot_mapping(
            self.mla_attn.layer_name
        )
        if attn_slot_mapping is not None:
            self.mla_attn.update_kv_cache(
                kv_c_global,
                k_pe_global,
                attn_slot_mapping[: kv_c_global.shape[0]],
            )
        # Shared top-k layers (IndexCache) have no indexer of their own; they
        # reuse the previous layer's top-k buffer and skip the Indexer-K cache.
        if self.indexer is None:
            return
        indexer_k_cache = getattr(self.indexer, "k_cache", None)
        indexer_prefix = getattr(
            indexer_k_cache, "prefix", f"{self.prefix}.indexer.k_cache"
        )
        indexer_slot_mapping = self._sharded_cp_global_slot_mapping(indexer_prefix)
        if indexer_slot_mapping is not None:
            self.indexer.update_local_k_cache(
                indexer_k_global,
                layer_slot_mapping=indexer_slot_mapping[: indexer_k_global.shape[0]],
            )

    def _forward_empty_sharded_cp(
        self,
        hidden_states: torch.Tensor,
        token_range: ShardedCPTokenRange,
    ) -> torch.Tensor:
        """A rank owning zero rows still joins the compact-KV all-gather and
        persists the gathered global rows into its local caches."""
        indexer_head_dim = self.indexer.head_dim if self.indexer is not None else 0
        kv_c_normed = hidden_states.new_empty((0, self.kv_lora_rank))
        k_pe = hidden_states.new_empty((0, 1, self.qk_rope_head_dim))
        indexer_k = hidden_states.new_empty((0, indexer_head_dim))
        kv_c_global, k_pe_global, indexer_k_global = all_gather_sharded_cp_compact_kv(
            kv_c_normed,
            k_pe,
            indexer_k,
            token_range,
            group=get_sharded_cp_group().device_group,
        )
        self._write_sharded_cp_global_caches(
            kv_c_global, k_pe_global, indexer_k_global
        )
        return hidden_states.new_empty((0, self.num_heads * self.v_head_dim))

    def _forward_sharded_cp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        token_range: ShardedCPTokenRange,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.q_lora_rank is None:
            raise RuntimeError(
                "Sharded-CP sparse MLA requires q_lora_rank for the Indexer."
            )
        use_global_compact_kv = self._sharded_cp_use_global_compact_kv()
        if token_range.num_tokens == 0:
            attn_out = self._forward_empty_sharded_cp(hidden_states, token_range)
            return self._sharded_cp_o_proj(attn_out, token_range)

        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        assert self.q_b_proj is not None
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)
        if self.rotary_emb is not None:
            # Rotate K before Q so the compact-KV all-gather can start early
            # and overlap with the q_b_proj GEMM and Q rope. DeepSeek's rope
            # rotates a query/key pair and asserts both are present, so the
            # unused side is a [rows, 1, rope_dim] scratch tensor; cos/sin
            # broadcast over heads, so it stays cheap.
            k_pe, _ = self.rotary_emb(positions, k_pe, self._rope_scratch(k_pe))

        compact_kv_handle = None
        try:
            if self.indexer is not None:
                indexer_k, indexer_weights_raw = self.indexer.project_kw(
                    hidden_states, positions, self.indexer_rope_emb
                )
            else:
                # Shared top-k layer: gather MLA KV only; the previous
                # layer's top-k buffer is reused as-is.
                indexer_k = hidden_states.new_empty((hidden_states.shape[0], 0))
                indexer_weights_raw = None
            compact_kv_handle = all_gather_sharded_cp_compact_kv_async(
                kv_c_normed,
                k_pe,
                indexer_k,
                token_range,
                group=get_sharded_cp_group().device_group,
            )

            q = self.q_b_proj(q_c)[0]
            q = q.view(-1, self.num_heads, self.qk_head_dim)
            if self.rotary_emb is not None:
                q_pe, _ = self.rotary_emb(
                    positions,
                    q[..., self.qk_nope_head_dim :],
                    self._rope_scratch(q),
                )
                q[..., self.qk_nope_head_dim :] = q_pe

            kv_c_global, k_pe_global, indexer_k_global = compact_kv_handle.wait()
            self._write_sharded_cp_global_caches(
                kv_c_global, k_pe_global, indexer_k_global
            )

            if self.indexer is not None and not self.skip_topk:
                assert indexer_weights_raw is not None
                q_fp8, q_scale = self.indexer.project_q(
                    q_c, positions, self.indexer_rope_emb
                )
                indexer_weights = self.indexer.scale_weights(
                    indexer_weights_raw, q_scale
                )
                if use_global_compact_kv:
                    self.indexer.forward_global_compact(
                        hidden_states, q_fp8, indexer_k_global, indexer_weights
                    )
                else:
                    # Score against the paged cache, which already holds every
                    # global row from _write_sharded_cp_global_caches. Reusing
                    # project_kw/project_q here avoids recomputing the fused
                    # wk GEMM and re-inserting the same rows into the cache.
                    self.indexer.forward_local_paged(
                        hidden_states, q_fp8, indexer_weights
                    )
        except Exception:
            if compact_kv_handle is not None:
                compact_kv_handle.release()
            raise

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        if use_global_compact_kv:
            attn_kv_c, attn_k_pe = kv_c_global, k_pe_global
        else:
            attn_kv_c, attn_k_pe = kv_c_normed, k_pe
        attn_out = self.mla_attn(
            q,
            attn_kv_c,
            attn_k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            use_global_kv=use_global_compact_kv,
        )
        return self._sharded_cp_o_proj(attn_out, token_range)

    def _sharded_cp_o_proj(
        self,
        attn_out: torch.Tensor,
        token_range: ShardedCPTokenRange,
    ) -> torch.Tensor:
        """Apply o_proj with its normal TP-sharded weight.

        Attention produced all heads for this rank's token rows. One all-to-all
        turns that into all token rows with this rank's head slice, i.e. the
        layout o_proj's TP weight expects. o_proj runs without its internal
        all-reduce and the partial sums are reduce-scattered, which both
        completes the TP reduction and returns to CP-local rows.
        """
        cp_group = get_sharded_cp_group()
        if cp_group.world_size == 1:
            return self.o_proj(attn_out)[0]
        # The all-to-all sends head group i to CP rank i, while o_proj narrows
        # its weight by the TP rank, and the reduce-scatter performs o_proj's
        # TP reduction over the CP group. All three only agree while the CP
        # group *is* the TP group.
        if (
            cp_group.world_size != get_tensor_model_parallel_world_size()
            or cp_group.rank_in_group != get_tensor_model_parallel_rank()
        ):
            raise RuntimeError(
                "Sharded-CP o_proj requires the CP group to be the TP group: "
                f"cp=({cp_group.rank_in_group}/{cp_group.world_size}) vs "
                f"tp=({get_tensor_model_parallel_rank()}/"
                f"{get_tensor_model_parallel_world_size()})."
            )

        attn_out = attn_out.view(-1, self.num_heads, self.v_head_dim)
        tp_layout = all_to_all_cp_rows_to_tp_heads(
            attn_out,
            token_range,
            group=cp_group.device_group,
        )
        local_heads = self.num_heads // cp_group.world_size
        partial = self.o_proj(
            tp_layout.reshape(tp_layout.shape[0], local_heads * self.v_head_dim)
        )[0]
        return reduce_scatter_padded_token_rows(
            partial,
            token_range,
            group=cp_group.device_group,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sharded_cp_token_range = self._sharded_cp_token_range()
        if sharded_cp_token_range is not None:
            return self._forward_sharded_cp(
                positions,
                hidden_states,
                sharded_cp_token_range,
                llama_4_scaling,
            )
        if self.enable_sharded_context_parallel and self.is_sparse:
            # o_proj was built with reduce_results=False because Sharded-CP
            # folds the TP reduction into its reduce-scatter. Falling through
            # to the plain path would silently skip that reduction.
            raise RuntimeError(
                "Sharded-CP is enabled for this layer but no CP token range "
                "is active for this forward."
            )

        q_c = None
        kv_lora = None

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
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
