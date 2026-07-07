# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_mqa_logits,
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_torch,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops

logger = init_logger(__name__)

_TORCH_FALLBACK_MAX_SCORE_BYTES = 64 * 1024 * 1024
_DEEP_GEMM_MAX_LOGITS_BYTES = 512 * 1024 * 1024


def _query_chunk_size_for_bytes(
    q: torch.Tensor,
    bytes_per_query_row: int,
    max_bytes: int,
) -> int:
    if q.shape[0] == 0:
        return 1
    if bytes_per_query_row <= 0:
        return q.shape[0]

    return max(1, min(q.shape[0], max_bytes // bytes_per_query_row))


def _mqa_logits_torch_query_chunk_size(
    q: torch.Tensor,
    seq_len_kv: int,
) -> int:
    if seq_len_kv <= 0:
        return q.shape[0]

    # fp8_mqa_logits_torch materializes score as float32 with shape
    # [num_heads, num_query_rows, seq_len_kv]. Bound that tensor so the
    # fallback remains usable for long contexts when DeepGEMM is unavailable.
    score_element_size = torch.tensor([], dtype=torch.float32).element_size()
    bytes_per_query_row = q.shape[1] * seq_len_kv * score_element_size
    return _query_chunk_size_for_bytes(
        q, bytes_per_query_row, _TORCH_FALLBACK_MAX_SCORE_BYTES
    )


def _mqa_logits_deep_gemm_query_chunk_size(
    q: torch.Tensor,
    seq_len_kv: int,
) -> int:
    if seq_len_kv <= 0:
        return q.shape[0]

    # DeepGEMM returns materialized [num_query_rows, seq_len_kv] float32
    # logits. Long profile/extend prefill chunks can have very large flattened
    # KV lengths, so bound the output tensor before top-k consumes it.
    logits_element_size = torch.tensor([], dtype=torch.float32).element_size()
    bytes_per_query_row = seq_len_kv * logits_element_size
    return _query_chunk_size_for_bytes(
        q, bytes_per_query_row, _DEEP_GEMM_MAX_LOGITS_BYTES
    )


def _topk_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    num_rows = logits.shape[0]
    if current_platform.is_xpu():
        ops.top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
    else:
        torch.ops._C.top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )


def _fill_prefill_topk_from_logits(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    *,
    add_cu_seqlen_ks: bool,
) -> None:
    _topk_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
        topk_tokens,
    )
    if add_cu_seqlen_ks:
        valid_topk = topk_indices >= 0
        topk_indices.copy_(
            torch.where(
                valid_topk,
                topk_indices + cu_seqlen_ks.unsqueeze(1),
                topk_indices,
            )
        )


def _fill_prefill_topk_from_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    *,
    add_cu_seqlen_ks: bool,
) -> None:
    kv_fp8, _ = kv
    query_chunk_size = _mqa_logits_torch_query_chunk_size(
        q_fp8,
        kv_fp8.shape[0],
    )
    for token_start in range(0, q_fp8.shape[0], query_chunk_size):
        token_end = min(token_start + query_chunk_size, q_fp8.shape[0])
        logits = fp8_mqa_logits_torch(
            q_fp8[token_start:token_end],
            kv,
            weights[token_start:token_end],
            cu_seqlen_ks[token_start:token_end],
            cu_seqlen_ke[token_start:token_end],
        )
        _fill_prefill_topk_from_logits(
            logits,
            cu_seqlen_ks[token_start:token_end],
            cu_seqlen_ke[token_start:token_end],
            topk_indices[token_start:token_end],
            topk_tokens,
            add_cu_seqlen_ks=add_cu_seqlen_ks,
        )


def _validate_prefill_topk_rows(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    num_rows = q_fp8.shape[0]
    if (
        weights.shape[0] == num_rows
        and cu_seqlen_ks.shape[0] == num_rows
        and cu_seqlen_ke.shape[0] == num_rows
        and topk_indices.shape[0] == num_rows
    ):
        return
    raise RuntimeError(
        "SparseAttnIndexer prefill metadata row mismatch: "
        f"q_rows={num_rows}, weights_rows={weights.shape[0]}, "
        f"cu_seqlen_ks_rows={cu_seqlen_ks.shape[0]}, "
        f"cu_seqlen_ke_rows={cu_seqlen_ke.shape[0]}, "
        f"topk_rows={topk_indices.shape[0]}."
    )


def _fill_prefill_topk_from_mqa_logits(
    q_fp8: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    *,
    add_cu_seqlen_ks: bool,
    use_deep_gemm: bool,
) -> None:
    _validate_prefill_topk_rows(
        q_fp8,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    )
    kv_fp8, _ = kv
    if use_deep_gemm:
        query_chunk_size = _mqa_logits_deep_gemm_query_chunk_size(
            q_fp8,
            kv_fp8.shape[0],
        )
        for token_start in range(0, q_fp8.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q_fp8.shape[0])
            logits = fp8_mqa_logits(
                q_fp8[token_start:token_end].contiguous(),
                kv,
                weights[token_start:token_end].contiguous(),
                cu_seqlen_ks[token_start:token_end].contiguous(),
                cu_seqlen_ke[token_start:token_end].contiguous(),
                clean_logits=False,
            )
            _fill_prefill_topk_from_logits(
                logits,
                cu_seqlen_ks[token_start:token_end],
                cu_seqlen_ke[token_start:token_end],
                topk_indices[token_start:token_end],
                topk_tokens,
                add_cu_seqlen_ks=add_cu_seqlen_ks,
            )
    else:
        _fill_prefill_topk_from_mqa_logits_torch(
            q_fp8,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_indices,
            topk_tokens,
            add_cu_seqlen_ks=add_cu_seqlen_ks,
        )


def _fill_topk_from_global_compact_k(
    *,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    attn_metadata: DeepseekV32IndexerMetadata,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    if attn_metadata.num_prefills == 0 or attn_metadata.prefill is None:
        return topk_indices_buffer
    if k.dim() != 2:
        raise RuntimeError(
            "Sharded-CP global compact Indexer-K must be a 2D tensor, got "
            f"{tuple(k.shape)}."
        )

    k_fp8, k_scale = per_token_group_quant_fp8(
        k.contiguous(),
        quant_block_size,
        column_major_scales=False,
        use_ue8m0=scale_fmt is not None,
    )

    for chunk in attn_metadata.prefill.chunks:
        if chunk.token_end <= chunk.token_start:
            continue
        topk_indices_buffer[chunk.token_start : chunk.token_end] = -1
        chunk_q = q_fp8[chunk.token_start : chunk.token_end]
        chunk_weights = weights[chunk.token_start : chunk.token_end]
        topk_indices = topk_indices_buffer[
            chunk.token_start : chunk.token_end, :topk_tokens
        ]
        _fill_prefill_topk_from_mqa_logits(
            chunk_q,
            (k_fp8, k_scale.view(torch.float32).flatten()),
            chunk_weights,
            chunk.cu_seqlen_ks,
            chunk.cu_seqlen_ke,
            topk_indices,
            topk_tokens,
            add_cu_seqlen_ks=True,
            use_deep_gemm=is_deep_gemm_supported(),
        )

    return topk_indices_buffer


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
        )
        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    if attn_metadata.k_is_global_compact:
        return _fill_topk_from_global_compact_k(
            q_fp8=q_fp8,
            k=k,
            weights=weights,
            attn_metadata=attn_metadata,
            quant_block_size=quant_block_size,
            scale_fmt=scale_fmt,
            topk_tokens=topk_tokens,
            topk_indices_buffer=topk_indices_buffer,
        )

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            _fill_prefill_topk_from_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32).flatten()),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                topk_tokens,
                add_cu_seqlen_ks=False,
                use_deep_gemm=is_deep_gemm_supported(),
            )

            # Compute lengths from row spans
            # lengths = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            # torch.ops._C.large_context_topk(
            #    logits,
            #    topk_indices,
            #    lengths,
            #    chunk.cu_seqlen_ks,  # row_starts
            # )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        if is_deep_gemm_supported():
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            logits = fp8_paged_mqa_logits_torch(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                max_model_len=max_model_len,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if decode_metadata.use_large_context_topk:
            if next_n == 1:
                lengths = decode_metadata.seq_lens
            else:
                # (bs,) -> (bs, 1) + (next_n,) -> (bs, next_n) -> (bs * next_n,)
                lengths = (
                    decode_metadata.seq_lens.unsqueeze(1)
                    - next_n
                    + 1
                    + decode_metadata.offsets
                ).flatten()

            torch.ops._C.large_context_topk(
                logits,
                topk_indices,
                lengths,
                None,
            )
        else:
            if current_platform.is_xpu():
                ops.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode_metadata.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode_metadata.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM is not supported or available. SparseAttnIndexer will use a "
                "less efficient PyTorch implementation. "
                "Please make sure you have the required hardware and software setup "
                "for DeepGEMM to achieve optimal performance."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_global_compact(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_native(hidden_states, q_fp8, k, weights)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
