# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # When has_weight=False, pass weight=None so implementations that
        # support a weightless path can skip the per-channel multiply.
        # Implementations that require weight (e.g. oink) fall back via IR
        # op priority when weight=None is unsupported.
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if 0 in x.shape[:-1]:
            if residual is None:
                return x
            return x, residual

        return self.forward_static(
            x,
            self.variance_epsilon,
            self.hidden_size,
            x.dtype,
            self.weight.data if self.has_weight else None,
            residual,
            self.variance_size_override,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
                residual=residual,
            )

        if 0 in x.shape[:-1]:
            if residual is None:
                return x
            return x, residual

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        # Optional Oink SM100 fast path (no residual). This path is
        # torch.compile-friendly via torch.ops.oink.rmsnorm and preserves
        # 2D layouts (including padded rows) when using the Oink
        # pointer-based kernel.
        if (
            residual is None
            and getattr(self, "_use_oink_rmsnorm", False)
            and x.is_cuda
            and x.dim() >= 2
            and self.has_weight
            and not vllm_is_batch_invariant()
            and self.weight.data.dtype == x.dtype
            and self.weight.data.is_contiguous()
        ):
            orig_shape = x.shape
            hidden_size = orig_shape[-1]
            if _can_view_as_2d(x):
                x_2d = x.view(-1, hidden_size)
                if _is_oink_stride_compatible_2d(x_2d):
                    y_2d = _oink_ops.rmsnorm(
                        x_2d,
                        self.weight.data,
                        self.variance_epsilon,
                    )
                    return y_2d.view(orig_shape)

        # Optional Oink SM100 fast path (fused residual-add + RMSNorm, in-place).
        # This mirrors vLLM's fused_add_rms_norm semantics by mutating both
        # `x` (normalized output) and `residual` (residual-out buffer).
        if (
            residual is not None
            and getattr(self, "_use_oink_fused_add_rmsnorm", False)
            and x.is_cuda
            and residual.is_cuda
            and x.shape == residual.shape
            and x.dtype == residual.dtype
            and x.dim() >= 2
            and self.has_weight
            and not vllm_is_batch_invariant()
            and self.weight.data.dtype == x.dtype
            and self.weight.data.is_contiguous()
        ):
            orig_shape = x.shape
            hidden_size = orig_shape[-1]
            if _can_view_as_2d(x) and _can_view_as_2d(residual):
                x_2d = x.view(-1, hidden_size)
                res_2d = residual.view(-1, hidden_size)

                # The Oink in-place pointer path supports the common vLLM
                # layout where:
                # - `x` may be strided/padded row-major (stride(1) == 1), and
                # - `residual` is contiguous row-major ([M, N] with stride(0) == N).
                # If these conditions are not met, fall back to vLLM's built-in
                # fused kernel.
                if (
                    _is_oink_stride_compatible_2d(x_2d)
                    and _is_oink_stride_compatible_2d(res_2d)
                    and res_2d.is_contiguous()
                ):
                    _oink_ops.fused_add_rms_norm_(
                        x_2d,
                        res_2d,
                        self.weight.data,
                        self.variance_epsilon,
                    )
                    return x, residual

        add_residual = residual is not None
        if add_residual:
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if 0 in x.shape[:-1]:
            if residual is None:
                return x
            return x, residual

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            return self.rocm_norm_func_with_add(
                x, residual, self.weight.data, self.variance_epsilon
            )

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        weight = self.weight.float() + 1.0
        if residual is None:
            return ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        z: torch.Tensor | None,
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ) -> torch.Tensor:
        """Pure-PyTorch RMS normalization with optional gating.

        This static method contains the full native logic so that both
        ``forward_native`` and ``MatcherRMSNormGated`` (used by the
        compilation pattern matcher) can share the same implementation.

        If *z* is not None and *norm_before_gate* is True:
            ``out = rms_norm(x) * act(z)``
        If *z* is not None and *norm_before_gate* is False:
            ``out = rms_norm(x * act(z))``
        """
        x = x.float()
        weight = weight.float()
        if z is not None:
            z = z.float()

        assert activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if activation == "sigmoid" else F.silu

        if z is not None and not norm_before_gate:
            x = x * act_fn(z)

        if group_size is None:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + epsilon)
            out = x_normed * weight
        else:
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + epsilon)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        if z is not None and norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
