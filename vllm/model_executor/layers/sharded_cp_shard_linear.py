# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard Linear helpers for Sharded Context Parallel.

A single CP rank owns the persistent full logical projection parameters for a
layer. Other ranks materialize those parameters before the layer consumes them,
either synchronously or by waiting on a previously issued async prefetch.
"""

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from vllm.distributed.sharded_cp_utils import get_sharded_cp_group


def get_sharded_cp_shard_linear_owner(layer_id: int, world_size: int) -> int:
    """Return the CP rank that owns a layer's persistent Shard Linear weights."""
    if layer_id < 0:
        raise ValueError(f"layer_id must be >= 0, got {layer_id}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    return layer_id % world_size


@dataclass(frozen=True)
class ShardedCPShardLinearParam:
    """Metadata for one materialized Shard Linear parameter."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    owner_rank: int


class ShardedCPShardLinearPrefetch:
    """A materialized Shard Linear prefetch that can be consumed once."""

    def __init__(
        self,
        layer: "ShardedCPShardLinearLayer",
        group,
        handles: list[dist.Work | None],
        streams: list[torch.cuda.Stream],
    ) -> None:
        self._layer = layer
        self._group = group
        self._handles = handles
        self._streams = streams
        self._waited = False
        self._released = False

    @property
    def layer_id(self) -> int:
        return self._layer.layer_id

    @property
    def released(self) -> bool:
        return self._released

    def wait(self) -> None:
        if self._released:
            raise RuntimeError(
                "Cannot wait on a released Sharded-CP Shard Linear prefetch."
            )
        if self._waited:
            return
        for handle in self._handles:
            if handle is not None:
                handle.wait()
        for stream in self._streams:
            stream.synchronize()
        self._waited = True

    def release(self) -> None:
        if self._released:
            raise RuntimeError(
                "Sharded-CP Shard Linear prefetch was released more than once."
            )
        self.wait()
        self._layer.release_non_owner(self._group)
        self._released = True

    @contextmanager
    def materialized(self):
        self.wait()
        try:
            yield
        finally:
            self.release()


class ShardedCPShardLinearLayer:
    """Manage materialization for one decoder layer.

    The registered modules are expected to be full logical projection modules
    such as MLA ``q_b_proj``/``q_proj``, ``kv_b_proj``, and ``o_proj``.
    Parameters are kept persistent only on ``layer_id % cp_world_size``. Other
    ranks allocate the full tensors before the layer runs, receive the owner
    broadcast, and release the tensors afterwards.
    """

    def __init__(
        self,
        layer_id: int,
        modules: Iterable[tuple[str, nn.Module | None]],
    ) -> None:
        if layer_id < 0:
            raise ValueError(f"layer_id must be >= 0, got {layer_id}.")
        self.layer_id = layer_id
        self.modules = tuple((name, module) for name, module in modules if module)

    def _iter_params(self):
        for module_name, module in self.modules:
            assert module is not None
            for param_name, param in module.named_parameters(recurse=False):
                yield f"{module_name}.{param_name}", param

    @staticmethod
    def _record_param_metadata(param: nn.Parameter) -> None:
        if not hasattr(param, "_sharded_cp_full_shape"):
            if param.numel() == 0:
                raise RuntimeError(
                    "Cannot infer Sharded-CP Shard Linear parameter shape from "
                    "an already released parameter."
                )
            param._sharded_cp_full_shape = torch.Size(param.shape)
            param._sharded_cp_full_dtype = param.dtype
            param._sharded_cp_full_device = param.device

    @staticmethod
    def _param_shape(param: nn.Parameter) -> torch.Size:
        return torch.Size(param._sharded_cp_full_shape)

    @staticmethod
    def _param_dtype(param: nn.Parameter) -> torch.dtype:
        return param._sharded_cp_full_dtype

    @staticmethod
    def _param_device(param: nn.Parameter) -> torch.device:
        return param._sharded_cp_full_device

    def describe_parameters(self, group=None) -> list[ShardedCPShardLinearParam]:
        """Return parameter metadata without forcing materialization."""
        group = group if group is not None else get_sharded_cp_group()
        owner_rank = get_sharded_cp_shard_linear_owner(
            self.layer_id, group.world_size
        )
        params: list[ShardedCPShardLinearParam] = []
        for name, param in self._iter_params():
            self._record_param_metadata(param)
            params.append(
                ShardedCPShardLinearParam(
                    name=name,
                    shape=self._param_shape(param),
                    dtype=self._param_dtype(param),
                    device=self._param_device(param),
                    owner_rank=owner_rank,
                )
            )
        return params

    def release_non_owner(self, group=None) -> None:
        """Release non-owner materialized tensors for this layer."""
        group = group if group is not None else get_sharded_cp_group()
        if group.world_size == 1:
            return
        owner_rank = get_sharded_cp_shard_linear_owner(
            self.layer_id, group.world_size
        )
        for _, param in self._iter_params():
            self._record_param_metadata(param)
            if group.rank_in_group == owner_rank:
                continue
            if param.numel() == 0:
                continue
            param.data = torch.empty(
                (0,),
                dtype=self._param_dtype(param),
                device=self._param_device(param),
            )

    def _broadcast(self, tensor: torch.Tensor, owner_rank: int, group) -> None:
        if hasattr(group, "broadcast"):
            group.broadcast(tensor, src=owner_rank)
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized for Sharded-CP "
                "Shard Linear broadcast."
            )
        dist.broadcast(tensor, src=owner_rank, group=group)

    @staticmethod
    def _can_async_broadcast(group) -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and getattr(group, "device_group", None) is not None
            and hasattr(group, "ranks")
        )

    def _broadcast_async(
        self,
        tensor: torch.Tensor,
        owner_rank: int,
        group,
        stream: torch.cuda.Stream | None,
    ) -> dist.Work | None:
        if not tensor.is_cuda or not self._can_async_broadcast(group):
            self._broadcast(tensor, owner_rank, group)
            return None
        if not 0 <= owner_rank < len(group.ranks):
            raise RuntimeError(
                "Sharded-CP Shard Linear owner rank is out of bounds for "
                f"the process group: owner={owner_rank}, "
                f"group_size={len(group.ranks)}."
            )
        if stream is None:
            return dist.broadcast(
                tensor,
                src=group.ranks[owner_rank],
                group=group.device_group,
                async_op=True,
            )
        stream.wait_stream(torch.cuda.current_stream(tensor.device))
        tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            return dist.broadcast(
                tensor,
                src=group.ranks[owner_rank],
                group=group.device_group,
                async_op=True,
            )

    def _materialize_params(
        self,
        group,
        *,
        async_broadcast: bool,
    ) -> tuple[list[dist.Work | None], list[torch.cuda.Stream]]:
        if group.world_size == 1:
            return [], []

        owner_rank = get_sharded_cp_shard_linear_owner(
            self.layer_id, group.world_size
        )
        handles: list[dist.Work | None] = []
        stream_by_device: dict[torch.device, torch.cuda.Stream] = {}
        try:
            for name, param in self._iter_params():
                self._record_param_metadata(param)
                shape = self._param_shape(param)
                dtype = self._param_dtype(param)
                device = self._param_device(param)
                if group.rank_in_group == owner_rank:
                    if torch.Size(param.shape) != shape:
                        raise RuntimeError(
                            "Owner Sharded-CP Shard Linear parameter was "
                            "released or reshaped unexpectedly: "
                            f"{name} has shape {tuple(param.shape)}, "
                            f"expected {tuple(shape)}."
                        )
                elif torch.Size(param.shape) != shape:
                    param.data = torch.empty(shape, dtype=dtype, device=device)
                if async_broadcast:
                    stream = None
                    if param.data.is_cuda:
                        stream = stream_by_device.setdefault(
                            param.data.device,
                            torch.cuda.Stream(device=param.data.device),
                        )
                    handles.append(
                        self._broadcast_async(
                            param.data, owner_rank, group, stream
                        )
                    )
                else:
                    self._broadcast(param.data, owner_rank, group)
        except Exception:
            for handle in handles:
                if handle is not None:
                    handle.wait()
            for stream in stream_by_device.values():
                stream.synchronize()
            self.release_non_owner(group)
            raise
        return handles, list(stream_by_device.values())

    def materialize(self, group=None) -> None:
        """Synchronously broadcast full logical params from the owner rank."""
        group = group if group is not None else get_sharded_cp_group()
        self._materialize_params(group, async_broadcast=False)

    def prefetch(self, group=None) -> ShardedCPShardLinearPrefetch:
        """Start materializing params and return a handle to consume later."""
        group = group if group is not None else get_sharded_cp_group()
        handles, streams = self._materialize_params(group, async_broadcast=True)
        return ShardedCPShardLinearPrefetch(self, group, handles, streams)

    @contextmanager
    def materialized(self, group=None):
        """Materialize registered params for one layer execution scope."""
        group = group if group is not None else get_sharded_cp_group()
        self.materialize(group)
        try:
            yield
        finally:
            self.release_non_owner(group)
