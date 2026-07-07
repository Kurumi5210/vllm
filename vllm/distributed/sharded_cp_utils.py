# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for Sharded Context Parallel token-row partitioning.

These helpers are intentionally layout-only. They do not depend on DeepSeek
model classes and can be unit-tested on CPU before the Sharded-CP model path is
implemented.
"""

from bisect import bisect_left
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group


@dataclass(frozen=True)
class ShardedCPTokenRange:
    """Contiguous token-row range owned by one Sharded-CP rank."""

    rank: int
    world_size: int
    start: int
    end: int
    padded_end: int
    total_tokens: int
    padded_num_rows: int | None = None
    rank_starts: tuple[int, ...] | None = None
    rank_ends: tuple[int, ...] | None = None
    local_request_starts: tuple[int, ...] | None = None
    local_request_ends: tuple[int, ...] | None = None
    local_request_global_starts: tuple[int, ...] | None = None
    local_request_indices: tuple[int, ...] | None = None

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    @property
    def padded_num_tokens(self) -> int:
        if self.padded_num_rows is not None:
            return self.padded_num_rows
        return self.padded_end - self.start

    @property
    def has_explicit_rank_ranges(self) -> bool:
        return self.rank_starts is not None and self.rank_ends is not None

    @property
    def has_local_request_fragments(self) -> bool:
        return (
            self.local_request_starts is not None
            and self.local_request_ends is not None
            and self.local_request_global_starts is not None
        )


def get_sharded_cp_group() -> GroupCoordinator:
    """Return the initial Sharded-CP group.

    The first implementation reuses the tensor-parallel process group while
    keeping Sharded-CP as a distinct layout concept.
    """
    return get_tp_group()


def get_sharded_cp_token_range(
    num_tokens: int, rank: int, world_size: int
) -> ShardedCPTokenRange:
    """Return the balanced contiguous token range for one CP rank.

    The partition uses a fixed padded chunk size so collective inputs have equal
    shapes on all ranks:

    ``chunk = ceil(num_tokens / world_size)``.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), got rank={rank}."
        )

    chunk = (num_tokens + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, num_tokens)
    # Ranks beyond num_tokens own an empty real range but still keep the padded
    # collective shape when chunk > 0.
    end = max(start, end)
    return ShardedCPTokenRange(
        rank=rank,
        world_size=world_size,
        start=start,
        end=end,
        padded_end=start + chunk,
        total_tokens=num_tokens,
    )


def get_request_aligned_sharded_cp_token_ranges(
    query_start_loc_cpu: torch.Tensor,
    world_size: int,
) -> tuple[ShardedCPTokenRange, ...]:
    """Return request-aligned token ranges for all Sharded-CP ranks.

    The first Sharded-CP implementation keeps each request on one rank. Nominal
    CP boundaries are rounded up to the next request boundary, so no request's
    query rows are split across ranks. The collective chunk size is the maximum
    real rows owned by any rank.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    if query_start_loc_cpu.dim() != 1:
        raise ValueError("query_start_loc_cpu must be a 1D tensor.")
    if query_start_loc_cpu.numel() == 0:
        raise ValueError("query_start_loc_cpu must contain at least one entry.")

    starts = [int(v) for v in query_start_loc_cpu.cpu().tolist()]
    if starts[0] != 0:
        raise ValueError("query_start_loc_cpu must start at 0.")
    if any(prev > cur for prev, cur in zip(starts, starts[1:])):
        raise ValueError("query_start_loc_cpu must be monotonically nondecreasing.")

    total_tokens = starts[-1]
    if total_tokens < 0:
        raise ValueError(f"total tokens must be >= 0, got {total_tokens}.")

    nominal_chunk = (total_tokens + world_size - 1) // world_size
    boundaries = [0]
    for rank in range(1, world_size):
        target = min(rank * nominal_chunk, total_tokens)
        boundary_idx = bisect_left(starts, target)
        if boundary_idx >= len(starts):
            boundary = total_tokens
        else:
            boundary = starts[boundary_idx]
        boundaries.append(boundary)
    boundaries.append(total_tokens)

    # Duplicate boundaries are allowed and represent empty ranks.
    if any(prev > cur for prev, cur in zip(boundaries, boundaries[1:])):
        raise ValueError(
            "request-aligned Sharded-CP boundaries must be monotonically "
            "nondecreasing."
        )

    rank_starts = tuple(boundaries[:-1])
    rank_ends = tuple(boundaries[1:])
    padded_rows = max(
        (end - start for start, end in zip(rank_starts, rank_ends)),
        default=0,
    )
    return tuple(
        ShardedCPTokenRange(
            rank=rank,
            world_size=world_size,
            start=start,
            end=end,
            padded_end=start + padded_rows,
            total_tokens=total_tokens,
            padded_num_rows=padded_rows,
            rank_starts=rank_starts,
            rank_ends=rank_ends,
        )
        for rank, (start, end) in enumerate(zip(rank_starts, rank_ends))
    )


def get_request_aligned_sharded_cp_token_range(
    query_start_loc_cpu: torch.Tensor,
    rank: int,
    world_size: int,
) -> ShardedCPTokenRange:
    """Return the request-aligned token range for one Sharded-CP rank."""
    if not 0 <= rank < world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), got rank={rank}."
        )
    return get_request_aligned_sharded_cp_token_ranges(
        query_start_loc_cpu, world_size
    )[rank]


def pad_for_token_all_gather(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad local token rows to the rank's fixed Sharded-CP chunk size."""
    expected_rows = token_range.num_tokens
    if x.shape[0] != expected_rows:
        raise ValueError(
            "local tensor row count must match token_range.num_tokens: "
            f"got {x.shape[0]}, expected {expected_rows}."
        )

    padded_rows = token_range.padded_num_tokens
    if padded_rows == expected_rows:
        return x
    if padded_rows < expected_rows:
        raise ValueError(
            "token_range padded rows must be >= real rows: "
            f"got padded={padded_rows}, real={expected_rows}."
        )

    pad_shape = (padded_rows - expected_rows, *x.shape[1:])
    padding = x.new_full(pad_shape, pad_value)
    return torch.cat((x, padding), dim=0)


def trim_token_all_gather(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
) -> torch.Tensor:
    """Trim gathered padded token rows back to the global token count."""
    return x[: token_range.total_tokens]


def assemble_token_all_gather_chunks(
    gathered: list[torch.Tensor],
    token_range: ShardedCPTokenRange,
) -> torch.Tensor:
    """Assemble padded all-gather chunks into global token-row order."""
    if len(gathered) != token_range.world_size:
        raise ValueError(
            "gathered chunk count must match token_range.world_size: "
            f"got {len(gathered)}, expected {token_range.world_size}."
        )

    if token_range.has_explicit_rank_ranges:
        assert token_range.rank_starts is not None
        assert token_range.rank_ends is not None
        real_chunks = [
            chunk[: end - start]
            for chunk, start, end in zip(
                gathered, token_range.rank_starts, token_range.rank_ends
            )
        ]
        if not real_chunks:
            return gathered[0].new_empty((0, *gathered[0].shape[1:]))
        return torch.cat(real_chunks, dim=0)

    return trim_token_all_gather(torch.cat(gathered, dim=0), token_range)


class TokenRowsAllGather:
    """Deferred result for a token-row all-gather."""

    def __init__(
        self,
        *,
        token_range: ShardedCPTokenRange,
        gathered: list[torch.Tensor] | None = None,
        work: dist.Work | None = None,
        stream: torch.cuda.Stream | None = None,
        result: torch.Tensor | None = None,
        input_tensor: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._token_range = token_range
        self._gathered = gathered
        self._work = work
        self._stream = stream
        self._result = result
        self._input_tensor = input_tensor
        self._device = device

    def wait(self) -> torch.Tensor:
        if self._result is not None:
            return self._result
        if self._work is not None:
            self._work.wait()
        if self._stream is not None:
            device = self._device
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.current_stream().wait_stream(self._stream)
            else:
                torch.cuda.current_stream().wait_stream(self._stream)
        assert self._gathered is not None
        self._result = assemble_token_all_gather_chunks(
            self._gathered, self._token_range
        )
        return self._result


def all_gather_token_rows_async(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
    pad_value: float = 0.0,
) -> TokenRowsAllGather:
    """Start an all-gather of local token rows and return a waitable handle."""
    if token_range.world_size == 1:
        if x.shape[0] != token_range.num_tokens:
            raise ValueError(
                "local tensor row count must match token_range.num_tokens: "
                f"got {x.shape[0]}, expected {token_range.num_tokens}."
            )
        return TokenRowsAllGather(token_range=token_range, result=x)

    padded = pad_for_token_all_gather(x, token_range, pad_value=pad_value)
    if padded.shape[0] == 0:
        return TokenRowsAllGather(token_range=token_range, result=padded)

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for all-gather.")

    world_size = dist.get_world_size(group)
    if world_size != token_range.world_size:
        raise ValueError(
            "distributed group size must match token_range.world_size: "
            f"got {world_size}, expected {token_range.world_size}."
        )
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    stream: torch.cuda.Stream | None = None
    if padded.is_cuda:
        stream = torch.cuda.Stream(device=padded.device)
        stream.wait_stream(torch.cuda.current_stream(padded.device))
        padded.record_stream(stream)
        for tensor in gathered:
            tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            work = dist.all_gather(
                gathered,
                padded,
                group=group,
                async_op=True,
            )
    else:
        work = dist.all_gather(gathered, padded, group=group, async_op=True)
    return TokenRowsAllGather(
        token_range=token_range,
        gathered=gathered,
        work=work,
        stream=stream,
        input_tensor=padded,
        device=padded.device if padded.is_cuda else None,
    )


def all_gather_token_rows(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """All-gather local token rows and trim padding.

    The first dimension of ``x`` must be the local real token count. All ranks
    are padded to the same chunk size before the collective.
    """
    return all_gather_token_rows_async(
        x,
        token_range,
        group=group,
        pad_value=pad_value,
    ).wait()



def _pad_token_rows(
    x: torch.Tensor,
    padded_rows: int,
    *,
    pad_value: float,
) -> torch.Tensor:
    if x.shape[0] > padded_rows:
        raise ValueError(
            "local tensor rows must not exceed padded rows: "
            f"got {x.shape[0]}, padded rows {padded_rows}."
        )
    if x.shape[0] == padded_rows:
        return x
    pad_shape = (padded_rows - x.shape[0], *x.shape[1:])
    return torch.cat((x, x.new_full(pad_shape, pad_value)), dim=0)


def make_reduce_scatter_token_chunks(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    pad_value: float = 0.0,
) -> list[torch.Tensor]:
    """Build equal-sized reduce-scatter chunks for Sharded-CP token rows."""
    if x.shape[0] != token_range.total_tokens:
        raise ValueError(
            "global tensor row count must match token_range.total_tokens: "
            f"got {x.shape[0]}, expected {token_range.total_tokens}."
        )

    padded_rows = token_range.padded_num_tokens
    if padded_rows == 0:
        return []

    if token_range.has_explicit_rank_ranges:
        assert token_range.rank_starts is not None
        assert token_range.rank_ends is not None
        return [
            _pad_token_rows(
                x[start:end],
                padded_rows,
                pad_value=pad_value,
            )
            for start, end in zip(token_range.rank_starts, token_range.rank_ends)
        ]

    padded_total_rows = padded_rows * token_range.world_size
    if x.shape[0] < padded_total_rows:
        pad_shape = (padded_total_rows - x.shape[0], *x.shape[1:])
        x = torch.cat((x, x.new_full(pad_shape, pad_value)), dim=0)
    input_chunks = list(x.split(padded_rows, dim=0))
    if len(input_chunks) != token_range.world_size:
        raise ValueError(
            "global tensor row count must form one padded chunk per rank: "
            f"got {len(input_chunks)} chunks, expected "
            f"{token_range.world_size}."
        )
    return input_chunks


def reduce_scatter_token_rows(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    group: dist.ProcessGroup | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Reduce-scatter global token rows into this rank's CP-local rows.

    ``x`` is a rank-local contribution to the global hidden states, such as the
    output of a vocab-parallel embedding before TP reduction. Each rank pads the
    token dimension to an equal chunk size, reduce-scatters along that dimension,
    and then trims local padding.
    """
    if x.shape[0] != token_range.total_tokens:
        raise ValueError(
            "global tensor row count must match token_range.total_tokens: "
            f"got {x.shape[0]}, expected {token_range.total_tokens}."
        )

    if token_range.world_size == 1:
        return x

    padded_rows = token_range.padded_num_tokens
    if padded_rows == 0:
        return x.new_empty((0, *x.shape[1:]))

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for reduce-scatter.")

    world_size = dist.get_world_size(group)
    if world_size != token_range.world_size:
        raise ValueError(
            "distributed group size must match token_range.world_size: "
            f"got {world_size}, expected {token_range.world_size}."
        )
    input_chunks = make_reduce_scatter_token_chunks(
        x, token_range, pad_value=pad_value
    )

    out = torch.empty((padded_rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter(out, input_chunks, group=group)
    return out[: token_range.num_tokens]


def shard_global_token_rows(
    x: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, ShardedCPTokenRange]:
    """Slice global token rows into the CP-local padded chunk for one rank."""
    token_range = get_sharded_cp_token_range(x.shape[0], rank, world_size)
    return (
        slice_for_token_reduce_scatter(x, token_range, pad_value=pad_value),
        token_range,
    )


def slice_for_token_reduce_scatter(
    x: torch.Tensor,
    token_range: ShardedCPTokenRange,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Return the padded local chunk for a future token-row reduce-scatter.

    This pure slicing helper lets tests validate the EmbeddingTP -> CP hidden
    layout before the actual distributed reduce-scatter path is wired in.
    """
    if x.shape[0] != token_range.total_tokens:
        raise ValueError(
            "global tensor row count must match token_range.total_tokens: "
            f"got {x.shape[0]}, expected {token_range.total_tokens}."
        )

    real = x[token_range.start : token_range.end]
    padded_rows = token_range.padded_num_tokens
    if real.shape[0] == padded_rows:
        return real

    pad_shape = (padded_rows - real.shape[0], *x.shape[1:])
    padding = x.new_full(pad_shape, pad_value)
    return torch.cat((real, padding), dim=0)
