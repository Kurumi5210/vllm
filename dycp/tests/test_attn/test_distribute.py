import torch

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def _avg_distribute_tokens_to_ranks(
        world_size: int,
        num_tokens: int) -> list[int]:
        
        print(f"num_tokens: {num_tokens}, block_size: 64",flush=True)
        total_blocks_required = cdiv(num_tokens, 64)
        # total_blocks_required = 3
        remain_tokens = num_tokens % 64
        # remain_tokens = 2
        quota = total_blocks_required // world_size
        # quota = 3
        remainder = total_blocks_required % world_size
        # remainder = 0
        print(f"total_blocks_required: {total_blocks_required}, remain_tokens: {remain_tokens}, quota: {quota}, remainder: {remainder}",flush=True)
        results = []
        for idx in range(world_size):
            local_blocks = quota + (1 if idx < remainder else 0)
            if remainder - 1 == idx:
                local_tokens = (local_blocks - 1) * 64 + remain_tokens
            else:
                local_tokens = local_blocks * 64
            results.append(local_tokens)
        print(f"------ : {results}",flush=True)
        return results

def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each dcp rank.
    Only consider dcp now, we can extend the case of cp based on this.
    """
    num_requests = seq_lens.size(0)
    if dcp_rank is None:
        rank_offsets = (
            torch.arange(dcp_size, dtype=torch.int32, device=seq_lens.device)
            .unsqueeze(0)
            .repeat(num_requests, 1)
        )
    else:
        rank_offsets = torch.tensor(
            [[dcp_rank]], dtype=torch.int32, device=seq_lens.device
        )
    seq_lens_tiled = (
        seq_lens.to(torch.int32).unsqueeze(-1).repeat(1, rank_offsets.shape[1])
    )
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    print(f"dcp_local_seq_lens: {dcp_local_seq_lens.squeeze(1)}",flush=True)
    return dcp_local_seq_lens.squeeze(1)


def _avg_distribute_tokens_to_ranks_2(
    seq_len: int,
    world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> list[list[int]]:
    """Calculate local seq_lens for all DCP ranks given a list of sequence lengths.
    
    While using dcp, kv_cache size stored on each rank may be different.
    This function calculates the split decode seq_lens for all dcp ranks.
    
    Args:
        seq_len: sequence lengths of the request
        dcp_size: Number of DCP ranks
        cp_kv_cache_interleave_size: Interleave size for KV cache
        
    Returns:
        List of lists, where each inner list contains the local seq_len for 
        all requests on that rank.
        Format: [[rank0_req0, rank0_req1, ...], [rank1_req0, rank1_req1, ...], ...]
    """
    
    # Initialize result: list of lists for each rank
    result = []
    
    # Process each request
    # for req_idx, seq_len in enumerate(seq_lens):
    # Calculate base: the part that's evenly distributed
    base = (
        (seq_len // cp_kv_cache_interleave_size // world_size)
        * cp_kv_cache_interleave_size
    )
    
    # Calculate remainder: the part that needs to be distributed
    remainder = seq_len - base * world_size
    
    # Distribute remainder across ranks
    for rank in range(world_size):
        rank_offset = rank * cp_kv_cache_interleave_size
        # Calculate how much of the remainder this rank gets
        # Clip to [0, cp_kv_cache_interleave_size]
        rank_remainder = max(
            0,
            min(
                cp_kv_cache_interleave_size,
                remainder - rank_offset
            )
        )
        local_seq_len = base + rank_remainder
        result.append(local_seq_len)
    print(f"result: {result}",flush=True)
    return result


def get_dcp_local_seq_lens_all_ranks(
    seq_lens: list[int],
    dcp_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> list[list[int]]:
    """Calculate local seq_lens for all DCP ranks given a list of sequence lengths.
    
    While using dcp, kv_cache size stored on each rank may be different.
    This function calculates the split decode seq_lens for all dcp ranks.
    
    Args:
        seq_lens: List of sequence lengths for each request
        dcp_size: Number of DCP ranks
        cp_kv_cache_interleave_size: Interleave size for KV cache
        
    Returns:
        List of lists, where each inner list contains the local seq_lens for 
        all requests on that rank.
        Format: [[rank0_req0, rank0_req1, ...], [rank1_req0, rank1_req1, ...], ...]
    """
    num_requests = len(seq_lens)
    
    # Initialize result: list of lists for each rank
    result = [[] for _ in range(dcp_size)]
    
    # Process each request
    for req_idx, seq_len in enumerate(seq_lens):
        # Calculate base: the part that's evenly distributed
        base = (
            (seq_len // cp_kv_cache_interleave_size // dcp_size)
            * cp_kv_cache_interleave_size
        )
        
        # Calculate remainder: the part that needs to be distributed
        remainder = seq_len - base * dcp_size
        
        # Distribute remainder across ranks
        for rank in range(dcp_size):
            rank_offset = rank * cp_kv_cache_interleave_size
            # Calculate how much of the remainder this rank gets
            # Clip to [0, cp_kv_cache_interleave_size]
            rank_remainder = max(
                0,
                min(
                    cp_kv_cache_interleave_size,
                    remainder - rank_offset
                )
            )
            local_seq_len = base + rank_remainder
            result[rank].append(local_seq_len)
    print(f"result: {result}",flush=True)
    return result

if __name__ == "__main__":
    results = _avg_distribute_tokens_to_ranks(1, 100)
    results = _avg_distribute_tokens_to_ranks(2, 100)
    results = _avg_distribute_tokens_to_ranks(3, 100)
    results = get_dcp_local_seq_lens(torch.tensor([100]), 1, 0, 64)
    results = get_dcp_local_seq_lens_all_ranks([100], 1, 64)
    results = get_dcp_local_seq_lens_all_ranks([100], 2, 64)
    results = get_dcp_local_seq_lens_all_ranks([1500], 4, 64)
    _avg_distribute_tokens_to_ranks_2(100, 1, 64)
    _avg_distribute_tokens_to_ranks_2(100, 2, 64)
    _avg_distribute_tokens_to_ranks_2(1500, 4, 64)