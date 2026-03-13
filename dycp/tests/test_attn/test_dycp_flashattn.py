import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from typing import Any
from vllm.attention.ops.common import cp_lse_ag_out_ar

class CPGroupWrapper:
    def __init__(self, group, ranks):
        self.group = group
        self.ranks = ranks
        self.world_size = len(ranks)
        
        global_rank = dist.get_rank()
        
        if global_rank in ranks:
            self.rank_in_group = ranks.index(global_rank)
        else:
            self.rank_in_group = dist.get_rank(group=group)

    def __getattr__(self, name):
        return getattr(self.group, name)

    def all_gather(self, tensor: torch.Tensor, dim: int = 0):
        gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, tensor, group=self.group)
        return torch.cat(gather_list, dim=dim)

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        dist.all_reduce(tensor, op=op, group=self.group)
        return tensor

    def barrier(self):
        dist.barrier(group=self.group)

_CONTEXT_PARALLEL_GROUP = None

def init_cp_group(world_size, cp_size):
    global _CONTEXT_PARALLEL_GROUP
    rank = dist.get_rank()
    
    num_cp_groups = world_size // cp_size
    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        raw_group = dist.new_group(ranks)
        
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = CPGroupWrapper(raw_group, ranks)

def get_cp_group():
    return _CONTEXT_PARALLEL_GROUP

try:
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    print("Please ensure vllm is installed in your CUDA environment.")
    raise

def setup(rank, world_size):
    """init distirbute env"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # CUDA 环境下使用 nccl 后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """clear distribute env"""
    dist.destroy_process_group()

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x_flat = x.to(torch.float64).flatten()
    y_flat = y.to(torch.float64).flatten()
    
    cos_sim = (x_flat @ y_flat) / (x_flat.norm() * y_flat.norm()).clamp_min(1e-12)
    cos_diff = 1 - cos_sim.item()
    
    print(f"[{name}] Cosine Difference: {cos_diff:.6e}")

def run_distributed_fa(rank, world_size, case):
    setup(rank, world_size)
    cp_size = 2 
    init_cp_group(world_size, cp_size)
    cp_group = get_cp_group()
    seq_lens = case["seq_lens"]
    num_heads = case["num_heads"]
    head_size = case["head_size"]
    dtype = case["dtype"]
    block_size = case["block_size"]
    num_blocks = case["num_blocks"]
    fa_version = case["fa_version"]

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5


    torch.manual_seed(42)
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype).cuda()
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32).cuda()
    

    global_key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype).cuda()
    global_value_cache = torch.randn_like(global_key_cache)


    blocks_per_rank = num_blocks // world_size
    start_block = rank * blocks_per_rank
    end_block = (rank + 1) * blocks_per_rank

    local_k_cache = global_key_cache[start_block:end_block].clone().contiguous()
    local_v_cache = global_value_cache[start_block:end_block].clone().contiguous()
    
    local_block_table = torch.arange(blocks_per_rank, dtype=torch.int32).view(num_seqs, -1).cuda()

    local_kv_lens = torch.full((num_seqs,), blocks_per_rank * block_size, dtype=torch.int32).cuda()


    out_local, lse_local = flash_attn_varlen_func(
        q=query,
        k=local_k_cache,
        v=local_v_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=local_kv_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=local_kv_lens.max().item(),
        softmax_scale=scale,
        causal=False,
        block_table=local_block_table,
        fa_version=fa_version,
        return_softmax_lse=True,
    )
    out_dist, out_lse = cp_lse_ag_out_ar(
            out_local,
            lse_local.transpose(0, 1),
            cp_group,
            return_lse=True,
        )


    if rank == 0:
        print("\n--- Verification on Rank 0 (CUDA) ---")
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32).cuda()
        max_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
        global_block_table = torch.arange(num_blocks, dtype=torch.int32).view(num_seqs, max_blocks_per_seq).cuda()

        out_ref, _ = flash_attn_varlen_func(
            q=query,
            k=global_key_cache,
            v=global_value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens_t,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(kv_lens),
            softmax_scale=scale,
            causal=False,
            block_table=global_block_table,
            fa_version=fa_version,
            return_softmax_lse=True,
        )

        cal_diff(out_ref, out_dist, "Distributed Attention Output")
        
        is_close = torch.allclose(out_ref, out_dist, rtol=1e-3, atol=1e-3)
        print(f"Are results close? -> {is_close}")

    cleanup()

if __name__ == "__main__":
    # 测试配置
    QUICK_CASE = dict(
        seq_lens=[(1, 4096)],
        num_heads=(8, 8),
        head_size=128,
        dtype=torch.bfloat16,
        block_size=128,
        num_blocks=32,
        fa_version=3,
    )

    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs.")
    else:
        mp.spawn(
            run_distributed_fa,
            args=(world_size, QUICK_CASE),
            nprocs=world_size,
            join=True
        )
