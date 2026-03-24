import pytest
import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr,):
    if STAGE == 1: # causal
        lo, hi = 0, start_m*BLOCK_M
    elif STAGE == 2: # causal: on the band
        lo, hi = start_m*BLOCK_M, (start_m+1)*BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else: # causal = False
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo

    for start_n in tl.range(lo,hi,BLOCK_N):
        k = desc_k.load([offsetk_y,0]).T
        qk = tl.dot(q,k)
        if STAGE == 2:
            mask = offs_m[:,None] >= (start_n + offs_n[None,:])
            qk = qk * qk_scale + tl.where(mask,0,-1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk,1))
            qk -= m_ij[:,None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p,1)
        acc = acc*alpha[:,None]
        v = desc_v.load([offsetv_y,0])
        p = p.to(dtype)
        acc = tl.dot(p,v,acc)

        l_i = l_i*alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i



def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]

NUM_STAGES_OPTIONS = [2, 3, 4]
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)

def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf for conf in configs 
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX 
        and (conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
        and not (N_CTX == conf.kwargs.get("BLOCK_N", 0) and conf.num_stages > 2)
    ]

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)
    


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale,
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    y_dim = Z*H*N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    
    offset_y = off_z*H*N_CTX + off_h*N_CTX
    qo_offset_y = offset_y + start_m*BLOCK_M
    offs_m = start_m*BLOCK_M + tl.arange(0,BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype = tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M,HEAD_DIM], dtype = tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y,0])
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX)
    
    #epilogue
    m_i += tl.math.log2(l_i)
    acc = acc/l_i[:,None]
    desc_o.store([qo_offset_y, 0], acc.to(dtype))



def attn_forward(q, k, v, causal, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    desc_q = q
    desc_v = v
    desc_k = k
    desc_o = o

    def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2],META["BLOCK_M"]), q.shape[0]*q.shape[1],1)
    
    _attn_fwd[grid](
            sm_scale,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage)
    return o



@pytest.fixture(autouse=True)
def cleanup():
    yield
    torch.cuda.empty_cache()


# test correctness
@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, 4*1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("provider", ["triton-fp16"])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, provider, dtype=torch.float16):
    # skip if estimated memory exceeds threshold
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
    ref_peak_bytes = Z * H * N_CTX * N_CTX * 4 * 2
    if ref_peak_bytes > total_gpu_mem * 0.8:
        pytest.skip(f"Skipping: estimated peak memory {ref_peak_bytes/1e9:.1f} GB exceeds threshold on {total_gpu_mem/1e9:.1f} GB GPU")

    torch.manual_seed(0)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    sm_scale = 0.5
    
    # reference implementation
    ref_dtype = dtype
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    ref_out = torch.matmul(p, v).half()
    
    tri_out = attn_forward(q, k, v, causal, sm_scale).half()

    best_config = _attn_fwd.best_config
    print(f"BLOCK_M={best_config.kwargs['BLOCK_M']}, "
        f"BLOCK_N={best_config.kwargs['BLOCK_N']}, "
        f"num_stages={best_config.num_stages}, "
        f"num_warps={best_config.num_warps}, "
        f"causal: {causal}")
    
    atol = 1e-2
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)


if __name__ == "__main__":
    test_op(4, 48, 128, 128, True, "triton-fp16")