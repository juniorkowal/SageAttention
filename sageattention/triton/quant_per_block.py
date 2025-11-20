"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl
from .utils import autotune_configs

MAX_BLOCK_M = max(c.kwargs.get('BLOCK_M', 128) for c in autotune_configs)
MAX_BLOCK_N = max(c.kwargs.get('BLOCK_N', 64) for c in autotune_configs)

@triton.autotune(
    configs=autotune_configs,
    key=["C", "L"],
)
@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale,
                                C: tl.constexpr, BLOCK_M: tl.constexpr,
                                 BLOCK_N: tl.constexpr
                                ):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=(offs_n[:, None] < L))
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=(offs_n[:, None] < L))
    tl.store(scale_ptrs, scale)


def per_block_int8(q, k, km=None, BLKQ=None, BLKK=None, sm_scale=None, tensor_layout="HND"):
    """
    Quantize q and k per-block into int8 with scales.

    BLKQ/BLKK are suggested defaults; if None we use MAX_BLOCK_M / MAX_BLOCK_N as conservative alloc size.
    The kernel launch will still autotune over configs (which may contain various BLOCK sizes).
    """
    if BLKQ is None:
        BLKQ = MAX_BLOCK_M
    if BLKK is None:
        BLKK = MAX_BLOCK_N

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + MAX_BLOCK_M - 1) // MAX_BLOCK_M), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + MAX_BLOCK_N - 1) // MAX_BLOCK_N), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    grid_q = lambda META: (triton.cdiv(qo_len, META['BLOCK_M']), h_qo, b)
    quant_per_block_int8_kernel[grid_q](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim,
    )

    grid_k = lambda META: (triton.cdiv(kv_len, META['BLOCK_M']), h_kv, b)
    quant_per_block_int8_kernel[grid_k](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
    )

    return q_int8, q_scale, k_int8, k_scale


