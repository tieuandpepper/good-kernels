"""
Compiles a CUDA kernel provided as a Python string and benchmarks it against
the PyTorch reference for speed and correctness exactly as described by
Ouyang et al. :contentReference[oaicite:8]{index=8}
"""
import time
import torch
from torch.utils.cpp_extension import load_inline

def compile_kernel(src: str, name: str):
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=src,
        functions=[name],
        verbose=False,
    )

def bench_kernel(fn, ref_out, *args, n_warmup=20, n_iter=100):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    return out, total / n_iter

def is_close(a, b, rtol=1e-2, atol=1e-2):
    return torch.allclose(a, b, rtol=rtol, atol=atol)
