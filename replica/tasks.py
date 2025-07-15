"""
Level-1 task metadata taken directly from KernelBench. :contentReference[oaicite:7]{index=7}
Only the five headline operators are listed for brevity.
"""
from pathlib import Path
import torch
BENCH_ROOT = Path(__file__).parent / "kernelbench_level1"

TASKS = {
    "matmul_fp32": {
        "size": 4096,
        "init": lambda: (
            torch.randn(4096, 4096, device="cuda", dtype=torch.float32),
            torch.randn(4096, 4096, device="cuda", dtype=torch.float32),
        ),
    },
    "conv2d_fp32": {
        "init": lambda: (
            torch.randn(100, 3, 224, 224, device="cuda", dtype=torch.float32),
            torch.nn.Conv2d(3, 96, 11, stride=4, padding=2).cuda().weight,
        ),
    },
    "softmax_fp32": {
        "init": lambda: (torch.randn(4096, 65536, device="cuda", dtype=torch.float32),),
    },
    "layernorm_fp32": {
        "shape": (16, 64, 256, 256),
        "init": lambda: (
            torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float32),
            torch.nn.LayerNorm((64, 256, 256)).cuda(),
        ),
    },
    "conv2d_relu_pool": {
        "init": lambda: (
            torch.randn(100, 3, 224, 224, device="cuda", dtype=torch.float32),
            torch.nn.Conv2d(3, 96, 11, stride=4, padding=2).cuda().weight,
        ),
    },
}
