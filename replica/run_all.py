"""
Entry-point that runs the optimizer over the five headline Level-1 operators
and reports speed-ups exactly as Table 1 of the blog. :contentReference[oaicite:12]{index=12}
"""
from tasks import TASKS
from search_agent import optimize
from pathlib import Path

out_dir = Path("generated_kernels")
out_dir.mkdir(exist_ok=True)

for name, cfg in TASKS.items():
    best_kernel = optimize(name, cfg)
    if best_kernel:
        (out_dir / f"{name}.cu").write_text(best_kernel)
        print(f"[✓] {name} saved.")
    else:
        print(f"[✗] {name} failed.")
