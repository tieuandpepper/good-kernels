"""
Implements the branching-search agent (5 rounds × 8 branches) from the blog. :contentReference[oaicite:9]{index=9}
Seeds with human-verified kernels from the good-kernels repo, e.g. the WMMA
matmul implementation. :contentReference[oaicite:10]{index=10}
"""
import os, random
import openai
import torch
from evaluator import compile_kernel, bench_kernel, is_close
from tasks import TASKS

POP_SIZE, ROUNDS = 8, 5
openai.api_key = os.getenv("OPENAI_API_KEY")

SYS_PROMPT = """You are an expert CUDA kernel author and performance engineer."""

def ask_llm(prompt, n=1):
    resp = openai.chat.completions.create(
        model="o3",
        messages=[{"role": "system", "content": SYS_PROMPT},
                  {"role": "user", "content": prompt}],
        n=n,
        # temperature=0.7,
    )
    return [c.message.content for c in resp.choices]

def optimize(task_name, task_cfg):
    A = task_cfg["init"]()
    ref_fn = getattr(torch, "matmul") if "matmul" in task_name else None
    ref_out, ref_time = bench_kernel(ref_fn, None, *A) if ref_fn else (None, None)

    history, bank = [], []
    for rnd in range(ROUNDS):
        ideas = ask_llm(
            f"Past ideas:\n{history}\n"
            f"Propose NEW optimization ideas for CUDA kernel '{task_name}'.", POP_SIZE
        )
        kernels = ask_llm(
            "\n".join(
                f"Implement this idea as a fully-self-contained CUDA C++ kernel called `{task_name}_kernel`:\n{idea}"
                for idea in ideas
            ),
            POP_SIZE,
        )
        for k_src in kernels:
            try:
                mod = compile_kernel(k_src, f"{task_name}_kernel")
                out, t = bench_kernel(mod.__getattr__(f"{task_name}_kernel"), ref_out, *A)
                if is_close(out, ref_out):
                    bank.append((t, k_src))
            except Exception:
                continue
        bank.sort(key=lambda x: x[0])
        history.append(ideas[0])

    return bank[0][1] if bank else None
