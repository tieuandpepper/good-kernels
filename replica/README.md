# Fast-Kernels Replica

This repo re-implements the search-time-only method from the Stanford CRFM blog  
“**Surprisingly Fast AI-Generated Kernels We Didn’t Mean to Publish (Yet)**”  
(Ouyang *et al.*, 28 May 2025). :contentReference[oaicite:0]{index=0}

* **Seed kernels** come from the public `good-kernels` examples. :contentReference[oaicite:1]{index=1}  
* **Benchmark tasks** follow Level-1 of **KernelBench**. :contentReference[oaicite:2]{index=2}  
* Runs are expected on an **NVIDIA L40S** (40 GB HBM3, ~90–120 TFLOPS FP32). :contentReference[oaicite:3]{index=3}  
* Requires **PyTorch ≥ 2.2 nightly (CUDA 12.4 build)**. :contentReference[oaicite:4]{index=4}  
* Uses the **OpenAI o3** (0614+) or any compatible code-capable LLM. :contentReference[oaicite:5]{index=5}  

