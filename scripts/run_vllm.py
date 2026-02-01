from __future__ import annotations

import time
from pathlib import Path
from contextlib import contextmanager
import ctypes

from vllm import LLM, SamplingParams

#            NVTX (labels only; does NOT control capture) 
try:
    import nvtx
except Exception:
    nvtx = None


@contextmanager
def nvtx_range(name: str):
    if nvtx is None:
        yield
        return
    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()


#            CUDA Profiler API (controls Nsight capture) 
def _load_cudart() -> ctypes.CDLL | None:
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_CUDART = _load_cudart()


def cuda_profiler_start() -> None:
    if _CUDART is None:
        return
    try:
        _CUDART.cudaProfilerStart()
    except Exception:
        pass


def cuda_profiler_stop() -> None:
    if _CUDART is None:
        return
    try:
        _CUDART.cudaProfilerStop()
    except Exception:
        pass


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "tinyllama"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}. Run scripts/download_model.py first."
        )

    #           NOT CAPTURED 
    with nvtx_range("phase:model_init"):
        llm = LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            trust_remote_code=False,
            enable_prefix_caching=False,
        )

    warmup_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    run_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)

    prompts = [
        "Write 5 bullet points about why caching improves LLM inference.",
        "Explain GPU memory vs compute bottlenecks in 4 sentences.",
        "Give a short checklist for debugging slow inference (no code).",
        "Summarize what H2D and D2H copies mean in CUDA in 3 lines.",
        "Create a tiny JSON object with keys latency_ms, throughput_tps, gpu_util.",
    ]

    with nvtx_range("phase:warmup"):
        print("Warmup (NOT captured)...")
        _ = llm.generate(["Warmup: say hello in one sentence."], warmup_params)

    repeats = 20
    workload = prompts * repeats
    print(f"Running workload: {len(prompts)} prompts × {repeats} = {len(workload)} total prompts")

    #                 CAPTURED ONLY HERE 
    # Starting capture FIRST, then entering NVTX range so NVTX push/pop are inside capture window.
    cuda_profiler_start()
    try:
        with nvtx_range("phase:inference"):
            t0 = time.time()
            outputs = llm.generate(workload, run_params)
            t1 = time.time()
    finally:
        cuda_profiler_stop()

    print(f"Total wall time (inference only): {(t1 - t0):.3f}s")
    if outputs:
        print("Sample output[0]:")
        print(outputs[0].outputs[0].text[:300].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
