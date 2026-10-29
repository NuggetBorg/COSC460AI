import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilbert/distilgpt2"
BATCH_SIZE = 16          # Upped from Jack's initial batch size of 4
WARMUP_STEPS = 5
EVAL_STEPS = 25
MAX_NEW_TOKENS = 32

def get_process_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(device_type="cpu"):
    print("=" * 60)
    print(f"BENCHMARKING ON: {device_type.upper()}")
    print("=" * 60)

    if device_type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable in this environment. Skipping GPU run.")
        print("To enable GPU, configure PCIe/vGPU passthrough in your VM hypervisor.\n")
        return

    device = torch.device(device_type)
    if device_type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer & Model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 2. Prepare Sample Wikipedia Q&A Prompts (Batched)
    base_prompts = [
        "Question: What is an operating system? Answer:",
        "Question: How does virtual memory work? Answer:",
        "Question: What causes a deadlock in concurrent programming? Answer:"
    ]
    # Multiply to reach target batch size
    prompts = (base_prompts * ((BATCH_SIZE // len(base_prompts)) + 1))[:BATCH_SIZE]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # 3. Warm-up
    print(f"Warming up ({WARMUP_STEPS} steps)...")
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
    
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 4. Timed Inference Runs
    print(f"Running benchmark ({EVAL_STEPS} steps, Batch Size = {BATCH_SIZE})...")
    ram_start = get_process_ram_mb()

    if device_type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        with torch.no_grad():
            for _ in range(EVAL_STEPS):
                _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        
        end_event.record()
        torch.cuda.synchronize()
        total_time_sec = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(EVAL_STEPS):
                _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        total_time_sec = time.perf_counter() - start_time

    ram_end = get_process_ram_mb()

    # 5. Output Resource Metrics
    avg_step_latency = (total_time_sec / EVAL_STEPS) * 1000
    throughput = (EVAL_STEPS * BATCH_SIZE) / total_time_sec

    print("\n--- RESULTS ---")
    print(f"Average Batch Latency:    {avg_step_latency:.2f} ms")
    print(f"Latency Per Sample:       {(avg_step_latency / BATCH_SIZE):.2f} ms")
    print(f"Throughput:               {throughput:.2f} samples/sec")
    print(f"Host RAM Footprint:       {ram_end:.2f} MB (Delta: +{max(0.0, ram_end - ram_start):.2f} MB)")
    
    if device_type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU VRAM Allocated:  {peak_vram:.2f} MB")
    print()

if __name__ == "__main__":
    # Benchmark CPU execution
    run_benchmark(device_type="cpu")
    
    # Benchmark GPU execution (auto-skipped if passthrough is not present)
    run_benchmark(device_type="cuda")
