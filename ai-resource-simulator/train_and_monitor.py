"""
train_and_monitor.py
CPU-only LoRA fine-tuning of a small model with live resource monitoring.
 
Outputs (in ./results):
  resource_log.csv   - one row per second: CPU, RAM, process memory, training step, loss
  resource_plot.png  - charts of the above
A summary of peak/average usage is printed at the end.
"""
import csv
import os
import threading
import time
 
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
 
# ------------------------- Config (edit these) -------------------------
MODEL_NAME = "distilgpt2"      # ~82M params; small enough for CPU
MAX_LENGTH = 128               # tokens per training example
NUM_EXAMPLES = 800             # training examples to use
BATCH_SIZE = 4
MAX_STEPS = 60                 # increase for a longer simulation
SAMPLE_INTERVAL = 1.0          # seconds between resource samples
OUT_DIR = "results"
# -----------------------------------------------------------------------
 
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "resource_log.csv")
PLOT_PATH = os.path.join(OUT_DIR, "resource_plot.png")
 
# State shared between the training callback and the monitor thread
shared = {"phase": "setup", "step": 0, "loss": None}
 
 
class ResourceMonitor(threading.Thread):
    """Samples system and process resources in the background and logs to CSV."""
 
    def __init__(self, path, interval=1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.proc = psutil.Process(os.getpid())
 
    def run(self):
        start = time.time()
        self.proc.cpu_percent(interval=None)  # prime the counter
        psutil.cpu_percent(interval=None)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_s", "phase", "step", "loss",
                "sys_cpu_percent", "sys_ram_used_gb", "sys_ram_percent",
                "proc_cpu_percent", "proc_rss_mb", "proc_threads",
            ])
            while not self.stop_event.is_set():
                time.sleep(self.interval)
                vm = psutil.virtual_memory()
                writer.writerow([
                    round(time.time() - start, 2),
                    shared["phase"],
                    shared["step"],
                    shared["loss"],
                    psutil.cpu_percent(interval=None),
                    round(vm.used / 1024**3, 3),
                    vm.percent,
                    self.proc.cpu_percent(interval=None),  # can exceed 100 on multi-core
                    round(self.proc.memory_info().rss / 1024**2, 1),
                    self.proc.num_threads(),
                ])
                f.flush()
 
    def stop(self):
        self.stop_event.set()
        self.join()
 
 
class StepLogger(TrainerCallback):
    """Pushes the current step and loss into the shared state."""
 
    def on_step_end(self, args, state, control, **kwargs):
        shared["step"] = state.global_step
 
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            shared["loss"] = logs["loss"]
 
 
def load_training_data(tokenizer):
    """Load a slice of WikiText-2; fall back to synthetic text if offline."""
    try:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        raw = raw.filter(lambda x: len(x["text"].strip()) > 50)
        raw = raw.select(range(min(NUM_EXAMPLES, len(raw))))
        print(f"Loaded {len(raw)} WikiText-2 examples.")
    except Exception as e:
        print(f"Could not download dataset ({e}). Using synthetic text instead.")
        sentence = "The quick brown fox jumps over the lazy dog while the system monitors resources. "
        raw = Dataset.from_dict({"text": [sentence * 8] * NUM_EXAMPLES})
 
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
 
    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)
 
 
def print_summary(train_seconds):
    df = pd.read_csv(CSV_PATH)
    train = df[df["phase"] == "training"]
    if train.empty:
        train = df
    print("\n" + "=" * 56)
    print(" RESOURCE SUMMARY (training phase)")
    print("=" * 56)
    print(f" Training time:        {train_seconds:.1f} s")
    print(f" System CPU:           avg {train['sys_cpu_percent'].mean():.1f}%   peak {train['sys_cpu_percent'].max():.1f}%")
    print(f" System RAM used:      avg {train['sys_ram_used_gb'].mean():.2f} GB   peak {train['sys_ram_used_gb'].max():.2f} GB")
    print(f" Process memory (RSS): avg {train['proc_rss_mb'].mean():.0f} MB   peak {train['proc_rss_mb'].max():.0f} MB")
    print(f" Process CPU:          avg {train['proc_cpu_percent'].mean():.0f}%   peak {train['proc_cpu_percent'].max():.0f}%  (100% = one core)")
    losses = train["loss"].dropna()
    if not losses.empty:
        print(f" Loss:                 first {losses.iloc[0]:.3f} -> last {losses.iloc[-1]:.3f}")
    print(f" Log: {CSV_PATH}")
    print("=" * 56)
 
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(df["elapsed_s"], df["sys_cpu_percent"], label="System CPU %")
    axes[0].set_ylabel("CPU %")
    axes[0].legend()
    axes[1].plot(df["elapsed_s"], df["proc_rss_mb"], color="tab:orange", label="Process RSS (MB)")
    axes[1].set_ylabel("MB")
    axes[1].legend()
    axes[2].plot(df["elapsed_s"], df["loss"], color="tab:green", marker=".", label="Loss")
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Elapsed (s)")
    axes[2].legend()
    # shade the training window
    t = df[df["phase"] == "training"]
    if not t.empty:
        for ax in axes:
            ax.axvspan(t["elapsed_s"].min(), t["elapsed_s"].max(), alpha=0.08, color="gray")
    fig.suptitle("Resource usage during LoRA fine-tuning (shaded = training)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f" Plot: {PLOT_PATH}")
 
 
def main():
    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | "
          f"CPU cores: {psutil.cpu_count(logical=True)} | RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
 
    monitor = ResourceMonitor(CSV_PATH, SAMPLE_INTERVAL)
    monitor.start()
 
    try:
        shared["phase"] = "loading"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
 
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],   # GPT-2 attention projection
            fan_in_fan_out=True,         # GPT-2 uses Conv1D layers
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
 
        dataset = load_training_data(tokenizer)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
 
        args = TrainingArguments(
            output_dir=os.path.join(OUT_DIR, "checkpoints"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[StepLogger()],
        )
 
        shared["phase"] = "training"
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
 
        shared["phase"] = "done"
        time.sleep(SAMPLE_INTERVAL * 1.5)  # capture a final sample
    finally:
        monitor.stop()
 
    print_summary(train_seconds)
 
 
if __name__ == "__main__":
    main()