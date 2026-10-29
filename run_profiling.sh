#!/bin/bash

mkdir -p metrics_output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CPU_LOG="metrics_output/cpu_samples_${TIMESTAMP}.txt"
VMSTAT_LOG="metrics_output/vmstat_${TIMESTAMP}.txt"

echo "=========================================================="
echo "LAUNCHING DISTILGPT-2 BENCHMARK + LIVE TELEMETRY"
echo "=========================================================="

# 1. Start vmstat in the background for global OS stats
vmstat 1 > "$VMSTAT_LOG" 2>&1 &
VMSTAT_PID=$!

# 2. Launch the PyTorch benchmark in background
python benchmark_distilgpt2.py &
MODEL_PID=$!
echo "[+] PyTorch Benchmark running (PID: $MODEL_PID)"

# 3. Sample exact CPU % of the model PID every second using ps
while kill -0 $MODEL_PID 2>/dev/null; do
    USAGE=$(ps -p $MODEL_PID -o %cpu= 2>/dev/null | tr -d ' ')
    if [ -n "$USAGE" ]; then
        echo "$USAGE" >> "$CPU_LOG"
    fi
    sleep 1
done

# Wait for process exit and clean up background logger
wait $MODEL_PID
kill $VMSTAT_PID 2>/dev/null
wait $VMSTAT_PID 2>/dev/null

echo "=========================================================="
echo "BENCHMARK COMPLETED"
echo "=========================================================="

# 4. Generate ASCII Graph using inline Python parser
python3 - << 'PYEOF'
import sys
import os
import glob

# Find latest sample log
log_files = sorted(glob.glob("metrics_output/cpu_samples_*.txt"))
if not log_files:
    print("No CPU logs found.")
    sys.exit(0)

with open(log_files[-1], "r") as f:
    raw_vals = [line.strip() for line in f if line.strip()]

if not raw_vals:
    print("No data collected in CPU log.")
    sys.exit(0)

data = [float(x) for x in raw_vals]
max_val = max(data) if data else 100.0
# Round max scale up to next 10%
upper_bound = max(100.0, ((int(max_val) // 10) + 1) * 10.0)

HEIGHT = 12
WIDTH = len(data)

print(f"\nCPU UTILIZATION OVER TIME (Total Duration: {WIDTH}s, Peak: {max(data):.1f}%)")
print(f"Target Process: DistilGPT-2 Inference\n")

# Build the grid
grid = [[" " for _ in range(WIDTH)] for _ in range(HEIGHT)]

for col_idx, val in enumerate(data):
    # Determine bar height proportional to upper_bound
    bar_h = int((val / upper_bound) * (HEIGHT - 1))
    bar_h = min(bar_h, HEIGHT - 1)
    for row_idx in range(bar_h + 1):
        grid[HEIGHT - 1 - row_idx][col_idx] = "█"

# Render with Y-axis markers
for r in range(HEIGHT):
    level = upper_bound * (1 - (r / (HEIGHT - 1)))
    row_str = "".join(grid[r])
    print(f"{level:5.1f}% | {row_str}")

# Render X-axis
print("       +" + "-" * (WIDTH + 2))
print(f"   sec: 0" + " " * max(0, WIDTH - 5) + f"{WIDTH}s\n")
PYEOF

echo "Log saved to: $CPU_LOG"
echo "=========================================================="
