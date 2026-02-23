#!/usr/bin/env bash
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-/home/ubuntu/gpu-profiling/venv/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-/home/ubuntu/gpu-profiling/models/tinyllama}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-160}"   # warmup can be ~75s+

LOG_DIR="/home/ubuntu/gpu-profiling/artifacts/vllm_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%s)"
LOG="$LOG_DIR/vllm_${TS}.log"

# stop old
if [ -f /home/ubuntu/vllm.pid ]; then
  kill "$(cat /home/ubuntu/vllm.pid)" 2>/dev/null || true
  rm -f /home/ubuntu/vllm.pid
fi
pkill -f "vllm.*serve" 2>/dev/null || true
sleep 2

echo "CMD: $VLLM_BIN serve $MODEL_DIR --host $HOST --port $PORT --dtype $DTYPE --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN $*" | tee -a "$LOG"

nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  "$@" > "$LOG" 2>&1 &

echo $! > /home/ubuntu/vllm.pid

echo "Waiting for server to become ready (timeout ${READY_TIMEOUT_S}s)..."
for ((i=1; i<=READY_TIMEOUT_S; i++)); do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "OK vLLM up. PID=$(cat /home/ubuntu/vllm.pid)"
    echo "LOG=$LOG"
    exit 0
  fi
  sleep 1
done

echo "ERROR: vLLM did not become ready in ${READY_TIMEOUT_S}s"
echo "Check log: $LOG"
exit 1
