#!/usr/bin/env bash
set -euo pipefail
if [ -f /home/ubuntu/vllm.pid ]; then
  PID="$(cat /home/ubuntu/vllm.pid)"
  echo "Stopping vLLM PID $PID"
  kill "$PID" 2>/dev/null || true
  rm -f /home/ubuntu/vllm.pid
else
  echo "No PID file; pkill fallback"
  pkill -f "vllm.*serve" 2>/dev/null || true
fi
