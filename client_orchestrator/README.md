# Client Orchestrator (EC2)

This folder contains the client-side orchestration code that:
- SSH’es into the GPU EC2 to restart vLLM with different flags
- Runs LLMPerf token benchmark against the GPU server
- Collects per-trial outputs + best run summaries

## Key files
- `tools/sweep_6knobs.py` — sweep runner
- `configs/sweep_6knobs.json` — host/port + sweep config

## Expected environment (example)
- `OPENAI_API_BASE=http://<GPU_PUBLIC_IP>:8000/v1/`
- SSH key path for GPU instance is referenced by the sweep script
