##under work

Project File Guide — What Each File Does
This project profiles vLLM GPU inference using NVIDIA Nsight Systems, converts the profiler output into SQLite, and generates a clean, human-readable JSON summary that matches Nsight’s own reports.

##scripts/
scripts/download_model.py

This script downloads the TinyLlama-1.1B-Chat model from Hugging Face and stores it locally.

It:

Downloads model weights and tokenizer files

Saves them under models/tinyllama/

Verifies that required files like config.json and tokenizer.json exist

This ensures the model is available locally before running vLLM or profiling.

scripts/env_profile.sh

This script sets up a stable profiling environment.

It:

Activates the Python virtual environment

Sets VLLM_WORKER_MULTIPROC_METHOD=spawn so Nsight can trace GPU work in vLLM worker processes

Defines the path to the Nsight Systems (nsys) binary

This avoids having to export environment variables manually each time.

scripts/run_vllm.py

This script runs a controlled vLLM inference workload with NVTX markers and explicit CUDA capture control.

It:

Loads the TinyLlama model using vLLM

Runs a small warmup inference (not captured)

Starts CUDA profiling using cudaProfilerStart()

Runs the real inference workload

Stops profiling using cudaProfilerStop()

Adds NVTX ranges such as phase:inference

This script defines exactly what gets profiled and ensures the captured trace is clean and focused.

scripts/profile_vllm_nsys.sh

This script launches Nsight Systems profiling around the vLLM workload.

It:

Runs nsys profile with CUDA, OS runtime, and NVTX tracing enabled

Uses --capture-range=cudaProfilerApi so only the intended inference region is captured

Produces:

vllm_capture.nsys-rep

vllm_capture.sqlite

This is the main entry point for collecting profiling data.

scripts/export_sqlite.sh

This script converts the Nsight Systems report into SQLite format.

It:

Uses nsys export --type sqlite

Overwrites any existing SQLite file

SQLite is used because it is easier to parse programmatically than .nsys-rep.

##parse/
parse/list_sqlite_tables.py

This script lists all tables present in the Nsight-generated SQLite file.

It is used to:

Explore the SQLite schema

Identify tables such as OSRT_API, CUPTI_ACTIVITY_KIND_KERNEL, and NVTX_EVENTS

This helps guide parser development.

parse/print_nvtx_schema.py

This script prints the schema of the NVTX_EVENTS table.

It is used to:

Inspect NVTX column names and data types

Correctly extract NVTX phase names and timing information

parse/parse_sqlite_baseline.py

This is the core parsing script of the project.

It:

Computes the total capture window duration

Extracts NVTX phases (e.g., phase:inference)

Summarizes:

GPU kernel execution time

GPU memory copies (H2D, D2H, D2D)

CPU waiting and synchronization calls

Produces a clean, structured summary.json

Adds ui_checks that directly match Nsight CLI reports:

NVTX range summary

Top GPU kernel summary

GPU memcpy totals

This script converts raw profiler data into clear performance insight.

parse/parse_sqlite_osrt.py (optional / legacy)

This script focuses only on CPU OS runtime calls.

It:

Extracts calls like epoll_wait, poll, and sem_wait

Calculates how much CPU time is spent waiting during GPU execution

It is kept for CPU-focused analysis and debugging.

##artifacts/
artifacts/vllm_capture.nsys-rep

This is the raw Nsight Systems profiling report.

It is used for:

Viewing timelines in the Nsight Systems UI

Manual inspection and validation

artifacts/vllm_capture.sqlite

This is the SQLite export of the Nsight report.

It is the primary input for all parsing and JSON generation.

artifacts/summary.json

This is the final output of the project.

It contains:

Inference phase timing

GPU kernel breakdown

GPU memcpy breakdown

CPU wait behavior

Cross-checks against Nsight CLI summaries

##nvtx_utils.py

This file contains a reusable NVTX helper.

It:

Provides a safe nvtx_range() context manager

Works even if the nvtx package is not installed

This allows NVTX markers to be added cleanly across scripts.


