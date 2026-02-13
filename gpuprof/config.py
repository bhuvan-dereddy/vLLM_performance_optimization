from typing import Any, Dict


def get_prompts_path(cfg: Dict[str, Any]) -> str:
    dataset = cfg.get("dataset", {})
    prompts = dataset.get("path") or dataset.get("prompts_jsonl")
    if not prompts:
        raise RuntimeError("dataset.path or dataset.prompts_jsonl missing")
    return str(prompts)


def get_endpoint(cfg: Dict[str, Any]) -> str:
    workload = cfg.get("workload", {})
    return str(workload.get("endpoint", "/v1/chat/completions"))


def get_num_requests(cfg: Dict[str, Any]) -> int:
    workload = cfg.get("workload", {})
    num_requests = int(workload.get("request_count", workload.get("num_requests", 0)))
    if num_requests <= 0:
        num_requests = int(cfg.get("experiment", {}).get("final_requests", 0))
    if num_requests <= 0:
        num_requests = int(cfg.get("experiment", {}).get("screening_requests", 20))
    return num_requests


def get_concurrency(cfg: Dict[str, Any]) -> Any:
    workload = cfg.get("workload", {})
    return workload.get("concurrency", 4)


def get_timeout_s(cfg: Dict[str, Any]) -> Any:
    workload = cfg.get("workload", {})
    return workload.get("timeout_s", 180.0)


def get_temperature(cfg: Dict[str, Any]) -> Any:
    workload = cfg.get("workload", {})
    return workload.get("temperature", 0.0)
