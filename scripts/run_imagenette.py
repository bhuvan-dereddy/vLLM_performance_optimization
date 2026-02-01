from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet50

from nvtx_utils import nvtx_range


#                   CUDA Profiler API (controls Nsight capture) 
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


#                   Opt 4: CUDA Prefetcher 
class CudaPrefetcher:
    def __init__(self, loader, device: str, non_blocking: bool):
        self.loader = iter(loader)
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            x, y = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            with nvtx_range("phase:copy_h2d"):
                x = x.to(self.device, non_blocking=self.non_blocking)
                y = y.to(self.device, non_blocking=self.non_blocking)

        self.next_batch = (x, y)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        self._preload()
        return batch


def build_loader(data_root: Path, batch_size: int, num_workers: int, pin_memory: bool, dataset_size: str):
    tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    ds = torchvision.datasets.Imagenette(
        root=str(data_root),
        split="train",
        size=dataset_size,  # "160px" / "320px" / "full"
        download=True,
        transform=tfm,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


def train_step(model, x, y, opt, loss_fn):
    with nvtx_range("phase:compute"):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return float(loss.detach().cpu())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="p2_baseline")
    ap.add_argument("--dataset_size", default="320px")  # 160px / 320px / full
    ap.add_argument("--batch_size", type=int, default=64)

    # Opt 1
    ap.add_argument("--num_workers", type=int, default=0)

    # Opt 2
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--non_blocking", action="store_true")

    # Opt 3
    ap.add_argument("--use_streams", action="store_true")

    # Opt 4
    ap.add_argument("--use_prefetcher", action="store_true")

    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--profile_steps", type=int, default=50)

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "datasets" / "imagenette"
    data_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    loader = build_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        dataset_size=args.dataset_size,
    )

    with nvtx_range("phase:model_init"):
        model = resnet50(weights=None).to(device)
        model.train()

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Opt 3 streams (only if not using prefetcher)
    copy_stream = torch.cuda.Stream() if (device == "cuda" and args.use_streams and not args.use_prefetcher) else None
    compute_stream = torch.cuda.Stream() if (device == "cuda" and args.use_streams and not args.use_prefetcher) else None

    # iterator selection
    if device == "cuda" and args.use_prefetcher:
        it = CudaPrefetcher(loader, device=device, non_blocking=args.non_blocking)
    else:
        it = iter(loader)

    # stage first batch for Opt 3
    staged = None
    if device == "cuda" and args.use_streams and not args.use_prefetcher:
        with nvtx_range("phase:get_batch"):
            x, y = next(it)
        with torch.cuda.stream(copy_stream):
            with nvtx_range("phase:copy_h2d"):
                x = x.to(device, non_blocking=args.non_blocking)
                y = y.to(device, non_blocking=args.non_blocking)
        staged = (x, y)

    if device == "cuda":
        torch.cuda.synchronize()

    total_steps = args.warmup_steps + args.profile_steps
    step = 0
    started = False
    t0 = None
    t1 = None
    last_loss = None

    while step < total_steps:
        if step == args.warmup_steps:
            if device == "cuda":
                torch.cuda.synchronize()
                cuda_profiler_start()
            started = True
            t0 = time.time()

        with nvtx_range("phase:step"):
            if device == "cuda" and args.use_prefetcher:
                with nvtx_range("phase:get_batch"):
                    x, y = next(it)  # already H2D in prefetcher
                last_loss = train_step(model, x, y, opt, loss_fn)

            elif device == "cuda" and args.use_streams and not args.use_prefetcher:
                compute_stream.wait_stream(copy_stream)
                x, y = staged

                with nvtx_range("phase:get_batch"):
                    nx, ny = next(it)

                with torch.cuda.stream(copy_stream):
                    with nvtx_range("phase:copy_h2d"):
                        nx = nx.to(device, non_blocking=args.non_blocking)
                        ny = ny.to(device, non_blocking=args.non_blocking)

                with torch.cuda.stream(compute_stream):
                    last_loss = train_step(model, x, y, opt, loss_fn)

                staged = (nx, ny)

            else:
                with nvtx_range("phase:get_batch"):
                    x, y = next(it)
                if device == "cuda":
                    with nvtx_range("phase:copy_h2d"):
                        x = x.to(device, non_blocking=args.non_blocking)
                        y = y.to(device, non_blocking=args.non_blocking)
                last_loss = train_step(model, x, y, opt, loss_fn)

        step += 1

    if started:
        t1 = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
            cuda_profiler_stop()

    if t0 and t1:
        wall = t1 - t0
        thr = args.profile_steps / wall if wall > 0 else None
        print(f"[{args.run_name}] prof_steps={args.profile_steps} wall_s={wall:.3f} thr_steps_per_s={thr} last_loss={last_loss}")


if __name__ == "__main__":
    main()
