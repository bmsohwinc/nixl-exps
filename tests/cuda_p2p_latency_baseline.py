#!/usr/bin/env python3

"""Measure GPU-to-GPU tensor copy latency with PyTorch.

This is a simple intra-node baseline for comparing against NIXL transfers. It
preallocates one source tensor on a source GPU and one destination tensor on a
destination GPU, then times repeated `dst.copy_(src)` operations.
"""

import argparse
import statistics
import time

import torch


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def parse_size(size_text):
    units = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    text = size_text.strip().lower()
    number = ""
    unit = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        else:
            unit += char
    if not number:
        raise argparse.ArgumentTypeError(f"invalid size: {size_text}")
    return int(float(number) * units.get(unit or "b", 1))


def parse_sizes(sizes_text):
    return [parse_size(item) for item in sizes_text.split(",") if item.strip()]


def percentile(values, pct):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def maybe_device_name(device):
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        return "unknown"


def can_access_peer(src_device, dst_device):
    fn = getattr(torch.cuda, "can_device_access_peer", None)
    if fn is None:
        return "unknown"
    try:
        return str(bool(fn(dst_device, src_device))).lower()
    except Exception:
        return "unknown"


def make_tensor(num_bytes, dtype, device, fill_value):
    element_size = torch.empty((), dtype=dtype).element_size()
    num_elements = max(1, (num_bytes + element_size - 1) // element_size)
    tensor = torch.empty(num_elements, dtype=dtype, device=device)
    tensor.fill_(fill_value)
    return tensor


def measure_size(num_bytes, dtype, src_device, dst_device, args):
    src = make_tensor(num_bytes, dtype, src_device, 1)
    dst = make_tensor(src.numel() * src.element_size(), dtype, dst_device, 0)
    actual_bytes = src.numel() * src.element_size()

    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)

    with torch.cuda.device(dst_device):
        stream = torch.cuda.Stream(device=dst_device)

    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()

    event_us = []
    wall_us = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        wall_start = time.perf_counter()
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(args.copies_per_iter):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
        end.synchronize()
        wall_end = time.perf_counter()

        event_us.append(start.elapsed_time(end) * 1000.0 / args.copies_per_iter)
        wall_us.append((wall_end - wall_start) * 1_000_000.0 / args.copies_per_iter)

    if args.verify:
        torch.cuda.synchronize(dst_device)
        expected = torch.ones_like(dst)
        if not torch.equal(dst, expected):
            raise RuntimeError("destination tensor verification failed")

    return {
        "bytes": actual_bytes,
        "event_mean_us": statistics.fmean(event_us),
        "event_p50_us": statistics.median(event_us),
        "event_p95_us": percentile(event_us, 95),
        "wall_mean_us": statistics.fmean(wall_us),
        "wall_p50_us": statistics.median(wall_us),
        "wall_p95_us": percentile(wall_us, 95),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline GPU-to-GPU tensor copy latency using PyTorch."
    )
    parser.add_argument("--src-device", type=int, default=0)
    parser.add_argument("--dst-device", type=int, default=1)
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes("4,64,256,1k,4k,64k,1m,16m,64m"),
        help="Comma-separated transfer sizes, e.g. 4,1k,1m,64m.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="uint8")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--copies-per-iter",
        type=int,
        default=1,
        help="Batch this many copies per timed iteration; useful for tiny sizes.",
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    if torch.cuda.device_count() <= max(args.src_device, args.dst_device):
        raise SystemExit(
            f"need devices {args.src_device} and {args.dst_device}, "
            f"but PyTorch sees {torch.cuda.device_count()} CUDA device(s)"
        )
    if args.src_device == args.dst_device:
        raise SystemExit("--src-device and --dst-device must be different")
    if args.iters <= 0 or args.warmup < 0 or args.copies_per_iter <= 0:
        raise SystemExit("--iters and --copies-per-iter must be > 0; --warmup >= 0")

    dtype = DTYPES[args.dtype]
    print("# PyTorch CUDA peer tensor-copy latency baseline")
    print(f"# torch_version={torch.__version__}")
    print(f"# src_device={args.src_device} name={maybe_device_name(args.src_device)}")
    print(f"# dst_device={args.dst_device} name={maybe_device_name(args.dst_device)}")
    print(f"# dst_can_access_src={can_access_peer(args.src_device, args.dst_device)}")
    print(f"# dtype={args.dtype} iters={args.iters} warmup={args.warmup}")
    print(f"# copies_per_iter={args.copies_per_iter}")
    print(
        "bytes,event_mean_us,event_p50_us,event_p95_us,"
        "wall_mean_us,wall_p50_us,wall_p95_us,effective_gib_s"
    )

    for num_bytes in args.sizes:
        result = measure_size(num_bytes, dtype, args.src_device, args.dst_device, args)
        gib = result["bytes"] / 1024**3
        seconds = result["event_mean_us"] / 1_000_000.0
        bandwidth = gib / seconds if seconds > 0 else 0.0
        print(
            f"{result['bytes']},"
            f"{result['event_mean_us']:.3f},"
            f"{result['event_p50_us']:.3f},"
            f"{result['event_p95_us']:.3f},"
            f"{result['wall_mean_us']:.3f},"
            f"{result['wall_p50_us']:.3f},"
            f"{result['wall_p95_us']:.3f},"
            f"{bandwidth:.3f}"
        )


if __name__ == "__main__":
    main()
