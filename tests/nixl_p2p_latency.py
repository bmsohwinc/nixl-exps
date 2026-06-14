#!/usr/bin/env python3

"""Measure GPU-to-GPU transfer latency through NIXL.

Run one target process and one initiator process on the same node. To match the
PyTorch CUDA-copy baseline, use the target as the source GPU and the initiator
as the destination GPU, then time NIXL READ transfers on the initiator.
"""

import argparse
import statistics
import time

import torch

from nixl import nixl_agent, nixl_agent_config


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

SIZE_DONE_PREFIX = b"SIZE_DONE:"


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


def make_tensor(num_bytes, dtype, device, fill_value):
    element_size = torch.empty((), dtype=dtype).element_size()
    num_elements = max(1, (num_bytes + element_size - 1) // element_size)
    tensor = torch.empty(num_elements, dtype=dtype, device=device)
    tensor.fill_(fill_value)
    return tensor


def wait_for_metadata(agent, peer_name):
    while not agent.check_remote_metadata(peer_name):
        pass


def wait_for_notification(agent, peer_name):
    while True:
        notifs = agent.get_new_notifs()
        if peer_name in notifs and notifs[peer_name]:
            return notifs[peer_name][0]


def wait_for_size_done(agent):
    while True:
        notifs = agent.get_new_notifs()
        for messages in notifs.values():
            for message in messages:
                if message.startswith(SIZE_DONE_PREFIX):
                    return message


def wait_for_xfer(agent, xfer_handle):
    while True:
        state = agent.check_xfer_state(xfer_handle)
        if state == "DONE":
            return
        if state == "ERR":
            raise RuntimeError("NIXL transfer entered ERR state")


def post_and_wait(agent, xfer_handle):
    state = agent.transfer(xfer_handle)
    if state == "ERR":
        raise RuntimeError("posting NIXL transfer failed")
    wait_for_xfer(agent, xfer_handle)


def target(args, dtype):
    torch.set_default_device(f"cuda:{args.src_device}")
    config = nixl_agent_config(True, True, args.port)
    agent = nixl_agent("target", config)

    wait_for_metadata(agent, "initiator")

    for num_bytes in args.sizes:
        tensor = make_tensor(num_bytes, dtype, f"cuda:{args.src_device}", 1)
        actual_bytes = tensor.numel() * tensor.element_size()
        torch.cuda.synchronize(args.src_device)

        reg_descs = agent.register_memory(tensor)
        if not reg_descs:
            raise RuntimeError("target memory registration failed")

        try:
            target_descs = agent.get_xfer_descs([tensor])
            if not target_descs:
                raise RuntimeError("target transfer descriptor creation failed")
            agent.send_notif("initiator", agent.get_serialized_descs(target_descs))

            message = wait_for_size_done(agent)
            expected = SIZE_DONE_PREFIX + str(actual_bytes).encode()
            if message != expected:
                raise RuntimeError(f"unexpected size completion message: {message!r}")
        finally:
            agent.deregister_memory(reg_descs)


def measure_size(agent, tensor, target_descs, args):
    local_descs = agent.get_xfer_descs([tensor])
    if not local_descs:
        raise RuntimeError("initiator transfer descriptor creation failed")

    xfer_handle = agent.initialize_xfer(
        "READ", local_descs, target_descs, "target", "Done_reading"
    )

    try:
        for _ in range(args.warmup):
            post_and_wait(agent, xfer_handle)

        samples_us = []
        for _ in range(args.iters):
            start = time.perf_counter()
            for _ in range(args.copies_per_iter):
                post_and_wait(agent, xfer_handle)
            end = time.perf_counter()
            samples_us.append((end - start) * 1_000_000.0 / args.copies_per_iter)
    finally:
        agent.release_xfer_handle(xfer_handle)

    return {
        "mean_us": statistics.fmean(samples_us),
        "p50_us": statistics.median(samples_us),
        "p95_us": percentile(samples_us, 95),
    }


def initiator(args, dtype):
    torch.set_default_device(f"cuda:{args.dst_device}")
    config = nixl_agent_config(True, True, 0)
    agent = nixl_agent("initiator", config)

    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)
    wait_for_metadata(agent, "target")

    print("# NIXL GPU-to-GPU READ latency")
    print(f"# torch_version={torch.__version__}")
    print(f"# src_device={args.src_device} name={maybe_device_name(args.src_device)}")
    print(f"# dst_device={args.dst_device} name={maybe_device_name(args.dst_device)}")
    print(f"# dtype={args.dtype} iters={args.iters} warmup={args.warmup}")
    print(f"# copies_per_iter={args.copies_per_iter}")
    print("bytes,nixl_mean_us,nixl_p50_us,nixl_p95_us,effective_gib_s")

    try:
        for num_bytes in args.sizes:
            tensor = make_tensor(num_bytes, dtype, f"cuda:{args.dst_device}", 0)
            actual_bytes = tensor.numel() * tensor.element_size()
            reg_descs = agent.register_memory(tensor)
            if not reg_descs:
                raise RuntimeError("initiator memory registration failed")

            try:
                target_descs = agent.deserialize_descs(
                    wait_for_notification(agent, "target")
                )
                result = measure_size(agent, tensor, target_descs, args)

                if args.verify:
                    torch.cuda.synchronize(args.dst_device)
                    expected = torch.ones_like(tensor)
                    if not torch.equal(tensor, expected):
                        raise RuntimeError("destination tensor verification failed")

                gib = actual_bytes / 1024**3
                seconds = result["mean_us"] / 1_000_000.0
                bandwidth = gib / seconds if seconds > 0 else 0.0
                print(
                    f"{actual_bytes},"
                    f"{result['mean_us']:.3f},"
                    f"{result['p50_us']:.3f},"
                    f"{result['p95_us']:.3f},"
                    f"{bandwidth:.3f}",
                    flush=True,
                )
                agent.send_notif("target", SIZE_DONE_PREFIX + str(actual_bytes).encode())
            finally:
                agent.deregister_memory(reg_descs)
    finally:
        agent.remove_remote_agent("target")
        agent.invalidate_local_metadata(args.ip, args.port)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NIXL GPU-to-GPU transfer latency."
    )
    parser.add_argument("--ip", required=True, help="target/listen IP address")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--mode", choices=("target", "initiator"), required=True)
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
        help="Batch this many completed NIXL transfers per timed iteration.",
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
    if args.mode == "target":
        target(args, dtype)
    else:
        initiator(args, dtype)


if __name__ == "__main__":
    main()
