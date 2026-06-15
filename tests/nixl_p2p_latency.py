#!/usr/bin/env python3

"""Measure GPU-to-GPU transfer latency through NIXL.

Run one target process and one initiator process on the same node. To match the
PyTorch CUDA-copy baseline, use the target as the source GPU and the initiator
as the destination GPU, then time NIXL READ transfers on the initiator.

For physical GPU0 -> physical GPU1 with CUDA-visible-device remapping, run the
target with `CUDA_VISIBLE_DEVICES=0 --src-device 0` and the initiator with
`CUDA_VISIBLE_DEVICES=1 --dst-device 0`.
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
RUN_DONE = b"RUN_DONE"


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


def configure_cuda_device(device):
    torch.cuda.set_device(device)
    torch.set_default_device(f"cuda:{device}")


def make_tensor(num_bytes, dtype, device, fill_value, row_bytes):
    element_size = torch.empty((), dtype=dtype).element_size()
    num_elements = max(1, (num_bytes + element_size - 1) // element_size)
    row_elements = max(1, row_bytes // element_size)
    row_elements = min(row_elements, num_elements)
    num_rows = (num_elements + row_elements - 1) // row_elements
    padded_elements = num_rows * row_elements
    tensor = torch.empty((num_rows, row_elements), dtype=dtype, device=device)
    tensor.fill_(fill_value)
    return tensor, num_elements, padded_elements * element_size


def get_tensor_rows(tensor):
    return [tensor[i, :] for i in range(tensor.shape[0])]


def wait_until(predicate, description, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")


def wait_for_metadata(agent, peer_name, timeout_seconds):
    wait_until(
        lambda: agent.check_remote_metadata(peer_name),
        f"metadata from {peer_name}",
        timeout_seconds,
    )


def wait_for_notification(agent, peer_name, timeout_seconds):
    def poll():
        notifs = agent.get_new_notifs()
        if peer_name in notifs and notifs[peer_name]:
            return notifs[peer_name][0]
        return None

    return wait_until(poll, f"notification from {peer_name}", timeout_seconds)


def wait_for_size_done(agent, timeout_seconds):
    def poll():
        notifs = agent.get_new_notifs()
        for messages in notifs.values():
            for message in messages:
                if message.startswith(SIZE_DONE_PREFIX):
                    return message
        return None

    return wait_until(poll, "size completion notification", timeout_seconds)


def wait_for_run_done(agent, timeout_seconds):
    def poll():
        notifs = agent.get_new_notifs()
        for messages in notifs.values():
            for message in messages:
                if message == RUN_DONE:
                    return message
        return None

    return wait_until(poll, "run completion notification", timeout_seconds)


def wait_for_xfer(agent, xfer_handle, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = agent.check_xfer_state(xfer_handle)
        if state == "DONE":
            return
        if state == "ERR":
            raise RuntimeError("NIXL transfer entered ERR state")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for NIXL transfer completion")


def post_and_wait(agent, xfer_handle, timeout_seconds):
    state = agent.transfer(xfer_handle)
    if state == "ERR":
        raise RuntimeError("posting NIXL transfer failed")
    wait_for_xfer(agent, xfer_handle, timeout_seconds)


def target(args, dtype):
    configure_cuda_device(args.src_device)
    config = nixl_agent_config(True, True, args.port)
    agent = nixl_agent("target", config)

    for num_bytes in args.sizes:
        tensor, _, actual_bytes = make_tensor(
            num_bytes, dtype, f"cuda:{args.src_device}", 1, args.row_bytes
        )
        torch.cuda.synchronize(args.src_device)

        reg_descs = agent.register_memory(tensor)
        if not reg_descs:
            raise RuntimeError("target memory registration failed")

        try:
            target_rows = get_tensor_rows(tensor)
            target_descs = agent.get_xfer_descs(target_rows)
            if not target_descs:
                raise RuntimeError("target transfer descriptor creation failed")

            wait_for_metadata(agent, "initiator", args.timeout)
            agent.send_notif("initiator", agent.get_serialized_descs(target_descs))

            message = wait_for_size_done(agent, args.timeout)
            expected = SIZE_DONE_PREFIX + str(actual_bytes).encode()
            if message != expected:
                raise RuntimeError(f"unexpected size completion message: {message!r}")
        finally:
            agent.deregister_memory(reg_descs)

    wait_for_run_done(agent, args.timeout)


def measure_size(agent, tensor, target_descs, args):
    local_rows = get_tensor_rows(tensor)
    local_descs = agent.get_xfer_descs(local_rows)
    if not local_descs:
        raise RuntimeError("initiator transfer descriptor creation failed")

    xfer_handle = agent.initialize_xfer(
        "READ", local_descs, target_descs, "target", "Done_reading"
    )

    try:
        for _ in range(args.warmup):
            post_and_wait(agent, xfer_handle, args.timeout)

        samples_us = []
        for _ in range(args.iters):
            start = time.perf_counter()
            for _ in range(args.copies_per_iter):
                post_and_wait(agent, xfer_handle, args.timeout)
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
    configure_cuda_device(args.dst_device)
    config = nixl_agent_config(True, True, 0)
    agent = nixl_agent("initiator", config)

    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)

    print("# NIXL GPU-to-GPU READ latency")
    print(f"# torch_version={torch.__version__}")
    print(f"# src_device={args.src_device}")
    print(f"# dst_device={args.dst_device} name={maybe_device_name(args.dst_device)}")
    print(f"# dtype={args.dtype} iters={args.iters} warmup={args.warmup}")
    print(f"# copies_per_iter={args.copies_per_iter}")
    print(f"# row_bytes={args.row_bytes}")
    print("bytes,nixl_mean_us,nixl_p50_us,nixl_p95_us,effective_gib_s")

    try:
        for num_bytes in args.sizes:
            tensor, logical_elements, actual_bytes = make_tensor(
                num_bytes, dtype, f"cuda:{args.dst_device}", 0, args.row_bytes
            )
            reg_descs = agent.register_memory(tensor)
            if not reg_descs:
                raise RuntimeError("initiator memory registration failed")

            try:
                target_descs = agent.deserialize_descs(
                    wait_for_notification(agent, "target", args.timeout)
                )
                wait_for_metadata(agent, "target", args.timeout)
                result = measure_size(agent, tensor, target_descs, args)

                if args.verify:
                    torch.cuda.synchronize(args.dst_device)
                    flat = tensor.reshape(-1)
                    expected = torch.ones_like(flat[:logical_elements])
                    if not torch.equal(flat[:logical_elements], expected):
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
        try:
            agent.send_notif("target", RUN_DONE)
        except Exception:
            pass
        try:
            agent.remove_remote_agent("target")
        except Exception:
            pass
        try:
            agent.invalidate_local_metadata(args.ip, args.port)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NIXL GPU-to-GPU transfer latency."
    )
    parser.add_argument("--ip", required=True, help="target/listen IP address")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--mode", choices=("target", "initiator"), required=True)
    parser.add_argument(
        "--src-device",
        type=int,
        default=0,
        help="process-local CUDA device used by the target/source process",
    )
    parser.add_argument(
        "--dst-device",
        type=int,
        default=1,
        help="process-local CUDA device used by the initiator/destination process",
    )
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
    parser.add_argument(
        "--row-bytes",
        type=parse_size,
        default=parse_size("64k"),
        help="row size for NIXL transfer descriptors; mirrors basic_two_peers row views",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait for metadata, notifications, or transfer completion",
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    active_device = args.src_device if args.mode == "target" else args.dst_device
    if torch.cuda.device_count() <= active_device:
        raise SystemExit(
            f"need visible CUDA device {active_device}, "
            f"but PyTorch sees {torch.cuda.device_count()} CUDA device(s)"
        )
    if args.iters <= 0 or args.warmup < 0 or args.copies_per_iter <= 0:
        raise SystemExit("--iters and --copies-per-iter must be > 0; --warmup >= 0")
    if args.row_bytes <= 0 or args.timeout <= 0:
        raise SystemExit("--row-bytes and --timeout must be > 0")

    dtype = DTYPES[args.dtype]
    if args.mode == "target":
        target(args, dtype)
    else:
        initiator(args, dtype)


if __name__ == "__main__":
    main()
