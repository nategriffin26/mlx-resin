"""Measure MLX slice writes with and without a live alias to the old arena."""

import argparse
import statistics
import time

import mlx.core as mx


def measure(elements, *, retain_alias, repeats):
    storage = mx.zeros(elements)
    update = mx.ones(512)
    mx.eval(storage, update)
    samples = []
    for trial in range(4):
        start = time.perf_counter()
        for step in range(repeats):
            old = storage[:] if retain_alias else None
            offset = (step * 512) % (elements - 512)
            storage[offset : offset + 512] = update
            mx.eval(storage)
            if old is not None:
                mx.eval(old)
        if trial:
            samples.append((time.perf_counter() - start) * 1e6 / repeats)
    assert float(storage.sum()) > 0
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    print(mx.device_info())
    for mib in (1, 16, 128):
        for alias in (False, True):
            samples = measure(
                mib * 1024 * 1024 // 4, retain_alias=alias, repeats=args.repeats
            )
            print(
                f"{mib=} {alias=} us/write={samples} median={statistics.median(samples):.1f}"
            )


if __name__ == "__main__":
    main()
