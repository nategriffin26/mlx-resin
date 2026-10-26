"""Measure paged decode kernels and count actual Metal library compilations."""

import ctypes
import hashlib
import json
import math
import statistics
import time

import mlx.core as mx

from resin import CacheGroupLayout, CacheLayout
from resin.backend.mlx import SlabArena, paged_attention


class MetalCompilations:
    """Observe newLibraryWithSource:options:error:, below MLX's library cache."""

    def __enter__(self):
        pointer = ctypes.c_void_p
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = pointer
        self.objc.object_getClass.argtypes = [pointer]
        self.objc.object_getClass.restype = pointer
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = pointer
        self.objc.class_getInstanceMethod.argtypes = [pointer, pointer]
        self.objc.class_getInstanceMethod.restype = pointer
        self.objc.method_setImplementation.argtypes = [pointer, pointer]
        self.objc.method_setImplementation.restype = pointer
        self.objc.method_getImplementation.argtypes = [pointer]
        self.objc.method_getImplementation.restype = pointer
        selector = self.objc.sel_registerName(b"newLibraryWithSource:options:error:")
        device_class = self.objc.object_getClass(metal.MTLCreateSystemDefaultDevice())
        self.method = self.objc.class_getInstanceMethod(device_class, selector)
        self.original = self.objc.method_getImplementation(self.method)
        signature = ctypes.CFUNCTYPE(
            pointer, pointer, pointer, pointer, pointer, pointer
        )
        original = signature(self.original)
        utf8 = ctypes.CFUNCTYPE(ctypes.c_char_p, pointer, pointer)(
            ("objc_msgSend", self.objc)
        )
        utf8_selector = self.objc.sel_registerName(b"UTF8String")
        self.sources = []

        def record(device, command, source, options, error):
            text = utf8(source, utf8_selector)
            self.sources.append((hashlib.sha256(text).hexdigest(), b"resin_" in text))
            return original(device, command, source, options, error)

        self.callback = signature(record)
        self.objc.method_setImplementation(self.method, self.callback)
        return self

    def counts(self):
        return {
            "metal_library_compilations": len(self.sources),
            "resin_library_compilations": sum(resin for _, resin in self.sources),
            "resin_unique_libraries": len(
                {digest for digest, resin in self.sources if resin}
            ),
        }

    def __exit__(self, *args):
        self.objc.method_setImplementation(self.method, self.original)


def main():
    with MetalCompilations() as compilation:
        initial_compilations = None
        for batch, length in ((1, 128), (3, 255), (8, 4096), (2, 32768), (5, 513)):
            width = max(1, math.ceil(length / 128))
            arena = SlabArena(
                CacheLayout(
                    parent_capacity=batch * width, groups=(CacheGroupLayout(128),)
                ),
                ((1, 128, 2, 2, 64),),
                group_dtypes=(mx.bfloat16,),
            )
            slabs = mx.arange(1, batch * width + 1, dtype=mx.uint32).reshape(
                batch, width
            )
            slots = mx.zeros_like(slabs)
            contexts = mx.full((batch,), length, mx.uint32)
            queries = mx.zeros((batch, 1, 8, 64), mx.bfloat16)
            keys = mx.zeros((batch, 1, 2, 64), mx.bfloat16)
            values = mx.ones_like(keys)
            mx.eval(slabs, slots, contexts, queries, keys, values)
            mx.eval(arena.read_layer(0, 0, slabs, slots))

            def execute():
                return paged_attention(
                    queries,
                    keys,
                    values,
                    arena,
                    0,
                    0,
                    slabs,
                    slots,
                    contexts,
                    scale=0.125,
                )

            output = execute()
            mx.eval(output)
            assert bool(
                mx.allclose(
                    output.astype(mx.float32), mx.array(1 / (length + 1)), atol=1e-5
                )
            )
            samples = []
            for _ in range(30):
                start = time.perf_counter()
                mx.eval(execute())
                samples.append((time.perf_counter() - start) * 1e6)
            count = compilation.counts()["resin_library_compilations"]
            if initial_compilations is None:
                initial_compilations = count
            assert count == initial_compilations, (
                "novel shapes compiled another Resin Metal library"
            )
            print(
                json.dumps(
                    {
                        "batch": batch,
                        "context": length,
                        "median_us": statistics.median(samples),
                        **compilation.counts(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
