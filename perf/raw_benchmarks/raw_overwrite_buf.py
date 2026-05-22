# Copyright 2026 Syed Basim Ali
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
import numpy as np
from numcircbuf import OverwriteCircBuffer
from numcircbuf.bench_utils import (
    RefPythonNumPyCircBuffer,
    raw_bench,
    temporary_benchmark_data,
    determine_num_runs,
)

DTYPE = np.float64
ELEM_BYTES = np.dtype(DTYPE).itemsize

MAXLEN_BYTE_LIMIT = 65_536
BLOCK_BYTE_LIMIT = 65_536

TOTAL_GiB_LIMIT = 0.5
TOTAL_BYTE_LIMIT = round(TOTAL_GiB_LIMIT * (1024**3))


class BenchDeque(deque):
    def __init__(self, maxlen, *_):
        super().__init__(maxlen=maxlen)

    extend_unchecked = deque.extend


class BenchList(list):
    def __init__(self, *_):
        super().__init__()

    extend_unchecked = list.extend


def _run_benchmark(
    dtype: type,
    maxlen: int,
    block_size: int,
    n_runs: int,
    warm_cache: bool,
    data: np.ndarray,
    warmup_data: np.ndarray,
    offset_data: np.ndarray,
    evict_arr: np.ndarray | None,
):
    def bench_buffer(buffer_cls):
        return raw_bench(
            buffer_cls(maxlen, "never", dtype),
            offset_data,
            warmup_data,
            data,
            n_runs,
            evict_arr,
            warm_cache,
        )

    results_ns = {
        buffer_cls.__name__: bench_buffer(buffer_cls)
        for buffer_cls in (
            OverwriteCircBuffer,
            RefPythonNumPyCircBuffer,
            BenchDeque,
            BenchList,
        )
    }

    print(f"\nWarm Cache: {warm_cache}")
    for name, time_ns in results_ns.items():
        throughput_gb_s = (block_size * ELEM_BYTES) / time_ns
        print(f"{name} throughput: {throughput_gb_s:.4g} GB/s")


def main(
    dtype=DTYPE,
    elem_bytes=ELEM_BYTES,
    maxlen_byte_limit=MAXLEN_BYTE_LIMIT,
    block_byte_limit=BLOCK_BYTE_LIMIT,
    total_byte_limit=TOTAL_BYTE_LIMIT,
):
    for block_size in (None, 1):
        if block_size is None:
            n_runs, block_size, maxlen = determine_num_runs(
                elem_bytes=elem_bytes,
                maxlen_byte_limit=maxlen_byte_limit,
                block_byte_limit=block_byte_limit,
                total_byte_limit=total_byte_limit,
                with_fill=False,
            )
            print("\n--- Extend ---")
        else:
            n_runs, block_size, maxlen = determine_num_runs(
                elem_bytes=elem_bytes,
                maxlen_byte_limit=maxlen_byte_limit,
                block_size=block_size,
                total_byte_limit=total_byte_limit,
                with_fill=False,
            )
            print("\n--- Append ---")

        with temporary_benchmark_data(
            dtype,
            maxlen,
            block_size,
            n_runs,
            create_offset_data=True,
            create_fill_data=False,
            create_evict_arr=True,
        ) as (
            _,
            data,
            _,
            warmup_data,
            _,
            offset_data,
            _,
            _,
            _,
            evict_arr,
        ):
            for warm_cache in (False, True):
                _run_benchmark(
                    dtype,
                    maxlen,
                    block_size,
                    n_runs,
                    warm_cache,
                    data,
                    warmup_data,
                    offset_data,
                    evict_arr if not warm_cache else None,
                )


if __name__ == "__main__":
    main()
