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

import numpy as np
from numcircbuf import BlockingCircBuffer
from numcircbuf.bench_utils import (
    raw_bench_write_read,
    temporary_benchmark_data,
    determine_num_runs,
)

DTYPE = np.float64
ELEM_BYTES = np.dtype(DTYPE).itemsize

MAXLEN_BYTE_LIMIT = 65_536
BLOCK_BYTE_LIMIT = 65_536

TOTAL_GiB_LIMIT = 0.5
TOTAL_BYTE_LIMIT = round(TOTAL_GiB_LIMIT * (1024**3))


def _run_benchmark(
    dtype: type,
    maxlen: int,
    block_size: int,
    n_runs: int,
    warm_cache: bool,
    data: np.ndarray,
    warmup_data: np.ndarray,
    offset_data: np.ndarray,
    evict_arr: np.ndarray,
    read_into_arr: np.ndarray | None,
):

    time_ns = raw_bench_write_read(
        BlockingCircBuffer(maxlen, dtype),
        offset_data,
        warmup_data,
        data,
        n_runs,
        evict_arr,
        warm_cache,
        read_into_arr,
    )

    print(
        f"\nWarm Cache: {warm_cache}\n"
        f"Read Into Array Used: {read_into_arr is not None}\n"
        "BlockingCircBuffer Write throughput: "
        f"{(block_size * ELEM_BYTES) / time_ns[0]:.2f} GB/s\n"
        "BlockingCircBuffer Read throughput: "
        f"{(block_size * ELEM_BYTES) / time_ns[1]:.2f} GB/s"
    )


def main(
    dtype=DTYPE,
    elem_bytes=ELEM_BYTES,
    maxlen_byte_limit=MAXLEN_BYTE_LIMIT,
    block_byte_limit=BLOCK_BYTE_LIMIT,
    total_byte_limit=TOTAL_BYTE_LIMIT,
):
    n_runs, block_size, maxlen = determine_num_runs(
        elem_bytes=elem_bytes,
        maxlen_byte_limit=maxlen_byte_limit,
        block_byte_limit=block_byte_limit,
        total_byte_limit=total_byte_limit,
        with_fill=False,
    )
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
            for read_into in (False, True):
                _run_benchmark(
                    dtype,
                    maxlen,
                    block_size,
                    n_runs,
                    warm_cache,
                    data,
                    warmup_data,
                    offset_data,
                    evict_arr,
                    np.zeros(block_size, dtype=DTYPE) if read_into else None,
                )


if __name__ == "__main__":
    main()
