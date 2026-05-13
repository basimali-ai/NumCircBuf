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

from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np
from numcircbuf import (
    RunningMeanSqBuffer,
    RunningMeanBuffer,
    IntegratedGatedBuffer,
    determine_operation_focus,
)
from numcircbuf.bench_utils import (
    raw_bench_with_calc,
    temporary_benchmark_data,
    determine_num_runs,
)

TOTAL_GiB_LIMIT = 0.1
TOTAL_BYTE_LIMIT = round(TOTAL_GiB_LIMIT * (1024**3))

CALC_EVERY = 1

BUFFER_METHODS = {
    RunningMeanSqBuffer: "mean_square",
    RunningMeanBuffer: "mean",
    IntegratedGatedBuffer: "gated_mean_square",
}
DTYPES = (np.float32, np.float64)
MAXLENS = (4096, 8192, 16_384, 32_768, 65_536)
BLOCK_SIZES = (4096, 8192, 16_384, 32_768, 65_536)


def _run_benchmark(
    buffer_class: type[RunningMeanSqBuffer]
    | type[RunningMeanBuffer]
    | type[IntegratedGatedBuffer],
    func_name: str,
    dtype: type,
    maxlen: int,
    block_size: int,
    calc_every: int,
    n_runs: int,
    warm_cache: bool,
    data,
    warmup_data,
    offset_data,
    fill_data,
    evict_arr,
):
    elem_bytes = np.dtype(dtype).itemsize

    if buffer_class in (RunningMeanSqBuffer, RunningMeanBuffer):
        buffer = buffer_class(
            maxlen,
            operation_focus=determine_operation_focus(
                buffer_type=buffer_class,
                dtype=dtype,
                buffer_maxlen=maxlen,
                block_size=block_size,
                calc_every=calc_every,
            ),
            dtype=dtype,
        )
    else:
        buffer = buffer_class(
            maxlen,
            -70.0,
            -10.0,
            dtype=dtype,
        )

    time_ns = raw_bench_with_calc(
        buffer,
        getattr(buffer, func_name),
        fill_data,
        offset_data,
        warmup_data,
        data,
        calc_every,
        n_runs,
        evict_arr,
        warm_cache,
    )

    time_s = time_ns * 1e-9
    total_bytes = block_size * elem_bytes
    processing_rate = block_size / time_s
    throughput_gbps = total_bytes / (time_s * 1e9)

    return (
        processing_rate,
        throughput_gbps,
    )


def params_for_dtype(
    dtype: type,
    total_byte_limit: int,
    maxlens: tuple[int, ...],
    block_sizes: tuple[int, ...],
):
    """Yield (maxlen, block_size, n_runs) for a given dtype."""
    elem_bytes = np.dtype(dtype).itemsize
    for maxlen in maxlens:
        for block_size in block_sizes:
            if block_size > maxlen:
                continue
            n_runs, _, _ = determine_num_runs(
                elem_bytes=elem_bytes,
                total_byte_limit=total_byte_limit,
                maxlen=maxlen,
                block_size=block_size,
                with_fill=True,
            )
            yield maxlen, block_size, n_runs


def print_separator():
    print("-" * 10)


def print_summary(warm_cache, dtype, processing_rates, throughputs):
    print(
        f"Warm Cache: {warm_cache}\n"
        f"~{(min(processing_rates) / 1_000_000):.0f}–"
        f"{(max(processing_rates) / 1_000_000):.0f}M "
        f"{dtype.__name__} elems/sec (extend + calculation)\n"
        f"{min(throughputs):_.2f}–{max(throughputs):_.2f} "
        "GB/s effective throughput"
    )
    print_separator()


def main(
    total_byte_limit: int = TOTAL_BYTE_LIMIT,
    maxlens: tuple[int, ...] = MAXLENS,
    block_sizes: tuple[int, ...] = BLOCK_SIZES,
    calc_every: int = CALC_EVERY,
):
    results: defaultdict[
        str, defaultdict[Tuple[type, bool], Dict[str, List[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: {"processing": [], "throughput": []}))
    for dtype in DTYPES:
        for maxlen, block_size, n_runs in params_for_dtype(
            dtype, total_byte_limit, maxlens, block_sizes
        ):
            with temporary_benchmark_data(
                dtype,
                maxlen,
                block_size,
                n_runs,
                create_offset_data=True,
                create_fill_data=True,
                create_evict_arr=True,
            ) as (
                _,
                data,
                _,
                warmup_data,
                _,
                offset_data,
                _,
                fill_data,
                _,
                evict_arr,
            ):
                for warm_cache in (False, True):
                    key = (dtype, warm_cache)
                    for buffer_class, func in BUFFER_METHODS.items():
                        p, t = _run_benchmark(
                            buffer_class,
                            func,
                            dtype,
                            maxlen,
                            block_size,
                            calc_every,
                            n_runs,
                            warm_cache,
                            data,
                            warmup_data,
                            offset_data,
                            fill_data,
                            evict_arr,
                        )
                        name = buffer_class.__name__
                        results[name][key]["processing"].append(p)
                        results[name][key]["throughput"].append(t)

    all_buffer_names = [buffer_class.__name__ for buffer_class in list(BUFFER_METHODS)]
    for buffer_name in all_buffer_names:
        print_separator()
        print(f"{buffer_name} :")
        for warm_cache in (False, True):
            for dtype in DTYPES:
                key = (dtype, warm_cache)
                proc = results[buffer_name].get(key, {}).get("processing", [])
                thru = results[buffer_name].get(key, {}).get("throughput", [])
                print_summary(warm_cache, dtype, proc, thru)


if __name__ == "__main__":
    main()
