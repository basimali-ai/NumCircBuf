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

import time
import os
from typing import Type, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from numcircbuf import RunningMeanBuffer, RunningMeanSqBuffer
from numcircbuf.system_info import get_available_ram
from numcircbuf.bench_utils import (
    temporary_benchmark_data,
    raw_bench_with_calc,
    BenchLogger,
    get_cpu_name,
    prepare_blocks,
    determine_num_runs,
)

NUM_THREADS = 1
GIB_SAFETY_BUFFER = 2
TOTAL_BYTE_LIMIT = max(0, get_available_ram() - (GIB_SAFETY_BUFFER * (1024**3)))


def _run_variant(
    buffer_class: Type[RunningMeanBuffer] | Type[RunningMeanSqBuffer],
    operation_focus: Literal["extend/append", "calculation"],
    maxlen: int,
    dtype: type,
    block_size: int,
    n_runs: int,
    calc_every: int,
    data_path: str,
    data_shape: tuple[int, int],
    warmup_path: str,
    warmup_data_shape: tuple[int, int],
    offset_path: str,
    offset_data_shape: tuple[int, int],
    fill_path: str,
    fill_data_shape: tuple[int, int],
    evict_path: str,
) -> tuple[str, float]:
    """Worker function to run a single benchmark variant."""
    blocks, warmup_block, offset_blocks, fill_blocks, evict_arr = prepare_blocks(
        block_size,
        maxlen,
        dtype,
        n_runs,
        data_path,
        data_shape,
        warmup_path,
        warmup_data_shape,
        single_offset=False,
        offset_path=offset_path,
        offset_data_shape=offset_data_shape,
        prepare_fill=True,
        fill_path=fill_path,
        fill_data_shape=fill_data_shape,
        prepare_evict=True,
        evict_path=evict_path,
    )

    buffer = buffer_class(maxlen, operation_focus, dtype=dtype)
    if isinstance(buffer, RunningMeanBuffer):
        func = buffer.mean
    else:
        func = buffer.mean_square

    t = raw_bench_with_calc(
        buffer,
        func,
        fill_blocks,
        offset_blocks,
        warmup_block,
        blocks,
        calc_every,
        n_runs,
        evict_arr,
    )

    return buffer_class.__name__, t


def benchmark_and_save(dtype: type):
    start = time.time()
    py_file_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpu_name = get_cpu_name()
    base_name = f"{py_file_name}_{np.dtype(dtype).name}_{cpu_name}_{timestamp}"

    progress_file = f"logs/{py_file_name}/{base_name}_progress.txt"
    results_file = f"results/{py_file_name}/{base_name}_results.csv"

    bench_logger = BenchLogger(progress_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    BLOCK_SIZE = (
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
    )
    MAXLEN = (
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
        2_097_152,
        4_194_304,
        8_388_608,
    )
    CALC_EVERY = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    VARIANTS: tuple[Type[RunningMeanBuffer] | Type[RunningMeanSqBuffer], ...] = (
        RunningMeanBuffer,
        RunningMeanSqBuffer,
    )
    MODES: tuple[
        Literal["extend/append", "calculation"],
        Literal["extend/append", "calculation"],
    ] = (
        "extend/append",
        "calculation",
    )

    combo_indices = [
        (block_idx, maxlen_idx, calc_idx, mode_idx, variant_idx)
        for block_idx in range(len(BLOCK_SIZE))
        for maxlen_idx in range(len(MAXLEN))
        for calc_idx in range(len(CALC_EVERY))
        for mode_idx in range(len(MODES))
        for variant_idx in range(len(VARIANTS))
        if BLOCK_SIZE[block_idx] <= MAXLEN[maxlen_idx]
    ]
    rows = []
    times = []

    max_block_size = max(BLOCK_SIZE)
    max_maxlen = max(MAXLEN)
    n_runs, _, _ = determine_num_runs(
        elem_bytes=np.dtype(dtype).itemsize,
        total_byte_limit=TOTAL_BYTE_LIMIT,
        maxlen=max_maxlen,
        block_size=max_block_size,
        with_fill=True,
    )

    bench_logger.log(
        f"\nGenerating data for:\n"
        f"    max(BLOCK_SIZE): {max_block_size:_}\n"
        f"    max(MAXLEN): {max_maxlen:_}\n"
        f"    n_runs: {n_runs:_}\n",
        True,
    )

    with temporary_benchmark_data(
        dtype,
        max_maxlen,
        max_block_size,
        n_runs,
        create_offset_data=True,
        create_fill_data=True,
        create_evict_arr=True,
    ) as (
        data_path,
        data,
        warmup_path,
        warmup_data,
        offset_path,
        offset_data,
        fill_path,
        fill_data,
        evict_path,
        _,
    ):
        total = len(combo_indices)
        bench_logger.log(f"Total tasks: {total}")
        bench_logger.log("Creating ProcessPoolExecutor...")
        pbar = tqdm(
            total=total,
            desc="Benchmark progress",
            unit="task",
            ncols=100,
            smoothing=0.1,
        )
        with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
            bench_logger.log("Executor created, submitting futures...")
            futures = {}
            completed_futures = []
            submit_pbar = tqdm(
                total=total,
                desc="Submitting",
                unit="task",
                leave=False,
                ncols=80,
                smoothing=0.1,
            )
            for idx, (
                block_idx,
                maxlen_idx,
                calc_idx,
                mode_idx,
                variant_idx,
            ) in enumerate(combo_indices):
                block_size = BLOCK_SIZE[block_idx]
                maxlen = MAXLEN[maxlen_idx]
                calc_every = CALC_EVERY[calc_idx]
                mode = MODES[mode_idx]
                variant = VARIANTS[variant_idx]

                fut = executor.submit(
                    _run_variant,
                    variant,
                    mode,
                    maxlen,
                    dtype,
                    block_size,
                    n_runs,
                    calc_every,
                    data_path,
                    data.shape,
                    warmup_path,
                    warmup_data.shape,
                    offset_path,
                    offset_data.shape,
                    fill_path,
                    fill_data.shape,
                    evict_path,
                )
                futures[fut] = (
                    mode,
                    maxlen,
                    block_size,
                    n_runs,
                    calc_every,
                )
                submit_pbar.update(1)

                # Check for any errors/completed futures while submitting
                if idx % 100:
                    done_futures = [
                        f
                        for f in futures.keys()
                        if f.done() and f not in completed_futures
                    ]
                    for done_fut in done_futures:
                        (
                            mode,
                            maxlen,
                            block_size,
                            n_runs,
                            calc_every,
                        ) = futures[done_fut]
                        try:
                            v, t = done_fut.result()
                            rows.append(
                                [
                                    v,
                                    mode,
                                    maxlen,
                                    t,
                                    block_size,
                                    n_runs,
                                    calc_every,
                                ]
                            )
                            times.append(t)
                            completed_futures.append(done_fut)
                            # bench_logger.log(
                            #     "Completed during submission: "
                            #     f"({variant}, {mode}), BLOCK_SIZE={block_size:_}, "
                            #     f"MAX_LEN={maxlen:_}, CALC_EVERY={calc_every:_}, "
                            #     f"BLOCKS={num_blocks:_}, time={t:_.8f}s"
                            # )
                        except Exception as e:
                            completed_futures.append(done_fut)
                            bench_logger.log(f"FAILED during submission: {e}", True)
                        pbar.update(1)

            submit_pbar.close()
            bench_logger.log(
                f"All {total} tasks submitted, {len(completed_futures)} already completed",
            )

            # Collect any remaining completions
            remaining = [f for f in futures.keys() if f not in completed_futures]
            bench_logger.log(f"Waiting for {len(remaining)} remaining tasks...")

            for i, future in enumerate(as_completed(remaining), 1):
                (
                    mode,
                    maxlen,
                    block_size,
                    n_runs,
                    calc_every,
                ) = futures[future]
                try:
                    v, t = future.result()
                    rows.append([v, mode, maxlen, t, block_size, n_runs, calc_every])
                    times.append(t)
                    # if i % 100:
                    #     bench_logger.log(
                    #         f"Remaining {i}/{len(remaining)}: "
                    #         f"({variant}, {mode}), BLOCK_SIZE={block_size:_}, "
                    #         f"MAX_LEN={maxlen:_}, CALC_EVERY={calc_every:_}, "
                    #         f"BLOCKS={num_blocks:_}, time={t:_.8f}s"
                    #     )
                except Exception as e:
                    bench_logger.log(
                        f"Remaining task {i}/{len(remaining)} FAILED: {e}",
                        True,
                    )
                pbar.update(1)

            bench_logger.log("Exited as_completed() loop")

        bench_logger.log("Executor closed")

        pbar.close()
        bench_logger.log("Executor closed")

        if times:
            total_time = sum(times)
            max_time = max(times)
            avg_time = total_time // len(times)
            bench_logger.log("\n--- Benchmark Summary ---", True)
            bench_logger.log(f"Max single task time: {max_time:_}ns", True)
            bench_logger.log(f"Average task time: {avg_time:_}ns", True)
        else:
            bench_logger.log("No task times were recorded.", True)

        df = pd.DataFrame(
            rows,
            columns=[
                "VARIANT",
                "MODE",
                "MAXLEN",
                "TIME",
                "BLOCK_SIZE",
                "NUM_RUNS",
                "CALC_EVERY",
            ],
        )
        df.to_csv(results_file, index=False)

        end = time.time()

        bench_logger.log(
            f"Total time taken for data generation + benchmark: {end - start:_.2f}s",
            True,
        )


if __name__ == "__main__":
    dtypes = (
        np.float32,
        np.float64,
    )
    for dtype in dtypes:
        benchmark_and_save(dtype)
