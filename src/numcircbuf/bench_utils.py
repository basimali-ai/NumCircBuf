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

"""
Benchmarking utilities for the NumCircBuf library.

This module provides a suite of performance tests to compare NumCircBuf
against custom buffers and structures defined in `numcircbuf.protocols`.
It includes tools for measuring the throughput of circular buffer operations
and statistical accumulators/functions.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import math
import os
import platform
import re
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter_ns
from typing import Any, Literal, overload

import numpy as np

from ._typing import Scalar, ScalarT, ShapeT
from .exceptions import (
    NumCircBufArithmeticError,
    NumCircBufTypeError,
    NumCircBufValueError,
)
from .protocols import (
    ReadWriteBenchmarkBufferProtocol,
    WriteBenchmarkBufferProtocol,
)
from .system_info import (
    PAGESIZE,
    get_cache_line_size,
    get_cpu_l3_cache,
)
from .utils import classproperty

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

RS = 25
ss = np.random.SeedSequence(RS)
child_ss = ss.spawn(5)
rngs = tuple(np.random.default_rng(s) for s in child_ss)

_cache_line_size: int | None = None


class EvictArrConfig:
    """
    Configuration for generating a CPU cache eviction array.

    Provides lazily initialized class-level settings for array generation.
    Values are cached and shared across all instances.
    """

    _dtype: type[Scalar] | None = None
    """Internal storage for the NumPy dtype. Initialized on first access or via set_dtype."""

    _bytes: int | None = None
    """Internal storage for the byte size. Initialized on first access or via set_bytes."""

    _item_size: int | None = None
    """Internal cache for item size. Resets whenever dtype is changed."""

    @classproperty
    def dtype(cls) -> type[Scalar]:
        """
        NumPy dtype used for the eviction array.

        Defaults to `np.float64`. Lazily initialized upon first access if not
        previously set via :meth:`set_dtype`.

        :rtype: type[np.floating] | type[np.integer]
        """
        if cls._dtype is None:
            cls._dtype = np.float64
        return cls._dtype

    @classproperty
    def item_size(cls) -> int:
        """
        Size in bytes of a single element based on the current `dtype`.

        Derived from `np.dtype(dtype).itemsize`. This value is cached and
        automatically invalidated if :meth:`set_dtype` is called.

        :rtype: int
        """
        if cls._item_size is None:
            cls._item_size = np.dtype(cls.dtype).itemsize
        return cls._item_size

    @classproperty
    def bytes(cls) -> int:
        """
        Total target size of the eviction array in bytes.

        Defaults to the CPU L3 cache size. Lazily initialized upon first access
        if not previously set via :meth:`set_bytes`.

        :rtype: int
        """
        if cls._bytes is None:
            cls._bytes = get_cpu_l3_cache()
        return cls._bytes

    @classmethod
    def set_dtype(cls, dtype: type[Scalar]) -> None:
        """
        Set the NumPy dtype for the eviction array and invalidate the cached item size.

        :param dtype: NumPy scalar type (e.g., `np.float32`, `np.int64`)
        :type dtype: type[np.floating] | type[np.integer]
        """
        cls._dtype = dtype
        cls._item_size = None

    @classmethod
    def set_bytes(cls, value: int) -> None:
        """
        Set the target size of the array in bytes.

        :param value: Target size in bytes
        :type value: int
        """
        cls._bytes = value

    @classproperty
    def shape(cls) -> tuple[int]:
        """
        The 1-dimensional shape required to reach the configured byte size.

        Computed using floor division as `(bytes // item_size,)`.

        :rtype: tuple[int]
        """
        return (cls.bytes // cls.item_size,)


def get_cpu_name() -> str:
    """
    Retrieve and sanitize the CPU name from the system.

    The name is normalized by converting it to lowercase, removing common trademarks
    symbols (e.g., (R), (TM)), and stripping clock speed information to create
    a filesystem-friendly string.

    :return:
        A sanitized string representing the CPU model, or "unknown_cpu"
        if the name cannot be determined.
    :rtype: str
    """
    name = platform.processor().strip()

    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            processor_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            name = processor_name.strip()

        elif sys.platform == "linux":
            with open("/proc/cpuinfo", "r", encoding="utf8") as f:
                cpuinfo = f.read()

            for line in cpuinfo.splitlines():
                if "model name" in line:
                    name = line.split(":", 1)[1].strip()
                    break
            else:
                for line in cpuinfo.splitlines():
                    if line.startswith(("Hardware", "Processor")):
                        name = line.split(":", 1)[1].strip()
                        break

        elif sys.platform == "darwin":
            import subprocess

            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()

    except Exception:  # pragma: no cover  # noqa: BLE001, S110
        pass

    if not name:
        return "unknown_cpu"

    name = name.lower()
    name = re.sub(r"\(r\)|\(tm\)|cpu|processor", "", name)
    name = re.sub(r"@.*", "", name)
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    return name if name else "unknown_cpu"


def touch_pages(
    arr: np.ndarray[tuple[int], np.dtype[Scalar]],
    *,
    warm_cache: bool = False,
    page_size: int = PAGESIZE,
    cache_line_bytes: int | None = None,
) -> None:
    """
    Iterate through a NumPy array to ensure memory pages are resident in RAM,
    and CPU cache if `warm_cache` is True.

    This function accesses the array at specific intervals to force the OS to
    map pages (page faulting) or to load data into the CPU cache.

    :param arr: The array to be touched.
    :type arr: np.ndarray

    :param warm_cache:
        If True, accesses the array at cache-line intervals
        rather than page-size intervals. Defaults to False.
    :type warm_cache: bool, optional

    :param page_size: The size of a memory page in bytes. Defaults to :meth:`PAGESIZE`.
    :type page_size: int, optional

    :param cache_line_bytes:
        The size of a CPU cache line in bytes. If None and
        `warm_cache` is True, it attempts to detect the
        system's cache line size. Defaults to None.
    :type cache_line_bytes: int | None, optional
    """
    if warm_cache:
        if cache_line_bytes is None:
            global _cache_line_size
            if _cache_line_size is None:
                _cache_line_size = get_cache_line_size()
            cache_line_bytes = _cache_line_size
        np.add.reduce(arr[:: cache_line_bytes // arr.itemsize])
        return
    np.add.reduce(arr[:: page_size // arr.itemsize])


def set_process_priority(priority: Literal["high", "normal"] = "high") -> None:
    """
    Adjust the scheduling priority of the current process.

    On Windows, this modifies the Process Priority Class.
    On Linux and macOS, it modifies the process "nice" value.
    This function fails silently if the process lacks the necessary permissions.

    :param priority:
        The target priority level. "high" increases priority,
        while "normal" resets it to system defaults.
        Defaults to "high".
    :type priority: Literal["high", "normal"], optional
    """
    try:
        if sys.platform == "win32":
            HIGH_PRIORITY_CLASS = 0x00000080
            NORMAL_PRIORITY_CLASS = 0x00000020

            handle = ctypes.windll.kernel32.GetCurrentProcess()

            if priority == "high":
                ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
            elif priority == "normal":
                ctypes.windll.kernel32.SetPriorityClass(handle, NORMAL_PRIORITY_CLASS)
        elif sys.platform == "linux" or sys.platform == "darwin":
            if priority == "high":
                os.nice(-10)
            elif priority == "normal":
                os.nice(0)
    except (PermissionError, OSError, AttributeError):  # pragma: no cover
        pass


@contextmanager
def no_gc() -> Generator[None, None, None]:
    """
    Context manager to temporarily disable the Python Garbage Collector.

    Upon entering, a full garbage collection is triggered and the GC is disabled.
    The GC is automatically re-enabled when exiting the context. This is useful
    for isolating performance benchmarks from GC-induced jitter.

    :yields: None
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@overload
def trimmed_mean_times(
    arr: np.ndarray[tuple[int], np.dtype[Scalar]], return_int: Literal[True]
) -> int: ...
@overload
def trimmed_mean_times(
    arr: np.ndarray[tuple[int], np.dtype[Scalar]], return_int: Literal[False]
) -> float: ...
@overload
def trimmed_mean_times(
    arr: np.ndarray[tuple[int], np.dtype[Scalar]], return_int: bool
) -> int | float: ...


def trimmed_mean_times(
    arr: np.ndarray[tuple[int], np.dtype[Scalar]], return_int: bool
) -> int | float:
    """
    Calculate the trimmed mean of an array.

    The function removes the first 10% of values (at least one) before
    calculating the average. This is used to remove outliers
    in timing benchmarks.

    :param arr: A NumPy array containing numeric timing data.
    :type arr: np.ndarray

    :param return_int: If True, the result is rounded to the nearest integer.
    :type return_int: bool

    :return: The calculated trimmed mean.
    :rtype: int | float

    :raises :exc:`NumCircBufArithmeticError`:
        If the result is NaN, or if an integer
        return is requested but the result
        is infinite.
    """
    arr_size = arr.size
    if arr_size == 0:
        return 0 if return_int else 0.0

    drop = max(1, round(arr_size / 10))
    if drop >= arr_size:
        result = float(arr.mean())
    else:
        result = float(arr[drop:].mean())

    if math.isnan(result):
        raise NumCircBufArithmeticError(message="The trimmed mean resulted in NaN.")

    if return_int:
        if math.isinf(result):
            raise NumCircBufArithmeticError(
                message="Cannot return an integer because the calculated mean is infinite."
            )
        return round(result)

    return result


def determine_num_runs(
    *,
    elem_bytes: int,
    total_byte_limit: int,
    maxlen_byte_limit: int | None = None,
    block_byte_limit: int | None = None,
    maxlen: int | None = None,
    block_size: int | None = None,
    with_fill: bool,
) -> tuple[int, int, int]:
    """
    Calculate the number of benchmark iterations possible within memory constraints.

    This function determines the optimal number of runs and the operational sizes
    (maxlen and block_size) for benchmarking, ensuring that the total memory
    footprint of generated datasets (including warmup, wrap-around offsets,
    and fill data) stays within the specified byte limit.

    :param elem_bytes: Size of a single array element in bytes.
    :type elem_bytes: int

    :param total_byte_limit: Maximum allowed memory for all benchmark datasets.
    :type total_byte_limit: int

    :param maxlen_byte_limit: Target byte size for the buffer's maxlen.
    :type maxlen_byte_limit: int | None

    :param block_byte_limit: Target byte size for the block operations.
    :type block_byte_limit: int | None

    :param maxlen: Explicit maximum length of the buffer.
    :type maxlen: int | None

    :param block_size: Explicit number of elements per block operation.
    :type block_size: int | None

    :param with_fill: Whether the benchmark includes full-buffer pre-filling.
    :type with_fill: bool

    :return: A tuple containing (n_runs, block_size, maxlen).
    :rtype: tuple[int, int, int]

    :raises :exc:`NumCircBufValueError`:
        If parameter constraints are violated (e.g., maxlen < block_size)
        or if the total_byte_limit is insufficient for a single run.
    """

    max_elems = total_byte_limit // elem_bytes

    if block_size is None:
        if block_byte_limit is None:
            raise NumCircBufValueError(
                message="neither block_byte_limit nor block_size provided"
            )
        block_size = block_byte_limit // elem_bytes

    if maxlen is None:
        if maxlen_byte_limit is None:
            raise NumCircBufValueError(
                message="neither maxlen_byte_limit nor maxlen provided"
            )
        maxlen = maxlen_byte_limit // elem_bytes

    if block_size <= 0:
        raise NumCircBufValueError(message="block_size cannot be <= 0")

    if maxlen <= 2:
        raise NumCircBufValueError(message="maxlen cannot be <= 2")

    if maxlen < block_size:
        raise NumCircBufValueError(message="maxlen cannot be < block_size")

    warmup_block_size = maxlen
    fixed = maxlen + warmup_block_size

    fill_size = maxlen if with_fill else 0
    offset_size = maxlen - max(1, block_size // 2)
    elems_per_pair_run = (block_size * 2) + offset_size + (fill_size * 2)

    if fixed + elems_per_pair_run > max_elems:
        raise NumCircBufValueError(message="total_byte_limit too low")

    k = (max_elems - fixed) // elems_per_pair_run
    n_runs = k * 2
    offset_num_blocks = n_runs // 2

    total_elems = (
        fixed
        + (n_runs * block_size)
        + (offset_num_blocks * offset_size)
        + (n_runs * fill_size)
    )

    if total_elems + (block_size + fill_size + offset_size) <= max_elems:
        n_runs += 1

    return n_runs, block_size, maxlen


def generate_rand_memmap_arr(
    dtype: type[ScalarT],
    shape: ShapeT,
    path: str,
    rng: np.random.Generator,
    *,
    multiply_by: Scalar | float | None = None,
    subtract: Scalar | float | None = None,
) -> np.memmap[ShapeT, np.dtype[ScalarT]]:
    """
    Create a memory-mapped file filled with random data, optionally transformed.

    The function generates random values according to the specified `dtype` and
    maps them to a file on disk. If integer types are used, the function
    calculates safe bounds to prevent overflow after applying `multiply_by`
    and `subtract`.

    :param dtype: The NumPy data type for the array.
    :type dtype: type[Scalar]

    :param shape: The dimensions of the array.
    :type shape: tuple[int, ...]

    :param path: Filesystem path where the memmap file will be created.
    :type path: str

    :param rng: NumPy random number generator instance.
    :type rng: np.random.Generator

    :param multiply_by: Factor to multiply each element by. Defaults to None.
    :type multiply_by: Scalar | float | int | None = None

    :param subtract: Value to subtract from each element. Defaults to None.
    :type subtract: Scalar | float | int | None = None

    :return: A reference to the memory-mapped NumPy array.
    :rtype: np.memmap

    :raises :exc:`NumCircBufValueError`:
        If transformations result in values outside the dtype's
        representable range or if invalid types are provided for integers.
    :raises :exc:`NumCircBufTypeError`: If the dtype is not supported.
    """

    if np.issubdtype(dtype, np.integer):
        is_int = True
    elif np.issubdtype(dtype, np.floating):
        is_int = False
    else:
        raise NumCircBufTypeError(message=f"Unsupported dtype: {dtype}")

    data = np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    if is_int:
        info = np.iinfo(dtype)  # type: ignore

        if subtract is not None:
            if not float(subtract).is_integer():
                raise NumCircBufValueError(
                    message="subtract must be an integer for integer dtype"
                )
            sub = int(subtract)
        else:
            sub = 0

        if multiply_by is not None:
            if not float(multiply_by).is_integer():
                raise NumCircBufValueError(
                    message="multiply_by must be an integer for integer dtype"
                )
            mul = int(multiply_by)
        else:
            mul = 1

        if mul == 0:
            value = 0 - sub
            if not (info.min <= value <= info.max):
                raise NumCircBufValueError(
                    message="Resulting constant out of dtype bounds"
                )
            data.fill(value)
            return data

        if mul > 0:
            low = -(-(int(info.min) + sub) // mul)
            high = (int(info.max) + sub) // mul
        else:
            if np.issubdtype(dtype, np.unsignedinteger):
                raise NumCircBufValueError(
                    message="`multiply_by` cannot be negative for unsigned integers."
                )

            low = -(-(int(info.max) + sub) // mul)
            high = (int(info.min) + sub) // mul

        low = max(int(info.min), min(low, int(info.max)))
        high = max(int(info.min), min(high, int(info.max)))

        if low > high:
            raise NumCircBufValueError(
                message="No valid integers exist for this dtype given the `multiply_by` and `subtract`."
            )

        data[:] = rng.integers(
            low=low,
            high=high,
            size=shape,
            dtype=dtype,
            endpoint=True,
        )

    else:
        if dtype == np.float32 or dtype == np.float64:
            rng.random(shape, dtype=dtype, out=data)  # type: ignore

    if multiply_by is not None:
        data *= multiply_by  # type: ignore[arg-type]
    if subtract is not None:
        data -= subtract  # type: ignore[arg-type]

    return data


def generate_bench_memmaps(
    dtype: type[ScalarT],
    maxlen: int,
    block_size: int,
    n_runs: int,
    *,
    create_offset_data: bool,
    create_fill_data: bool,
    create_evict_arr: bool,
    data_path: str = "temp_bench_data.dat",
    warmup_path: str = "temp_bench_warmup.dat",
    offset_path: str = "temp_bench_offset.dat",
    fill_path: str = "temp_bench_fill.dat",
    evict_path: str = "temp_bench_evict.dat",
) -> tuple[
    str,
    np.memmap[tuple[int, int], np.dtype[ScalarT]],
    str,
    np.memmap[tuple[int], np.dtype[ScalarT]],
    str,
    np.memmap[tuple[int, int], np.dtype[ScalarT]] | None,
    str,
    np.memmap[tuple[int, int], np.dtype[ScalarT]] | None,
    str,
    np.memmap[tuple[int], np.dtype[Scalar]] | None,
]:
    """
    Orchestrate the generation of multiple memmap files for benchmarking.

    Produces a suite of datasets required for comprehensive testing, including
    primary data blocks, warmup data, and specialized data for testing
    wrap-around behavior and cache eviction.

    :param dtype: NumPy dtype for the datasets.
    :type dtype: type[np.floating] | type[np.integer]

    :param maxlen: Maximum length of the circular buffer being tested.
    :type maxlen: int

    :param block_size: Elements per write operation.
    :type block_size: int

    :param n_runs: Total number of runs/iterations to generate data for.
    :type n_runs: int

    :param create_offset_data: Whether to generate data for wrap-around testing.
    :type create_offset_data: bool

    :param create_fill_data: Whether to generate data to pre-fill the buffer.
    :type create_fill_data: bool

    :param create_evict_arr: Whether to generate an array for cache eviction.
    :type create_evict_arr: bool

    :param data_path: Filename for the primary data.
    :type data_path: str

    :param warmup_path: Filename for the warmup data.
    :type warmup_path: str

    :param offset_path: Filename for the offset data.
    :type offset_path: str

    :param fill_path: Filename for the fill data.
    :type fill_path: str

    :param evict_path: Filename for the eviction array.
    :type evict_path: str

    :return:
        A tuple containing all file paths and their corresponding memmap objects.
        (
            data_path, data,
            warmup_path, warmup_data,
            offset_path, offset_data,
            fill_path, fill_data,
            evict_path, evict_arr,
        )
    :rtype: tuple

    :raises :exc:`NumCircBufValueError`: If `n_runs <= 1`.
    """

    if n_runs <= 1:
        raise NumCircBufValueError(message="n_runs must be > 1")

    multiply_by = dtype(2)
    subtract = dtype(1)

    data = generate_rand_memmap_arr(
        dtype,
        (n_runs, block_size),
        data_path,
        rngs[0],
        multiply_by=multiply_by,
        subtract=subtract,
    )

    warmup_data = generate_rand_memmap_arr(
        dtype,
        (maxlen,),
        warmup_path,
        rngs[1],
        multiply_by=multiply_by,
        subtract=subtract,
    )

    if create_offset_data:
        offset_data = generate_rand_memmap_arr(
            dtype,
            (n_runs // 2, maxlen - (block_size // 2)),
            offset_path,
            rngs[2],
            multiply_by=multiply_by,
            subtract=subtract,
        )
    else:
        offset_path, offset_data = "", None

    if create_fill_data:
        fill_data = generate_rand_memmap_arr(
            dtype,
            (n_runs, maxlen),
            fill_path,
            rngs[3],
            multiply_by=multiply_by,
            subtract=subtract,
        )
    else:
        fill_path, fill_data = "", None

    if create_evict_arr:
        evict_arr = generate_rand_memmap_arr(
            EvictArrConfig.dtype,
            EvictArrConfig.shape,
            evict_path,
            rngs[4],
        )
    else:
        evict_path, evict_arr = "", None

    return (
        data_path,
        data,
        warmup_path,
        warmup_data,
        offset_path,
        offset_data,
        fill_path,
        fill_data,
        evict_path,
        evict_arr,
    )


@contextmanager
def temporary_benchmark_data(
    dtype: type[ScalarT],
    buffer_maxlen: int,
    block_size: int,
    n_runs: int,
    *,
    create_offset_data: bool,
    create_fill_data: bool,
    create_evict_arr: bool,
    data_path: str = "temp_bench_data.dat",
    warmup_path: str = "temp_bench_warmup.dat",
    offset_path: str = "temp_bench_offset.dat",
    fill_path: str = "temp_bench_fill.dat",
    evict_path: str = "temp_bench_evict.dat",
    log_delete_errors: bool = True,
) -> Generator[
    tuple[
        str,
        np.memmap[tuple[int, int], np.dtype[ScalarT]],
        str,
        np.memmap[tuple[int], np.dtype[ScalarT]],
        str,
        np.memmap[tuple[int, int], np.dtype[ScalarT]] | None,
        str,
        np.memmap[tuple[int, int], np.dtype[ScalarT]] | None,
        str,
        np.memmap[tuple[int], np.dtype[Scalar]] | None,
    ],
    None,
    None,
]:
    """
    Context manager to handle lifecycle of temporary benchmark data files.

    Generates all necessary datasets upon entry and ensures all memmap files are
    closed and deleted from the filesystem upon exit; However if the files
    could not be deleted and `log_delete_errors = True`, it will log with severity 'WARNING'

    :param dtype: NumPy dtype for the datasets.
    :type dtype: type[np.floating] | type[np.integer]

    :param buffer_maxlen: Maximum length of the circular buffer being tested.
    :type buffer_maxlen: int

    :param block_size: Elements per write operation.
    :type block_size: int

    :param n_runs: Total number of runs/iterations to generate data for.
    :type n_runs: int

    :param create_offset_data: Whether to generate data for wrap-around testing.
    :type create_offset_data: bool

    :param create_fill_data: Whether to generate data to pre-fill the buffer.
    :type create_fill_data: bool

    :param create_evict_arr: Whether to generate an array for cache eviction.
    :type create_evict_arr: bool

    :param data_path: Filename for the primary data.
    :type data_path: str

    :param warmup_path: Filename for the warmup data.
    :type warmup_path: str

    :param offset_path: Filename for the offset data.
    :type offset_path: str

    :param fill_path: Filename for the fill data.
    :type fill_path: str

    :param evict_path: Filename for the eviction array.
    :type evict_path: str

    :yields:
        A tuple containing all file paths and their corresponding memmap objects.
        (
            data_path, data,
            warmup_path, warmup_data,
            offset_path, offset_data,
            fill_path, fill_data,
            evict_path, evict_arr,
        )
    """

    _data_path = _warmup_path = _offset_path = _fill_path = _evict_path = None
    data = warmup_data = offset_data = fill_data = evict_arr = None

    try:
        (
            _data_path,
            data,
            _warmup_path,
            warmup_data,
            _offset_path,
            offset_data,
            _fill_path,
            fill_data,
            _evict_path,
            evict_arr,
        ) = generate_bench_memmaps(
            dtype,
            buffer_maxlen,
            block_size,
            n_runs,
            create_offset_data=create_offset_data,
            create_fill_data=create_fill_data,
            create_evict_arr=create_evict_arr,
            data_path=data_path,
            warmup_path=warmup_path,
            offset_path=offset_path,
            fill_path=fill_path,
            evict_path=evict_path,
        )
        yield (
            _data_path,
            data,
            _warmup_path,
            warmup_data,
            _offset_path,
            offset_data,
            _fill_path,
            fill_data,
            _evict_path,
            evict_arr,
        )

    finally:
        for var in (data, warmup_data, offset_data, fill_data, evict_arr):
            try:
                mmap_obj = getattr(var, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except Exception:  # pragma: no cover  # noqa: BLE001, S110
                pass

        for path in (
            _data_path,
            _warmup_path,
            _offset_path,
            _fill_path,
            _evict_path,
        ):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                if log_delete_errors:
                    logger.warning(
                        "--- Could not delete temp file. Delete manually. ---\n"
                        f"Path: {path}",
                        exc_info=True,
                    )


@overload
def prepare_blocks(
    block_size: int,
    maxlen: int,
    dtype: type[ScalarT],
    n_runs: int,
    data_path: str,
    data_shape: tuple[int, int],
    warmup_path: str,
    warmup_data_shape: tuple[int],
    *,
    single_offset: Literal[True],
    offset_path: str = "",
    offset_data_shape: tuple[int, int] = (0, 0),
    prepare_fill: bool,
    fill_path: str = "",
    fill_data_shape: tuple[int, int] = (0, 0),
    prepare_evict: bool,
    evict_path: str = "",
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[ScalarT]],
    np.ndarray[tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int], np.dtype[ScalarT]] | None,
    np.memmap[tuple[int], np.dtype[Scalar]] | None,
]: ...
@overload
def prepare_blocks(
    block_size: int,
    maxlen: int,
    dtype: type[ScalarT],
    n_runs: int,
    data_path: str,
    data_shape: tuple[int, int],
    warmup_path: str,
    warmup_data_shape: tuple[int],
    *,
    single_offset: Literal[False],
    offset_path: str = "",
    offset_data_shape: tuple[int, int] = (0, 0),
    prepare_fill: bool,
    fill_path: str = "",
    fill_data_shape: tuple[int, int] = (0, 0),
    prepare_evict: bool,
    evict_path: str = "",
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[ScalarT]],
    np.ndarray[tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int], np.dtype[ScalarT]] | None,
    np.memmap[tuple[int], np.dtype[Scalar]] | None,
]: ...
@overload
def prepare_blocks(
    block_size: int,
    maxlen: int,
    dtype: type[ScalarT],
    n_runs: int,
    data_path: str,
    data_shape: tuple[int, int],
    warmup_path: str,
    warmup_data_shape: tuple[int],
    *,
    single_offset: bool,
    offset_path: str = "",
    offset_data_shape: tuple[int, int] = (0, 0),
    prepare_fill: bool,
    fill_path: str = "",
    fill_data_shape: tuple[int, int] = (0, 0),
    prepare_evict: bool,
    evict_path: str = "",
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[ScalarT]],
    np.ndarray[tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int] | tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int], np.dtype[ScalarT]] | None,
    np.memmap[tuple[int], np.dtype[Scalar]] | None,
]: ...


def prepare_blocks(
    block_size: int,
    maxlen: int,
    dtype: type[ScalarT],
    n_runs: int,
    data_path: str,
    data_shape: tuple[int, int],
    warmup_path: str,
    warmup_data_shape: tuple[int],
    *,
    single_offset: bool,
    offset_path: str = "",
    offset_data_shape: tuple[int, int] = (0, 0),
    prepare_fill: bool,
    fill_path: str = "",
    fill_data_shape: tuple[int, int] = (0, 0),
    prepare_evict: bool,
    evict_path: str = "",
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[ScalarT]],
    np.ndarray[tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int] | tuple[int], np.dtype[ScalarT]],
    np.ndarray[tuple[int, int], np.dtype[ScalarT]] | None,
    np.memmap[tuple[int], np.dtype[Scalar]] | None,
]:
    """
    Re-map existing benchmark files into read-only NumPy arrays.

    This is used to load previously generated datasets into memory-mapped
    arrays ready for use in benchmarking functions.

    :param block_size: Elements per block.
    :type block_size: int

    :param maxlen: Maximum length of the buffer.
    :type maxlen: int

    :param dtype: NumPy dtype of the data.
    :type dtype: type[np.floating] | type[np.integer]

    :param n_runs: Number of runs/iterations to prepare.
    :type n_runs: int

    :param data_path: Path to primary data file.
    :type data_path: str

    :param data_shape: Dimensions of primary data.
    :type data_shape: tuple[int, int]

    :param warmup_path: Path to warmup data file.
    :type warmup_path: str

    :param warmup_data_shape: Dimensions of warmup data.
    :type warmup_data_shape: tuple[int, int]

    :param single_offset: If True, uses a slice of warmup data as offset.
    :type single_offset: bool

    :param offset_path: Path to offset data file.
    :type offset_path: str

    :param offset_data_shape: Dimensions of offset data.
    :type offset_data_shape: tuple[int, int]

    :param prepare_fill: Whether to map fill data.
    :type prepare_fill: bool

    :param fill_path: Path to fill data file.
    :type fill_path: str

    :param fill_data_shape: Dimensions of fill data.
    :type fill_data_shape: tuple[int, int]

    :param prepare_evict: Whether to map the eviction array.
    :type prepare_evict: bool

    :param evict_path: Path to eviction file.
    :type evict_path: str

    :return: A tuple of (blocks, warmup_block, offset, fill, evict_arr).
    :rtype: tuple

    :raises :exc:`NumCircBufValueError`: If required paths or shapes are missing.
    """

    blocks_full = np.memmap(data_path, dtype=dtype, mode="r", shape=data_shape)
    blocks: np.ndarray[tuple[int, int], np.dtype[ScalarT]] = blocks_full[
        :n_runs, :block_size
    ]

    warmup_data = np.memmap(
        warmup_path,
        dtype=dtype,
        mode="r",
        shape=warmup_data_shape,
    )
    warmup_block: np.ndarray[tuple[int], np.dtype[ScalarT]] = warmup_data[:maxlen]

    offset: np.ndarray[tuple[int, int] | tuple[int], np.dtype[ScalarT]]
    if single_offset:
        offset = warmup_block[: maxlen - (block_size // 2)]
    elif offset_path and offset_data_shape != (0, 0):
        offset_blocks_full = np.memmap(
            offset_path,
            dtype=dtype,
            mode="r",
            shape=offset_data_shape,
        )
        offset = offset_blocks_full[: n_runs // 2, : maxlen - (block_size // 2)]
    else:
        raise NumCircBufValueError(
            message="Multiple offset blocks requested but the "
            "offset path and/or shape was not provided or is invalid."
        )

    fill: np.ndarray[tuple[int, int], np.dtype[ScalarT]] | None
    if prepare_fill:
        if fill_path and fill_data_shape != (0, 0):
            fill_blocks_full = np.memmap(
                fill_path,
                dtype=dtype,
                mode="r",
                shape=fill_data_shape,
            )
            fill = fill_blocks_full[:n_runs, :maxlen]
        else:
            raise NumCircBufValueError(
                message="Fill blocks preparation requested but the "
                "fill path and/or shape was not provided or is invalid."
            )
    else:
        fill = None

    evict_arr: np.memmap[tuple[int], np.dtype[Scalar]] | None
    if prepare_evict:
        if evict_path:
            evict_arr = np.memmap(
                evict_path,
                dtype=EvictArrConfig.dtype,
                mode="r",
                shape=EvictArrConfig.shape,
            )
        else:
            raise NumCircBufValueError(
                message="Evict array preparation requested but the "
                "evict path was not provided or is invalid."
            )
    else:
        evict_arr = None

    return blocks, warmup_block, offset, fill, evict_arr


def _append_timed_with_calc(
    buffer: WriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
    func: Callable[[], Any],
) -> int:
    val = block.item()
    t0 = perf_counter_ns()
    buffer.append(val)
    func()
    return perf_counter_ns() - t0


def _append_timed(
    buffer: WriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
) -> int:
    val = block.item()
    t0 = perf_counter_ns()
    buffer.append(val)
    return perf_counter_ns() - t0


def _extend_timed_with_calc(
    buffer: WriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
    func: Callable[[], Any],
) -> int:
    t0 = perf_counter_ns()
    buffer.extend_unchecked(block)
    func()
    return perf_counter_ns() - t0


def _extend_timed(
    buffer: WriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
) -> int:
    t0 = perf_counter_ns()
    buffer.extend_unchecked(block)
    return perf_counter_ns() - t0


def _read_timed(buffer: ReadWriteBenchmarkBufferProtocol, _: Any) -> int:
    t0 = perf_counter_ns()
    buffer.read()
    return perf_counter_ns() - t0


def _read_into_timed(
    buffer: ReadWriteBenchmarkBufferProtocol,
    read_into_arr: np.ndarray[tuple[int], np.dtype[Scalar]],
) -> int:
    t0 = perf_counter_ns()
    buffer.read_into(read_into_arr)
    return perf_counter_ns() - t0


def _write_extend_timed(
    buffer: ReadWriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
) -> int:
    t0 = perf_counter_ns()
    buffer.write_extend_unchecked(block)
    return perf_counter_ns() - t0


def _write_append_timed(
    buffer: ReadWriteBenchmarkBufferProtocol,
    block: np.ndarray[tuple[int], np.dtype[Scalar]],
) -> int:
    val = block.item()
    t0 = perf_counter_ns()
    buffer.write_append(val)
    return perf_counter_ns() - t0


def raw_bench_with_calc(
    buffer: WriteBenchmarkBufferProtocol,
    func: Callable[[], Any],
    fill_blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    offset_blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    warmup_block: np.ndarray[tuple[int], np.dtype[Scalar]],
    blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    calc_every: int,
    n_runs: int,
    evict_arr: np.ndarray[tuple[int], np.dtype[Scalar]] | None,
    warm_cache: bool = False,
) -> int:
    """
    Execute a write benchmark that includes a calculation callback every N steps.

    This function measures the time taken to write data to a circular buffer
    while periodically executing an arbitrary function (`func`). It separates
    measurements for "wrap-around" writes and "linear" writes, returning
    the average of their trimmed means.

    :param buffer: The circular buffer instance to test.
    :type buffer: WriteBenchmarkBufferProtocol

    :param func: A callback function to execute (e.g., a signal processing task).
    :type func: Callable[[], Any]

    :param fill_blocks: Data used to pre-fill the buffer. Expected shape: `(n_runs, n)`
    :type fill_blocks: np.ndarray

    :param offset_blocks:
        Data used to force writes into a wrap-around state.
        Expected shape: `(n_runs - (n_runs // 2), n)`
    :type offset_blocks: np.ndarray

    :param warmup_block: Data used for initial buffer warmup. Expected shape: `(n,)`
    :type warmup_block: np.ndarray

    :param blocks:
        Data blocks to write during timing. Expected shape: `(n,)`
    :type blocks: np.ndarray

    :param calc_every: Frequency of the callback execution.
    :type calc_every: int

    :param n_runs: Total number of runs/iterations.
    :type n_runs: int

    :param evict_arr: Array used to flush CPU cache if provided. Expected shape: `(n,)`
    :type evict_arr: np.ndarray | None

    :param warm_cache: If True, ensures data is warm in CPU cache before timing.
    :type warm_cache: bool

    :return: The combined trimmed mean time in nanoseconds.
    :rtype: int

    :raises :exc:`NumCircBufValueError`:
        If there is a first dimension mismatch for `blocks`, `fill_blocks`,
        or `offset_blocks`
    """

    n_wrap_runs = n_runs // 2
    n_nowrap_runs = n_runs - n_wrap_runs

    if blocks.shape[0] != n_runs:
        raise NumCircBufValueError(
            message=f"`blocks` first dimension is {blocks.shape[0]}; expected {n_runs}"
        )
    if fill_blocks.shape[0] != n_runs:
        raise NumCircBufValueError(
            message=f"`fill_blocks` first dimension is {fill_blocks.shape[0]}; expected {n_runs}"
        )
    if offset_blocks.shape[0] != n_wrap_runs:
        raise NumCircBufValueError(
            message=f"`offset_blocks` first dimension is {offset_blocks.shape[0]}; expected {n_wrap_runs}"
        )

    if blocks.shape[1] == 1:
        bench_func = _append_timed
        bench_func_calc = _append_timed_with_calc
    else:
        bench_func = _extend_timed
        bench_func_calc = _extend_timed_with_calc

    set_process_priority()
    with no_gc():
        if evict_arr is not None:
            touch_pages(evict_arr, warm_cache=True)

        wrap_times = np.empty(n_wrap_runs, dtype=np.uint64)
        nowrap_times = np.empty(n_nowrap_runs, dtype=np.uint64)

        wrap_blocks = blocks[:n_wrap_runs]
        nowrap_blocks = blocks[n_wrap_runs:]

        wrap_fill = fill_blocks[:n_wrap_runs]
        nowrap_fill = fill_blocks[n_wrap_runs:]

        def make_calc_mask(
            length: int, calc_every: int
        ) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
            mask = np.zeros(length, dtype=bool)
            mask[calc_every - 1 :: calc_every] = True
            return mask

        wrap_calc_mask = make_calc_mask(n_wrap_runs, calc_every)
        nowrap_calc_mask = make_calc_mask(n_nowrap_runs, calc_every)

        buffer.extend_unchecked(warmup_block)
        for i, block in enumerate(wrap_blocks):
            buffer.clear()
            buffer.extend_unchecked(wrap_fill[i])
            buffer.extend_unchecked(offset_blocks[i])
            touch_pages(block, warm_cache=warm_cache)
            if wrap_calc_mask[i]:
                wrap_times[i] = bench_func_calc(buffer, block, func)
            else:
                wrap_times[i] = bench_func(buffer, block)
        for i, block in enumerate(nowrap_blocks):
            buffer.clear()
            buffer.extend_unchecked(nowrap_fill[i])
            touch_pages(block, warm_cache=warm_cache)
            if nowrap_calc_mask[i]:
                nowrap_times[i] = bench_func_calc(buffer, block, func)
            else:
                nowrap_times[i] = bench_func(buffer, block)
    return (
        trimmed_mean_times(wrap_times, return_int=True)
        + trimmed_mean_times(nowrap_times, return_int=True)
    ) // 2


def raw_bench(
    buffer: WriteBenchmarkBufferProtocol,
    offset_blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    warmup_block: np.ndarray[tuple[int], np.dtype[Scalar]],
    blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    n_runs: int,
    evict_arr: np.ndarray[tuple[int], np.dtype[Scalar]] | None,
    warm_cache: bool = False,
) -> int:
    """
    Execute a standard write benchmark for the circular buffer.

    Measures the performance of `append` or `extend_unchecked` by alternating
    between linear and wrap-around write scenarios.

    :param buffer: The circular buffer instance to test.
    :type buffer: WriteBenchmarkBufferProtocol

    :param offset_blocks:
        Data used to force writes into a wrap-around state.
        Expected shape: `(n_runs - (n_runs // 2), n)`
    :type offset_blocks: np.ndarray

    :param warmup_block: Data used for initial buffer warmup. Expected shape: `(n,)`
    :type warmup_block: np.ndarray

    :param blocks:
        Data blocks to write during timing. Expected shape: `(n,)`
    :type blocks: np.ndarray

    :param n_runs: Total number of runs/iterations.
    :type n_runs: int

    :param evict_arr: Array used to flush CPU cache if provided. Expected shape: `(n,)`
    :type evict_arr: np.ndarray | None

    :param warm_cache: If True, ensures data is warm in CPU cache before timing.
    :type warm_cache: bool

    :return: Combined trimmed mean time in nanoseconds.
    :rtype: int

    :raises :exc:`NumCircBufValueError`:
        If there is a first dimension mismatch for `blocks`, or `offset_blocks`
    """

    n_wrap_runs = n_runs // 2
    n_nowrap_runs = n_runs - n_wrap_runs

    if blocks.shape[0] != n_runs:
        raise NumCircBufValueError(
            message=f"`blocks` first dimension is {blocks.shape[0]}; expected {n_runs}"
        )
    if offset_blocks.shape[0] != n_wrap_runs:
        raise NumCircBufValueError(
            message=f"`offset_blocks` first dimension is {offset_blocks.shape[0]}; expected {n_wrap_runs}"
        )

    if blocks.shape[1] == 1:
        bench_func = _append_timed
    else:
        bench_func = _extend_timed

    set_process_priority()
    with no_gc():
        if evict_arr is not None:
            touch_pages(evict_arr, warm_cache=True)

        wrap_times = np.empty(n_wrap_runs, dtype=np.uint64)
        nowrap_times = np.empty(n_nowrap_runs, dtype=np.uint64)

        wrap_blocks = blocks[:n_wrap_runs]
        nowrap_blocks = blocks[n_wrap_runs:]

        buffer.extend_unchecked(warmup_block)
        for i, block in enumerate(wrap_blocks):
            buffer.clear()
            buffer.extend_unchecked(offset_blocks[i])
            touch_pages(block, warm_cache=warm_cache)
            wrap_times[i] = bench_func(buffer, block)
        for i, block in enumerate(nowrap_blocks):
            buffer.clear()
            touch_pages(block, warm_cache=warm_cache)
            nowrap_times[i] = bench_func(buffer, block)

    return (
        trimmed_mean_times(wrap_times, return_int=True)
        + trimmed_mean_times(nowrap_times, return_int=True)
    ) // 2


def raw_bench_write_read(
    buffer: ReadWriteBenchmarkBufferProtocol,
    offset_blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    warmup_block: np.ndarray[tuple[int], np.dtype[Scalar]],
    blocks: np.ndarray[tuple[int, int], np.dtype[Scalar]],
    n_runs: int,
    evict_arr: np.ndarray[tuple[int], np.dtype[Scalar]] | None,
    warm_cache: bool = False,
    read_into_arr: np.ndarray[tuple[int], np.dtype[Scalar]] | None = None,
) -> tuple[int, int]:
    """
    Execute a combined write and read benchmark.

    Sequentially times a write operation followed by a read operation. This
    simulates a real-world producer-consumer cycle.

    :param buffer: The circular buffer instance to test.
    :type buffer: ReadWriteBenchmarkBufferProtocol

    :param offset_blocks:
        Data used to force writes into a wrap-around state.
        Expected shape: `(n_runs - (n_runs // 2), n)`
    :type offset_blocks: np.ndarray

    :param warmup_block: Data used for initial buffer warmup. Expected shape: `(n,)`
    :type warmup_block: np.ndarray

    :param blocks:
        Data blocks to write during timing. Expected shape: `(n,)`
    :type blocks: np.ndarray

    :param n_runs: Total number of runs/iterations.
    :type n_runs: int

    :param evict_arr: Array used to flush CPU cache if provided. Expected shape: `(n,)`
    :type evict_arr: np.ndarray | None

    :param warm_cache: If True, ensures data is warm in CPU cache before timing.
    :type warm_cache: bool

    :param read_into_arr:
        If provided, uses `read_into`. If None, uses `read`.
        Defaults to None.
    :type read_into_arr: np.ndarray | None, optional

    :return: A tuple of (mean_write_time, mean_read_time) in nanoseconds.
    :rtype: tuple[int, int]

    :raises :exc:`NumCircBufValueError`:
        If there is a first dimension mismatch for `blocks`, or `offset_blocks`
    """

    n_wrap_runs = n_runs // 2
    n_nowrap_runs = n_runs - n_wrap_runs

    if blocks.shape[0] != n_runs:
        raise NumCircBufValueError(
            message=f"`blocks` first dimension is {blocks.shape[0]}; expected {n_runs}"
        )
    if offset_blocks.shape[0] != n_wrap_runs:
        raise NumCircBufValueError(
            message=f"`offset_blocks` first dimension is {offset_blocks.shape[0]}; expected {n_wrap_runs}"
        )

    if blocks.shape[1] == 1:
        write_bench_func = _write_append_timed
    else:
        write_bench_func = _write_extend_timed

    if read_into_arr is None:
        read_bench_func: Callable[[ReadWriteBenchmarkBufferProtocol, Any], int] = (
            _read_timed
        )
    else:
        read_bench_func = _read_into_timed

    set_process_priority()
    with no_gc():
        if evict_arr is not None:
            touch_pages(evict_arr, warm_cache=True)

        wrap_write_times = np.empty(n_wrap_runs, dtype=np.uint64)
        wrap_read_times = np.empty(n_wrap_runs, dtype=np.uint64)

        nowrap_write_times = np.empty(n_nowrap_runs, dtype=np.uint64)
        nowrap_read_times = np.empty(n_nowrap_runs, dtype=np.uint64)

        wrap_blocks = blocks[:n_wrap_runs]
        nowrap_blocks = blocks[n_wrap_runs:]

        buffer.clear()
        buffer.write_extend_unchecked(warmup_block)

        for i, block in enumerate(wrap_blocks):
            buffer.clear()
            buffer.write_extend_unchecked(offset_blocks[i])
            buffer.read()
            touch_pages(block, warm_cache=warm_cache)
            wrap_write_times[i] = write_bench_func(buffer, block)
            wrap_read_times[i] = read_bench_func(buffer, read_into_arr)
        for i, block in enumerate(nowrap_blocks):
            buffer.clear()
            touch_pages(block, warm_cache=warm_cache)
            nowrap_write_times[i] = write_bench_func(buffer, block)
            nowrap_read_times[i] = read_bench_func(buffer, read_into_arr)

    return (
        trimmed_mean_times(wrap_write_times, return_int=True)
        + trimmed_mean_times(nowrap_write_times, return_int=True)
    ) // 2, (
        trimmed_mean_times(wrap_read_times, return_int=True)
        + trimmed_mean_times(nowrap_read_times, return_int=True)
    ) // 2


class BenchLogger:
    """
    Utility for logging benchmark results to a file and console.

    Encapsulates timestamping and filesystem synchronization (`fsync`) to
    ensure logs are preserved even in the event of a system crash.
    """

    def __init__(self, file_name: str):
        """
        Initialize the logger and create the log file with a header.

        :param file_name: Path to the log file.
        :type file_name: str
        """
        self.file_name = file_name
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(f"Logging started at {datetime.now()}\n")
            f.write("=" * 80 + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log(self, msg: str, to_console: bool = False) -> None:
        """
        Write a timestamped message to the log file and optionally the console.

        :param msg: Message to log.
        :type msg: str

        :param to_console: If True, also prints the message to stdout.
        :type to_console: bool
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        with open(self.file_name, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        if to_console:
            print(line.strip(), flush=True)
            sys.stdout.flush()


class RefPythonNumPyCircBuffer:
    """
    An optimized reference implementation of a circular buffer using pure Python and NumPy.

    This class serves as a baseline for performance comparisons. It implements
    optimized circular buffer logic with manual head management and adds padding
    to reduce CPU cache index aliasing and conflict misses.
    """

    __slots__ = ["_raw", "buffer", "maxlen", "size", "write_head"]

    def __init__(
        self,
        maxlen: int,
        return_overwritten_policy: str | None = None,
        dtype: type[Scalar] = np.float32,
        *,
        cache_line_size: int | None = None,
    ) -> None:
        """
        Initialize the reference circular buffer.

        :param maxlen: Maximum number of elements the buffer can hold.
        :type maxlen: int

        :param return_overwritten_policy: Ignored (maintained for API compatibility).
        :type return_overwritten_policy: str | None, optional

        :param dtype: NumPy scalar type for the data.
        :type dtype: type[np.floating] | type[np.integer]

        :param cache_line_size:
            Size of cache line for memory alignment.
            If None, detected automatically.
        :type cache_line_size: int | None, optional
        """

        if cache_line_size is None:
            global _cache_line_size
            if _cache_line_size is None:
                _cache_line_size = get_cache_line_size()
            cache_line_size = _cache_line_size

        itemsize = np.dtype(dtype).itemsize
        offset = (cache_line_size * 2) // itemsize
        padding = offset + (cache_line_size // itemsize)

        self._raw: np.ndarray = np.zeros(maxlen + padding, dtype=dtype)
        self.buffer = self._raw[offset : offset + maxlen]

        self.write_head = 0
        self.size = 0
        self.maxlen = maxlen

    def append(self, value: float) -> None:
        """
        Add a single value to the buffer, overwriting the oldest data if full.

        :param value: Numeric value to append.
        :type value: float | int
        """

        write_head = self.write_head
        maxlen = self.maxlen

        if self.size < maxlen:
            self.size += 1

        self.buffer[write_head] = value

        write_head += 1
        if write_head >= maxlen:
            write_head = 0
        self.write_head = write_head

    def extend_unchecked(
        self, block_np: np.ndarray[tuple[int], np.dtype[Scalar]]
    ) -> None:
        """
        Add multiple values from a NumPy array to the buffer.

        Does not perform bounds checking on the input array size relative
        to maxlen.

        :param block_np: Array of values to add.
        :type block_np: np.ndarray
        """

        write_head = self.write_head
        maxlen = self.maxlen
        n = len(block_np)

        if not n:
            return

        if write_head + n <= maxlen:
            self.buffer[write_head : write_head + n] = block_np
        else:
            buffer = self.buffer
            first_part = maxlen - write_head
            buffer[write_head : write_head + first_part] = block_np[:first_part]
            buffer[: n - first_part] = block_np[first_part:]

        write_head += n
        if write_head >= maxlen:
            write_head -= maxlen
        self.write_head = write_head

        if self.size + n <= maxlen:
            self.size += n

    def clear(self) -> None:
        """Reset the buffer's write head and logical size to zero."""

        self.write_head = 0
        self.size = 0
