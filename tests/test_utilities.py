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

from __future__ import annotations
import os
import sys
from io import StringIO
from pathlib import Path
import time
from itertools import product
import ctypes
import platform  # noqa: F401
import logging
import warnings

import numpy as np
import pytest

from numcircbuf import (
    system_info,
    determine_operation_focus,
    RunningMeanSqBuffer,
    RunningMeanBuffer,
    OverwriteCircBuffer,
    BlockingCircBuffer,
    bench_utils,
    constants,
)
from numcircbuf.exceptions import (
    NumCircBufValueError,
    NumCircBufTypeError,
    NumCircBufDeprecationWarning,
    NumCircBufFutureWarning,
    NumCircBufArithmeticError,
    NumCircBufError,
    NumCircBufWarning,
    NumCircBufRuntimeError,
    NumCircBufOSError,
)

from .constants import SUPPORTED_DTYPES_ALL, SUPPORTED_DTYPES_FP

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

RS = 25


def time_func(func):
    start = time.perf_counter_ns()
    res = func()
    delta = time.perf_counter_ns() - start
    return delta, res


def _helper(func):
    times_cache_hit = []
    times_no_hit = []
    results = []

    for _ in range(100):
        func.cache_clear()
        time_1, res_1 = time_func(func)
        time_2, res_2 = time_func(func)

        times_no_hit.append(time_1)
        times_cache_hit.append(time_2)
        results.extend([res_1, res_2])

    assert min(times_cache_hit) < min(times_no_hit)
    for res in results:
        assert res == results[0]
        assert isinstance(res, int)


def test_page_size():
    assert isinstance(system_info.PAGESIZE, int)


def test_get_cache_line_size():
    _helper(system_info.get_cache_line_size)


def test_get_cpu_l3_cache_mib():
    _helper(system_info.get_cpu_l3_cache)


def test_get_available_ram():
    res = system_info.get_available_ram()
    assert isinstance(res, int) or res is None


@pytest.mark.parametrize("buffer_type", (RunningMeanSqBuffer, RunningMeanBuffer))
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
@pytest.mark.parametrize("block_size", (1, 2))
@pytest.mark.parametrize("verbose", (True, False))
def test_determine_operation_focus(caplog, buffer_type, dtype, block_size, verbose):
    if verbose:
        caplog.set_level(logging.INFO)

    expected_results = ("calculation", "extend/append")
    res = determine_operation_focus(
        buffer_type=buffer_type,
        dtype=dtype,
        buffer_maxlen=4,
        block_size=block_size,
        calc_every=2,
        verbose=verbose,
    )
    assert res in expected_results

    if verbose:
        assert any(
            r.levelno == logging.INFO and "Speed Comparison" in r.message
            for r in caplog.records
        )


def test_determine_operation_focus_exceptions():
    cases = (
        # calc_every <= 0
        (NumCircBufValueError, RunningMeanBuffer, np.float64, 10, 5, 0),
        (NumCircBufValueError, RunningMeanBuffer, np.float64, 10, 5, -1),
        # block_size <= 0
        (NumCircBufValueError, RunningMeanBuffer, np.float64, 10, 0, 5),
        (NumCircBufValueError, RunningMeanBuffer, np.float64, 10, -5, 5),
        # buffer_maxlen <= 2
        (NumCircBufValueError, RunningMeanBuffer, np.float64, 2, 5, 5),
        (NumCircBufValueError, RunningMeanBuffer, np.float64, -1, 5, 5),
        # buffer_maxlen > PY_SSIZE_T_MAX
        (
            NumCircBufValueError,
            RunningMeanBuffer,
            np.float64,
            constants.Limits.PY_SSIZE_T_MAX.value + 1,
            5,
            5,
        ),
        # dtype not np.float32 or np.float64
        (NumCircBufTypeError, RunningMeanBuffer, np.int32, 10, 5, 5),
        (NumCircBufTypeError, RunningMeanBuffer, float, 10, 5, 5),
        # buffer_type not a subclass of RunningMeanBuffer or RunningMeanSqBuffer
        (NumCircBufTypeError, OverwriteCircBuffer, np.float64, 10, 5, 5),
        (NumCircBufTypeError, BlockingCircBuffer, np.float64, 10, 5, 5),
    )
    for (
        exc,
        buffer_type,
        dtype,
        buffer_maxlen,
        block_size,
        calc_every,
    ) in cases:
        with pytest.raises(exc) as exc_info:
            determine_operation_focus(
                buffer_type=buffer_type,
                dtype=dtype,
                buffer_maxlen=buffer_maxlen,
                block_size=block_size,
                calc_every=calc_every,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


def test_determine_operation_focus_fallback():
    # buffer_maxlen < block_size
    result = determine_operation_focus(
        buffer_type=RunningMeanBuffer,
        dtype=np.float64,
        buffer_maxlen=10,
        block_size=20,
        calc_every=5,
    )
    assert result == "extend/append"

    # maxlen asks for too much memory
    result = determine_operation_focus(
        buffer_type=RunningMeanBuffer,
        dtype=np.float64,
        buffer_maxlen=10**9,
        block_size=10,
        calc_every=5,
    )
    assert result == "extend/append"


def test_evict_arr():
    initial_bytes = bench_utils.EvictArrConfig.bytes
    assert isinstance(initial_bytes, int)
    new_bytes = 1024
    bench_utils.EvictArrConfig.set_bytes(new_bytes)
    assert bench_utils.EvictArrConfig.bytes == new_bytes

    initial_dtype = bench_utils.EvictArrConfig.dtype
    assert issubclass(initial_dtype, (np.floating, np.integer))
    new_dtype = np.float16
    bench_utils.EvictArrConfig.set_dtype(new_dtype)
    assert bench_utils.EvictArrConfig.dtype is new_dtype
    assert bench_utils.EvictArrConfig.shape == (
        (new_bytes // np.dtype(new_dtype).itemsize),
    )

    bench_utils.EvictArrConfig.set_bytes(initial_bytes)
    bench_utils.EvictArrConfig.set_dtype(initial_dtype)


def test_get_cpu_name():
    res = bench_utils.get_cpu_name()
    assert isinstance(res, str)


def test_set_process_priority():
    for priority in ("normal", "high"):
        bench_utils.set_process_priority(priority)


def test_determine_num_runs():
    n_runs, block_size, maxlen = bench_utils.determine_num_runs(
        elem_bytes=8,
        total_byte_limit=1024,
        maxlen_byte_limit=24,
        block_byte_limit=16,
        with_fill=True,
    )
    assert all(isinstance(x, int) for x in (n_runs, block_size, maxlen))
    assert n_runs == 20
    assert block_size == 2
    assert maxlen == 3

    n_runs, block_size, maxlen = bench_utils.determine_num_runs(
        elem_bytes=1,
        total_byte_limit=48,
        maxlen=10,
        block_size=4,
        with_fill=False,
    )
    assert all(isinstance(x, int) for x in (n_runs, block_size, maxlen))
    assert n_runs == 3
    assert block_size == 4
    assert maxlen == 10


def test_determine_num_runs_exceptions():
    for total_byte_limit, maxlen, block_size in (
        (1024, None, 2),  # block_size and block_byte_limit are None
        (1024, 4, None),  # maxlen and maxlen_byte_limit are None
        (1024, 4, -1),  # block_size <= 0
        (1024, 2, 1),  # maxlen <= 2
        (1024, 4, 8),  # maxlen < block_size
        (8, 4, 2),  # total_byte_limit too low
    ):
        with pytest.raises(NumCircBufValueError) as exc_info:
            bench_utils.determine_num_runs(
                elem_bytes=8,
                total_byte_limit=total_byte_limit,
                maxlen=maxlen,
                block_size=block_size,
                with_fill=True,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_generate_rand_memmap_arr_not_covered(dtype):
    path = "temp_test_generate_rand_memmap_arr.dat"

    ss = np.random.SeedSequence(RS)
    child_ss = ss.spawn(1)
    rng = np.random.default_rng(child_ss[0])

    test_args_tuple = (
        (None, 1),
        (2, None),
        (None, None),
        (0, None),
    )

    if not np.issubdtype(dtype, np.unsignedinteger):
        test_args_tuple += ((-1, None),)

    for multiply_by, subtract in test_args_tuple:
        arr = bench_utils.generate_rand_memmap_arr(
            dtype,
            (1,),
            path,
            rng,
            multiply_by=multiply_by,
            subtract=subtract,
        )

        assert isinstance(arr, np.memmap)
        assert arr.dtype == dtype
        assert arr.shape == (1,)

        try:
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        except Exception:
            pass

        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.warning(
                f"--- Could not delete temp file. Delete manually. ---\nPath: {path}",
                exc_info=True,
            )


def test_generate_rand_memmap_arr_exceptions():
    path = "temp_test_generate_rand_memmap_arr.dat"

    ss = np.random.SeedSequence(RS)
    child_ss = ss.spawn(1)
    rng = np.random.default_rng(child_ss[0])

    for multiply_by, subtract, dtype, exception in (
        (
            1000,
            10500,
            np.uint8,
            NumCircBufValueError,
        ),
        (
            0,
            1,
            np.uint8,
            NumCircBufValueError,
        ),
        (
            None,
            2.5,
            np.uint8,
            NumCircBufValueError,
        ),
        (
            2.5,
            None,
            np.uint8,
            NumCircBufValueError,
        ),
        (
            -1,
            None,
            np.uint64,
            NumCircBufValueError,
        ),
        (
            None,
            None,
            np.str_,
            NumCircBufTypeError,
        ),
    ):
        with pytest.raises(exception) as exc_info:
            bench_utils.generate_rand_memmap_arr(
                dtype,
                (1,),
                path,
                rng,
                multiply_by=multiply_by,
                subtract=subtract,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message

        del exc, exc_info

        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.warning(
                f"--- Could not delete temp file. Delete manually. ---\nPath: {path}",
                exc_info=True,
            )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_generate_bench_memmaps(dtype):
    maxlen = 2
    block_size = 2
    num_blocks = 2

    expected_shape = {
        "data": (num_blocks, block_size),
        "warmup_data": (maxlen,),
        "offset_data": (num_blocks // 2, maxlen - (block_size // 2)),
        "fill_data": (num_blocks, maxlen),
        "evict_arr": bench_utils.EvictArrConfig.shape,
    }

    for create_offset_data, create_fill_data, create_evict_arr in tuple(
        product([True, False], repeat=3)
    ):
        (
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
        ) = bench_utils.generate_bench_memmaps(
            dtype,
            2,
            2,
            2,
            create_offset_data=create_offset_data,
            create_fill_data=create_fill_data,
            create_evict_arr=create_evict_arr,
        )

        cases = [
            ("data", data, data_path, True),
            ("warmup_data", warmup_data, warmup_path, True),
            ("offset_data", offset_data, offset_path, create_offset_data),
            ("fill_data", fill_data, fill_path, create_fill_data),
            ("evict_arr", evict_arr, evict_path, create_evict_arr),
        ]

        for name, arr, path, valid in cases:
            if valid:
                assert isinstance(arr, np.memmap)
                assert (
                    arr.dtype == dtype
                    if arr is not evict_arr
                    else arr.dtype == bench_utils.EvictArrConfig.dtype
                )
                assert arr.shape == expected_shape[name]
                assert isinstance(path, str)
                assert os.path.exists(path)
            else:
                assert arr is None
                assert path == ""

        for _, arr, _, valid in cases:
            try:
                if valid:
                    mmap_obj = getattr(arr, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
            except Exception:
                pass

        for _, _, path, valid in cases:
            try:
                if valid:
                    os.remove(path)
            except Exception:
                logger.warning(
                    "--- Could not delete temp file. Delete manually. ---\n"
                    f"Path: {path}",
                    exc_info=True,
                )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_raw_bench_with_calc_exceptions(dtype):
    def _mock_func(): ...

    n_runs = 2
    irrelevant_dim_size = 1

    buf = RunningMeanSqBuffer(irrelevant_dim_size, "extend/append", dtype=dtype)
    warmup_block = np.empty((n_runs, irrelevant_dim_size), dtype=dtype)

    for blocks, fill_blocks, offset_blocks in (
        (
            np.empty((n_runs + 1, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs // 2, irrelevant_dim_size), dtype=dtype),
        ),
        (
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs + 1, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs // 2, irrelevant_dim_size), dtype=dtype),
        ),
        (
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
        ),
    ):
        with pytest.raises(NumCircBufValueError) as exc_info:
            bench_utils.raw_bench_with_calc(
                buf,
                _mock_func,
                fill_blocks,
                offset_blocks,
                warmup_block,
                blocks,
                calc_every=1,
                n_runs=n_runs,
                evict_arr=None,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_raw_bench_exceptions(dtype):
    n_runs = 2
    irrelevant_dim_size = 1

    buf = OverwriteCircBuffer(irrelevant_dim_size, "never", dtype=dtype)
    warmup_block = np.empty((n_runs, irrelevant_dim_size), dtype=dtype)

    for blocks, offset_blocks in (
        (
            np.empty((n_runs + 1, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs // 2, irrelevant_dim_size), dtype=dtype),
        ),
        (
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
        ),
    ):
        with pytest.raises(NumCircBufValueError) as exc_info:
            bench_utils.raw_bench(
                buf,
                offset_blocks,
                warmup_block,
                blocks,
                n_runs=n_runs,
                evict_arr=None,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_raw_bench_write_read_exceptions(dtype):
    n_runs = 2
    irrelevant_dim_size = 1

    buf = BlockingCircBuffer(irrelevant_dim_size, dtype=dtype)
    warmup_block = np.empty((n_runs, irrelevant_dim_size), dtype=dtype)

    for blocks, offset_blocks in (
        (
            np.empty((n_runs + 1, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs // 2, irrelevant_dim_size), dtype=dtype),
        ),
        (
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
            np.empty((n_runs, irrelevant_dim_size), dtype=dtype),
        ),
    ):
        with pytest.raises(NumCircBufValueError) as exc_info:
            bench_utils.raw_bench_write_read(
                buf,
                offset_blocks,
                warmup_block,
                blocks,
                n_runs=n_runs,
                evict_arr=None,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


def test_generate_bench_memmaps_exceptions():
    for n_runs in (0, -1):
        with pytest.raises(NumCircBufValueError) as exc_info:
            bench_utils.generate_bench_memmaps(
                np.float64,
                1,
                1,
                n_runs,
                create_offset_data=False,
                create_fill_data=False,
                create_evict_arr=False,
            )

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


def _helper_mocker_edge_cases(
    *,
    mocker,
    platform_name: str,
    patch_target_func: str | None,
    call_target_func,
    return_val=None,
    supports_default_overwriting: bool,
    has_cache: bool,
):
    mocker.patch("sys.platform", platform_name)

    if supports_default_overwriting:
        return_val = default_val = -1

    if patch_target_func is not None:
        mocked_query = mocker.patch(patch_target_func)

        cases = [
            ("OSError", OSError("Mock OSError")),
            ("None return", None),
            ("Zero return", 0),
            ("Negative return", -1),
        ]

        for label, value in cases:
            if has_cache:
                call_target_func.cache_clear()

            if isinstance(value, Exception):
                mocked_query.side_effect = value
            else:
                mocked_query.side_effect = None
                mocked_query.return_value = value

            result = (
                call_target_func(default=default_val)
                if supports_default_overwriting
                else call_target_func()
            )
            assert result == return_val, (
                f"Failed on {label}: expected {default_val}, got {result}"
            )

    else:
        if has_cache:
            call_target_func.cache_clear()

        result = (
            call_target_func(default=default_val)
            if supports_default_overwriting
            else call_target_func()
        )
        assert result == return_val, (
            f"Failed on 'unknown' platform: expected {default_val}, got {result}"
        )


@pytest.mark.parametrize(
    "platform_name, target_func",
    (
        ("win32", "_get_cache_line_size_windows"),
        ("linux", "_get_cache_line_size_linux"),
        ("darwin", "_get_cache_line_size_darwin"),
        ("unknown", None),
    ),
)
def test_get_cache_line_size_edge_cases(mocker, platform_name, target_func):
    _helper_mocker_edge_cases(
        mocker=mocker,
        platform_name=platform_name,
        patch_target_func=f"numcircbuf.system_info.{target_func}",
        call_target_func=system_info.get_cache_line_size,
        supports_default_overwriting=True,
        has_cache=True,
    )


@pytest.mark.parametrize(
    "platform_name, target_func",
    (
        ("win32", "_get_l3_win32"),
        ("linux", "_get_l3_linux"),
        ("darwin", "_get_l3_darwin"),
        ("unknown", None),
    ),
)
def test_get_cpu_l3_cache_mib_edge_cases(mocker, platform_name, target_func):
    _helper_mocker_edge_cases(
        mocker=mocker,
        platform_name=platform_name,
        patch_target_func=f"numcircbuf.system_info.{target_func}",
        call_target_func=system_info.get_cpu_l3_cache,
        supports_default_overwriting=True,
        has_cache=True,
    )


@pytest.mark.parametrize(
    "platform_name, target_func",
    (
        ("win32", "_get_available_ram_windows"),
        ("linux", "_get_available_ram_linux"),
        ("darwin", "_get_available_ram_darwin"),
        ("unknown", None),
    ),
)
def test_get_available_ram_edge_cases(mocker, platform_name, target_func):
    _helper_mocker_edge_cases(
        mocker=mocker,
        platform_name=platform_name,
        patch_target_func=f"numcircbuf.system_info.{target_func}",
        call_target_func=system_info.get_available_ram,
        supports_default_overwriting=False,
        has_cache=False,
    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_temporary_benchmark_data(dtype):
    test_data_path = "_test_temp_bench_data.dat"
    test_warmup_path = "_test_temp_bench_warmup.dat"
    test_offset_path = "_test_temp_bench_offset.dat"
    test_fill_path = "_test_temp_bench_fill.dat"
    test_evict_path = "_test_temp_bench_evict.dat"

    buffer_maxlen, block_size, n_runs = 4, 2, 4

    evict_arr_dtype, evict_arr_shape = (
        bench_utils.EvictArrConfig.dtype,
        bench_utils.EvictArrConfig.shape,
    )

    with bench_utils.temporary_benchmark_data(
        dtype=dtype,
        buffer_maxlen=buffer_maxlen,
        block_size=block_size,
        n_runs=n_runs,
        create_offset_data=True,
        create_fill_data=True,
        create_evict_arr=True,
        data_path=test_data_path,
        warmup_path=test_warmup_path,
        offset_path=test_offset_path,
        fill_path=test_fill_path,
        evict_path=test_evict_path,
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
        evict_arr,
    ):
        for out_path, in_path in (
            (data_path, test_data_path),
            (warmup_path, test_warmup_path),
            (offset_path, test_offset_path),
            (fill_path, test_fill_path),
            (evict_path, test_evict_path),
        ):
            assert Path(out_path).is_file()
            assert out_path == in_path

        for arr, expected_dtype, expected_shape in (
            (
                data,
                dtype,
                (n_runs, block_size),
            ),
            (
                warmup_data,
                dtype,
                (buffer_maxlen,),
            ),
            (
                offset_data,
                dtype,
                (n_runs // 2, buffer_maxlen - (block_size // 2)),
            ),
            (
                fill_data,
                dtype,
                (n_runs, buffer_maxlen),
            ),
            (
                evict_arr,
                evict_arr_dtype,
                evict_arr_shape,
            ),
        ):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == expected_dtype
            assert arr.shape == expected_shape


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_raw_bench(dtype):
    buffer_maxlen, n_runs = 4, 4
    for mode in ("never", "always", "conditional"):
        buffer = OverwriteCircBuffer(buffer_maxlen, mode, dtype)
        for block_size in (2, 1):
            with bench_utils.temporary_benchmark_data(
                dtype=dtype,
                buffer_maxlen=buffer_maxlen,
                block_size=block_size,
                n_runs=n_runs,
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
                for warm_cache in (True, False):
                    bench_utils.raw_bench(
                        buffer,
                        offset_data,
                        warmup_data,
                        data,
                        n_runs,
                        evict_arr,
                        warm_cache=warm_cache,
                    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_raw_bench_write_read(dtype):
    buffer_maxlen, n_runs = 4, 4
    buffer = BlockingCircBuffer(buffer_maxlen, dtype)
    for block_size in (2, 1):
        with bench_utils.temporary_benchmark_data(
            dtype=dtype,
            buffer_maxlen=buffer_maxlen,
            block_size=block_size,
            n_runs=n_runs,
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
            for read_into_arr in (None, np.zeros(buffer_maxlen, dtype=dtype)):
                for warm_cache in (True, False):
                    bench_utils.raw_bench_write_read(
                        buffer,
                        offset_data,
                        warmup_data,
                        data,
                        n_runs,
                        evict_arr,
                        warm_cache=warm_cache,
                        read_into_arr=read_into_arr,
                    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_reference_buffer(dtype):
    maxlen = 4

    bench_utils._cache_line_size = None
    buf = bench_utils.RefPythonNumPyCircBuffer(maxlen, dtype=dtype)
    assert buf.maxlen == maxlen
    assert not np.any(buf.buffer)

    buf.append(1)
    assert buf.write_head == 1
    assert buf.size == 1
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 1

    buf.extend_unchecked(np.array([2, 3], dtype=dtype))
    assert buf.write_head == 3
    assert buf.size == 3
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 3

    buf.append(4)
    assert buf.write_head == 0
    assert buf.size == 4
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 4

    buf.extend_unchecked(np.array([5], dtype=dtype))
    assert buf.write_head == 1
    assert buf.size == 4
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 5

    buf.extend_unchecked(np.array([6, 7, 8, 9], dtype=dtype))
    assert buf.write_head == 1
    assert buf.size == 4
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 9

    buf.extend_unchecked(np.array([], dtype=dtype))
    assert buf.write_head == 1
    assert buf.size == 4
    assert buf.buffer[(buf.write_head - 1) % maxlen] == 9

    buf.clear()
    assert buf.write_head == 0
    assert buf.size == 0


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_prepare_blocks(dtype):
    buffer_maxlen, block_size, n_runs = 12, 4, 8

    evict_arr_dtype, evict_arr_shape = (
        bench_utils.EvictArrConfig.dtype,
        bench_utils.EvictArrConfig.shape,
    )

    def _run_test_cases_in_context():
        for test_maxlen, test_block_size, test_n_runs in (
            (buffer_maxlen, block_size, n_runs),
            (buffer_maxlen // 2, block_size // 2, n_runs // 2),
            (buffer_maxlen // 4, block_size // 4, n_runs // 4),
        ):
            for single_offset, prepare_fill, prepare_evict in tuple(
                product([True, False], repeat=3)
            ):
                (
                    test_data,
                    test_warmup_data,
                    test_offset_data,
                    test_fill_data,
                    test_evict_arr,
                ) = bench_utils.prepare_blocks(
                    test_block_size,
                    test_maxlen,
                    dtype,
                    test_n_runs,
                    data_path,
                    data.shape,
                    warmup_path,
                    warmup_data.shape,
                    single_offset=single_offset,
                    offset_path=offset_path,
                    offset_data_shape=offset_data.shape,
                    prepare_fill=prepare_fill,
                    fill_path=fill_path,
                    fill_data_shape=fill_data.shape,
                    prepare_evict=prepare_evict,
                    evict_path=evict_path,
                )
                cases = (
                    (
                        test_data,
                        dtype,
                        (test_n_runs, test_block_size),
                    ),
                    (
                        test_warmup_data,
                        dtype,
                        (test_maxlen,),
                    ),
                )

                if single_offset:
                    cases += (
                        (
                            test_offset_data,
                            dtype,
                            (test_maxlen - (test_block_size // 2),),
                        ),
                    )
                else:
                    cases += (
                        (
                            test_offset_data,
                            dtype,
                            (
                                test_n_runs // 2,
                                test_maxlen - (test_block_size // 2),
                            ),
                        ),
                    )

                if prepare_fill:
                    cases += (
                        (
                            test_fill_data,
                            dtype,
                            (test_n_runs, test_maxlen),
                        ),
                    )

                if prepare_evict:
                    cases += (
                        (
                            test_evict_arr,
                            evict_arr_dtype,
                            evict_arr_shape,
                        ),
                    )

                for arr, expected_dtype, expected_shape in cases:
                    assert isinstance(arr, np.ndarray)
                    assert arr.dtype == expected_dtype
                    assert arr.shape == expected_shape

    def _run_exception_cases_in_context():
        for (
            _offset_path,
            offset_data_shape,
            _fill_path,
            fill_data_shape,
            _evict_path,
        ) in (
            ("", offset_data.shape, fill_path, fill_data.shape, evict_path),
            (offset_path, (0, 0), fill_path, fill_data.shape, evict_path),
            (offset_path, offset_data.shape, "", fill_data.shape, evict_path),
            (offset_path, offset_data.shape, fill_path, (0, 0), evict_path),
            (offset_path, offset_data.shape, fill_path, fill_data.shape, ""),
        ):
            with pytest.raises(NumCircBufValueError) as exc_info:
                bench_utils.prepare_blocks(
                    block_size,
                    buffer_maxlen,
                    dtype,
                    n_runs,
                    data_path,
                    data.shape,
                    warmup_path,
                    warmup_data.shape,
                    single_offset=False,
                    offset_path=_offset_path,
                    offset_data_shape=offset_data_shape,
                    prepare_fill=True,
                    fill_path=_fill_path,
                    fill_data_shape=fill_data_shape,
                    prepare_evict=True,
                    evict_path=_evict_path,
                )

            exc = exc_info.value
            assert exc.class_obj is None
            assert exc.obj is None
            assert exc.message

            del exc, exc_info

    with bench_utils.temporary_benchmark_data(
        dtype=dtype,
        buffer_maxlen=buffer_maxlen,
        block_size=block_size,
        n_runs=n_runs,
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
        evict_arr,
    ):
        _run_test_cases_in_context()
        _run_exception_cases_in_context()


def test_bench_logger():
    log_file = Path("tmp") / "logs" / "bench.log"

    logger = bench_utils.BenchLogger(str(log_file))

    logger.log("first message")
    logger.log("second message", to_console=True)

    assert log_file.exists()

    content = log_file.read_text().splitlines()

    assert "Logging started at" in content[0]
    assert "=" * 80 in content[1]

    assert any("first message" in line for line in content)
    assert any("second message" in line for line in content)

    assert content[-1].endswith("second message")


def test_get_cache_line_linux_from_proc_cpuinfo(mocker):

    def mock_exists(path):
        if "/sys/devices/system/cpu/cpu0/cache/index0" in path:
            return False
        if path == "/proc/cpuinfo":
            return True
        return False

    mocker.patch("os.path.exists", side_effect=mock_exists)

    mock_data = "processor: 0\nvendor_id: GenuineIntel\nclflush size: 128\n"
    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_data))

    assert system_info._get_cache_line_size_linux() == 128


def test_get_cache_line_linux_returns_none_if_no_files_exist(mocker):
    mocker.patch("os.path.exists", return_value=False)

    assert system_info._get_cache_line_size_linux() is None


def test_get_l3_linux_m_suffix(mocker):
    mocker.patch("os.path.isdir", return_value=True)

    fake_index_path = "/sys/devices/system/cpu/cpu0/cache/index3"
    mocker.patch("glob.glob", return_value=[fake_index_path])

    def side_effect(path, mode="r"):
        if "level" in path:
            return StringIO("3")
        if "size" in path:
            return StringIO("16M")
        return StringIO("")

    mocker.patch("builtins.open", side_effect=side_effect)

    assert system_info._get_l3_linux() == 16


def test_get_l3_linux_raw_bytes(mocker):
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch(
        "glob.glob", return_value=["/sys/devices/system/cpu/cpu0/cache/index3"]
    )

    def side_effect(path, mode="r"):
        if "level" in path:
            return StringIO("3")
        if "size" in path:
            return StringIO("33554432")
        return StringIO("")

    mocker.patch("builtins.open", side_effect=side_effect)

    assert system_info._get_l3_linux() == 33554432


def test_get_l3_linux_no_dir(mocker):
    mocker.patch("os.path.isdir", return_value=False)
    assert system_info._get_l3_linux() is None


def test_get_l3_linux_no_indices(mocker):
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("glob.glob", return_value=[])
    assert system_info._get_l3_linux() is None


def test_get_l3_linux_no_level_3(mocker):
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("glob.glob", return_value=["/sys/.../index0", "/sys/.../index1"])

    mock_file = mocker.mock_open()
    mock_file.side_effect = [
        mocker.mock_open(read_data="1").return_value,  # index0 level
        mocker.mock_open(read_data="2").return_value,  # index1 level
    ]
    mocker.patch("builtins.open", mock_file)

    assert system_info._get_l3_linux() is None


def test_get_l3_linux_exception_handling(mocker):
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("glob.glob", return_value=["/sys/.../index3"])

    mock_open = mocker.patch("builtins.open")
    handle_level = mocker.MagicMock()
    handle_level.__enter__.return_value.read.return_value = "3"

    handle_size = mocker.MagicMock()
    handle_size.__enter__.side_effect = IOError("Mock permissions denied")

    mock_open.side_effect = [handle_level, handle_size]

    assert system_info._get_l3_linux() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_return_none_initial_call_fails_win32(mocker):
    mock_windll = mocker.patch("ctypes.windll", create=True)
    mock_k32 = mock_windll.kernel32
    mock_k32.GetLogicalProcessorInformation.return_value = 0
    mock_k32.GetLastError.return_value = 999

    assert system_info._get_cache_line_size_windows() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_return_none_byte_count_zero_win32(mocker):
    mock_windll = mocker.patch("ctypes.windll", create=True)
    mock_k32 = mock_windll.kernel32

    ERROR_INSUFFICIENT_BUFFER = 122
    mock_k32.GetLogicalProcessorInformation.return_value = 0
    mock_k32.GetLastError.return_value = ERROR_INSUFFICIENT_BUFFER

    assert system_info._get_cache_line_size_windows() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_return_none_second_call_fails_win32(mocker):
    mock_windll = mocker.patch("ctypes.windll", create=True)
    mock_k32 = mock_windll.kernel32

    def side_effect(ptr, buf_size_ref):
        if ptr is None:
            buf_size_ref._obj.value = 64
            mock_k32.GetLastError.return_value = 122
            return 0
        else:
            return 0

    mock_k32.GetLogicalProcessorInformation.side_effect = side_effect

    assert system_info._get_cache_line_size_windows() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_return_none_no_cache_found_win32(mocker):
    mock_windll = mocker.patch("ctypes.windll", create=True)
    mock_k32 = mock_windll.kernel32

    def side_effect(ptr, buf_size_ref):
        struct_size = 32
        if ptr is None:
            buf_size_ref._obj.value = struct_size
            mock_k32.GetLastError.return_value = 122
            return 0
        else:
            return 1

    mock_k32.GetLogicalProcessorInformation.side_effect = side_effect

    assert system_info._get_cache_line_size_windows() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_l3_returns_none_on_api_fail_win32(mocker):
    mock_dll = mocker.patch("ctypes.WinDLL")
    mock_dll.return_value.GetLogicalProcessorInformationEx.side_effect = [
        True,
        False,
    ]

    assert system_info._get_l3_win32() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
def test_l3_returns_none_when_no_l3_exists_win32(mocker):
    mock_dll = mocker.patch("ctypes.WinDLL")
    mock_api = mock_dll.return_value.GetLogicalProcessorInformationEx

    def side_effect(rel, ptr, size_ref):
        if size_ref:
            if hasattr(size_ref, "_obj"):
                size_ref._obj.value = 32
            else:
                size_ref.contents.value = 32
        return True

    mock_api.side_effect = side_effect

    mock_info = mocker.MagicMock()
    mock_info.Relationship = 2
    mock_info.u.Cache.Level = 2
    mock_info.Size = 32

    mocker.patch("ctypes.cast").return_value.contents = mock_info

    assert system_info._get_l3_win32() is None


def test_get_cache_line_size_darwin(mocker):
    mock_cdll = mocker.patch("ctypes.CDLL")
    mock_libc = mock_cdll.return_value

    mock_libc.sysctlbyname.return_value = -1
    assert system_info._get_cache_line_size_darwin() is None

    def simulate_c_success(name, val_ptr, size_ptr, newp, newlen):
        val = ctypes.cast(val_ptr, ctypes.POINTER(ctypes.c_uint64))
        val.contents.value = 64
        return 0

    mock_libc.sysctlbyname.side_effect = simulate_c_success
    assert system_info._get_cache_line_size_darwin() == 64


def test_get_l3_darwin(mocker):
    mock_cdll = mocker.patch("ctypes.CDLL")
    mock_libc = mock_cdll.return_value

    mock_libc.sysctlbyname.return_value = -1
    assert system_info._get_l3_darwin() is None

    def simulate_c_success_l2_only(name, val_ptr, size_ptr, newp, newlen):
        if name == b"hw.l3cachesize":
            return -1
        val = ctypes.cast(val_ptr, ctypes.POINTER(ctypes.c_uint64))
        val.contents.value = 33_554_432
        return 0

    def simulate_c_success_l3_only(name, val_ptr, size_ptr, newp, newlen):
        if name == b"hw.l2cachesize":
            return -1
        val = ctypes.cast(val_ptr, ctypes.POINTER(ctypes.c_uint64))
        val.contents.value = 67_108_864
        return 0

    for side_effect, expected_res in (
        (simulate_c_success_l2_only, 33_554_432),
        (simulate_c_success_l3_only, 67_108_864),
    ):
        mock_libc.sysctlbyname.side_effect = side_effect
        assert system_info._get_l3_darwin() == expected_res


def test_get_available_ram_linux_exceptions(mocker):
    mock_file_content = (
        "MemTotal:       16393304 kB\n"
        "MemFree:         8194452 kB\n"
        "Buffers:          123456 kB\n"
    )

    mocked_open = mocker.mock_open(read_data=mock_file_content)
    mocker.patch("builtins.open", mocked_open)

    with pytest.raises(NumCircBufRuntimeError) as exc_info:
        system_info._get_available_ram_linux()

    exc = exc_info.value
    assert exc.class_obj is None
    assert exc.obj is None
    assert exc.message


def test_get_available_ram_darwin_exceptions(mocker):
    mock_cdll = mocker.patch("ctypes.CDLL")
    mock_libc = mock_cdll.return_value

    mock_libc.mach_host_self.return_value = 123
    mock_libc.host_statistics64.return_value = 1

    with pytest.raises(NumCircBufOSError) as exc_info:
        system_info._get_available_ram_darwin()

    exc = exc_info.value
    assert exc.class_obj is None
    assert exc.obj is None
    assert exc.message


def test_deprecation_and_future_warning():
    mock_obj = "mock_obj"
    mock_feature = "mock_feature"
    mock_replacement = "mock_replacement"
    mock_remove_in_version = "mock_remove_in_version"

    def _raise_deprecation_warning():
        warnings.warn(
            NumCircBufDeprecationWarning(
                mock_obj,
                mock_feature,
                mock_replacement,
                mock_remove_in_version,
            ),
            stacklevel=2,
        )

    def _raise_future_warning():
        warnings.warn(
            NumCircBufFutureWarning(
                mock_obj,
                mock_feature,
                mock_replacement,
                mock_remove_in_version,
            ),
            stacklevel=2,
        )

    with pytest.warns(
        (NumCircBufDeprecationWarning, NumCircBufFutureWarning)
    ) as record:
        _raise_deprecation_warning()
        _raise_future_warning()

    assert isinstance(record[0].message, NumCircBufDeprecationWarning)
    assert isinstance(record[1].message, NumCircBufFutureWarning)

    for warning_info in record:
        w = warning_info.message
        assert w.obj == mock_obj
        assert w.feature == mock_feature
        assert w.replacement == mock_replacement
        assert w.remove_in_version == mock_remove_in_version


def test_get_cpu_name_edge_cases(mocker):
    mocker.patch("platform.processor", side_effect=["", " ", "    "])
    for _ in range(3):
        assert bench_utils.get_cpu_name() == "unknown_cpu"


def test_trimmed_mean_times():
    for arr, expected_res_int, expected_res_float in (
        (np.array([]), 0, 0.0),
        (np.array([25]), 25, 25.0),
        (np.array([x for x in range(5)]), 2, 2.5),
        (np.array([x for x in range(10)]), 5, 5.0),
        (np.array([x for x in range(20)]), 10, 10.5),
    ):
        res_int = bench_utils.trimmed_mean_times(arr, return_int=True)
        assert isinstance(res_int, int)
        assert res_int == expected_res_int

        res_float = bench_utils.trimmed_mean_times(arr, return_int=False)
        assert isinstance(res_float, float)
        assert res_float == pytest.approx(expected_res_float)


def test_trimmed_mean_times_exceptions():
    for arr, return_int in (
        (np.array([0.0, float("inf"), float("inf")]), True),
        (np.array([0.0, float("-inf"), float("-inf")]), True),
        (np.array([0.0, float("-inf"), float("inf")]), False),
        (np.array([0.0, float("-inf"), float("inf")]), True),
        (np.array([0.0, float("nan"), float("inf")]), False),
        (np.array([0.0, float("nan"), float("inf")]), True),
    ):
        with pytest.raises(NumCircBufArithmeticError) as exc_info:
            with np.errstate(invalid="ignore"):
                bench_utils.trimmed_mean_times(arr, return_int)

        exc = exc_info.value
        assert exc.class_obj is None
        assert exc.obj is None
        assert exc.message


def test_base_exception_and_warning_no_message():
    exc = NumCircBufError()
    assert exc.class_obj is None
    assert exc.obj is None
    assert exc.message is None

    with pytest.warns(NumCircBufWarning) as record:
        warnings.warn(
            NumCircBufWarning(),
            stacklevel=2,
        )

    w = record[0].message
    assert w.class_obj is None
    assert w.obj is None
    assert w.message is None


@pytest.mark.parametrize("log_delete_errors", (False, True))
def test_temporary_benchmark_data_error_logging(caplog, mocker, log_delete_errors):
    mock = mocker.patch(
        "numcircbuf.bench_utils.os.remove",
        side_effect=OSError("--- Mocked OSError ---"),
    )

    with bench_utils.temporary_benchmark_data(
        dtype=np.float64,
        buffer_maxlen=2,
        block_size=1,
        n_runs=2,
        create_offset_data=False,
        create_fill_data=False,
        create_evict_arr=False,
        log_delete_errors=log_delete_errors,
    ) as (
        data_path,
        _,
        warmup_path,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ):
        pass

    mocker.stop(mock)

    if log_delete_errors:
        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == "WARNING"
        assert caplog.records[0].exc_info is not None
    else:
        assert len(caplog.records) == 0

    for path in (data_path, warmup_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.warning(
                f"--- Could not delete temp file. Delete manually. ---\nPath: {path}",
                exc_info=True,
            )
