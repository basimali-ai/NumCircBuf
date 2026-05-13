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
from typing import Callable, Any
import os
import gc
import time
import math

import pytest
import numpy as np

from numcircbuf.bench_utils import EvictArrConfig, touch_pages
from numcircbuf.exceptions import (
    ConfigurationValueError,
    ConfigurationTypeError,
    NumCircBufNotImplementedError,
)
from numcircbuf import (
    RunningMeanSqBuffer,
    RunningMeanBuffer,
    IntegratedGatedBuffer,
)
from numcircbuf.core import _UtilityBufferBP

try:
    import numcircbuf_test_cython_api as _test_cython_api  # pyright: ignore[reportMissingImports]

    HAS_C_TESTS = True

except ImportError:
    try:
        import numcircbuf._test_cython_api as _test_cython_api  # pyright: ignore[reportMissingImports]

        HAS_C_TESTS = True

    except ImportError:
        HAS_C_TESTS = False

from .constants import SUPPORTED_DTYPES_FP, CAPACITIES, Limits

is_valgrind = os.getenv("IS_VALGRIND", "0") == "1"

if is_valgrind:
    BLOCK_SIZE = 32
    MAXLEN = 64
    NUM_BLOCKS = NUM_RUNS = 100
    RECALC_THRESHOLDS = (BLOCK_SIZE * 2, BLOCK_SIZE * 3)
else:
    BLOCK_SIZE = 1024
    MAXLEN = BLOCK_SIZE * 10
    NUM_BLOCKS = NUM_RUNS = 8192
    RECALC_THRESHOLDS = (BLOCK_SIZE * 2, BLOCK_SIZE * 10)

ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0


def _mock_gated_mean_square(
    abs_gate_lufs: float,
    rel_gate_lu: float,
    signal: np.ndarray,
    is_squared=False,
):
    abs_gate_lin_sq = 10 ** ((abs_gate_lufs - 0.691) / 10.0)
    rel_gate_lin_sq_factor = 10 ** (rel_gate_lu / 10.0)

    if is_squared:
        samples_sq = signal
    else:
        samples_sq = np.array(signal, dtype=signal.dtype) ** 2

    abs_gated_mask = samples_sq >= abs_gate_lin_sq
    abs_gated_values = samples_sq[abs_gated_mask]

    if len(abs_gated_values) == 0:
        return 0.0

    avg_after_abs_gate = np.mean(abs_gated_values)
    rel_gate_threshold = avg_after_abs_gate * rel_gate_lin_sq_factor

    final_gate_threshold = max(abs_gate_lin_sq, rel_gate_threshold)
    final_values = samples_sq[samples_sq >= final_gate_threshold]

    if len(final_values) == 0:
        return 0.0

    return np.mean(final_values)


def _mock_abs_gated_mean_square(
    abs_gate_lufs: float, signal: np.ndarray, is_squared: bool = False
):
    abs_gate_lin_sq = 10 ** ((abs_gate_lufs - 0.691) / 10.0)

    if is_squared:
        samples_sq = signal
    else:
        samples_sq = np.array(signal, dtype=signal.dtype) ** 2

    abs_gated_mask = samples_sq >= abs_gate_lin_sq
    abs_gated_values = samples_sq[abs_gated_mask]

    if len(abs_gated_values) == 0:
        return 0.0

    return np.mean(abs_gated_values)


CALC_BUFFERS = {
    "RunningMeanSqBuffer_calculation": {
        "init": lambda maxlen, dtype, mode="calculation", recalc_threshold=None: (
            RunningMeanSqBuffer(maxlen, mode, dtype, recalc_threshold)
        ),
        "metric_fn": lambda buf: buf.mean_square(),
        "ground_fn": lambda buf: np.mean(buf.view().to_numpy() ** 2),
        "direct_ground_fn": lambda data: np.mean(data**2),
    },
    "RunningMeanBuffer_calculation": {
        "init": lambda maxlen, dtype, mode="calculation", recalc_threshold=None: (
            RunningMeanBuffer(maxlen, mode, dtype, recalc_threshold)
        ),
        "metric_fn": lambda buf: buf.mean(),
        "ground_fn": lambda buf: np.mean(buf.view().to_numpy()),
        "direct_ground_fn": lambda data: np.mean(data),
    },
}

EXTEND_BUFFERS = {
    "RunningMeanSqBuffer_extend/append": {
        "init": lambda maxlen, dtype, mode="extend/append", recalc_threshold=None: (
            RunningMeanSqBuffer(maxlen, mode, dtype, recalc_threshold)
        ),
        "metric_fn": lambda buf: buf.mean_square(),
        "ground_fn": lambda buf: np.mean(buf.view().to_numpy() ** 2),
        "direct_ground_fn": lambda data: np.mean(data**2),
    },
    "RunningMeanBuffer_extend/append": {
        "init": lambda maxlen, dtype, mode="extend/append", recalc_threshold=None: (
            RunningMeanBuffer(maxlen, mode, dtype, recalc_threshold)
        ),
        "metric_fn": lambda buf: buf.mean(),
        "ground_fn": lambda buf: np.mean(buf.view().to_numpy()),
        "direct_ground_fn": lambda data: np.mean(data),
    },
}

INTEGRATED_BUFFER = {
    "IntegratedGatedBuffer": {
        "init": lambda maxlen, dtype, recalc_threshold=None: IntegratedGatedBuffer(
            maxlen,
            ABS_GATE_LUFS,
            REL_GATE_LU,
            dtype,
            recalc_threshold,
        ),
        "metric_fn": lambda buf: buf.gated_mean_square(),
        "ground_fn": lambda buf: _mock_gated_mean_square(
            ABS_GATE_LUFS, REL_GATE_LU, buf.view().to_numpy(), is_squared=True
        ),
        "direct_ground_fn": lambda data: _mock_gated_mean_square(
            ABS_GATE_LUFS, REL_GATE_LU, data, is_squared=False
        ),
        "metric_fn_internal": lambda buf: buf._current_abs_gated_mean_sq(),
        "ground_fn_internal": lambda buf: _mock_abs_gated_mean_square(
            ABS_GATE_LUFS, buf.view().to_numpy(), is_squared=True
        ),
    },
}

CALC_BUFFERS_PARAMS = list(CALC_BUFFERS.items())
CALC_BUFFERS_IDS = [name for name, _ in CALC_BUFFERS_PARAMS]

EXTEND_BUFFERS_PARAMS = list(EXTEND_BUFFERS.items())
EXTEND_BUFFERS_IDS = [name for name, _ in EXTEND_BUFFERS_PARAMS]

INTEGRATED_BUFFER_PARAMS = list(INTEGRATED_BUFFER.items())
INTEGRATED_BUFFER_IDS = [name for name, _ in INTEGRATED_BUFFER_PARAMS]

np.random.seed(25)
blocks = []
for i in range(NUM_BLOCKS):
    if i % 10 < 5:
        signs = np.random.choice([-1, 1], size=BLOCK_SIZE)
        magnitudes = 0.75 + np.random.rand(BLOCK_SIZE) * 0.25
    else:
        signs = np.random.choice([-1, 1], size=BLOCK_SIZE)
        magnitudes = np.random.rand(BLOCK_SIZE) * 0.01
    blocks.append((signs * magnitudes).astype(np.float32))
evict_arr = np.random.rand(EvictArrConfig.shape[0])


def calculate_drift(
    buf: RunningMeanBuffer | RunningMeanSqBuffer | IntegratedGatedBuffer,
    metric_fn: Callable[
        [RunningMeanBuffer | RunningMeanSqBuffer | IntegratedGatedBuffer],
        float,
    ],
    ground_fn: Callable[
        [RunningMeanBuffer | RunningMeanSqBuffer | IntegratedGatedBuffer],
        float,
    ],
    manually_recalculate: bool,
):
    max_ground_truth = 0.0
    max_metric = 0.0
    max_drift = 0.0
    for block in blocks:
        buf.extend(block)

        if manually_recalculate:
            buf.recalculate()
        ground_truth = ground_fn(buf)
        metric = metric_fn(buf)
        drift = abs(metric - ground_truth)

        if drift > max_drift:
            max_drift = drift
            max_ground_truth = ground_truth
            max_metric = metric
    return max_ground_truth, max_metric, max_drift


def test_c_utility_buffer_bp():
    buf = _UtilityBufferBP()
    np.dtype(np.float32).num

    func_set = {
        lambda: buf.append(1),
        lambda: buf.extend_unchecked(np.empty(1)),
        buf.recalculate,
        buf.clear_cache,
    }

    if HAS_C_TESTS:
        write_head_diff = _test_cython_api.ubb_spy_append_c(buf, 1)
        assert write_head_diff == 0

        size_diff, write_head_diff = _test_cython_api.ubb_spy_clear_c(buf)
        assert size_diff == 0
        assert write_head_diff == 0

        _test_cython_api.ubb_test_init(buf, np.dtype(np.float32).num)
        func_set.add(lambda: buf.extend([1]))

    for func in func_set:
        with pytest.raises(NumCircBufNotImplementedError) as exc_info:
            func()

        exc = exc_info.value
        assert exc.class_obj is _UtilityBufferBP
        assert exc.obj is buf
        assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_nan_inf_handling(buf_name, buf_funcs, dtype):
    for val, val_assert_fn in (
        (float("INF"), lambda: buf_funcs["metric_fn"](buf) == val),
        (float("NAN"), lambda: math.isnan(buf_funcs["metric_fn"](buf))),
    ):
        buf = buf_funcs["init"](6, dtype)

        for fn in (
            lambda: buf.append(val),
            lambda: buf.extend([val] * 2),
            lambda: buf.extend_unchecked(np.array([val] * 3, dtype=dtype)),
            lambda: buf.append(1),
            lambda: buf.extend([1] * 2),
            lambda: buf.extend_unchecked(np.array([1] * 2, dtype=dtype)),
        ):
            fn()
            assert val_assert_fn()

        ###

        buf.append(1)
        assert buf_funcs["metric_fn"](buf) == 1

        for fn in (
            lambda block: buf.extend(block),
            lambda block: buf.extend_unchecked(np.array(block, dtype=dtype)),
        ):
            fn([val] * 6)
            fn([1] * 4)
            assert val_assert_fn()
            fn([1] * 2)
            assert buf_funcs["metric_fn"](buf) == 1

        ###

        full_val_arr = np.array([val] * 6, dtype=dtype)

        for fn in (
            lambda: buf.extend_unchecked(full_val_arr),
            buf.recalculate,
            buf.clear_cache,
        ):
            fn()
            assert val_assert_fn()

        buf.clear()
        assert buf_funcs["metric_fn"](buf) == 0

        buf.extend_unchecked(full_val_arr)

        if math.isnan(val):
            buf.clear_infs()
            assert val_assert_fn()

            buf.clear_nans()
            assert buf_funcs["metric_fn"](buf) == 0

        elif val == float("INF"):
            buf.clear_nans()
            assert val_assert_fn()

            buf.clear_infs()
            assert buf_funcs["metric_fn"](buf) == 0

        ###

        buf.extend([1] * 6)
        buf._set_accum_value(val)
        assert buf_funcs["metric_fn"](buf) == 1

        ###

        buf = buf_funcs["init"](maxlen=6, dtype=dtype, recalc_threshold=1)

        buf.extend([1] * 3)
        assert buf_funcs["metric_fn"](buf) == 1
        buf.extend([val] * 3)
        assert val_assert_fn()

        buf.append(1)
        assert val_assert_fn()
        buf.extend_unchecked(np.array([1], dtype=dtype))
        assert val_assert_fn()
        buf.extend([1])
        assert val_assert_fn()

        buf.extend([1] * 3)
        assert buf_funcs["metric_fn"](buf) == 1

        ###

        buf = buf_funcs["init"](maxlen=2, dtype=dtype)
        buf.append(val)
        buf.append(1)

        buf.recalculate()
        buf.append(1)

        assert buf_funcs["metric_fn"](buf) == 1

        buf.extend([val, 1])

        buf.recalculate()
        buf.extend([1])

        assert buf_funcs["metric_fn"](buf) == 1

        buf.extend_unchecked(np.array([val, 1], dtype=dtype))

        buf.recalculate()
        buf.extend_unchecked(np.array([1], dtype=dtype))

        assert buf_funcs["metric_fn"](buf) == 1


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS,
)
@pytest.mark.parametrize("mode", ["abc", "never", "always", "conditional"])
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_invalid_mode_value(buf_name, buf_funcs, capacity, mode, dtype):
    with pytest.raises(ConfigurationValueError) as exc_info:
        buf_funcs["init"](capacity, dtype, mode)

    class_obj = buf_funcs["init"](1, SUPPORTED_DTYPES_FP[0]).__class__

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.parameter == "operation_focus"
    assert exc.received_value == mode
    assert exc.valid_values == ("extend/append", "calculation")
    assert exc.message


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_special_and_invalid_thresholds(capacity, dtype):
    IntegratedGatedBuffer(1, float("-INF"), float("-INF"))
    valid_values = {
        "min": float("-INF"),
        "max": 1.7976931348623157e308,
    }
    for invalid_threshold in (float("INF"), float("NAN")):
        for a, b, name in (
            (invalid_threshold, 1, "abs_gate_lufs"),
            (1, invalid_threshold, "rel_gate_lu"),
        ):
            with pytest.raises(ConfigurationValueError) as exc_info:
                IntegratedGatedBuffer(1, a, b)

            exc = exc_info.value
            assert exc.class_obj is IntegratedGatedBuffer
            assert exc.obj is None
            assert exc.parameter == name

            if math.isnan(invalid_threshold):
                assert math.isnan(exc.received_value)
            else:
                assert exc.received_value == invalid_threshold

            assert exc.valid_values == valid_values
            assert exc.message


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_internal_current_abs_gated_mean_sq(capacity, dtype):
    buf = IntegratedGatedBuffer(capacity, ABS_GATE_LUFS, REL_GATE_LU, dtype=dtype)
    assert buf._current_abs_gated_mean_sq() == 0
    buf.append(1)
    assert buf._current_abs_gated_mean_sq() == 1


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_threshold_being_applied(dtype):
    abs_gate_lufs = 0  # 10**((0 - 0.691) / 10.0) = 0.852

    buf = IntegratedGatedBuffer(10, abs_gate_lufs, REL_GATE_LU, dtype=dtype)

    arr = np.array(([0.1] * 5) + ([1.0] * 5), dtype=dtype)
    buf.extend_unchecked(arr)

    assert buf.view()[:] == pytest.approx(arr * arr)
    assert buf.gated_mean_square() == pytest.approx(1.0)

    arr = np.array(([1.0] * 10), dtype=dtype)
    buf.extend_unchecked(arr)

    assert buf.view()[:] == pytest.approx(arr)
    assert buf.gated_mean_square() == pytest.approx(1.0)

    arr = np.array(([0.1] * 10), dtype=dtype)
    buf.extend_unchecked(arr)

    assert buf.view()[:] == pytest.approx(arr * arr)
    assert buf.gated_mean_square() == 0.0

    abs_gate_lufs_edge = float("-INF")  # 10**((-inf - 0.691)/10) = 0.0
    buf = IntegratedGatedBuffer(10, abs_gate_lufs_edge, REL_GATE_LU, dtype=dtype)

    arr = np.zeros(10, dtype=dtype)
    buf.extend_unchecked(arr)

    assert np.array_equal(buf.view()[:], arr)
    assert buf.gated_mean_square() == 0.0

    rel_gate_lu_edge = 0  # 10**(0 / 10) = 1.0
    buf = IntegratedGatedBuffer(10, ABS_GATE_LUFS, rel_gate_lu_edge, dtype=dtype)

    arr = np.array(([0.5] * 10), dtype=dtype)
    buf.extend_unchecked(arr)

    assert buf.view()[:] == pytest.approx(arr * arr)
    assert buf.gated_mean_square() == 0.0


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS,
)
@pytest.mark.parametrize(
    "mode",
    [
        -1,
        0,
        1.5,
        None,
        np.float32,
        RunningMeanSqBuffer,
        RunningMeanSqBuffer(1, "extend/append"),
        np.float64,
    ],
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_invalid_mode_type(buf_name, buf_funcs, capacity, mode, dtype):
    with pytest.raises(ConfigurationTypeError) as exc_info:
        buf_funcs["init"](capacity, dtype, mode)

    class_obj = buf_funcs["init"](1, SUPPORTED_DTYPES_FP[0]).__class__

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.parameter == "operation_focus"
    assert exc.received_type is type(mode)
    assert exc.valid_types == (str,)
    assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize(
    "recalc_threshold",
    [
        -1,
        -10,
        Limits.SIZE_MAX.value + 1,
    ],
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_invalid_recalc_value(buf_name, buf_funcs, recalc_threshold, capacity, dtype):
    with pytest.raises(ConfigurationValueError) as exc_info:
        buf_funcs["init"](capacity, dtype, recalc_threshold=recalc_threshold)

    class_obj = buf_funcs["init"](1, SUPPORTED_DTYPES_FP[0]).__class__

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.parameter == "recalc_threshold"
    assert exc.received_value == recalc_threshold
    assert exc.valid_values == {"min": 0, "max": Limits.UINT64_MAX.value}
    assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize(
    "recalc_threshold",
    [
        1.5,
        np.float32,
        RunningMeanSqBuffer,
        RunningMeanSqBuffer(1, "extend/append"),
        np.float64,
    ],
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_invalid_recalc_type(buf_name, buf_funcs, recalc_threshold, capacity, dtype):
    with pytest.raises(ConfigurationTypeError) as exc_info:
        buf_funcs["init"](capacity, dtype, recalc_threshold=recalc_threshold)

    class_obj = buf_funcs["init"](1, SUPPORTED_DTYPES_FP[0]).__class__

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.parameter == "recalc_threshold"
    assert exc.received_type is type(recalc_threshold)
    assert exc.valid_types == (int, None)
    assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_initialization(buf_name, buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)
    assert len(buf) == 0
    assert buf.maxlen == capacity
    assert buf_funcs["metric_fn"](buf) == 0.0


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_extend_and_metric(buf_name, buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)
    data = np.random.uniform(-1, 1, size=capacity).astype(dtype)
    buf.extend(data)
    assert len(buf) == capacity
    assert math.isclose(
        buf_funcs["metric_fn"](buf),
        buf_funcs["direct_ground_fn"](data),
        rel_tol=1e-5 if dtype is np.float32 else 1e-9,
    )


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + EXTEND_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + EXTEND_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_metric_when_wraps_and_cache(buf_name, buf_funcs, dtype):
    batch_size = NUM_RUNS // 2
    buf = buf_funcs["init"](MAXLEN, dtype)

    def _time(buf, clear_cache: bool):
        gc.disable()
        try:
            if clear_cache:
                start = time.perf_counter_ns()
                for _ in range(batch_size):
                    buf.clear_cache()
                overhead = time.perf_counter_ns() - start

                touch_pages(evict_arr, warm_cache=True)
                start = time.perf_counter_ns()
                for _ in range(batch_size):
                    buf.clear_cache()
                    buf_funcs["metric_fn"](buf)

            else:
                start = time.perf_counter_ns()
                for _ in range(batch_size):
                    pass
                overhead = time.perf_counter_ns() - start

                touch_pages(evict_arr, warm_cache=True)
                start = time.perf_counter_ns()
                for _ in range(batch_size):
                    buf_funcs["metric_fn"](buf)

            total_time = time.perf_counter_ns() - start - overhead

        finally:
            gc.enable()

        return total_time / batch_size

    test_size = int(MAXLEN * 1.5) // BLOCK_SIZE
    test_data = np.concatenate(blocks[:test_size], dtype=dtype)
    expected_data = test_data[-MAXLEN:]

    buf.extend(test_data[:MAXLEN])
    buf.extend(test_data[MAXLEN:])
    assert len(buf) == MAXLEN

    buf_funcs["metric_fn"](buf)
    max_attempts = 3
    for _ in range(max_attempts):
        hit_time = _time(buf, clear_cache=False)
        miss_time = _time(buf, clear_cache=True)

        if is_valgrind or buf_name in (
            "RunningMeanSqBuffer_calculation",
            "RunningMeanBuffer_calculation",
        ):
            break

        if miss_time > hit_time * 1.2:
            break
    else:
        pytest.fail(
            f"Cache timing failed after {max_attempts} attempts. "
            f"miss_time ({miss_time}) was not > hit_time ({hit_time}) * 1.2"
        )

    assert math.isclose(
        buf_funcs["metric_fn"](buf),
        buf_funcs["direct_ground_fn"](expected_data),
        rel_tol=1e-5 if dtype is np.float32 else 1e-9,
    )


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
@pytest.mark.parametrize("recalc_threshold", RECALC_THRESHOLDS)
def test_when_recalc(buf_name, buf_funcs, dtype, recalc_threshold):
    test_maxlen = (BLOCK_SIZE * NUM_BLOCKS) // 4
    buf = buf_funcs["init"](test_maxlen, dtype=dtype, recalc_threshold=recalc_threshold)
    block_len = len(blocks[0])
    fill_data = np.tile(blocks[0], math.ceil(test_maxlen / BLOCK_SIZE)).astype(dtype)

    buf.extend(fill_data)
    if len(fill_data) >= test_maxlen:
        n_ops = 0
    else:
        n_ops = len(fill_data) % recalc_threshold

    recalc_times_extend = []
    normal_times_extend = []
    recalc_times_append = []
    normal_times_append = []

    gc.disable()
    try:
        touch_pages(evict_arr, warm_cache=True)
        for block in blocks:
            start = time.perf_counter_ns()
            buf.extend(block)
            dt = time.perf_counter_ns() - start

            n_ops += block_len
            if n_ops >= recalc_threshold:
                recalc_times_extend.append(dt)
                n_ops = 0
            else:
                normal_times_extend.append(dt)

        touch_pages(evict_arr, warm_cache=True)
        target_recalcs = 25
        ff_data = np.zeros(recalc_threshold, dtype=dtype)

        for _ in range(target_recalcs):
            ops_to_recalc = recalc_threshold - n_ops
            if ops_to_recalc > 5:
                fast_forward_amount = ops_to_recalc - 5
                buf.extend_unchecked(ff_data[:fast_forward_amount])
                n_ops += fast_forward_amount

            for i in range(10):
                val = blocks[0][i % block_len]

                start = time.perf_counter_ns()
                buf.append(val)
                dt = time.perf_counter_ns() - start

                n_ops += 1
                if n_ops >= recalc_threshold:
                    recalc_times_append.append(dt)
                    n_ops = 0
                else:
                    normal_times_append.append(dt)

    finally:
        gc.enable()

    for times_list in (
        normal_times_extend,
        recalc_times_extend,
        normal_times_append,
        recalc_times_append,
    ):
        assert times_list, f"{times_list} should not be empty"

    if not is_valgrind:
        for normal_times, recalc_times in (
            (normal_times_extend, recalc_times_extend),
            (normal_times_append, recalc_times_append),
        ):
            min_normal = min(normal_times)
            min_recalc = min(recalc_times)
            assert min_normal * 1.2 < min_recalc


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    CALC_BUFFERS_PARAMS + INTEGRATED_BUFFER_PARAMS,
    ids=CALC_BUFFERS_IDS + INTEGRATED_BUFFER_IDS,
)
def test_recalc_caps_drift_with_high_dynamic_range_signal(
    buf_name: str, buf_funcs: dict[str, Callable]
):
    if buf_name == "IntegratedGatedBuffer":
        metric_fn = buf_funcs["metric_fn_internal"]
        ground_fn = buf_funcs["ground_fn_internal"]
    else:
        metric_fn = buf_funcs["metric_fn"]
        ground_fn = buf_funcs["ground_fn"]

    max_drifts: dict[Any, float] = {}
    for recalc_threshold in (None, 1):
        buf = buf_funcs["init"](MAXLEN, np.float32, recalc_threshold=recalc_threshold)
        max_ground_truth, max_metric, max_drift = calculate_drift(
            buf=buf,
            metric_fn=metric_fn,
            ground_fn=ground_fn,
            manually_recalculate=False,
        )
        max_drifts[recalc_threshold] = max_drift

        print(
            f"\n[buf_name = {buf_name}, recalc_threshold = {recalc_threshold}] "
            f"Max drift after {NUM_RUNS:_} iterations: {max_drift}, "
            f"Ground Value = {max_ground_truth}, Buffer Value = {max_metric}"
        )

        if recalc_threshold is None:
            buf.clear()
            max_ground_truth, max_metric, max_drift = calculate_drift(
                buf=buf,
                metric_fn=metric_fn,
                ground_fn=ground_fn,
                manually_recalculate=True,
            )
            max_drifts["manual"] = max_drift
            print(
                f"\n[buf_name = {buf_name}, recalc_threshold = manual] "
                f"Max drift after {NUM_RUNS:_} iterations: {max_drift}, "
                f"Ground Value = {max_ground_truth}, Buffer Value = {max_metric}"
            )

    assert max_drifts[None] < 0.01

    for drift in (max_drifts[1], max_drifts["manual"]):
        if is_valgrind:
            assert drift < 0.01
        else:
            assert drift < max_drifts[None]

    assert max_drifts[1] == max_drifts["manual"]
