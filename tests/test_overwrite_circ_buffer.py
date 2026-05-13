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

import math

import pytest
import numpy as np

from numcircbuf import OverwriteCircBuffer
from numcircbuf.exceptions import (
    ConfigurationValueError,
    ConfigurationTypeError,
    UnsupportedOperation,
)

try:
    import numcircbuf_test_cython_api as _test_cython_api  # pyright: ignore[reportMissingImports]

    HAS_C_TESTS = True

except ImportError:
    try:
        import numcircbuf._test_cython_api as _test_cython_api  # pyright: ignore[reportMissingImports]

        HAS_C_TESTS = True

    except ImportError:
        HAS_C_TESTS = False

from .constants import (
    CAPACITIES,
    SUPPORTED_DTYPES_ALL,
    SUPPORTED_DTYPES_FP,
    DTYPE_TO_SUFFIX,
)

MODES = (
    "never",
    "always",
    "conditional",
)

SUPPORTED_DTYPES_CALC = (np.float64, np.float32, np.int32, np.uint32)
UNSUPPORTED_DTYPES_CALC = (np.int64, np.uint64)


@pytest.mark.parametrize("mode", ["extend/append", "abc", "calculation"])
def test_invalid_mode_value(mode):
    with pytest.raises(ConfigurationValueError) as exc_info:
        OverwriteCircBuffer(CAPACITIES[0], mode, SUPPORTED_DTYPES_ALL[0])

    exc = exc_info.value
    assert exc.class_obj is OverwriteCircBuffer
    assert exc.obj is None
    assert exc.parameter == "return_overwritten_policy"
    assert exc.received_value == mode
    assert exc.valid_values == MODES
    assert exc.message


@pytest.mark.parametrize(
    "mode",
    [
        -1,
        0,
        1.5,
        None,
        np.float32,
        OverwriteCircBuffer,
        OverwriteCircBuffer(1, "never"),
        np.float64,
    ],
)
def test_invalid_mode_type(mode):
    with pytest.raises(ConfigurationTypeError) as exc_info:
        OverwriteCircBuffer(CAPACITIES[0], mode, SUPPORTED_DTYPES_ALL[0])

    exc = exc_info.value
    assert exc.class_obj is OverwriteCircBuffer
    assert exc.obj is None
    assert exc.parameter == "return_overwritten_policy"
    assert exc.received_type is type(mode)
    assert exc.valid_types == (str,)
    assert exc.message


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_overwrite_returns(dtype):
    """Test overwrite returns."""
    empty_expected = np.array([], dtype)
    if HAS_C_TESTS:
        c_append = getattr(_test_cython_api, f"ocb_spy_append_{DTYPE_TO_SUFFIX[dtype]}")
        c_append_capture = getattr(
            _test_cython_api,
            f"ocb_spy_append_{DTYPE_TO_SUFFIX[dtype]}_capture",
        )

    def test_c_api_then_clear(buffer):
        if HAS_C_TESTS:
            assert c_append(buffer, 10) is None
            assert c_append_capture(buffer, 20) is None
            assert np.array_equal(buffer.extend([30, 40, 50]), empty_expected)
            assert c_append(buffer, 60) is None
            assert c_append_capture(buffer, 70) == 20
            buffer.clear()

    # Test "never" policy

    buffer = OverwriteCircBuffer(5, "never", dtype)

    test_c_api_then_clear(buffer)

    assert buffer.append(10) is None

    result = buffer.extend([20, 30, 40])
    expected = empty_expected
    assert np.array_equal(result, expected)

    result = buffer.extend([50, 60, 70, 80, 90])
    expected = empty_expected
    assert np.array_equal(result, expected)

    assert buffer.append(100) is None

    # Test "always" policy

    buffer = OverwriteCircBuffer(5, "always", dtype)

    test_c_api_then_clear(buffer)

    assert buffer.append(10) is None

    result = buffer.extend([20, 30, 40, 50])
    expected = empty_expected
    assert np.array_equal(result, expected)

    buffer.clear()

    result = buffer.extend([10, 20, 30, 40, 50])
    expected = empty_expected
    assert np.array_equal(result, expected)

    result = buffer.extend([60, 70, 80])
    expected = np.array([10, 20, 30], dtype)
    assert np.array_equal(result, expected)

    result = buffer.extend([90, 100, 110, 120, 130])
    expected = np.array([40, 50, 60, 70, 80], dtype)
    assert np.array_equal(result, expected)

    assert buffer.append(130) == 90

    # Test "conditional" policy

    buffer = OverwriteCircBuffer(5, "conditional", dtype)

    test_c_api_then_clear(buffer)

    assert buffer.append(10) is None

    result = buffer.extend([20, 30, 40])
    expected = empty_expected
    assert np.array_equal(result, expected)

    result = buffer.extend([50, 60, 70, 80, 90])
    expected = empty_expected
    assert np.array_equal(result, expected)

    result = buffer.extend([100, 110, 120], True)
    expected = np.array([50, 60, 70], dtype)
    assert np.array_equal(result, expected)

    result = buffer.extend([130, 140, 150, 160, 170], True)
    expected = np.array([80, 90, 100, 110, 120], dtype)
    assert np.array_equal(result, expected)

    assert buffer.append(180) is None
    assert buffer.append(190, True) == 140


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_CALC)
def test_math_supported_dtypes(mode, dtype):
    """Test math operations on supported dtypes."""
    buffer = OverwriteCircBuffer(10, mode, dtype)
    assert buffer.sum() == 0
    assert buffer.mean() == 0
    result_sum, result_count = buffer.sum_and_count_gt(50)
    assert result_sum == 0
    assert result_count == 0

    buffer.extend(list(range(10, 101, 10)))

    assert math.isclose(buffer.sum(), 550)
    assert math.isclose(buffer.mean(), 55)

    result_sum, result_count = buffer.sum_and_count_gt(50)
    assert math.isclose(result_sum, 400)
    assert math.isclose(result_count, 5)

    buffer.extend(list(range(110, 160, 10)))

    assert math.isclose(buffer.sum(), 1050)
    assert math.isclose(buffer.mean(), 105)

    result_sum, result_count = buffer.sum_and_count_gt(50)
    assert math.isclose(result_sum, 1050)
    assert math.isclose(result_count, 10)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES_CALC)
@pytest.mark.parametrize(
    "func_name",
    ("mean_squares", "sum_squares", "mean", "sum", "sum_and_count_gt"),
)
def test_math_unsupported_dtypes(mode, dtype, func_name):
    buffer = OverwriteCircBuffer(1, mode, dtype)
    func = getattr(buffer, func_name)

    with pytest.raises(UnsupportedOperation) as exc_info:
        func() if func_name != "sum_and_count_gt" else func(50)

    exc = exc_info.value
    assert exc.class_obj is OverwriteCircBuffer
    assert exc.obj is buffer
    assert exc.func == func
    assert exc.func_str == func_name
    assert exc.message


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", set(SUPPORTED_DTYPES_ALL) - set(SUPPORTED_DTYPES_FP))
@pytest.mark.parametrize("func_name", ("mean_squares", "sum_squares"))
def test_math_fp_unsupported(mode, dtype, func_name):
    buffer = OverwriteCircBuffer(1, mode, dtype)
    func = getattr(buffer, func_name)

    with pytest.raises(UnsupportedOperation) as exc_info:
        func()

    exc = exc_info.value
    assert exc.class_obj is OverwriteCircBuffer
    assert exc.obj is buffer
    assert exc.func == func
    assert exc.func_str == func_name
    assert exc.message


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_math_fp(mode, dtype):
    """Test math operations available for fp."""
    buffer = OverwriteCircBuffer(10, mode, dtype)
    assert buffer.sum_squares() == 0
    assert buffer.mean_squares() == 0

    buffer.extend(list(range(10, 101, 10)))
    assert math.isclose(buffer.sum_squares(), 38500)
    assert math.isclose(buffer.mean_squares(), 3850)

    buffer.extend(list(range(110, 160, 10)))
    assert math.isclose(buffer.sum_squares(), 118500)
    assert math.isclose(buffer.mean_squares(), 11850)
