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

import warnings

import pytest
import numpy as np

from numcircbuf import (
    OverwriteCircBuffer,
    BlockingCircBuffer,
    RunningMeanSqBuffer,
    RunningMeanBuffer,
    IntegratedGatedBuffer,
)
from numcircbuf.exceptions import (
    IndexOutOfBounds,
    BufferCapacityValueError,
    BufferCapacityTypeError,
    DataTypeError,
    DataSizeWarning,
    NumCircBufValueError,
    InvalidModification,
)

from .constants import (
    CAPACITIES,
    SUPPORTED_DTYPES_ALL,
    SUPPORTED_DTYPES_FP,
    Limits,
)

OVERFLOW_CASES_FP = [
    # dtype, value to append, expected result
    (np.float32, 3.5e38, float("inf")),
    (np.float64, 3.5e38, 3.5e38),
    (np.float32, -3.5e38, float("-inf")),
    (np.float64, -3.5e38, -3.5e38),
    (np.float64, 1e309, float("inf")),
    (np.float64, -1e309, float("-inf")),
]
OVERFLOW_CASES_ALL = OVERFLOW_CASES_FP + [
    # dtype, value to append, exception
    (np.int32, Limits.INT32_MAX.value + 1, OverflowError),
    (np.int32, Limits.INT32_MIN.value - 1, OverflowError),
    (np.int64, Limits.INT64_MAX.value + 1, OverflowError),
    (np.int64, Limits.INT64_MIN.value - 1, OverflowError),
    (np.uint32, Limits.UINT32_MAX.value + 1, OverflowError),
    (np.uint32, -1, OverflowError),
    (np.uint64, Limits.UINT64_MAX.value + 1, OverflowError),
    (np.uint64, -1, OverflowError),
]

_append = lambda buf, value: buf.append(value)
_extend = lambda buf, block: buf.extend(block)
_extend_unchecked = lambda buf, np_block: buf.extend_unchecked(np_block)

_write_append = lambda buf, value: buf.write_append(value)
_write_extend = lambda buf, block: buf.write_extend(block)
_write_extend_unchecked = lambda buf, np_block: buf.write_extend_unchecked(
    np_block
)
MAIN_BUFFERS = {
    "OverwriteCircBuffer_never": {
        "init": lambda maxlen, dtype=None: OverwriteCircBuffer(
            maxlen, "never", **({"dtype": dtype} if dtype is not None else {})
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "OverwriteCircBuffer_always": {
        "init": lambda maxlen, dtype=None: OverwriteCircBuffer(
            maxlen, "always", **({"dtype": dtype} if dtype is not None else {})
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "OverwriteCircBuffer_conditional": {
        "init": lambda maxlen, dtype=None: OverwriteCircBuffer(
            maxlen,
            "conditional",
            **({"dtype": dtype} if dtype is not None else {}),
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "BlockingCircBuffer": {
        "init": lambda maxlen, dtype=None: BlockingCircBuffer(
            maxlen, **({"dtype": dtype} if dtype is not None else {})
        ),
        "append": _write_append,
        "extend": _write_extend,
        "extend_unchecked": _write_extend_unchecked,
    },
}
UTIL_BUFFERS = {
    "RunningMeanSqBuffer_extend/append": {
        "init": lambda maxlen, dtype=None: RunningMeanSqBuffer(
            maxlen,
            "extend/append",
            **({"dtype": dtype} if dtype is not None else {}),
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "RunningMeanBuffer_extend/append": {
        "init": lambda maxlen, dtype=None: RunningMeanBuffer(
            maxlen,
            "extend/append",
            **({"dtype": dtype} if dtype is not None else {}),
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "RunningMeanSqBuffer_calculation": {
        "init": lambda maxlen, dtype=None: RunningMeanSqBuffer(
            maxlen,
            "calculation",
            **({"dtype": dtype} if dtype is not None else {}),
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "RunningMeanBuffer_calculation": {
        "init": lambda maxlen, dtype=None: RunningMeanBuffer(
            maxlen,
            "calculation",
            **({"dtype": dtype} if dtype is not None else {}),
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
    "IntegratedGatedBuffer": {
        "init": lambda maxlen, dtype=None: IntegratedGatedBuffer(
            maxlen, -70, 10, **({"dtype": dtype} if dtype is not None else {})
        ),
        "append": _append,
        "extend": _extend,
        "extend_unchecked": _extend_unchecked,
    },
}

MAIN_BUFFERS_PARAMS = list(MAIN_BUFFERS.items())
MAIN_BUFFERS_IDS = [name for name, _ in MAIN_BUFFERS_PARAMS]

UTIL_BUFFERS_PARAMS = list(UTIL_BUFFERS.items())
UTIL_BUFFERS_IDS = [name for name, _ in UTIL_BUFFERS_PARAMS]

ALL_BUFFERS_PARAMS = MAIN_BUFFERS_PARAMS + UTIL_BUFFERS_PARAMS
ALL_BUFFERS_IDS = MAIN_BUFFERS_IDS + UTIL_BUFFERS_IDS


@pytest.mark.parametrize(
    "buf_name, buf_funcs", ALL_BUFFERS_PARAMS, ids=ALL_BUFFERS_IDS
)
def test_default_dtype(buf_name, buf_funcs):
    expected_dtype = np.float64

    buf = buf_funcs["init"](5)
    assert buf.dtype is expected_dtype

    arr = np.array([(i + 1) for i in range(5)], dtype=expected_dtype)
    buf_funcs["extend_unchecked"](buf, arr)

    if buf_name != "IntegratedGatedBuffer":
        assert np.array_equal(buf.view()[:], arr)
    else:
        assert np.allclose(buf.view()[:], arr**2)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", ALL_BUFFERS_PARAMS, ids=ALL_BUFFERS_IDS
)
@pytest.mark.parametrize(
    "capacity",
    [Limits.PY_SSIZE_T_MAX.value + 1, Limits.SIZE_MAX.value + 1, -1, -10, 0],
)
def test_invalid_capacity_value(buf_name, buf_funcs, capacity):
    with pytest.raises(BufferCapacityValueError) as exc_info:
        buf_funcs["init"](capacity)

    class_obj = buf_funcs["init"](1).__class__
    overflow = capacity in (
        Limits.PY_SSIZE_T_MAX.value + 1,
        Limits.SIZE_MAX.value + 1,
    )
    max_maxlen = Limits.PY_SSIZE_T_MAX.value

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.overflow is overflow
    assert exc.received_value == capacity
    assert exc.max_maxlen == max_maxlen
    assert exc.valid_values == {"min": 1, "max": max_maxlen}
    assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs", ALL_BUFFERS_PARAMS, ids=ALL_BUFFERS_IDS
)
@pytest.mark.parametrize(
    "capacity",
    [
        1.5,
        np.float32,
        np.float64,
        None,
        "abc",
        BlockingCircBuffer,
        BlockingCircBuffer(1),
    ],
)
def test_invalid_capacity_type(buf_name, buf_funcs, capacity):
    with pytest.raises(BufferCapacityTypeError) as exc_info:
        buf_funcs["init"](capacity)

    class_obj = buf_funcs["init"](1).__class__

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.received_type == type(capacity)
    assert exc.valid_types == (int,)
    assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs", ALL_BUFFERS_PARAMS, ids=ALL_BUFFERS_IDS
)
@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.int16, np.uint16, np.bool, -1, 0, 1.5, "abc"],
)
def test_invalid_dtype(buf_name, buf_funcs, dtype):
    with pytest.raises(DataTypeError) as exc_info:
        buf_funcs["init"](1, dtype)

    class_obj = buf_funcs["init"](1).__class__
    valid_values = (
        SUPPORTED_DTYPES_ALL
        if buf_name in MAIN_BUFFERS_IDS
        else SUPPORTED_DTYPES_FP
    )

    exc = exc_info.value
    assert exc.class_obj is class_obj
    assert exc.obj is None
    assert exc.received_value == dtype
    assert exc.valid_values == valid_values
    assert exc.message


def _test_initialization(buf_funcs, dtype, capacity):
    buffer = buf_funcs["init"](capacity, dtype)
    assert buffer.maxlen == capacity
    assert len(buffer) == 0


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buffer_initialization(buf_name, buf_funcs, capacity, dtype):
    _test_initialization(buf_funcs, dtype, capacity)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buffer_initialization(buf_name, buf_funcs, capacity, dtype):
    _test_initialization(buf_funcs, dtype, capacity)


def _view_asserts(buf_name, buf_funcs, buf, expected, capacity):
    if capacity == 1:
        return  # DEBUG

    view = buf.view()
    assert len(view) == capacity
    assert view.maxlen == buf.maxlen
    assert view.dtype == buf.dtype

    def test_iter():
        for i, x in enumerate(view):
            assert x == expected[i]

    def test_indexing():
        for i in range(len(expected)):
            assert view[i] == expected[i]
            assert view[-len(expected) + i] == expected[i]

    def test_slicing():
        # Basic, big and prime steps
        for step in (2, 3, 10, 11, 100, 311):
            assert np.array_equal(view[::step], expected[::step])
            assert np.array_equal(view[::-step], expected[::-step])

        # Negative slicing
        i = min(3, capacity)
        assert np.array_equal(view[-i:], expected[-i:])
        assert np.array_equal(view[:-i], expected[:-i])

        # Empty slices
        assert np.array_equal(view[capacity:0:1], expected[capacity:0:1])
        assert np.array_equal(view[0:capacity:-1], expected[0:capacity:-1])
        assert np.array_equal(view[3:3], expected[3:3])

        # Out of bounds clipping
        assert np.array_equal(
            view[-capacity * 2 : capacity * 2],
            expected[-capacity * 2 : capacity * 2],
        )
        assert np.array_equal(
            view[capacity * 2 : -capacity * 2 : -1],
            expected[capacity * 2 : -capacity * 2 : -1],
        )

        # Mid-slice negative step
        assert np.array_equal(
            view[capacity - 1 : 1 : -2], expected[capacity - 1 : 1 : -2]
        )

    def test_to_numpy():
        assert np.array_equal(view.to_numpy(), expected)

    def run_tests():
        test_iter()
        test_indexing()
        test_slicing()
        test_to_numpy()

    # Test out of bounds
    positive_out_of_bounds = len(expected)
    negative_out_of_bounds = -len(expected) - 1

    with pytest.raises(IndexOutOfBounds) as exc_info_1:
        _ = view[positive_out_of_bounds]
    with pytest.raises(IndexOutOfBounds) as exc_info_2:
        _ = view[negative_out_of_bounds]

    for exc_info, index in (
        (exc_info_1, positive_out_of_bounds),
        (exc_info_2, negative_out_of_bounds),
    ):
        exc = exc_info.value
        assert exc.class_obj is view.__class__
        assert exc.obj is view
        assert exc.index == index
        assert exc.message

    run_tests()

    if capacity > 1:
        n_appends = min(capacity - 1, 5)
        if buf_name == "BlockingCircBuffer":
            buf.read(n_appends)
        for _ in range(n_appends):
            buf_funcs["append"](buf, 1)
            expected = np.append(expected[1:], 1)
        run_tests()

    buf.clear()
    assert np.array_equal(view.to_numpy(), np.array([], dtype=buf.dtype))


def _test_indexing(buf_name, buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)

    data = [i * 10 for i in range(capacity)]
    n_appends = capacity // 2

    for i in range(n_appends):
        buf_funcs["append"](buf, data[i])
    buf_funcs["extend"](buf, data[n_appends:])

    expected = np.array(data, dtype=dtype)
    if buf_name == "IntegratedGatedBuffer":
        expected *= expected

    _view_asserts(buf_name, buf_funcs, buf, expected, capacity)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_indexing(buf_name, buf_funcs, capacity, dtype):
    """Test indexing operations."""
    _test_indexing(buf_name, buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_indexing(buf_name, buf_funcs, capacity, dtype):
    """Test indexing operations."""
    _test_indexing(buf_name, buf_funcs, capacity, dtype)


def _test_invalid_modifications(buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)
    view = buf.view()
    buf_class = buf.__class__
    view_class = view.__class__
    for op, obj in (
        (lambda: buf[0], buf),
        (lambda: buf[:], buf),
        (lambda: buf[::2], buf),
        (lambda: buf.__setitem__(0, 25), buf),
        (lambda: buf.__delitem__(0), buf),
        (lambda: next(iter(buf)), buf),
        (lambda: view.__setitem__(0, 25), view),
        (lambda: view.__delitem__(0), view),
    ):
        with pytest.raises(InvalidModification) as exc_info:
            op()

        exc = exc_info.value

        if obj is buf:
            assert exc.class_obj is buf_class
        else:
            assert exc.class_obj is view_class

        assert exc.obj is obj
        assert exc.recommendation
        assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_invalid_modifications(buf_name, buf_funcs, capacity, dtype):
    _test_invalid_modifications(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_invalid_modifications(buf_name, buf_funcs, capacity, dtype):
    _test_invalid_modifications(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs",
    ALL_BUFFERS_PARAMS,
    ids=ALL_BUFFERS_IDS,
)
@pytest.mark.parametrize("fp_dtype", SUPPORTED_DTYPES_FP)
def test_main_buf_clears(buf_name, buf_funcs, fp_dtype):

    buf = buf_funcs["init"](7, fp_dtype)
    buf_funcs["extend"](buf, [10, float("inf"), 30, float("nan")])
    buf_funcs["append"](buf, float("inf"))
    buf_funcs["append"](buf, float("nan"))
    buf_funcs["append"](buf, 70)

    # Test clear_nans

    result = buf.view()[:]
    expected = np.array(
        [10, float("inf"), 30, float("nan"), float("inf"), float("nan"), 70],
        fp_dtype,
    )
    if buf_name == "IntegratedGatedBuffer":
        expected *= expected
    assert np.array_equal(result, expected, equal_nan=True)

    buf.clear_nans()
    result = buf.view()[:]
    expected = np.array([10, float("inf"), 30, float("inf"), 70], fp_dtype)
    if buf_name == "IntegratedGatedBuffer":
        expected *= expected
    assert np.array_equal(result, expected)

    # Test clear

    buf.clear()
    assert len(buf) == 0
    result = buf.view()[:]
    expected = np.array([], fp_dtype)
    assert np.array_equal(result, expected)

    # Test clear_infs

    buf_funcs["extend"](
        buf,
        [10, float("inf"), 30, float("nan"), float("inf"), float("nan"), 70],
    )
    buf.clear_infs()
    result = buf.view()[:]
    expected = np.array([10, 30, float("nan"), float("nan"), 70], fp_dtype)
    if buf_name == "IntegratedGatedBuffer":
        expected *= expected
    assert np.array_equal(result, expected, equal_nan=True)

    # Test empty buffer clear_nans and clear_infs
    buf.clear()
    buf.clear_nans()
    buf.clear_infs()


def _test_append_extend(buf_name, buf_funcs, capacity, dtype):
    quarter_capacity = capacity // 4

    buf = buf_funcs["init"](capacity, dtype)

    full_sequence = np.arange(capacity, dtype=dtype)
    quarter_sequence = np.arange(quarter_capacity, dtype=dtype)
    quarter_sequence_int16 = quarter_sequence.astype(np.int16)

    # Test append
    for i in range(capacity):
        buf_funcs["append"](buf, i)
        assert len(buf) == i + 1
    assert len(buf) == capacity
    if buf_name == "IntegratedGatedBuffer":
        assert np.array_equal(buf.view()[:], full_sequence * full_sequence)
    else:
        assert np.array_equal(buf.view()[:], full_sequence)

    buf.clear()

    # Test extend

    buf_funcs["extend"](buf, list(range(quarter_capacity)))
    assert len(buf) == quarter_capacity
    if buf_name == "IntegratedGatedBuffer":
        assert np.array_equal(
            buf.view()[:], quarter_sequence * quarter_sequence
        )
    else:
        assert np.array_equal(buf.view()[:], quarter_sequence)

    buf_funcs["extend"](buf, quarter_sequence_int16)
    assert len(buf) == 2 * quarter_capacity
    if buf_name == "IntegratedGatedBuffer":
        assert np.array_equal(
            buf.view()[:],
            np.tile(quarter_sequence, 2) * np.tile(quarter_sequence, 2),
        )
    else:
        assert np.array_equal(buf.view()[:], np.tile(quarter_sequence, 2))

    if buf_name == "BlockingCircBuffer":
        if len(buf):
            buf.read()
    buf_funcs["extend_unchecked"](buf, full_sequence)
    assert len(buf) == capacity
    if buf_name == "IntegratedGatedBuffer":
        assert np.array_equal(buf.view()[:], full_sequence * full_sequence)
    else:
        assert np.array_equal(buf.view()[:], full_sequence)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_append_extend(buf_name, buf_funcs, capacity, dtype):
    """Test basic append and extend operations."""
    _test_append_extend(buf_name, buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_append_extend(buf_name, buf_funcs, capacity, dtype):
    """Test basic append and extend operations."""
    _test_append_extend(buf_name, buf_funcs, capacity, dtype)


def _test_edge_cases_append_extend(buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)

    with pytest.raises((ValueError, TypeError)):
        buf_funcs["append"](buf, "abc")

    with pytest.raises((ValueError, TypeError)):
        buf_funcs["extend"](buf, ["a", "b", "c"])

    with pytest.raises((ValueError, TypeError)):
        buf_funcs["extend"](buf, "abc")


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_edge_cases_append_extend(
    buf_name, buf_funcs, capacity, dtype
):
    _test_edge_cases_append_extend(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_edge_cases_append_extend(
    buf_name, buf_funcs, capacity, dtype
):
    _test_edge_cases_append_extend(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
def test_main_buf_overflow_situations(buf_name, buf_funcs, capacity):
    for dtype, val, expected in OVERFLOW_CASES_ALL:
        buf = buf_funcs["init"](capacity, dtype)
        with np.errstate(over="ignore"):  # ignore np warnings
            if isinstance(expected, type) and issubclass(expected, Exception):
                with pytest.raises(expected):
                    buf_funcs["append"](buf, val)
                buf.clear()
                with pytest.raises(expected):
                    buf_funcs["extend"](buf, [val])
            else:
                buf_funcs["append"](buf, val)
                assert buf.view()[0] == expected
                buf.clear()
                buf_funcs["extend"](buf, [val])
                assert buf.view()[0] == expected


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
def test_util_buf_overflow_situations(buf_name, buf_funcs, capacity):
    for dtype, val, expected in OVERFLOW_CASES_FP:
        buf = buf_funcs["init"](capacity, dtype)
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                buf_funcs["append"](buf, val)
        else:
            buf_funcs["append"](buf, val)
            if buf_name == "IntegratedGatedBuffer":
                assert buf.view()[0] == expected * expected
            else:
                assert buf.view()[0] == expected


def _test_data_size_warnings(buf_name, buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)

    exceeds_capacity_list = list(range(capacity * 2))
    exceeds_capacity_arr = np.array(exceeds_capacity_list, dtype=dtype)
    with pytest.warns(DataSizeWarning) as record:
        buf_funcs["extend"](buf, exceeds_capacity_list)
        if buf_name == "BlockingCircBuffer":
            buf.read()
        buf_funcs["extend_unchecked"](buf, exceeds_capacity_arr)

    expected_record_len = 2
    assert len(record) == expected_record_len
    for i in range(expected_record_len):
        w = record[i].message
        assert isinstance(w, DataSizeWarning)
        assert w.obj is buf
        assert w.data_size == len(exceeds_capacity_list)
        assert w.maxlen == buf.maxlen

    if buf_name == "BlockingCircBuffer":
        buf.read()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        buf_funcs["extend"](buf, list(range(capacity)))
    assert len(record) == 0, f"Expected no warnings, but got {len(record)}"


def _test_extend_dim_exceptions(buf_funcs, capacity, dtype):
    buf = buf_funcs["init"](capacity, dtype)
    with pytest.raises(NumCircBufValueError) as exc_info_1:
        buf_funcs["extend"](buf, 1)
    with pytest.raises(NumCircBufValueError) as exc_info_2:
        buf_funcs["extend"](buf, np.zeros((3, 3)))

    for exc_info in (exc_info_1, exc_info_2):
        exc = exc_info.value
        assert exc.class_obj is buf.__class__
        assert exc.obj is buf
        assert exc.message


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_extend_dim_exceptions(buf_name, buf_funcs, capacity, dtype):
    _test_extend_dim_exceptions(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_extend_dim_exceptions(buf_name, buf_funcs, capacity, dtype):
    _test_extend_dim_exceptions(buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", MAIN_BUFFERS_PARAMS, ids=MAIN_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_main_buf_data_size_warnings(buf_name, buf_funcs, capacity, dtype):
    _test_data_size_warnings(buf_name, buf_funcs, capacity, dtype)


@pytest.mark.parametrize(
    "buf_name, buf_funcs", UTIL_BUFFERS_PARAMS, ids=UTIL_BUFFERS_IDS
)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_FP)
def test_util_buf_data_size_warnings(buf_name, buf_funcs, capacity, dtype):
    _test_data_size_warnings(buf_name, buf_funcs, capacity, dtype)
