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

import os
import threading
import time

import pytest
import numpy as np

from numcircbuf import BlockingCircBuffer
from numcircbuf.exceptions import NumCircBufValueError

try:
    import numcircbuf_test_cython_api as _test_cython_api

    HAS_C_TESTS = True

except ImportError:
    try:
        import numcircbuf._test_cython_api as _test_cython_api

        HAS_C_TESTS = True

    except ImportError:

        HAS_C_TESTS = False

from .constants import CAPACITIES, SUPPORTED_DTYPES_ALL, DTYPE_TO_SUFFIX

is_valgrind = os.getenv("IS_VALGRIND", "0") == "1"

TIMEOUT = 120 if is_valgrind else 60


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_read_into_dim_error(capacity, dtype):
    buf = BlockingCircBuffer(capacity, dtype)

    arr_not_contig = np.empty(4, dtype=dtype)[::2]
    arr_wrong_dtype = np.empty(1, dtype=np.int8)
    arr_not_1d = np.empty((1, 1), dtype=dtype)

    for invalid_arr in (arr_not_contig, arr_wrong_dtype, arr_not_1d):
        with pytest.raises(NumCircBufValueError) as exc_info:
            buf.read_into(invalid_arr)

        exc = exc_info.value
        assert exc.class_obj is buf.__class__
        assert exc.obj is buf
        assert exc.message


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_blocking_read(capacity, dtype):
    data = list(range(capacity))
    buf = BlockingCircBuffer(capacity, dtype)
    started = threading.Event()
    results = []

    def _test(target):
        started.clear()
        t = threading.Thread(target=target)
        t.start()
        started.wait(timeout=TIMEOUT)
        buf.write_extend(data)
        t.join(timeout=TIMEOUT)

    def consumer_read():
        started.set()
        val = buf.read()
        results.append(val)

    _test(consumer_read)
    assert list(results[0]) == data

    def consumer_read_into():
        started.set()
        buf.read_into(read_into_arr)

    def consumer_read_into_unchecked():
        started.set()
        buf.read_into_unchecked(read_into_arr)

    read_into_arr = np.zeros(capacity, dtype=dtype)
    data_arr = np.array(data, dtype=dtype)
    for c in (consumer_read_into, consumer_read_into_unchecked):
        read_into_arr[:] = 0
        _test(c)
        assert np.array_equal(data_arr, read_into_arr)


def _time_func(func):
    start = time.perf_counter_ns()
    res = func()
    return time.perf_counter_ns() - start, res


def _test_read_timeout(buf, timeout_s, min_expected_ns):
    elapsed, res = _time_func(lambda: buf.read(timeout=timeout_s))
    assert len(res) == 0
    assert elapsed >= min_expected_ns

    elapsed, res = _time_func(
        lambda: buf.read_into(np.empty(1, dtype=buf.dtype), timeout=timeout_s)
    )
    assert res == 0
    assert elapsed >= min_expected_ns


def _test_write_timeout(buf, timeout_s, min_expected_ns, dtype):
    func_list = [
        lambda: buf.write_append(1, timeout=timeout_s),
        lambda: buf.write_extend([1], timeout=timeout_s),
        lambda: buf.write_extend_unchecked(
            np.empty(1, dtype=buf.dtype), timeout=timeout_s
        ),
    ]

    if HAS_C_TESTS:
        c_append = getattr(
            _test_cython_api, f"bcb_spy_write_append_{DTYPE_TO_SUFFIX[dtype]}"
        )
        func_list.append(lambda: c_append(buf, 1, timeout_s))

    for func in func_list:
        elapsed, res = _time_func(func)
        assert res is False
        assert elapsed >= min_expected_ns


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_timeout(capacity, dtype):
    buf = BlockingCircBuffer(capacity, dtype)
    timeout_s = 0.01
    min_expected_ns = timeout_s * 1e9 * 0.75

    _test_read_timeout(buf, timeout_s, min_expected_ns)
    buf.write_extend(list(range(capacity)))
    _test_write_timeout(buf, timeout_s, min_expected_ns, dtype)


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_high_speed_handoff(capacity, dtype):
    buf = BlockingCircBuffer(capacity, dtype)

    block_size = capacity
    n_blocks = 10
    n_items = block_size * n_blocks
    blocks_flat = np.arange(n_blocks * block_size, dtype=dtype)
    blocks = blocks_flat.reshape(n_blocks, block_size)
    results = []

    def producer():
        for block in blocks:
            buf.write_extend_unchecked(block)

    def consumer():
        count = 0
        while count < n_items:
            data = buf.read()
            results.append(data)
            count += len(data)

    c = threading.Thread(target=consumer)
    p = threading.Thread(target=producer)

    c.start()
    p.start()

    p.join(timeout=TIMEOUT)
    c.join(timeout=TIMEOUT)

    assert not c.is_alive()
    assert not p.is_alive()

    for res in results:
        assert res.dtype == dtype

    results_flat = np.concatenate(results)
    assert np.array_equal(results_flat, blocks_flat)


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_multi_thread_ticket_skips(capacity, dtype):
    if capacity < 8:
        return

    buf = BlockingCircBuffer(capacity, dtype)
    n = 1
    started = threading.Event()

    def consumer_read(timeout, expected_return):
        started.set()
        res = buf.read(n, timeout, partial_read=False)
        assert np.array_equal(res, expected_return)

    def consumer_read_into(timeout, expected_return):
        arr = np.zeros(n, dtype=dtype)
        started.set()
        buf.read_into(arr, timeout, partial_read=False)
        assert np.array_equal(arr, expected_return)

    def consumer_read_into_unchecked(timeout, expected_return):
        arr = np.zeros(n, dtype=dtype)
        started.set()
        buf.read_into_unchecked(arr, timeout, partial_read=False)
        assert np.array_equal(arr, expected_return)

    def test_read_skips():
        started.clear()
        threads = []
        target_args_tuple = (
            (consumer_read, (-1, np.array([1], dtype=dtype))),
            (consumer_read, (0, np.array([], dtype=dtype))),
            (consumer_read, (-1, np.array([2], dtype=dtype))),
            (consumer_read_into, (-1, np.array([3], dtype=dtype))),
            (consumer_read_into, (0, np.array([0], dtype=dtype))),
            (consumer_read_into, (-1, np.array([4], dtype=dtype))),
            (consumer_read_into_unchecked, (-1, np.array([5], dtype=dtype))),
            (consumer_read_into_unchecked, (0, np.array([0], dtype=dtype))),
            (consumer_read_into_unchecked, (-1, np.array([6], dtype=dtype))),
        )
        for target, args in target_args_tuple:
            t = threading.Thread(
                target=target,
                args=args,
            )
            t.start()
            started.wait(timeout=TIMEOUT)
            threads.append(t)
            started.clear()
        buf.write_extend_unchecked(np.array([1, 2, 3, 4, 5, 6], dtype=dtype))
        for t in threads:
            t.join(timeout=TIMEOUT)

    def producer_append(value, timeout, expected_return: bool):
        started.set()
        res = buf.write_append(value, timeout=timeout)
        assert res is expected_return

    def producer_extend(block, timeout, expected_return: bool):
        started.set()
        res = buf.write_extend(block, timeout=timeout)
        assert res is expected_return

    def producer_extend_unchecked(block_np, timeout, expected_return: bool):
        started.set()
        res = buf.write_extend_unchecked(block_np, timeout=timeout)
        assert res is expected_return

    def test_write_skips():
        started.clear()
        target_args_tuple = (
            (producer_append, (1, -1, True)),
            (producer_append, (25, 0, False)),
            (producer_append, (2, -1, True)),
            (producer_extend, ([3], -1, True)),
            (producer_extend, ([25], 0, False)),
            (producer_extend, ([4], -1, True)),
            (
                producer_extend_unchecked,
                (np.array([5], dtype=dtype), -1, True),
            ),
            (
                producer_extend_unchecked,
                (np.array([25], dtype=dtype), 0, False),
            ),
            (
                producer_extend_unchecked,
                (np.array([6], dtype=dtype), -1, True),
            ),
        )

        expected = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)

        if HAS_C_TESTS:
            c_append = getattr(
                _test_cython_api,
                f"bcb_spy_write_append_{DTYPE_TO_SUFFIX[dtype]}",
            )

            def producer_c_append(value, timeout, expected_return: bool):
                started.set()
                res = c_append(buf, value, timeout=timeout)
                assert res is expected_return

            target_args_tuple += (
                (producer_c_append, (7, -1, True)),
                (producer_c_append, (25, 0, False)),
                (producer_c_append, (8, -1, True)),
            )

            expected = np.concatenate(
                [expected, np.array([7, 8], dtype=dtype)]
            )

        threads = []
        buf.write_extend_unchecked(np.empty(capacity, dtype=dtype))
        for target, args in target_args_tuple:
            t = threading.Thread(
                target=target,
                args=args,
            )
            t.start()
            started.wait(timeout=TIMEOUT)
            threads.append(t)
            started.clear()
        buf.read()
        for t in threads:
            t.join(timeout=TIMEOUT)
        res = buf.read()
        assert np.array_equal(res, expected)

    test_read_skips()
    test_write_skips()


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_multi_thread_stress_with_partial_reads(capacity, dtype):
    buf = BlockingCircBuffer(capacity, dtype)

    block_size = capacity
    n_blocks = 10
    n_items = block_size * n_blocks

    if HAS_C_TESTS:
        c_append = getattr(
            _test_cython_api, f"bcb_spy_write_append_{DTYPE_TO_SUFFIX[dtype]}"
        )

    def producer_append():
        if HAS_C_TESTS:
            for i in range(n_items):
                if i % 2:
                    buf.write_append(i)
                else:
                    c_append(buf, i)
        else:
            for i in range(n_items):
                buf.write_append(i)

    def producer_extend():
        blocks = [[i] * block_size for i in range(n_blocks)]
        for block in blocks:
            buf.write_extend(block)

    def producer_extend_unchecked():
        blocks = np.empty((n_blocks, block_size), dtype=dtype)
        for block in blocks:
            buf.write_extend_unchecked(block)

    def consumer_read():
        count = 0
        while count < n_items:
            remaining = n_items - count
            data = buf.read(remaining)
            count += len(data)

    def consumer_read_into():
        arr = np.zeros(n_items, dtype=dtype)
        total = 0
        while total < n_items:
            n = buf.read_into(arr[total:])
            total += n

    def consumer_read_into_unchecked():
        arr = np.zeros(n_items, dtype=dtype)
        total = 0
        while total < n_items:
            n = buf.read_into_unchecked(arr[total:])
            total += n

    consumers = (
        consumer_read,
        consumer_read_into,
        consumer_read_into_unchecked,
    )
    producers = (
        producer_append,
        producer_extend,
        producer_extend_unchecked,
    )

    threads = []
    for _ in range(2):
        for target in consumers + producers:
            threads.append(threading.Thread(target=target))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=TIMEOUT)

    for thread in threads:
        assert not thread.is_alive()


_n_test_threads = 3


@pytest.mark.parametrize(
    "capacity", tuple(c * _n_test_threads for c in CAPACITIES)
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_read_ordering(capacity, dtype):
    data = np.arange(capacity, dtype=dtype)
    parts = np.array_split(data, _n_test_threads)
    buf = BlockingCircBuffer(capacity, dtype)
    started = threading.Event()
    results = []

    def _test(target):
        threads = []
        started.clear()
        for _ in range(_n_test_threads):
            t = threading.Thread(target=target)
            t.start()
            started.wait(timeout=TIMEOUT)
            threads.append(t)
            started.clear()
        buf.write_extend_unchecked(data)
        for t in threads:
            t.join(timeout=TIMEOUT)

    def consumer_read():
        started.set()
        val = buf.read(n=len(parts[0]))
        results.append(val)

    _test(consumer_read)
    assert np.array_equal(data, np.concatenate(results))

    def consumer_read_into():
        arr = np.empty(len(parts[0]), dtype=dtype)
        started.set()
        buf.read_into(arr)
        results.append(arr)

    def consumer_read_into_unchecked():
        arr = np.empty(len(parts[0]), dtype=dtype)
        started.set()
        buf.read_into_unchecked(arr)
        results.append(arr)

    for c in (consumer_read_into, consumer_read_into_unchecked):
        results.clear()
        _test(c)
        assert np.array_equal(data, np.concatenate(results))


@pytest.mark.parametrize(
    "capacity", tuple(c * _n_test_threads for c in CAPACITIES)
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ALL)
def test_write_ordering(capacity, dtype):
    data = np.arange(capacity, dtype=dtype)
    parts = np.array_split(data, _n_test_threads)
    garbage_fill_arr = np.empty(capacity, dtype=dtype)
    buf = BlockingCircBuffer(capacity, dtype)
    started = threading.Event()

    def _test(target, append_mode):
        threads = []
        started.clear()
        buf.clear()
        buf.write_extend_unchecked(garbage_fill_arr)
        for i in range(_n_test_threads):
            t = threading.Thread(
                target=target,
                args=(parts[i],) if not append_mode else (parts[i][0],),
            )
            t.start()
            started.wait(timeout=TIMEOUT)
            threads.append(t)
            started.clear()
        buf.clear()
        for t in threads:
            t.join(timeout=TIMEOUT)

    def producer_extend(arr):
        started.set()
        buf.write_extend(arr)

    def producer_extend_unchecked(arr):
        started.set()
        buf.write_extend_unchecked(arr)

    def producer_append(val):
        started.set()
        buf.write_append(val)

    producer_list = [
        producer_extend,
        producer_extend_unchecked,
        producer_append,
    ]

    if HAS_C_TESTS:
        c_append = getattr(
            _test_cython_api, f"bcb_spy_write_append_{DTYPE_TO_SUFFIX[dtype]}"
        )

        def producer_c_append(val):
            started.set()
            c_append(buf, val)

        producer_list.append(producer_c_append)

    for p in producer_list:
        is_append = p == producer_append or (
            HAS_C_TESTS and p == producer_c_append
        )
        _test(p, is_append)
        if not is_append:
            assert np.array_equal(data, buf.read())
        else:
            arr = np.array(
                [parts[i][0] for i in range(_n_test_threads)], dtype=dtype
            )
            assert np.array_equal(arr, buf.read())
