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
General utility helpers for the NumCircBuf library.

Contains internal helper tools, including a custom `classproperty`
decorator and the `determine_operation_focus` logic used to optimize
buffer behavior based on specific workload types.
"""

from __future__ import annotations
import uuid
import logging
import time
from typing import Callable, Literal, TYPE_CHECKING, Any, TypeVar, Generic, TypedDict

import numpy as np

from .exceptions import NumCircBufTypeError, NumCircBufValueError
from .constants import Limits

if TYPE_CHECKING:
    from .core import RunningMeanBuffer, RunningMeanSqBuffer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

R = TypeVar("R")
PY_SSIZE_T_MAX = Limits.PY_SSIZE_T_MAX.value


class classproperty(Generic[R]):
    """
    A decorator that converts a method into a read-only property of the class.

    This allows a method to be accessed as an attribute on the class itself
    (and its instances), similar to how @property works for instances. The
    decorated method will receive the class (cls) as its only argument.

    Example:
    >>> class MyClass:
    ...     @classproperty
    ...     def name(cls):
    ...         return cls.__name__
    >>> MyClass.name
    'MyClass'
    """

    def __init__(self, f: Callable[[Any], R]):
        """
        Initialize the class property.

        :param f: The getter function to be wrapped.
        """
        self.f = f

    def __get__(self, obj: Any, cls: type) -> R:
        """
        Descriptor protocol to return the value of the property.

        Dispatches the call to the underlying function, passing the class
        regardless of whether the access is via the class or an instance.

        :param obj: The instance that the attribute was accessed through, or None.
        :param cls: The class that the attribute was accessed through.
        :return: The result of the wrapped function.
        """
        return self.f(cls)


def determine_operation_focus(
    buffer_type: "type[RunningMeanSqBuffer] | type[RunningMeanBuffer]",
    dtype: type[np.float32] | type[np.float64],
    buffer_maxlen: int,
    block_size: int,
    calc_every: int,
    verbose: bool = False,
) -> Literal["calculation", "extend/append"]:
    """
    Empirically determine the optimal performance strategy for a specific buffer configuration.

    This function executes a micro-benchmark comparing the two internal optimization
    strategies: `"calculation"` (which optimizes the mathematical reduction) and
    `"extend/append"` (which optimizes the data insertion logic). The result is
    based on the specific hardware, data type, buffer size, and the frequency of
    calculation calls.

    The benchmark size is determined deterministically using system RAM constraints
    and the frequency of calculation. The iteration count is forced to a power of
    two to maintain benchmark consistency.

    The returned string is intended to be passed directly to the `operation_focus`
    parameter during the instantiation of a `RunningMeanBuffer` or
    `RunningMeanSqBuffer`.

    :param buffer_type: The buffer class to be benchmarked.
    :type buffer_type: Type[RunningMeanSqBuffer] | Type[RunningMeanBuffer]

    :param dtype: The NumPy floating-point data type to use for the benchmark.
    :type dtype: Type[np.float32] | Type[np.float64]

    :param buffer_maxlen: The maximum capacity of the circular buffer.
    :type buffer_maxlen: int

    :param block_size: The number of elements to be written in each write operation.
    :type block_size: int

    :param calc_every: The frequency of calculation calls (e.g., every N writes).
    :type calc_every: int

    :param verbose: If True, logs detailed speedup comparisons and execution time
        to the logger. Defaults to False.
    :type verbose: bool, optional

    :return: The optimization focus that yielded the lowest total execution time.
    :rtype: Literal["calculation", "extend/append"]

    :raises NumCircBufTypeError:
        If an unsupported `buffer_type` or `dtype` is provided.
    :raises NumCircBufValueError:
        If `calc_every <= 0` or `block_size <= 0` or
        `buffer_maxlen <= 2` or `buffer_maxlen` > :meth:`PY_SSIZE_T_MAX`
    """

    if calc_every <= 0:
        raise NumCircBufValueError(message="`calc_every` must be greater than 0")

    if block_size <= 0:
        raise NumCircBufValueError(message="`block_size` must be greater than 0")

    if buffer_maxlen > PY_SSIZE_T_MAX:
        raise NumCircBufValueError(
            message="`buffer_maxlen` must be less than `PY_SSIZE_T_MAX`"
        )

    if buffer_maxlen <= 2:
        raise NumCircBufValueError(message="`buffer_maxlen` must be greater than 2")

    if dtype not in (np.float64, np.float32):
        raise NumCircBufTypeError(
            message="`dtype` must be `np.float64` or `np.float32`"
        )

    from .core import RunningMeanBuffer, RunningMeanSqBuffer

    if issubclass(buffer_type, RunningMeanSqBuffer):
        func_str = "mean_square"
    elif issubclass(buffer_type, RunningMeanBuffer):
        func_str = "mean"
    else:
        raise NumCircBufTypeError(
            message=f"Invalid `buffer_type` passed: {buffer_type}; "
            f"expected {RunningMeanSqBuffer} or {RunningMeanBuffer}"
        )

    from .bench_utils import (
        temporary_benchmark_data,
        raw_bench_with_calc,
        determine_num_runs,
    )
    from .system_info import get_available_ram

    ram_bytes = get_available_ram() or (1024**3)
    budget = min(ram_bytes // 10, 512 * 1024**2)
    elem_bytes = np.dtype(dtype).itemsize

    try:
        max_possible_runs, _, _ = determine_num_runs(
            elem_bytes=elem_bytes,
            total_byte_limit=budget,
            maxlen=buffer_maxlen,
            block_size=block_size,
            with_fill=True,
        )

        target = min(max(128, calc_every * 10), max_possible_runs)
        n_runs = 1 << (target.bit_length() - 1)

    except NumCircBufValueError:
        n_runs = 0

    if n_runs < calc_every:
        logger.warning(
            f"Benchmark precision warning: `calc_every` ({calc_every}) is greater "
            f"than the calculated `n_runs` ({n_runs}). The calculation "
            "overhead cannot be safely measured within memory limits. "
            "Defaulting to 'extend/append' optimization."
        )
        return "extend/append"

    start_time = time.time()
    unique_id = uuid.uuid4().hex[:6]

    if TYPE_CHECKING:

        class TempDataPaths(TypedDict):
            data_path: str
            warmup_path: str
            offset_path: str
            fill_path: str
            evict_path: str

    temp_data_paths: "TempDataPaths" = {
        "data_path": f"temp_bench_data_{unique_id}.dat",
        "warmup_path": f"temp_bench_warmup_{unique_id}.dat",
        "offset_path": f"temp_bench_offset_{unique_id}.dat",
        "fill_path": f"temp_bench_fill_{unique_id}.dat",
        "evict_path": f"temp_bench_evict_{unique_id}.dat",
    }

    with temporary_benchmark_data(
        dtype,
        buffer_maxlen,
        block_size,
        n_runs,
        create_offset_data=True,
        create_fill_data=True,
        create_evict_arr=True,
        **temp_data_paths,
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
        times = {}

        for operation_focus in ("calculation", "extend/append"):
            buffer = buffer_type(
                maxlen=buffer_maxlen,
                operation_focus=operation_focus,
                dtype=dtype,
            )
            times[operation_focus] = raw_bench_with_calc(
                buffer,  # type: ignore[arg-type]
                getattr(buffer, func_str),
                fill_data,
                offset_data,
                warmup_data,
                data,
                calc_every,
                n_runs,
                evict_arr,
            )

        if verbose:
            elapsed = time.time() - start_time

            calculation_speedup = times["extend/append"] / times["calculation"]
            extend_speedup = times["calculation"] / times["extend/append"]

            logger.info(
                "\nSpeed Comparison ('calculation' vs 'extend/append'):"
                f"\n  'calculation' speedup = {calculation_speedup:.2f}×"
                f"\n  'extend/append' speedup = {extend_speedup:.2f}×"
                f"\nTotal time taken for data generation + benchmark: {elapsed:.3f} s\n"
            )

        if times["calculation"] < times["extend/append"]:
            return "calculation"
        else:
            return "extend/append"
