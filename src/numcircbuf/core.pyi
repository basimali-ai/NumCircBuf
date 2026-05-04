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
Core implementations of high-performance circular buffers and statistical accumulators.

This module provides the primary Cython-optimized classes for the NumCircBuf
library. It contains self-contained buffer structures designed for high-speed
numerical data ingestion and provides O(1) complexity for rolling
statistical calculations.
"""

from __future__ import annotations
from typing import Iterable, Literal

import numpy as np

from .constants import Limits
from .protocols import ViewProtocol

class BlockingCircBuffer:
    """
    Blocking multi-producer, multi-consumer (MPMC) circular buffer,
    suitable for multi-threaded applications.

    This buffer blocks under these conditions:
    - Writers wait if the buffer is full or does not have enough space
      for the write.
    - Readers wait if the buffer is empty.

    Operations are mutually exclusive and will not corrupt buffer state.

    Arrival order among waiting writers and among waiting readers
    is guaranteed (FIFO fairness).

    **Note on Thread Integrity and Liveness:**

    To ensure strict FIFO ordering, deterministic execution, and
    maximum performance, this buffer requires that threads performing or
    waiting on operations (read/write) are not terminated abruptly
    (e.g., via SIGKILL or hard-cancellation).

    If a thread is terminated abruptly, the buffer will enter a permanent
    deadlock state. This design choice avoids the inherent unreliability of
    system-level thread monitoring, prioritizing correctness and order
    guarantees under standard operating conditions.
    """

    def __init__(
        self,
        maxlen: int,
        dtype: (
            type[np.float32]
            | type[np.float64]
            | type[np.int32]
            | type[np.int64]
            | type[np.uint32]
            | type[np.uint64]
        ) = np.float32,
    ) -> None:
        """
        :param maxlen: Maximum capacity of the buffer.
        :type maxlen: int

        :param dtype: NumPy dtype to use for the buffer.
        :type dtype: (
            type[np.float32]
            | type[np.float64]
            | type[np.int32]
            | type[np.int64]
            | type[np.uint32]
            | type[np.uint64]
        )
        """

    def write_append(self, value: float | int, timeout: float = -1) -> bool:
        """
        Writes a single value to the buffer.

        Blocks if the buffer doesn't have enough free space.

        Returns True on success, False on timeout.

        :param value: Value to write to the buffer.
        :type value: float | int

        :param timeout:
            Maximum time to wait for space in seconds.

            Use -1 to wait forever, 0 to not wait at all.
        :type timeout: float

        :return:
            True if data was written successfully,
            False if timeout occurred.
        :rtype: bool
        """

    def write_extend(
        self,
        data: Iterable[float | int],
        timeout: float = -1,
        warn_size: bool = True,
    ) -> bool:
        """
        Writes a block of data to the buffer.

        Blocks if the buffer doesn't have enough free space.
        Returns True on success, False on timeout.

        :param data: Block of data to write to the buffer.
        :type data: Iterable[float | int]

        :param timeout:
            Maximum time to wait for space in seconds.

            Use -1 to wait forever, 0 to not wait at all.
        :type timeout: float

        :param warn_size:
            If you want to receive warnings when the size of `data` exceeds the
            buffer's maximum capacity.
        :type warn_size: bool

        :return:
            True if data was written successfully,
            False if timeout occurred.
        :rtype: bool
        """

    def write_extend_unchecked(
        self,
        data: np.ndarray,
        timeout: float = -1,
        warn_size: bool = True,
    ) -> bool:
        """
        Fast writes a 1-D C-Contiguous NumPy array
        of the same dtype as the buffer to the buffer.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        Blocks if the buffer doesn't have enough free space.
        Returns True on success, False on timeout.

        :param data: NumPy array of data to write to the buffer.
        :type data: np.ndarray

        :param timeout:
            Maximum time to wait for space in seconds.

            Use -1 to wait forever, 0 to not wait at all.
        :type timeout: float

        :param warn_size:
            If you want to receive warnings when the size of `data` exceeds the
            buffer's maximum capacity.
        :type warn_size: bool

        :return:
            True if data was written successfully,
            False if timeout occurred.
        :rtype: bool
        """

    def read(
        self,
        n: int = Limits.SIZE_MAX.value,
        timeout: float = -1,
        partial_read: bool = True,
    ) -> np.ndarray:
        """
        Reads data from the buffer into a new NumPy array.

        This method blocks if the buffer does not satisfy the requirements
        specified by `n` and `partial_read`.

        :param n:
            Number of items to read.
            Defaults to the maximum possible buffer size.
        :type n: int

        :param timeout:
            Maximum time to wait for data in seconds.

            Use -1 to wait forever, 0 to not wait at all.

            Returns an empty array if the timeout expires.
        :type timeout: float

        :param partial_read:
            If True (default), the method returns as soon as at least one item
            is available, reading up to `n` items.
            If False, the method blocks until at least `n` items are
            available in the buffer.
        :type partial_read: bool

        :return:
            A 1-D NumPy array containing the read data.
            Returns an empty array if a timeout occurs or if `n=0`.
        :rtype: np.ndarray
        """

    def read_into(
        self,
        out_array_np: np.ndarray,
        timeout: float = -1,
        partial_read: bool = True,
    ) -> int:
        """
        High-performance read into an existing NumPy array (zero allocation).
        Returns number of items read, or 0 on timeout.

        This method validates that the input array is 1-D, C-Contiguous, and
        matches the buffer's dtype before proceeding.

        :param out_array_np:
            Pre-allocated NumPy array to fill with data. The number of items
            to read is determined by the length of this array.
        :type out_array_np: np.ndarray

        :param timeout:
            Maximum time to wait for data in seconds.

            Use -1 to wait forever, 0 to not wait at all.
        :type timeout: float

        :param partial_read:
            If True (default), fills the array with available data as soon as
            at least one item is ready.

            If False, blocks until the buffer contains enough items to
            completely fill `out_array_np`.
        :type partial_read: bool

        :return: Number of items successfully read, or 0 if timeout occurred.
        :rtype: int

        :raises NumCircBufValueError:
            If the input array is not 1-D, not C-Contiguous, or has a
            mismatched dtype.
        """

    def read_into_unchecked(
        self,
        out_array_np: np.ndarray,
        timeout: float = -1,
        partial_read: bool = True,
    ) -> int:
        """
        High-performance read into an existing NumPy array (zero allocation)
        without safety checks.
        Returns number of items read, or 0 on timeout.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        :param out_array: Pre-allocated NumPy array to fill with data.
        :type out_array: np.ndarray

        :param timeout:
            Maximum time to wait for data in seconds.

            Use -1 to wait forever, 0 to not wait at all.
        :type timeout: float

        :param partial_read:
            If True (default), fills the array with available data as soon as
            at least one item is ready.

            If False, blocks until the buffer contains enough items to
            completely fill `out_array_np`.
        :type partial_read: bool

        :return: Number of items successfully read, or 0 if timeout occurred.
        :rtype: int
        """

    def view(self) -> ViewProtocol:
        """
        Returns a live View of the buffer. Using this does not remove any data.

        Useful for real-time inspection of signal levels or history.

        **Note:** This view does not apply locks. While it provides real-time
        access to the underlying buffer, values may be inconsistent if the
        buffer is being modified by another thread during a read operation.
        """

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[np.generic]:
        """Returns the dtype of the buffer."""

    def clear(self) -> None:
        """Atomically clears all data from the buffer."""

    def clear_nans(self) -> None:
        """Removes NaN values from the buffer."""

    def clear_infs(self) -> None:
        """Removes infinite (Inf) values from the buffer."""

class OverwriteCircBuffer:
    """
    Write-optimized circular buffer with auto-overwrite
    and non-destructive live view reads.

    Provides simple vectorized mathematical metrics on all elements, including:
      - `.mean()`: Mean of all elements.
      - `.sum()`: Sum of all elements.
      - `.sum_squares()`: Sum of squares.
      - `.mean_squares()`: Mean of squares.
      - `.sum_and_count_gt(threshold)`: Sum and count of elements above a threshold.

    **Note:** This buffer is not thread safe.
    - Concurrent reads during writes can provide inaccurate data
    - Concurrent writes can cause data corruption.

    """

    def __init__(
        self,
        maxlen: int,
        return_overwritten_policy: Literal["never", "always", "conditional"],
        dtype: (
            type[np.float32]
            | type[np.float64]
            | type[np.int32]
            | type[np.int64]
            | type[np.uint32]
            | type[np.uint64]
        ) = np.float32,
    ) -> None:
        """
        :param maxlen: Maximum capacity of the buffer.
        :type maxlen: int

        :param return_overwritten_policy: If you'd like to receive
            overwritten/popped values per append/extend.

            The available choices are:

            - `"always"` -
            gives you a slower buffer but it always returns any overwritten
            values.

            - `"never"` -
            makes the buffer fastest but it never returns any overwritten
            values.

            - `"conditional"` -
            allows you to set a flag on each extend;
            returns values when the flag is True and vice versa.
            This is best in neither category in terms of speed
            but makes up for it by being more convenient in
            certain situations.

        :type return_overwritten_policy: Literal[
            "never", "always", "conditional"
        ]

        :param dtype: NumPy dtype to use for the buffer.
        :type dtype: (
            type[np.float32]
            | type[np.float64]
            | type[np.int32]
            | type[np.int64]
            | type[np.uint32]
            | type[np.uint64]
        )
        """

    def append(
        self, value: float | int, return_overwritten: bool = False
    ) -> float | None:
        """
        Appends a value, returning the overwritten value if the buffer
        was full, depending on your selected policy.

        (**Not Recommended for large data, Use extend if possible**)

        :param value: value to append
        :type value: float

        :param return_overwritten:
            True if you need overwritten value.
            (Only applies for "conditional" policy)
        :type return_overwritten: bool

        :return: Overwritten value or None
        :rtype: Optional[float]
        """

    def extend(
        self,
        block: Iterable[float | int],
        return_overwritten: bool = False,
        warn_size: bool = True,
    ) -> np.ndarray:
        """
        Extends the buffer with a block of data, returning an array of
        any values that were overwritten or an empty array.

        Always returns an empty array if return_overwritten_policy was "never".

        :param block: Block of data to extend the buffer with
        :type block: Iterable[float | int]

        :param return_overwritten:
            True if you need overwritten values.

            (Only applies for "conditional" policy)
        :type return_overwritten: bool

        :param warn_size:
            If you want to receive warnings when block size exceeds the
            buffer's maximum capacity.
        :type warn_size: bool

        :return:
            If not return_overwritten then always returns an empty np.ndarray
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """

    def extend_unchecked(
        self,
        block_np: np.ndarray,
        return_overwritten: bool = False,
        warn_size: bool = True,
    ) -> np.ndarray:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer, returning an array of
        any values that were overwritten or an empty array.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        Always returns an empty array if return_overwritten_policy was "never".

        :param block_np: NumPy array to extend the buffer with
        :type block_np: np.ndarray

        :param return_overwritten:
            True if you need overwritten values.

            (Only applies for "conditional" policy)
        :type return_overwritten: bool

        :param warn_size:
            If you want to receive warnings when block size exceeds the
            buffer's maximum capacity.
        :type warn_size: bool

        :return:
            If not return_overwritten then always returns an empty np.ndarray
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """

    def view(self) -> ViewProtocol:
        """Returns a read-only, zero-copy live view of the buffer in logical order."""

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[np.generic]:
        """Returns the dtype of the buffer."""

    def sum_squares(self) -> float:
        """
        Calculates sum of squares (x^2).

        **Note:** Only supported for floating-point buffers.
        """

    def mean_squares(self) -> float:
        """
        Calculates mean of squares (x^2).

        **Note:** Only supported for floating-point buffers.
        """

    def sum(self) -> float:
        """
        Calculates the sum of all elements.

        **Note:** Not supported for `int64` and `uint64` due to the risk
        of overflow.
        """

    def mean(self) -> float:
        """
        Calculates the mean of all elements.

        **Note:** Not supported for `int64` and `uint64` due to the risk
        of overflow.
        """

    def sum_and_count_gt(
        self, threshold: float | int
    ) -> tuple[float | int, int]:
        """
        Calculates sum and count of values greater than `threshold`.

        **Note:** Not supported for `int64` and `uint64` due to the risk
        of overflow.
        """

    def clear(self) -> None:
        """Clears all data from the buffer."""

    def clear_nans(self) -> None:
        """Removes NaN values from the buffer."""

    def clear_infs(self) -> None:
        """Removes infinite (Inf) values from the buffer."""

class RunningMeanSqBuffer:
    """
    Accumulator-capable circular buffer optimized for mean-square calculations.

    Features fully vectorized operations, float-drift protection,
    and caching for mean-square.

    **Note:** This buffer is not thread safe.
    - Concurrent reads during writes can return stale mean-square values
      and inaccurate data
    - Concurrent writes can cause data corruption.
    """

    def __init__(
        self,
        maxlen: int,
        operation_focus: Literal["extend/append", "calculation"],
        recalc_threshold: int | None = 0,
        dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> None:
        """
        :param maxlen: Maximum capacity of the buffer.
        :type maxlen: int

        :param operation_focus:
            Which operation should the buffer focus on optimizing the most.

            The available choices are:

            - `"extend/append"` - O(n) statistics, lower write cost.

            - `"calculation"` - O(1) statistics, higher per-write cost.

            Use the library utility `determine_operation_focus` to
            automatically select the best `operation_focus`.
            This function runs a small runtime benchmark
            and returns the appropriate Literal value.
        :type operation_focus: Literal["extend/append", "calculation"]

        :param recalc_threshold:
            Recalculate the sum from scratch every
            N operations to prevent float precision drift.
            0 means no threshold/off.
        :type recalc_threshold: Optional[int]

        :param dtype: NumPy dtype to use for the buffer.
        :type dtype: type[np.float32] | type[np.float64] = np.float32
        """

    def clear_cache(self) -> None:
        """Clears the cached mean_square value."""

    def recalculate(self) -> None:
        """Recalculates sum of squares from the buffer to correct drift."""

    def append(self, value: float | int) -> None:
        """
        Appends a single value.

        (**Not Recommended for large data, Use extend if possible**)

        :param value: Value to append
        :type value: float | int

        """

    def extend(
        self, block: Iterable[float | int], warn_size: bool = True
    ) -> None:
        """
        Extends the buffer with a block of elements using
        vectorized operations.

        :param block: Block of data to extend the buffer with.
        :type block: Iterable[float | int]

        :param warn_size: If you want to receive warnings when block size
         exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def extend_unchecked(
        self, block_np: np.ndarray, warn_size: bool = True
    ) -> None:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        :param block_np: NumPy array to extend the buffer with
        :type block_np: np.ndarray

        :param warn_size: If you want to receive warnings when block size
         exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def mean_square(self) -> float:
        """
        Calculates the mean square.

        Result is cached until the buffer is modified.
        """

    def clear(self) -> None:
        """Resets the buffer's state."""

    def clear_nans(self) -> None:
        """Removes NaN values from the buffer."""

    def clear_infs(self) -> None:
        """Removes infinite (Inf) values from the buffer."""

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[np.generic]:
        """Returns the dtype of the buffer."""

    def view(self) -> ViewProtocol:
        """Returns a read-only, zero-copy live view of the buffer in logical order."""

class RunningMeanBuffer:
    """
    Accumulator-capable circular buffer optimized for mean calculations.

    Features fully vectorized operations, and caching for mean.

    **Note:** This buffer is not thread safe.

    - Concurrent reads during writes can return stale mean-square values
      and inaccurate data
    - Concurrent writes can cause data corruption.
    """

    def __init__(
        self,
        maxlen: int,
        operation_focus: Literal["extend/append", "calculation"],
        dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> None:
        """
        :param maxlen: Maximum capacity of the buffer.
        :type maxlen: int

        :param operation_focus:
            Which operation should the buffer focus on optimizing the most.

            The available choices are:

            - `"extend/append"` - O(n) statistics, lower write cost.

            - `"calculation"` - O(1) statistics, higher per-write cost.

            Use the library utility `determine_operation_focus` to
            automatically select the best `operation_focus`.
            This function runs a small runtime benchmark
            and returns the appropriate Literal value.
        :type operation_focus: Literal["extend/append", "calculation"]

        :param dtype: NumPy dtype to use for the buffer.
        :type dtype: type[np.float32] | type[np.float64] = np.float32
        """

    def clear_cache(self) -> None:
        """Clears the cached mean value."""

    def recalculate(self) -> None:
        """Recalculates sum from the buffer to correct corruption or drift."""

    def append(self, value: float | int) -> None:
        """
        Appends a single value.

        (**Not Recommended for large data, Use extend if possible**)

        :param value: Value to append
        :type value: float | int
        """

    def extend(
        self, block: Iterable[float | int], warn_size: bool = True
    ) -> None:
        """
        Extends the buffer with a block of elements using
        vectorized operations.

        :param block: Block of data to extend the buffer with.
        :type block: Iterable[float | int]

        :param warn_size: If you want to receive warnings when block size
         exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def extend_unchecked(
        self, block_np: np.ndarray, warn_size: bool = True
    ) -> None:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        :param block_np: NumPy array to extend the buffer with
        :type block_np: np.ndarray

        :param warn_size: If you want to receive warnings when block size
         exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def mean(self) -> float:
        """
        Calculates the mean of the values.

        Result is cached until the buffer is modified.
        """

    def clear(self) -> None:
        """Resets the buffer's state."""

    def clear_nans(self) -> None:
        """Removes NaN values from the buffer."""

    def clear_infs(self) -> None:
        """Removes infinite (Inf) values from the buffer."""

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[np.generic]:
        """Returns the dtype of the buffer."""

    def view(self) -> ViewProtocol:
        """Returns a read-only, zero-copy live view of the buffer in logical order."""

    def __repr__(self) -> str:
        """Returns a string representation with the mean value."""

class IntegratedGatedBuffer:
    """
    A specialized circular buffer for calculating gated loudness.

    Features fully vectorized operations, and caching for gated mean-square.

    **Internal Storage:** This buffer stores the **square** of the input values.
    Values retrieved via views will represent squared values,
    not the original linear amplitude.

    **Note:** This buffer is not thread safe.

    - Concurrent reads during writes can return stale gated mean-square values
      and inaccurate data
    - Concurrent writes can cause data corruption.
    """

    def __init__(
        self,
        maxlen: int,
        abs_gate_lufs: float,
        rel_gate_lu: float,
        recalc_threshold: int | None = 0,
        dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> None:
        """
        :param maxlen: Maximum capacity of the buffer.
        :type maxlen: int

        :param abs_gate_lufs:
            The absolute loudness threshold in LUFS.

            Blocks with a mean-square power below this threshold
            are ignored during the first stage of the gating process.
        :type abs_gate_lufs: float

        :param rel_gate_lu:
            The relative loudness threshold in LU.

            Blocks with a mean-square power more than this many decibels below
            the absolute-gated mean-square are ignored in the final integrated
            loudness calculation.
        :type rel_gate_lu: float

        :param recalc_threshold:
            Recalculate the sum from scratch every
            N operations to prevent float precision drift.

            0 means no threshold/off.
        :type recalc_threshold: Optional[int]

        :param dtype: NumPy dtype to use for the buffer.
        :type dtype: type[np.float32] | type[np.float64] = np.float32
        """

    def clear_cache(self) -> None:
        """Clears the cached gated-mean-sq value."""

    def recalculate(self) -> None:
        """Recalculates gated sum and count from the buffer."""

    def append(self, value: float | int) -> None:
        """
        Appends a single linear amplitude value.

        The value is squared internally, then stored in the buffer.

        (**Not Recommended for large data, Use extend if possible**)

        :param value: Value to append
        :type value: float | int
        """

    def extend(
        self, block: Iterable[float | int], warn_size: bool = True
    ) -> None:
        """
        Extends the buffer with a block of linear amplitude values.

        Values are squared internally using vectorized operations,
        then stored in the buffer.

        :param block: Block of linear values to extend the buffer with.
        :type block: Iterable[float | int]

        :param warn_size:
            If you want to receive warnings when block size
            exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def extend_unchecked(
        self, block_np: np.ndarray, warn_size: bool = True
    ) -> None:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer, that contains linear values.

        **WARNING:** This method skips all checks and will cause silent data
        corruption or crashes if the input array has the wrong dtype, shape, or
        memory layout.

        :param block_np:
            NumPy array of already linear values to extend
            the buffer with.
        :type block_np: np.ndarray

        :param warn_size:
            If you want to receive warnings when block size
            exceeds the buffer's maximum capacity.
        :type warn_size: bool
        """

    def gated_mean_square(self) -> float:
        """
        Calculates the final gated mean square value.

        Result is cached until the buffer is modified.
        """

    def clear(self) -> None:
        """
        Resets the buffer's state.
        """

    def clear_nans(self) -> None:
        """Removes NaN values from the buffer."""

    def clear_infs(self) -> None:
        """Removes infinite (Inf) values from the buffer."""

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[np.generic]:
        """Returns the dtype of the buffer."""

    def view(self) -> ViewProtocol:
        """Returns a read-only, zero-copy live view of the buffer in logical order."""
