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
Type protocols and interfaces for the NumCircBuf library.

Defines PEP 544 structural protocols used for static type checking and
runtime validation. These protocols ensure compatibility between buffer
implementations and view interfaces. Most protocols are runtime-checkable,
allowing the use of `isinstance()` to verify implementation compliance.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Generic, Protocol, overload, runtime_checkable

import numpy as np

from ._typing import ConcreteFloatingT, ConcreteIntegerT, ConcreteScalarT_co

if TYPE_CHECKING:
    from .exceptions import IndexOutOfBounds, InvalidModification  # noqa: F401


@runtime_checkable
class ViewProtocol(Protocol, Generic[ConcreteScalarT_co]):
    """
    Provides a zero-copy, read-only logical view of a circular buffer.

    Behaves like a 1D NumPy array in logical order.

    Has a `to_numpy()` function which provides a copy of the buffer in
    logical order.

    Any slicing produces a copy containing only the selected elements.

    Indexing or iteration preserves logical order, but yields Python objects
    for each element.

    All returned arrays are independent (no shared memory).

    Note:
        This view is strictly read-only. Attempts to modify the view via
        indexing or deletion will raise :exc:`InvalidModification`
        at runtime.
    """

    def __len__(self) -> int:
        """Returns the current number of items in the buffer."""

    @overload
    def __getitem__(self: ViewProtocol[ConcreteFloatingT], idx: int) -> float:
        """
        Retrieves an item or a slice in logical order.

        - If `idx` is an integer:
            Returns an object of the corresponding
            native Python type (`int`, `float`).
        - If `idx` is a slice:
            Returns a 1-D NumPy array of the buffer's dtype.

        :raises :exc:`IndexOutOfBounds`: If the index is out of bounds.
        """

    @overload
    def __getitem__(self: ViewProtocol[ConcreteIntegerT], idx: int) -> int: ...

    @overload
    def __getitem__(
        self: ViewProtocol[ConcreteScalarT_co], idx: slice
    ) -> np.ndarray[tuple[int], np.dtype[ConcreteScalarT_co]]: ...

    @overload
    def __iter__(self: ViewProtocol[ConcreteFloatingT]) -> Iterator[float]:
        """
        Iterates over the buffer in logical order.

        Returns objects of the corresponding native Python type
        (`int`, `float`).
        """

    @overload
    def __iter__(self: ViewProtocol[ConcreteIntegerT]) -> Iterator[int]: ...

    @property
    def maxlen(self) -> int:
        """Returns the maximum capacity of the buffer."""

    @property
    def dtype(self) -> type[ConcreteScalarT_co]:
        """Returns the dtype of the buffer."""

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[ConcreteScalarT_co]]:
        """
        Returns a contiguous NumPy array copy of the data in logical order.
        """


@runtime_checkable
class WriteBenchmarkBufferProtocol(Protocol):
    """
    Protocol for Write Benchmark of a Buffer
    """

    def extend_unchecked(self, block_np: np.ndarray) -> Any:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer.

        This method should be skipping all checks.

        :param block_np: NumPy array to extend the buffer with
        :type block_np: np.ndarray
        """

    def append(self, value: float | int) -> Any:
        """
        Appends a single value to the buffer.

        :param value: value to append
        :type value: float
        """

    def clear(self) -> Any:
        """
        Clears all data from the buffer.

        Keep in mind this should just be setting `size` and `write_head` to 0
        """


@runtime_checkable
class ReadWriteBenchmarkBufferProtocol(Protocol):
    """
    Protocol for Read and Write Benchmark of a Buffer
    """

    def read(self) -> Any:
        """
        Reads all valid items from the buffer.
        """

    def read_into(self, out_array_np: np.ndarray) -> Any:
        """
        High-Performance Read. Fills existing np array (zero allocation).

        :param out_array_np: Pre-allocated NumPy array to fill with data.
        :type out_array_np: np.ndarray
        """

    def write_extend_unchecked(self, block_np: np.ndarray) -> Any:
        """
        Fast extends the buffer with a 1-D C-Contiguous NumPy Array
        of the same dtype as the buffer.

        This method should be skipping all checks.

        :param block_np: NumPy array to extend the buffer with
        :type block_np: np.ndarray
        """

    def write_append(self, value: float | int) -> Any:
        """
        Appends a single value to the buffer.

        :param value: value to append
        :type value: float
        """

    def clear(self) -> Any:
        """
        Clears all data from the buffer.

        Keep in mind this should just be setting `size`, `write_head` and `read_head` to 0
        """
