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
Internal and public constants for the NumCircBuf library.

Defines minimum/maximum bounds for common numeric types and supported
NumPy-compatible data types. Includes typical architecture-standard
constants, such as common cache line and memory page sizes.
"""

from enum import IntEnum, Enum
import ctypes
import sys

import numpy as np

TYPICAL_CACHE_LINE = 64
"""Typical CPU cache line size in bytes."""

TYPICAL_PAGESIZE = 4096
"""Typical memory page size in bytes."""


class SupportedDtypesFP(Enum):
    """Supported floating-point NumPy dtypes"""

    FLOAT64 = np.float64
    FLOAT32 = np.float32


class SupportedDtypesINT(Enum):
    """Supported integer NumPy dtypes"""

    INT64 = np.int64
    INT32 = np.int32


class SupportedDtypesUINT(Enum):
    """Supported unsigned integer NumPy dtypes"""

    UINT64 = np.uint64
    UINT32 = np.uint32


class SupportedDtypesAll(Enum):
    """All Supported NumPy dtypes"""

    FLOAT64 = np.float64
    FLOAT32 = np.float32
    INT64 = np.int64
    INT32 = np.int32
    UINT64 = np.uint64
    UINT32 = np.uint32


class Limits(IntEnum):
    """Min/max constants for common numeric types"""

    # Runtime-dependent
    PY_SSIZE_T_MIN = -sys.maxsize - 1
    PY_SSIZE_T_MAX = sys.maxsize

    SIZE_MIN = 0
    SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1

    # Signed integers
    INT64_MIN = -9_223_372_036_854_775_808
    INT64_MAX = 9_223_372_036_854_775_807

    INT32_MIN = -2_147_483_648
    INT32_MAX = 2_147_483_647

    # Unsigned integers
    UINT64_MIN = 0
    UINT64_MAX = 18_446_744_073_709_551_615

    UINT32_MIN = 0
    UINT32_MAX = 4_294_967_295
