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

from enum import Enum


class ClearTargets(str, Enum):
    NANS = "nans"
    INFS = "infs"


class Suffixes(str, Enum):
    FP = "f"
    F64 = "f64"
    F32 = "f32"
    INT = "i"
    I64 = "i64"
    I32 = "i32"
    UINT = "u"
    U64 = "u64"
    U32 = "u32"


class Ctypes(str, Enum):
    F64 = "double"
    F32 = "float"
    I64 = "int64_t"
    I32 = "int32_t"
    U64 = "uint64_t"
    U32 = "uint32_t"


class NpyConstName(str, Enum):
    F64 = "NPY_FLOAT64"
    F32 = "NPY_FLOAT32"
    I64 = "NPY_INT64"
    I32 = "NPY_INT32"
    U64 = "NPY_UINT64"
    U32 = "NPY_UINT32"


class SupportedDtype(str, Enum):
    F64 = "SUPPORTED_DTYPE_FLOAT64"
    F32 = "SUPPORTED_DTYPE_FLOAT32"
    I64 = "SUPPORTED_DTYPE_INT64"
    I32 = "SUPPORTED_DTYPE_INT32"
    U64 = "SUPPORTED_DTYPE_UINT64"
    U32 = "SUPPORTED_DTYPE_UINT32"


class SumCountStructType(str, Enum):
    F64 = "SumCountDouble"
    F32 = "SumCountFloat"
    I64 = "SumCountInt64"
    U64 = "SumCountUint64"


CLEAR_TARGETS = ("nans", "infs")

FP_TYPE_DEFINITIONS_EXPANDED = (
    (
        Suffixes.F64.value,
        Ctypes.F64.value,
        NpyConstName.F64.value,
        SupportedDtype.F64.value,
    ),
    (
        Suffixes.F32.value,
        Ctypes.F32.value,
        NpyConstName.F32.value,
        SupportedDtype.F32.value,
    ),
)
TYPE_DEFINITIONS_EXPANDED = FP_TYPE_DEFINITIONS_EXPANDED + (
    (
        Suffixes.I64.value,
        Ctypes.I64.value,
        NpyConstName.I64.value,
        SupportedDtype.I64.value,
    ),
    (
        Suffixes.I32.value,
        Ctypes.I32.value,
        NpyConstName.I32.value,
        SupportedDtype.I32.value,
    ),
    (
        Suffixes.U64.value,
        Ctypes.U64.value,
        NpyConstName.U64.value,
        SupportedDtype.U64.value,
    ),
    (
        Suffixes.U32.value,
        Ctypes.U32.value,
        NpyConstName.U32.value,
        SupportedDtype.U32.value,
    ),
)
MATH_GROUPS = (
    (Suffixes.FP.value, Ctypes.F64.value, "self._dtype"),
    (Suffixes.INT.value, Ctypes.I64.value, "np.int64"),
    (Suffixes.UINT.value, Ctypes.U64.value, "np.uint64"),
)
FP_SUM_COUNT_GROUPS = (
    (
        Suffixes.F64.value,
        Ctypes.F64.value,
        SumCountStructType.F64.value,
        NpyConstName.F64.value,
    ),
    (
        Suffixes.F32.value,
        Ctypes.F32.value,
        SumCountStructType.F32.value,
        NpyConstName.F32.value,
    ),
)
SUM_COUNT_GROUPS = FP_SUM_COUNT_GROUPS + (
    (
        Suffixes.INT.value,
        Ctypes.I64.value,
        SumCountStructType.I64.value,
        NpyConstName.I32.value,
    ),
    (
        Suffixes.UINT.value,
        Ctypes.U64.value,
        SumCountStructType.U64.value,
        NpyConstName.U32.value,
    ),
)
