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

from numcircbuf.constants import (
    SupportedDtypesFP,
    SupportedDtypesAll,
    Limits,  # noqa: F401
)

CAPACITIES = (1, 3, 10, 15, 64, 100, 128, 1024)

SUPPORTED_DTYPES_ALL = tuple(a.value for a in SupportedDtypesAll)
SUPPORTED_DTYPES_FP = tuple(a.value for a in SupportedDtypesFP)

DTYPE_TO_SUFFIX = {
    SupportedDtypesAll.FLOAT64.value: "f64",
    SupportedDtypesAll.FLOAT32.value: "f32",
    SupportedDtypesAll.INT64.value: "i64",
    SupportedDtypesAll.INT32.value: "i32",
    SupportedDtypesAll.UINT64.value: "u64",
    SupportedDtypesAll.UINT32.value: "u32",
}
