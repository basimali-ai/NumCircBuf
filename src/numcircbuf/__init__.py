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
NumCircBuf: High-performance numerical circular buffers for Python, featuring O(1) statistical accumulators.

Source Code & Documentation:
https://github.com/basimali-ai/NumCircBuf
"""

from importlib.metadata import version

from .core import (
    BlockingCircBuffer,
    IntegratedGatedBuffer,
    OverwriteCircBuffer,
    RunningMeanBuffer,
    RunningMeanSqBuffer,
)
from .utils import determine_operation_focus

__version__ = version("numcircbuf")

__all__ = [
    "BlockingCircBuffer",
    "IntegratedGatedBuffer",
    "OverwriteCircBuffer",
    "RunningMeanBuffer",
    "RunningMeanSqBuffer",
    "determine_operation_focus",
]
