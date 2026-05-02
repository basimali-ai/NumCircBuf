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

from __future__ import annotations
import os
import sys
import glob
import logging
import ctypes
from functools import lru_cache
from mmap import PAGESIZE as _MMAP_PAGESIZE

from .constants import TYPICAL_CACHE_LINE
from .exceptions import NumCircBufRuntimeError, NumCircBufOSError

logger = logging.getLogger(__name__)

if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


PAGESIZE = _MMAP_PAGESIZE
"""Memory page size in bytes for the current system."""


# --- Cache Line Size ---

# - Platform Specific Getters -


def _get_cache_line_size_windows() -> int | None:
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    GetLogicalProcessorInformation = kernel32.GetLogicalProcessorInformation
    GetLastError = kernel32.GetLastError

    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("Level", wintypes.BYTE),
            ("Associativity", wintypes.BYTE),
            ("LineSize", wintypes.USHORT),
            ("Size", wintypes.DWORD),
            ("Type", wintypes.DWORD),  # PROCESSOR_CACHE_TYPE
        ]

    class PROCESSORCORE(ctypes.Structure):
        _fields_ = [("Flags", wintypes.BYTE)]

    class NUMANODE(ctypes.Structure):
        _fields_ = [("NodeNumber", wintypes.DWORD)]

    class _U(ctypes.Union):
        _fields_ = [
            ("ProcessorCore", PROCESSORCORE),
            ("NumaNode", NUMANODE),
            ("Cache", CACHE_DESCRIPTOR),
            ("Reserved", ctypes.c_ulonglong),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("ProcessorMask", ctypes.c_ulonglong),
            ("Relationship", wintypes.DWORD),
            ("u", _U),
        ]

    buf_size = wintypes.DWORD(0)
    res = GetLogicalProcessorInformation(None, ctypes.byref(buf_size))
    ERROR_INSUFFICIENT_BUFFER = 122
    if res == 0 and GetLastError() != ERROR_INSUFFICIENT_BUFFER:
        return None

    byte_count = buf_size.value
    if byte_count == 0:
        return None
    raw = (ctypes.c_byte * byte_count)()
    ptr = ctypes.cast(
        raw, ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
    )

    res = GetLogicalProcessorInformation(ptr, ctypes.byref(buf_size))
    if res == 0:
        return None

    entry_size = ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
    entries = buf_size.value // entry_size
    found = set()
    RelationCache = 2  # from SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX enum

    for i in range(entries):
        entry = ptr[i]
        if entry.Relationship == RelationCache:
            line = int(entry.Cache.LineSize)
            if line > 0:
                found.add(line)

    if found:
        return int(min(found))

    return None


def _get_cache_line_size_linux() -> int | None:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r") as f:
            val = f.read().strip()
            if val:
                return int(val)

    path = "/proc/cpuinfo"
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                low = line.lower()
                if "clflush size" in low:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        return int(parts[1].strip())

    return None


def _get_cache_line_size_darwin() -> int | None:
    libc = ctypes.CDLL(None)
    name = b"hw.cachelinesize"
    val = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(val))
    if (
        libc.sysctlbyname(name, ctypes.byref(val), ctypes.byref(size), None, 0)
        == 0
    ):
        return val.value

    return None


# - Unified Getters -


def _get_cache_line_size() -> int | None:
    try:
        if sys.platform == "win32":
            return _get_cache_line_size_windows()
        elif sys.platform == "linux":
            return _get_cache_line_size_linux()
        elif sys.platform == "darwin":
            return _get_cache_line_size_darwin()

    except Exception:
        logger.exception(
            "Exception occured while trying to determine Cache Line size."
        )

    return None


@lru_cache(maxsize=1)
def get_cache_line_size(default: int = TYPICAL_CACHE_LINE) -> int:
    """
    Return the CPU cache line size in bytes if successfully queried,
    otherwise `default`.

    Result is cached so repeated calls are comparatively low cost.
    """
    res = _get_cache_line_size()
    if res is not None and res > 0:
        return res
    return default


# --- CPU L3 Cache Size ---

# - Platform Specific Getters -


def _get_l3_win32() -> int | None:
    from ctypes import wintypes

    RelationCache = 2  # from SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX enum

    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("Level", ctypes.c_byte),
            ("Associativity", ctypes.c_byte),
            ("LineSize", ctypes.c_ushort),
            ("Size", ctypes.c_uint),
            ("Type", ctypes.c_uint),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_UNION(ctypes.Union):
        _fields_ = [
            ("Cache", CACHE_DESCRIPTOR),
            ("Reserved", ctypes.c_ulonglong * 2),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
        _fields_ = [
            ("Relationship", ctypes.c_int),
            ("Size", ctypes.c_uint),
            ("u", SYSTEM_LOGICAL_PROCESSOR_INFORMATION_UNION),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetLogicalProcessorInformationEx = (
        kernel32.GetLogicalProcessorInformationEx
    )
    GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
        ctypes.POINTER(ctypes.c_uint),
    ]
    GetLogicalProcessorInformationEx.restype = wintypes.BOOL

    buffer_size = ctypes.c_uint(0)
    GetLogicalProcessorInformationEx(
        RelationCache, None, ctypes.byref(buffer_size)
    )
    buf = (ctypes.c_byte * buffer_size.value)()
    res = GetLogicalProcessorInformationEx(
        RelationCache,
        ctypes.cast(
            buf, ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
        ),
        ctypes.byref(buffer_size),
    )
    if not res:
        return None

    offset = 0
    while offset < buffer_size.value:
        info = ctypes.cast(
            ctypes.byref(buf, offset),
            ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
        ).contents
        if info.Relationship == RelationCache and info.u.Cache.Level == 3:
            return int(info.u.Cache.Size)
        offset += info.Size
    return None


def _get_l3_linux() -> int | None:
    base = "/sys/devices/system/cpu/cpu0/cache"
    if not os.path.isdir(base):
        return None
    for idx in sorted(glob.glob(os.path.join(base, "index*"))):
        try:
            with open(os.path.join(idx, "level")) as f:
                level = int(f.read().strip())
            if level != 3:
                continue
            with open(os.path.join(idx, "size")) as f:
                s = f.read().strip()
            if s.endswith("K"):
                return int(s[:-1])
            if s.endswith("M"):
                return int(s[:-1])
            return int(s)
        except Exception:  # pragma: no cover
            continue
    return None


def _get_l3_darwin() -> int | None:
    libc = ctypes.CDLL(None)

    for name in [b"hw.l3cachesize", b"hw.l2cachesize"]:
        val = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(val))

        if (
            libc.sysctlbyname(
                name, ctypes.byref(val), ctypes.byref(size), None, 0
            )
            == 0
        ):
            return val.value

    return None


# - Unified Getters -


def _get_l3_cache() -> int | None:
    try:
        if sys.platform == "win32":
            return _get_l3_win32()
        elif sys.platform == "linux":
            return _get_l3_linux()
        elif sys.platform == "darwin":
            return _get_l3_darwin()

    except Exception:
        logger.exception(
            "Exception occured while trying to determine CPU L3 Cache."
        )

    return None


@lru_cache(maxsize=1)
def get_cpu_l3_cache(default: int = 268_435_456) -> int:
    """
    Return the CPU L3 cache in bytes if successfully queried,
    otherwise `default`.

    Result is cached so repeated calls are comparatively low cost.
    """
    res = _get_l3_cache()
    if res is not None and res > 0:
        return res
    return default


# --- Current available RAM  ---

# - Platform Specific Getters -


def _get_available_ram_windows() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

    return int(stat.ullAvailPhys)


def _get_available_ram_linux() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) * 1024

    raise NumCircBufRuntimeError(message="MemAvailable not found")


def _get_available_ram_darwin() -> int:
    libc = ctypes.CDLL("libSystem.dylib", use_errno=True)

    class vm_statistics64(ctypes.Structure):
        _fields_ = [
            ("free_count", ctypes.c_uint32),
            ("active_count", ctypes.c_uint32),
            ("inactive_count", ctypes.c_uint32),
            ("wire_count", ctypes.c_uint32),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint32),
            ("speculative_count", ctypes.c_uint32),
        ]

    HOST_VM_INFO64 = 4

    libc.mach_host_self.restype = ctypes.c_uint
    host_port = libc.mach_host_self()

    vm_stat = vm_statistics64()
    count = ctypes.c_uint32(ctypes.sizeof(vm_stat) // 4)

    if (
        libc.host_statistics64(
            host_port,
            HOST_VM_INFO64,
            ctypes.byref(vm_stat),
            ctypes.byref(count),
        )
        != 0
    ):
        raise NumCircBufOSError(message="host_statistics64 failed")

    return int(vm_stat.free_count + vm_stat.inactive_count) * PAGESIZE


def get_available_ram() -> int | None:
    """
    Return the current available RAM in bytes if successfully queried and available,
    otherwise `None`
    """
    try:
        if sys.platform == "win32":
            res = _get_available_ram_windows()
            if res > 0:
                return res

        elif sys.platform == "linux":
            res = _get_available_ram_linux()
            if res > 0:
                return res

        elif sys.platform == "darwin":
            res = _get_available_ram_darwin()
            if res > 0:
                return res

    except Exception:
        logger.exception(
            "Exception occured while trying to get available RAM."
        )

    return None
