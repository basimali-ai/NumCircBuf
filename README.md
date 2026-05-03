# NumCircBuf: High-performance numerical circular buffers for Python, featuring O(1) statistical accumulators.

[![PyPI version](https://badge.fury.io/py/NumCircBuf.svg)](https://badge.fury.io/py/NumCircBuf)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/NumCircBuf.svg)](https://pypi.org/project/NumCircBuf/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/basimali-ai/NumCircBuf/build_wheels.yml)](https://github.com/basimali-ai/NumCircBuf/actions)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Common Buffer API](#common-buffer-api)
  - [The Buffer View (ViewProtocol)](#the-buffer-view-viewprotocol)
- [Main Buffers](#main-buffers)
  - [1. OverwriteCircBuffer](#1-overwritecircbuffer)
  - [2. BlockingCircBuffer](#2-blockingcircbuffer)
- [Utility Buffers](#utility-buffers)
  - [Common Utility Buffer API](#common-utility-buffer-api)
  - [1. RunningMeanSqBuffer](#1-runningmeansqbuffer)
  - [2. RunningMeanBuffer](#2-runningmeanbuffer)
  - [3. IntegratedGatedBuffer](#3-integratedgatedbuffer)
- [Exception Handling](#exception-handling)
  - [Exceptions](#exceptions)
  - [Warnings](#warnings)
- [Performance](#performance)
- [Documentation](#documentation)
- [Changelog](#changelog)
- [License](#license)
- [Support](#support)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

NumCircBuf is a high-performance Python library providing numerical circular buffers, including O(1) accumulators and specialized calculation variants.
Built with Cython for a balance between performance and maintainability, it provides efficient data structures for real-time signal processing, time-series analysis, and other performance-critical applications.

## Features

- **Multiple Buffer Types** – includes specialized buffers for different needs:
  - **BlockingCircBuffer** – blocking producer/consumer circular buffer for multi-threaded applications.
  - **OverwriteCircBuffer** – optimized for high-throughput writes
  - **IntegratedGatedBuffer** – specialized for calculating gated loudness/energy statistics
  - **O(1) Accumulators** – constant-time operations for statistics, implemented in specialized buffers:
    - **RunningMeanBuffer** – O(1) mean
    - **RunningMeanSqBuffer** – O(1) mean-square
- **NumPy Integration**: Seamless integration with NumPy arrays.
- **Type Safety**: supports fp32, fp64, int32, int64, uint32, uint64.
- **Comprehensive Documentation**: Detailed documentation and performance benchmarks.

## Installation

```bash
pip install numcircbuf
```

Or install from source:

```bash
git clone https://github.com/basimali-ai/NumCircBuf.git
cd NumCircBuf
pip install .
```

For development installation with all dependencies (from source):

```bash
git clone https://github.com/basimali-ai/NumCircBuf.git
cd NumCircBuf
pip install -e .[dev]
```

### Verification

To verify the installation and check the version:

```bash
pip show numcircbuf && python -c "import numcircbuf; print('\n' + '-'*20 + '\nRunning smoke test...'); print(f'Library version: {numcircbuf.__version__}'); print(numcircbuf.OverwriteCircBuffer(10, 'never')); print(numcircbuf.RunningMeanBuffer(10, 'calculation')); print('-'*20 + '\n--- Installation verification successful ---')"
```

## Common Buffer API

All buffer types expose the following methods:

- `clear()`: Clears the buffer.
- `clear_nan()`: Removes NaN values.
- `clear_infs()`: Removes Inf values.
- `__len__()`: Current buffer size.
- `maxlen`: Maximum buffer capacity.
- `view()`: Returns a read-only logical view of the buffer (implements [ViewProtocol](#the-buffer-view-viewprotocol)).

All bulk `extend` methods have an additional `warn_size` boolean argument which enables or disables warnings when the block size exceeds the buffer's `maxlen`. It is enabled by default.

**Note:** In all buffers nan/inf values are allowed to be appended/extended with, as they are valuable data points.

### The Buffer View (ViewProtocol)

The object returned by the `view()` method provides a zero-copy, read-only logical view of the circular buffer. It is exposed as a structural `Protocol` to provide strong type hints and an explicit contract, while the underlying implementation remains internal.

- Behaves like a 1D NumPy array in logical order.
- Has a `to_numpy()` function which provides a contiguous NumPy array copy of the data in logical order.
- All returned arrays are independent (no shared memory).
- Any slicing produces a 1D NumPy array copy containing only the selected elements.
- Indexing or iteration preserves logical order, but yields native Python objects (`int`, `float`) for each element.

**Note:** This view is strictly read-only. Attempts to modify the view via indexing or deletion will raise an `InvalidModification` exception at runtime.

#### Usage Example

```python
# Assuming `buffer` is a populated circular buffer instance from numcircbuf
view = buffer.view()

def view_usage(v):
    print("All elements (slice):", v[:])
    print("Single element:", v[0])
    print("Number of elements:", len(v))
    print("Max capacity:", v.maxlen)
    print("Dtype:", v.dtype)
    print("-" * 20)
    print("Iterating over view:")
    for value in v:
        print("-" * 10)
        print(f"Value = {value}")
        print(f"Type = {type(value)}")  # Will be native <class 'int'> or <class 'float'>
    print("-" * 20)

print("\n--- View Usage ---")
view_usage(view)

# Copy behavior
# All returned arrays are independent; no shared memory
arr = view.to_numpy()
print("Before: ", view[:])
arr *= 2.0  # modify the array copy
print("After: ", view[:])  # View remains unchanged
```

## Main Buffers

### 1. OverwriteCircBuffer

Write-optimized circular buffer with auto-overwrite and non-destructive reads.

Provides simple vectorized mathematical metrics over the buffer contents.

**Note:** This buffer is not thread safe. Concurrent writes can cause data corruption.

#### Constructor

```python
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
```

##### Overwrite Return Policy

Controls whether overwritten elements are returned when the buffer wraps.

**`return_overwritten_policy: Literal["always", "never", "conditional"]`**

- **`"never"`**

  ```python
  from numcircbuf import OverwriteCircBuffer
  buf = OverwriteCircBuffer(maxlen=3, return_overwritten_policy="never")
  print(buf.extend([1, 2]))  # Empty NumPy array []
  print(buf.extend([3, 4]))  # Empty NumPy array []
  print(buf.append(5))  # None
  ```

- **`"always"`**

  ```python
  from numcircbuf import OverwriteCircBuffer
  buf = OverwriteCircBuffer(maxlen=3, return_overwritten_policy="always")
  print(buf.extend([1, 2]))  # Empty NumPy array []
  print(buf.extend([3, 4]))  # NumPy array [1.0]
  print(buf.append(5))  # float 2.0
  ```

- **`"conditional"`**

  > The returned values depend on the `return_overwritten` flag, it defaults to `False`.

  ```python
  from numcircbuf import OverwriteCircBuffer
  buf = OverwriteCircBuffer(maxlen=3, return_overwritten_policy="conditional")
  print(buf.extend([1, 2]))  # Empty NumPy array []
  print(buf.extend([3, 4]))  # Empty NumPy array []
  print(buf.extend([5, 6], return_overwritten=True))  # NumPy array [2.0, 3.0]
  print(buf.append(7))  # None
  print(buf.append(8, return_overwritten=True))  # float 5.0
  ```

#### Supported input types

- **`.append(float | int)`**
  - Accepts a single numeric value.
  - Value is stored internally as the buffer’s `dtype`.
- **`.extend(Iterable[float | int])`**
  - Accepts any `Iterable` of numeric values.
  - The iterable is converted to a contiguous NumPy array of the buffer’s `dtype`.
  - If you pass an `np.ndarray`, it is only copied if,
    - Its `dtype` does not match the buffer’s
    - It is not C-contiguous
- **`.extend_unchecked(np.ndarray)`**
  - Expects only a C-contiguous 1-D `np.ndarray` with the same `dtype` as the buffer.
  - This method skips `dtype`, contiguous array and dimension conversions/checks.

  **Note:** Using this yields the best performance, but if the array's `dtype` is different or if it is not contiguous, it will cause silent data corruption or crashes.

#### Mathematical Metrics

These metrics are computed on all elements.

- `.mean()`: Mean of all elements.
- `.sum()`: Sum of all elements.
- `.sum_squares()`: Sum of squares.
- `.mean_squares()`: Mean of squares.
- `.sum_and_count_gt(threshold: float | int)`: Returns a sum and count of elements above a threshold.

### 2. BlockingCircBuffer

Blocking producer/consumer circular buffer, suitable for multi-threaded applications.

This buffer blocks under these conditions:

- The writer will wait if the buffer is full.
- The reader will wait if the buffer is empty.

#### Constructor

```python
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
```

#### Writing to the buffer

- **`.write_append(value: float | int, timeout: float = -1.0)`**
  - Accepts a single numeric value.
  - Value is stored internally as the buffer’s `dtype`.
  - `timeout`: Time in seconds to wait if the buffer is empty. Default is `-1.0` (waits indefinitely). Use `0.0` to make it non-blocking.

- **`.write_extend(data: Iterable[float | int], timeout: float = -1.0)`**
  - `data`: Data block to write.
    - Accepts any `Iterable` of numeric values.
    - The iterable is converted to a contiguous NumPy array of the buffer’s `dtype`.
    - If you pass an `np.ndarray`, it is only copied if,
      - Its `dtype` does not match the buffer’s
      - It is not C-contiguous
  - `timeout`: Same as `.write_append`

- **`.write_extend_unchecked(block_np: np.ndarray, timeout: float = -1.0)`**
  - `block_np`: `np.ndarray` to write.
    - Expects only a C-contiguous 1-D `np.ndarray` with the same `dtype` as the buffer.
    - Skips `dtype`, contiguous array, and dimension checks.
  - `timeout`: Same as `.write_append`

  **Note**: Using this yields the best performance, but if the array's `dtype` is different or if it is not contiguous, it will cause silent data corruption or crashes.

#### Reading from the buffer

- **`.read(n: int = NumCircBuf.constants.Limits.SIZE_MAX.value, timeout: float = -1.0, partial_read: bool = True) -> np.ndarray`**

  Reads items from the buffer.
  - `n`: Number of items to read. Defaults to the maximum possible buffer size.
  - `timeout`: Time in seconds to wait if the buffer is empty. Default is `-1.0` (waits indefinitely). Use `0.0` to make it non-blocking.
  - `partial_read`: If `True` (default), returns available items (up to `n`) immediately. If `False`, blocks until all `n` items are available.

- **`.read_into(out_array_np: np.ndarray, timeout: float = -1.0, partial_read: bool = True) -> int`**

  Reads directly into a pre-allocated `np.ndarray` of the same `dtype` as the buffer.
  - Returns the number of items actually read.
  - The number of items to read is determined by the length of `out_array_np`.
  - `timeout` and `partial_read`: Same as `.read()`.

- **`.read_into_unchecked(out_array_np: np.ndarray, timeout: float = -1.0, partial_read: bool = True) -> int`**

  Similar to `.read_into`
  - Skips `dtype`, contiguous array, and dimension checks.

  **Note**: Using this yields the best performance, but if the array's `dtype` is different or if it is not contiguous, it will cause silent data corruption or crashes.

#### Usage Example

```python
import threading
import time
import numpy as np
from numcircbuf import BlockingCircBuffer

buf = BlockingCircBuffer(maxlen=1000, dtype=np.float32)
data = np.random.randn(1000).astype(np.float32)

# Pre-allocate a NumPy array with zeros for read_into.
# We use zeros instead of np.empty so that any unwritten elements are clearly visible
# — useful when inspecting random data.
read_into_arr = np.zeros(100, dtype=np.float32)


def producer():
    # Write an initial batch of 150 items
    buf.write_extend(data[:150])

    # Simulate a delay (e.g., waiting for network packets or sensor data)
    time.sleep(0.5)

    # Write the remaining 850 items
    buf.write_extend(data[150:])


def consumer():
    # We ask for 200 items, but the producer has only written 150 so far.
    # Because partial_read=True, it returns the 150 items immediately instead of waiting.
    received = buf.read(n=200, partial_read=True)
    print(f"1. Partial read: asked for 200, got {len(received)}")  # 150
    print(f"   Items left in buffer: {len(buf)}\n")  # 0

    # We ask for 300 items. The buffer is currently empty.
    # Because partial_read=False, this will block until the producer writes the next batch,
    # ensuring we get exactly 300 items.
    received = buf.read(n=300, partial_read=False)
    print(f"2. Strict read: asked for 300, got {len(received)}")  # 300
    print(
        f"   Items left in buffer: {len(buf)}\n"
    )  # 550 (850 new items - 300 read)

    # Reads directly into the pre-allocated array (size 100)
    received_count = buf.read_into(out_array_np=read_into_arr)
    received = read_into_arr[:received_count]
    print(f"3. Read into array: got {len(received)}")  # 100
    print(f"   Items left in buffer: {len(buf)}\n")  # 450

    # Reads everything left in the buffer
    received = buf.read()
    print(f"4. Read remaining: got {len(received)}")  # 450
    print(f"   Items left in buffer: {len(buf)}\n")  # 0

    # Non-blocking read: buffer is empty, timeout=0 makes it return immediately
    received = buf.read(timeout=0)
    print(f"5. Non-blocking read: got {len(received)}")  # 0

    # Non-blocking read_into: buffer is empty, returns immediately
    received_count = buf.read_into_unchecked(
        out_array_np=read_into_arr, timeout=0
    )
    print(f"6. Non-blocking read_into: got {received_count}")  # 0


# Start producer and consumer threads
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
```

## Utility Buffers

### Common Utility Buffer API

All utility buffer types expose the methods defined in [Common Buffer API](#common-buffer-api), and the following:

- `clear_cache()`: Clears the cached metric.

Furthermore, all the buffers cache their metric value when the metric function is called and clear the cached metric value whenever the buffer is extended or appended to.

### 1. RunningMeanSqBuffer

Accumulator-capable circular buffer optimized for mean-square calculations.

Features fully vectorized operations, float-drift protection, and caching for mean-square.

**Note:** This buffer is not thread safe.

- Concurrent reads during writes can return stale mean-square values
- Concurrent writes can cause data corruption.

#### Constructor

```python
def __init__(
    self,
    maxlen: int,
    operation_focus: Literal["extend/append", "calculation"],
    recalc_threshold: int | None = 0,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> None:
```

##### Operation Focus

`operation_focus: Literal["calculation", "extend/append"]`

- **`"calculation"`: O(1) statistics, higher per-write cost.**

- **`"extend/append"`: O(n) statistics, lower write cost.**

Use the library utility `determine_operation_focus` to automatically select the best `operation_focus`.
This function runs a small runtime benchmark and returns the appropriate Literal value for your use case:

```python
def determine_operation_focus(
    buffer_type: type[RunningMeanSqBuffer] | type[RunningMeanBuffer],
    dtype: type[np.float32] | type[np.float64],
    buffer_maxlen: int,
    block_size: int, # Use 1 if you will be appending a single element only.
    calc_every: int, # Calculate every n blocks.
    test_blocks: int = 128, # How many blocks to run the benchmark for,
                            # more = (higher memory usage)
                            # + (potentially more accurate results).
    verbose: bool = False, # Logs the exact relative multipliers,
                           # as well as total time spent in the function.
) -> Literal["calculation", "extend/append"]: # Outputs the best operation focus
                                              # for this (use case) + (system),
                                              # This can be directly passed to
                                              # the buffer init
```

Actual performance depends on buffer's `maxlen`, block size, overwrite rate, and statistics frequency.
See [PERFORMANCE.md](https://github.com/basimali-ai/NumCircBuf/blob/main/docs/PERFORMANCE.md) for relative benchmarks.

##### Recalculation Threshold

`recalc_threshold: int`: Number of operations after which the accumulator is recalculated on all values in the buffer to reduce floating-point drift. Use `0` or `None` to disable.

#### Usage Example

```python
import numpy as np
from numcircbuf import RunningMeanSqBuffer, determine_operation_focus

DTYPE = np.float32
CALC_EVERY = 2
BUFFER_MAXLEN = 1000
BLOCK_SIZE = 100

buffer = RunningMeanSqBuffer(
    maxlen=BUFFER_MAXLEN,
    operation_focus=determine_operation_focus(
        buffer_type=RunningMeanSqBuffer,
        dtype=DTYPE,
        buffer_maxlen=BUFFER_MAXLEN,
        block_size=BLOCK_SIZE,
        calc_every=CALC_EVERY,
        verbose=True,
    ),
    dtype=DTYPE,
)

# Add arrays/blocks
rng = np.random.default_rng(25)  # For random number generation
for i in range(10):  # Simulate time
    block = rng.random(BLOCK_SIZE, dtype=DTYPE)  # Mock block arriving
    buffer.extend(block)  # Extend with the block
    if i % CALC_EVERY:  # Calc every n
        print("-" * 10)
        print(buffer.mean_square())  # Get mean-square
        print(buffer.mean_square())  # Uses cached value
```

### 2. RunningMeanBuffer

Accumulator-capable circular buffer optimized for mean calculations.

Features fully vectorized operations, and caching for mean.

**Note:** This buffer is not thread safe.

- Concurrent reads during writes can return stale mean values
- Concurrent writes can cause data corruption.

#### Constructor

```python
def __init__(
    self,
    maxlen: int,
    operation_focus: Literal["extend/append", "calculation"],
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> None:
```

`operation_focus` works the same as in [RunningMeanSqBuffer](#operation-focus).

#### Usage Example

Works identically to [RunningMeanSqBuffer](#usage-example-2) — use `.mean()` instead of `.mean_square()`

### 3. IntegratedGatedBuffer

A specialized circular buffer for calculating gated loudness.

Features fully vectorized operations, and caching for gated mean-square.

**Internal Storage:** This buffer stores the **square** of the input values.
Values retrieved via views will represent squared values, not the original linear amplitude.

**Note:** This buffer is not thread safe.

- Concurrent reads during writes can return stale gated mean-square values
  and inaccurate data
- Concurrent writes can cause data corruption.

#### Constructor

```python
def __init__(
    self,
    maxlen: int,
    abs_gate_lufs: float,
    rel_gate_lu: float,
    recalc_threshold: int | None = 0,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> None:
```

`recalc_threshold` works the same as in [RunningMeanSqBuffer](#recalculation-threshold).

##### Threshold Parameters

- **`abs_gate_lufs: float`**
  The absolute loudness threshold in LUFS.
  Blocks with a mean-square power below this threshold are ignored during the first stage of the gating process.

- **`rel_gate_lu: float`**
  The relative loudness threshold in LU.
  Blocks with a mean-square power more than this many decibels below the absolute-gated mean-square are ignored in the final integrated loudness calculation.

#### Usage Example

```python
import numpy as np
from numcircbuf import IntegratedGatedBuffer

# Parameters for gated loudness calculation
# (We use the ITU-R BS.1770 standard constants for the example)
ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0

buffer = IntegratedGatedBuffer(
    maxlen=3000,
    abs_gate_lufs=ABS_GATE_LUFS,
    rel_gate_lu=REL_GATE_LU
)

# Add the audio signal to the buffer.
# Input should be K-weighted and normalized to [-1.0, 1.0] as per ITU-R BS.1770.
block = np.random.uniform(-1.0, 1.0, 1000)
buffer.extend(block)

# Retrieve the final gated mean-square
gated_mean_sq = buffer.gated_mean_square()
print(f"Gated mean square: {gated_mean_sq}")
```

## Exception Handling

### Exceptions

```text
NumCircBufError(Exception)
├── NumCircBufValueError(NumCircBufError, ValueError) ────────────────────────────┬─────┐
├── NumCircBufTypeError(NumCircBufError, TypeError) ─────────────────────┬─────┬─ │ ─┐  │
│   └── InvalidModification(NumCircBufTypeError)                         │     │  │  │  │
├── NumCircBufIndexError(NumCircBufError, IndexError)                    │     │  │  │  │
│   └── IndexOutOfBounds(NumCircBufIndexError)                           │     │  │  │  │
├── NumCircBufArithmeticError(NumCircBufError, ArithmeticError)          │     │  │  │  │
│   └── UnsupportedOperation(NumCircBufArithmeticError)                  │     │  │  │  │
├── NumCircBufRuntimeError(NumCircBufError, RuntimeError)                │     │  │  │  │
├── NumCircBufOSError(NumCircBufError, OSError)                          │     │  │  │  │
├── NumCircBufNotImplementedError(NumCircBufError, NotImplementedError)  │     │  │  │  │
└── NumCircBufInitError(NumCircBufError)                                 │     │  │  │  │
    ├── DataTypeError(NumCircBufInitError, NumCircBufTypeError) ─────────┘     │  │  │  │
    ├── BufferCapacityError(NumCircBufInitError)                               │  │  │  │
    │   ├── BufferCapacityTypeError(BufferCapacityError, NumCircBufTypeError) ─┘  │  │  │
    │   └── BufferCapacityValueError(BufferCapacityError, NumCircBufValueError) ──┘  │  │
    └── ConfigurationError(NumCircBufInitError)                                      │  │
        ├── ConfigurationTypeError(ConfigurationError, NumCircBufTypeError) ─────────┘  │
        └── ConfigurationValueError(ConfigurationError, NumCircBufValueError) ──────────┘
```

> NumCircBuf exceptions are comprehensive and include context (e.g., the object/class that caused the error):
>
> - `class_obj` – the class where the exception occurred (may be `None` if no class is associated with the error).
> - `obj` – the instance that caused the error (may be `None` if instantiation failed or no specific instance is involved).
> - Various different attributes available based on the exception type.
>
> **Performance note:**
> Some low-level Cython/NumPy errors may still propagate as native Python exceptions (`ValueError`, `TypeError`, etc.) to avoid redundant checks in performance-critical code.

#### Usage Example

```python
try:
    buf.extend(data)
except exceptions.NumCircBufError as e:
    # Structured library errors; you can access `e.class_obj` and `e.obj`
    handle_numcircbuf_error(e)
except Exception as e:
    # Low-level errors from NumPy/Cython
    handle_low_level_error(e)
```

### Warnings

```text
NumCircBufWarning(Warning)
├── NumCircBufDeprecationWarning(NumCircBufWarning, DeprecationWarning)
├── NumCircBufFutureWarning(NumCircBufWarning, FutureWarning)
└── NumCircBufRuntimeWarning(NumCircBufWarning, RuntimeWarning)
    └── DataSizeWarning(NumCircBufRuntimeWarning)
```

> All NumCircBuf warnings include `class_obj` and `obj` attributes (like exceptions).

#### Usage Example

```python
import warnings
from numcircbuf import OverwriteCircBuffer, exceptions

buffer = OverwriteCircBuffer(5, "never")


# Example function that catches a NumCircBufWarning
def extend_and_catch(buffer, data):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")  # Capture all warnings
        buffer.extend(data)
        for w in caught_warnings:
            if isinstance(w.message, exceptions.NumCircBufWarning):
                print("Warning type:", type(w.message))
                print("Class of buffer:", w.message.class_obj)
                print("Buffer instance:", w.message.obj)
                print("Full message:", w.message)


# Trigger a warning
extend_and_catch(buffer, range(10))
```

## Performance

NumCircBuf is optimized for speed and efficiency:

- **Cython + raw C pointers** – for near-C performance
- **Minimal Python object creation** – slices/views reused, caching reduces repeated calculations
- **BLAS-backed NumPy operations** – for efficient array math (dot products, sum-of-squares)
- **O(1) accumulator operations** – for real-time applications
- **Handles high-throughput buffers** – for streaming data like audio or sensor signals

### Benchmark Results

For detailed performance benchmarks, see the [PERFORMANCE.md](https://github.com/basimali-ai/NumCircBuf/blob/main/docs/PERFORMANCE.md) document which includes comprehensive testing results across different buffer types and use cases.

**Key Performance Highlights:**

- **Throughput** (extending 64 KiB of data to `OverwriteCircBuffer`):
  - **Cold cache (data)**:
    - AMD R7700x (DDR5): ~50 GB/s
    - AMD R5600 (DDR4): ~30 GB/s

  - **Warm cache (data)**:
    - AMD R7700x (DDR5): ~73 GB/s
    - AMD R5600 (DDR4): ~50 GB/s

- **Memory efficiency and consistency**: ~4 bytes per element (32-bit) and ~8 bytes per element (64-bit)

## Documentation

- **[Performance Benchmarks](https://github.com/basimali-ai/NumCircBuf/blob/main/docs/PERFORMANCE.md):** Comprehensive performance testing results and optimization guide.
- **[Versioning Strategy](https://github.com/basimali-ai/NumCircBuf/blob/main/docs/VERSIONING.md):** Versioning policy, release process, and compatibility guarantees.

## Changelog

See [CHANGELOG.md](https://github.com/basimali-ai/NumCircBuf/blob/main/docs/CHANGELOG.md) for a detailed history of changes, including new features, bug fixes, and performance improvements.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/basimali-ai/NumCircBuf/blob/main/LICENSE) file for details.

## Support

For issues, questions, or feature requests:

- **GitHub Issues**: Open an issue on our [GitHub repository](https://github.com/basimali-ai/NumCircBuf/issues)
- **GitHub Discussions**: Join the conversation in our [Discussions forum](https://github.com/basimali-ai/NumCircBuf/discussions)

## Citation

If you use NumCircBuf in your research or projects, please cite it as:

```bibtex
@software{NumCircBuf,
  author = {Syed Basim Ali},
  title = {NumCircBuf: High-Performance Numerical Circular Buffers for Python},
  year = {2026},
  url = {https://github.com/basimali-ai/NumCircBuf},
  version = {1.0.0}
}
```

## Acknowledgements

NumCircBuf is built on top of these technologies:

- **[Cython](https://cython.org/)** for performance optimization
- **[NumPy](https://numpy.org/)** for numerical computing and integration
