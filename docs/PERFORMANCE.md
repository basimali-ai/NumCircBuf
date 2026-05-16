# Performance Benchmarks and Optimization Guide

This document provides performance benchmarks and optimization guidelines for the NumCircBuf library.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Performance Characteristics](#performance-characteristics)
- [Core Benchmarking Methods](#core-benchmarking-methods)
- [Raw Benchmarks](#raw-benchmarks)
- [Relative Benchmarks](#relative-benchmarks)
- [Optimization Guide](#optimization-guide)
- [Conclusion](#conclusion)
- [Benchmarking Your Application](#benchmarking-your-application)
- [Additional Resources](#additional-resources)

## Performance Overview

NumCircBuf provides a suite of pre-allocated, contiguous-memory circular buffers engineered for low-latency ingestion and O(1) windowed analytics.

- **Constant-Time Analytics:** **O(1) complexity** for specific statistical accumulators (mean, mean-square), decoupling computational cost from buffer depth.
- **Deterministic Memory Management:** Utilizes pre-allocated contiguous memory blocks to eliminate allocation jitter and heap fragmentation during high-frequency ingestion.
- **Low-Level Execution:** Employs Cython/C with raw pointer arithmetic to bypass Python's object-model overhead and bounds-checking in critical execution paths.
- **Hardware-Aware Vectorization:** Integrates BLAS-backed kernels, NumPy SIMD dispatch, and libc-optimized memcpy routines for architecture-specific throughput scaling.
- **Optimized Data Ingestion:** Specifically engineered for bulk data movement via `.extend()`, maximizing hardware bandwidth saturation.

**Target Workloads:** Real-time digital signal processing (DSP), high-frequency telemetry ingestion, audio loudness analysis (EBU R128), and low-latency scientific computing.

## Performance Characteristics

### Time Complexity

| Operation Category | OverwriteCircBuffer  | BlockingCircBuffer   | Utility Buffers                |
| ------------------ | -------------------- | -------------------- | ------------------------------ |
| extend ops         | O(n)                 | O(n)                 | O(n)                           |
| append ops         | O(1)                 | O(1)                 | O(1)                           |
| `view()`           | O(1) view, O(n) copy | O(1) view, O(n) copy | O(1) view, O(n) copy           |
| mathematical ops   | O(n)                 | N/A                  | O(n) / O(1) for some buffers\* |
| `clear()`          | O(1)                 | O(1)                 | O(1)                           |
| clear NaN or Infs  | O(n)                 | O(n)                 | O(n)                           |

\* RunningMeanBuffer / RunningMeanSqBuffer may be O(1) or O(n) depending on focus; IntegratedGatedBuffer is always O(n)

### Space Complexity

- **Primary Storage**: $O(N)$ where $N$ is the buffer capacity.
- **Transient Workspace**: Certain mathematical operations require temporary $O(K)$ workspace (where $K \leq N$) to ensure higher computational throughput. These allocations are transient and exist only for the duration of the specific operation.

## Core Benchmarking Methods

### Main Buffers

#### OverwriteCircBuffer

1. Times are measured in nanoseconds using Python’s arbitrary-precision integers for maximum precision.
2. Each test uses a block, whose size never exceeds the buffer's maxlen.
3. We benchmark a single block-extend operation per timing.
4. At the start of the benchmark, CPU cache is thrashed, and a single warm-up block is extended to the buffer to ensure the buffer’s internal memory is mapped.
5. There are multiple runs; for each run we use a separate block.
6. Before each run:
   - The buffer is cleared.
   - For the current block, one element per memory page is touched (`page_size // element_size` stride) to fault the block into RAM while introducing only negligible CPU cache warming (one cache line per page).
7. Runs are split into two groups:
   - **Wrap Runs:** extend the buffer with a unique offset block of size `buffer_maxlen - (block_size // 2)`, so half the block wraps around and half does not. Record the timing in `wrap_times`.
   - **No-Wrap Runs:** perform the single block-extend (no wrap) and record the timing in `nowrap_times`.
8. For each list (`wrap_times`, `nowrap_times`) discard the first 10% of measurements (warmup) and compute the mean of the remaining values. The final benchmark time is the average of those two means.

#### BlockingCircBuffer

Follows the OverwriteCircBuffer method, only difference is we measure reads as well.

### Utility Buffers

Follows the OverwriteCircBuffer method, with additional steps.

1. At the start of each run we now also fill the buffer with unique data.
2. A mask determines after how many blocks the calculation function is called.
3. On 'Calculate' runs, we measure the extend operation plus calculation function call.

## Benchmarking System

### CPU

**AMD Ryzen 5 5600:**

- **Cores / Threads:** 6 / 12
- **Frequency (All Cores):** ~4.44 GHz
- **Cache L1 (per core) / L2 (per core) / L3 (shared):** 64 KB / 512 KB / 32 MB

> CPU maintains ~4.44 GHz on all cores under load, ensuring relatively consistent performance regardless of thread scheduling.

### RAM

**XPG SPECTRIX D35G:**

- **Configuration:** 16 GB x 2
- **Form Factor:** UDIMM
- **Frequency:** 3666 MT/s
- **CAS Latency:** 18-22-22-44
- **Channels:** 2

### Environment

- **Windows 10 Pro:** 22H2 (19045.6456)
- **Python:** 3.12.12
- **NumPy:** 2.4.1

## Raw Benchmarks

### OverwriteCircBuffer

**Source File:** [raw_overwrite_buf.py](../perf/raw_benchmarks/raw_overwrite_buf.py)

**Explored parameter space:**

```text
DTYPE = np.float64
MAXLEN_BYTE_LIMIT = 65_536
BLOCK_BYTE_LIMIT = 65_536
```

**Results:**

```text
--- Extend ---

Warm Cache: False
OverwriteCircBuffer throughput: 31.57 GB/s
RefPythonNumPyCircBuffer throughput: 14.65 GB/s
BenchDeque throughput: 0.03 GB/s
BenchList throughput: 0.03 GB/s

Warm Cache: True
OverwriteCircBuffer throughput: 48.73 GB/s
RefPythonNumPyCircBuffer throughput: 17.21 GB/s
BenchDeque throughput: 0.03 GB/s
BenchList throughput: 0.03 GB/s

--- Append ---

Warm Cache: False
OverwriteCircBuffer throughput: 0.08 GB/s
RefPythonNumPyCircBuffer throughput: 0.02 GB/s
BenchDeque throughput: 0.05 GB/s
BenchList throughput: 0.03 GB/s

Warm Cache: True
OverwriteCircBuffer throughput: 0.09 GB/s
RefPythonNumPyCircBuffer throughput: 0.03 GB/s
BenchDeque throughput: 0.05 GB/s
BenchList throughput: 0.03 GB/s
```

### BlockingCircBuffer

**Source File:** [raw_blocking_buf.py](../perf/raw_benchmarks/raw_blocking_buf.py)

**Explored parameter space:**

```text
DTYPE = np.float64
MAXLEN_BYTE_LIMIT = 65_536
BLOCK_BYTE_LIMIT = 65_536
```

**Results:**

```text
Warm Cache: False
Read Into Array Used: False
BlockingCircBuffer Write throughput: 25.81 GB/s
BlockingCircBuffer Read throughput: 28.21 GB/s

Warm Cache: False
Read Into Array Used: True
BlockingCircBuffer Write throughput: 23.36 GB/s
BlockingCircBuffer Read throughput: 35.39 GB/s

Warm Cache: True
Read Into Array Used: False
BlockingCircBuffer Write throughput: 35.29 GB/s
BlockingCircBuffer Read throughput: 28.25 GB/s

Warm Cache: True
Read Into Array Used: True
BlockingCircBuffer Write throughput: 33.78 GB/s
BlockingCircBuffer Read throughput: 35.97 GB/s
```

### Utility/Calculation Buffers

**Source File:** [raw_util_buffers.py](../perf/raw_benchmarks/raw_util_buffers.py)

**Explored parameter space:**

```text
CALC_EVERY = 1
MAXLENS = (4096, 8192, 16_384, 32_768, 65_536)
BLOCK_SIZES = (4096, 8192, 16_384, 32_768, 65_536)
```

**Results:**

```text
----------
RunningMeanSqBuffer :
Warm Cache: False
~835–5524M float32 elems/sec (extend + calculation)
3.34–22.10 GB/s effective throughput
----------
Warm Cache: False
~159–1879M float64 elems/sec (extend + calculation)
1.28–15.03 GB/s effective throughput
----------
Warm Cache: True
~861–6121M float32 elems/sec (extend + calculation)
3.44–24.49 GB/s effective throughput
----------
Warm Cache: True
~162–2472M float64 elems/sec (extend + calculation)
1.30–19.78 GB/s effective throughput
----------
----------
RunningMeanBuffer :
Warm Cache: False
~347–2566M float32 elems/sec (extend + calculation)
1.39–10.26 GB/s effective throughput
----------
Warm Cache: False
~304–1926M float64 elems/sec (extend + calculation)
2.43–15.41 GB/s effective throughput
----------
Warm Cache: True
~297–2749M float32 elems/sec (extend + calculation)
1.19–11.00 GB/s effective throughput
----------
Warm Cache: True
~316–2157M float64 elems/sec (extend + calculation)
2.53–17.26 GB/s effective throughput
----------
----------
IntegratedGatedBuffer :
Warm Cache: False
~34–503M float32 elems/sec (extend + calculation)
0.13–2.01 GB/s effective throughput
----------
Warm Cache: False
~15–255M float64 elems/sec (extend + calculation)
0.12–2.04 GB/s effective throughput
----------
Warm Cache: True
~33–516M float32 elems/sec (extend + calculation)
0.13–2.06 GB/s effective throughput
----------
Warm Cache: True
~14–265M float64 elems/sec (extend + calculation)
0.11–2.12 GB/s effective throughput
----------
```

## Relative Benchmarks

### OverwriteCircBuffer vs RefPythonNumPyRingBuffer

`RefPythonNumPyRingBuffer` is an optimized reference implementation written in pure Python/NumPy and used as a performance baseline.

**Source File:** [bench_raw_buf.py](../perf/relative_benchmarks/bench_raw_buf.py)

**Explored parameter space:**

```text
    BLOCK_SIZE = (
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
    )
    MAXLEN = (
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
    )
```

#### float64

##### block size

![df_64_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_raw_buf/df_64_BLOCK_SIZE.png "fp64 block size relative graph")

#### float32

##### block size

![df_32_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_raw_buf/df_32_BLOCK_SIZE.png "fp32 block size relative graph")

### Operation Focus ("calculation" vs "extend/append")

**Source File:** [bench_op_f.py](../perf/relative_benchmarks/bench_op_f.py)

**Explored parameter space:**

```text
    BLOCK_SIZE = (
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
    )
    MAXLEN = (
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
        2_097_152,
        4_194_304,
        8_388_608,
    )
    CALC_EVERY = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
```

#### RunningMeanSqBuffer

##### float64

###### block size

![df_64_mean_sq_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_sq_BLOCK_SIZE.png)

###### maxlen

![df_64_mean_sq_MAXLEN.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_sq_MAXLEN.png)

###### calc every

![df_64_mean_sq_CALC_EVERY.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_sq_CALC_EVERY.png)

##### float32

###### block size

![df_32_mean_sq_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_sq_BLOCK_SIZE.png)

###### maxlen

![df_32_mean_sq_MAXLEN.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_sq_MAXLEN.png)

###### calc every

![df_32_mean_sq_CALC_EVERY.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_sq_CALC_EVERY.png)

#### RunningMeanBuffer

##### float64

###### block size

![df_64_mean_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_BLOCK_SIZE.png)

###### maxlen

![df_64_mean_MAXLEN.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_MAXLEN.png)

###### calc every

![df_64_mean_CALC_EVERY.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_64_mean_CALC_EVERY.png)

##### float32

###### block size

![df_32_mean_BLOCK_SIZE.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_BLOCK_SIZE.png)

###### maxlen

![df_32_mean_MAXLEN.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_MAXLEN.png)

###### calc every

![df_32_mean_CALC_EVERY.png](../perf/relative_benchmarks/eda/plots/bench_op_f/df_32_mean_CALC_EVERY.png)

## Optimization Guide

### Optimal Buffer Selection

| Application Requirement       | Recommended Buffer      | Technical Justification                                                                                                                            |
| :---------------------------- | :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **High-Throughput Ingestion** | `OverwriteCircBuffer`   | Minimum-overhead implementation using raw pointer arithmetic; optimized for single-threaded write-heavy workloads.                                 |
| **Producer-Consumer Sync**    | `BlockingCircBuffer`    | Implements thread-safe synchronization primitives and strict FIFO ordering for multi-threaded environments without manual locking/ticketing logic. |
| **Windowed Statistics**       | `RunningMeanBuffer`     | Maintains an internal $\sum x$ accumulator to provide **O(1) mean** calculation, decoupling latency from window size.                              |
| **Power/Energy Analysis**     | `RunningMeanSqBuffer`   | Maintains an internal $\sum x^2$ accumulator for **O(1) mean-square** updates, ideal for real-time RMS calculations.                               |
| **Non-Linear Analytics**      | `IntegratedGatedBuffer` | Specialized implementation for gated accumulation, minimizing branching overhead in integrated loudness pipelines.                                 |

### Performance Tips

1. **Use `extend()` for bulk operations:** Bulk operations minimize the frequency of Python-to-C context switching and allow the underlying engine to utilize vectorized memory copy routines.
2. **Choose operation focus at runtime:** For `RunningMeanBuffer`/`RunningMeanSqBuffer`, choose operation focus using `determine_operation_focus` helper provided. This allows the internal logic to optimize for the required use-case on available hardware.
3. **Precision and Cache Efficiency:** Use `np.float32` where full 64-bit precision is not mathematically required. Reducing the word size from 8 to 4 bytes effectively doubles the number of elements that can fit within the CPU caches and halves the required memory bus bandwidth.
4. **Extend with NumPy Arrays of the same dtype**: The library supports conversion but the extends will be substantially slower as it triggers implicit type-casting and temporary array creation, which incurs a significant CPU and memory allocation penalty.

### Thread Safety Considerations

- **Atomic Constraints:** Standard buffer variants are not thread-safe and omit internal locking mechanisms to maximize single-threaded throughput. If multi-threaded access is required for these variants, external synchronization (e.g., `threading.Lock`) must be managed by the caller.
- **Concurrency Scalability:** Increasing the thread count does not provide linear throughput scaling for thread-safe variants. High contention for the internal lock can degrade performance due to increased kernel-level context switching and synchronization overhead.
- **Timeouts:** When utilizing `BlockingCircBuffer`, implement appropriate timeout strategies to prevent indefinite pipeline stalls. Unbounded blocking can lead to global deadlocks if a thread in the producer/consumer chain fails to release its dependency.
- **Thread Integrity and Liveness:** To ensure strict FIFO ordering and deterministic execution, `BlockingCircBuffer` requires standard thread lifecycle management. Abrupt thread termination (e.g., via SIGKILL or hard cancellation) while a thread is performing or waiting on operations (read/write) will result in a permanent deadlock state. This design prioritizes high-performance state transitions and data integrity over the inherent unreliability of system-level thread monitoring.

### Best Practices

#### Code Examples

```python
# Good: Use extend for bulk operations

buffer = OverwriteCircBuffer(10000)

# Data in bulk
data = np.random.rand(10000)

# Efficient: Use extend
buffer.extend(data)

# Bad: Use individual appends in loop
for value in data:
    buffer.append(value)  # Much slower
```

```python
# Memory-efficient usage patterns

# Good: Reuse buffers
buffer = OverwriteCircBuffer(10000)
for epoch in range(100):
    buffer.clear()  # Reset without reallocating
    # Process data...

# Bad: Create new buffers frequently
for epoch in range(100):
    buffer = OverwriteCircBuffer(10000)  # New allocation each time
    # Process data...
```

## Conclusion

Standard Python containers like `collections.deque` are optimized for general-purpose use, but they don't perform as desired under the demands of high-throughput numerical workloads. The data above confirms that NumCircBuf effectively solves this problem.

### Technical Summary

- **Hardware-Adaptive SIMD Strategy:** The library utilizes NumPy’s internal runtime dispatch mechanism (Universal Intrinsics) to execute mathematical operations. By delegating execution to NumPy internals, NumCircBuf ensures that the most efficient SIMD kernels (such as AVX-512, AVX2, or NEON) are selected dynamically based on the host CPU. In addition, raw memory movement operations benefit from platform-optimized `memcpy` routines provided by the system C runtime, which utilize architecture-specific vectorized code paths. This combined approach provides a significant performance advantage over static Cython/C implementations, which often lack the cross-architecture portability and sophisticated runtime probing required to saturate modern CPU pipelines.
- **Memory & Workspace Strategy:** The library utilizes a pre-allocated contiguous memory architecture for primary storage to prevent heap fragmentation. For some specific math operations, the implementations utilize temporary copies. This trade-off allows the library to perform mathematical operations at a higher throughput.
- **Algorithmic Complexity:** Rolling statistical metrics (mean, mean-square) are implemented via **O(1) recurrence relations**. By maintaining internal accumulators, the computational cost of windowed analytics is decoupled from the buffer size, ensuring fixed-time execution even with extremely large window depths.
- **Numerical Integrity & Precision:** Precision is enforced through type-strict execution paths. fp32 buffers perform explicit single-precision arithmetic to ensure bit-level determinism. The library implements a specialized handling policy for **IEEE 754 non-finite values (inf, nan)** that prioritizes data transparency over silent error correction. While the implementation performs a single-pass resynchronization of the accumulator to correct for numerical drift, it does not engage in persistent suppression of non-finite values. This ensures that if the input stream is corrupted, the internal state faithfully reflects that corruption rather than masking it through silent removal.

### Performance Comparison

NumCircBuf is designed to saturate the hardware bandwidth, providing a massive performance leap over standard Python containers:

- **vs. `collections.deque` & Python lists**: **1000–1600× faster** for bulk `extend` and **2–4× faster** for single `append`.
- **vs. Optimized NumPy Ring Buffers**: **Up to 10× faster**, reaching speeds where performance is limited primarily by hardware bandwidth.

### When to Use NumCircBuf

**✅ Ideal for:**

- High-frequency data ingestion (e.g., telemetry, sensors).
- Real-time digital signal processing (DSP) and audio analysis.
- Latency-sensitive workloads requiring O(1) statistical updates.
- Scenarios requiring strict memory stability and zero-allocation loops.

**❌ Consider alternatives for:**

- Applications requiring dynamic resizing of buffers during runtime.
- Storage of non-numeric or heterogeneous Python objects.
- Simple use cases where the overhead of the Python interpreter is not a bottleneck.

> Performance figures were obtained on representative workloads using the benchmark code above.
> Actual performance may vary depending on hardware, data layout, and workload characteristics.

## Benchmarking Your Application

You can benchmark NumCircBuf for your specific use case using utilities from the `numcircbuf.bench_utils` sub-module.
Alternatively, you can run the example benchmark scripts provided in the GitHub repository.

## Additional Resources

- **Project Overview**: [README.md](../README.md) — Installation, usage, and examples.
- **Versioning Policy**: [VERSIONING.md](VERSIONING.md) — Stability policy and migration strategy.
- **Change History**: [CHANGELOG.md](CHANGELOG.md) — Detailed list of changes per version.
- **Support**: [Support Policy](../README.md#support) — How to get help and report issues.
