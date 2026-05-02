# Performance Benchmarks and Optimization Guide

This document provides comprehensive performance benchmarks and optimization guidelines for the NumCircBuf library.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Performance Characteristics](#performance-characteristics)
- [Raw Benchmarks](#raw-benchmarks)
- [Relative Benchmarks](#relative-benchmarks)
- [Optimization Guide](#optimization-guide)
- [Thread Safety Considerations](#thread-safety-considerations)
- [Memory Management](#memory-management)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)
- [Benchmarking Your Application](#benchmarking-your-application)
- [Additional Resources](#additional-resources)

## Performance Overview

NumCircBuf is designed for high-performance numerical computing with:

- **O(1) time complexity** for most accumulator operations (mean, mean-square, gated accumulators)
- **Memory efficiency** through pre-allocation
- **Cython optimizations** with raw pointers for near-native speed
- **BLAS-backed NumPy operations** for fast array math
- **Thread-safe options** for concurrent read/write usage
- **Exceptional bulk operation performance** with `.extend()` methods

**Target Use Cases**: Real-time signal processing, audio analysis, time-series data, scientific computing, and any application requiring efficient numerical buffers. The library is particularly optimized for bulk operations, making it ideal for high-throughput data processing.

## Performance Characteristics

### Time Complexity

| Operation Type    | OverwriteCircBuffer  | BlockingCircBuffer   | Utility Buffers                |
| ----------------- | -------------------- | -------------------- | ------------------------------ |
| extend ops        | O(n)                 | O(n)                 | O(n)                           |
| append ops        | O(1)                 | O(1)                 | O(1)                           |
| `view()`          | O(1) view, O(n) copy | O(1) view, O(n) copy | O(1) view, O(n) copy           |
| mathematical ops  | O(n)                 | N/A                  | O(n) / O(1) for some buffers\* |
| `clear()`         | O(1)                 | O(1)                 | O(1)                           |
| clear NaN or Infs | O(n)                 | O(n)                 | O(n)                           |

\* RunningMeanBuffer / RunningMeanSqBuffer may be O(1) or O(n) depending on focus; IntegratedGatedBuffer is always O(n)

### Space Complexity

All buffers have O(n) space complexity where n is the buffer capacity.

### Memory Efficiency

- **Pre-allocated memory:** Buffers pre-allocate memory for maximum capacity
- **Cython optimizations:** Minimizes Python object overhead

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

**AMD Ryzen 5 5600**:

- **Cores / Threads**: 6 / 12
- **Frequency (All Cores)**: ~4.44 GHz
- **Cache L1 (per core) / L2 (per core) / L3 (shared)**: 64 KB / 512 KB / 32 MB

> CPU maintains ~4.44 GHz on all cores under load, ensuring relatively consistent performance regardless of thread scheduling.

### RAM

**XPG SPECTRIX D35G**:

- **Configuration**: 16 GB x 2
- **Form Factor**: UDIMM
- **Frequency**: 3666 MT/s
- **CAS Latency**: 18-22-22-44
- **Channels**: 2

### Environment

- **Python**: 3.12.12
- **NumPy**: 2.4.1

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
Warm Cache: False
OverwriteCircBuffer throughput: 32.38 GB/s
NaiveNumpyRingBuffer throughput: 13.71 GB/s

Warm Cache: True
OverwriteCircBuffer throughput: 51.93 GB/s
NaiveNumpyRingBuffer throughput: 15.71 GB/s
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
~608–5283M float32 elems/sec (extend + calculation)
2.43–21.13 GB/s effective throughput
----------
Warm Cache: False
~58–1855M float64 elems/sec (extend + calculation)
0.46–14.84 GB/s effective throughput
----------
Warm Cache: True
~618–6426M float32 elems/sec (extend + calculation)
2.47–25.70 GB/s effective throughput
----------
Warm Cache: True
~57–2330M float64 elems/sec (extend + calculation)
0.45–18.64 GB/s effective throughput
----------
----------
RunningMeanBuffer :
Warm Cache: False
~173–2469M float32 elems/sec (extend + calculation)
0.69–9.88 GB/s effective throughput
----------
Warm Cache: False
~160–1878M float64 elems/sec (extend + calculation)
1.28–15.03 GB/s effective throughput
----------
Warm Cache: True
~173–2632M float32 elems/sec (extend + calculation)
0.69–10.53 GB/s effective throughput
----------
Warm Cache: True
~163–2083M float64 elems/sec (extend + calculation)
1.30–16.67 GB/s effective throughput
----------
----------
IntegratedGatedBuffer :
Warm Cache: False
~39–588M float32 elems/sec (extend + calculation)
0.16–2.35 GB/s effective throughput
----------
Warm Cache: False
~17–319M float64 elems/sec (extend + calculation)
0.14–2.55 GB/s effective throughput
----------
Warm Cache: True
~39–570M float32 elems/sec (extend + calculation)
0.16–2.28 GB/s effective throughput
----------
Warm Cache: True
~17–317M float64 elems/sec (extend + calculation)
0.14–2.54 GB/s effective throughput
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

### Choosing the Right Buffer Type

| Use Case          | Recommended Buffer      | Reason                                          |
| ----------------- | ----------------------- | ----------------------------------------------- |
| General purpose   | `OverwriteCircBuffer`   | Write-focused balanced performance and features |
| Multi-threaded    | `BlockingCircBuffer`    | Read-Write Thread-safe with blocking operations |
| Running averages  | `RunningMeanBuffer`     | Optimized mean calculations                     |
| Signal processing | `RunningMeanSqBuffer`   | Optimized mean-square calculations              |
| Audio loudness    | `IntegratedGatedBuffer` | Specialized for gated loudness calculations     |

### Performance Tips

1. **Use `extend()` for bulk operations:** This is the library's specialty, offering exceptional performance for bulk data insertion. Much faster than individual `append()` calls.
2. **Choose operation focus on runtime:** For `RunningMeanBuffer`/`RunningMeanSqBuffer`, choose operation focus using `determine_operation_focus` helper provided.
3. **Use appropriate data types:** `np.float32` uses half the memory of `np.float64` with minimal precision loss for many applications.
4. **Extend with NumPy Arrays of the same dtype**: The library supports conversion but the extends will be substantially slower as it causes a memory copy and wastes CPU cycles.

### Thread Safety Considerations

- **BlockingCircBuffer**: Use for multi-threaded applications with proper timeout handling.
- **Other buffers**: Not thread-safe by default - use external synchronization if needed.
- **Lock contention**: Minimize time spent holding locks in thread-safe operations.
- **Timeout strategies**: Use appropriate timeouts to avoid deadlocks.

### Memory Management

#### Memory Allocation Strategy

1. **Pre-allocation**: All buffers pre-allocate memory for their maximum capacity.
2. **Efficient growth**: Buffers don't reallocate - they have fixed capacity.

#### Memory Cleanup

- **Automatic cleanup**: Python's garbage collector and OS handles memory cleanup
- **Manual cleanup**: Call `clear()` to reset buffer state without deallocation
- **No memory leaks**: Proper reference counting ensures cleanup

## Best Practices

### Code Examples

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

NumCircBuf is designed for high-performance numerical buffers with:

- **O(1) accumulator operations** for supported buffer types (mean, mean-square)
- **Memory-efficient algorithms** using pre-allocation, and minimal Python object creation
- **Cython optimizations** with raw pointers for near-native speed
- **BLAS-backed NumPy operations** for efficient array math
- **Thread-safe options** available for specific buffers (e.g., BlockingCircBuffer)

By following the usage guidelines and choosing the appropriate buffer type, you can achieve optimal performance for real-time, high-throughput, or numerical computation workloads.

### Performance Comparison

NumCircBuf outperforms traditional approaches under equivalent workloads:

- **vs. Python lists**: 150–200× faster for extend numerical operations
- **vs. Optimized Pure Python NumPy ring buffers**: 1.05–4× faster due to Cython/C optimizations for most use cases
- **vs. Manual implementations**: More reliable, maintainable, and easier to use

### When to Use NumCircBuf

**✅ Ideal for:**

- Real-time data processing
- Signal processing applications
- Audio analysis and loudness measurement
- High-frequency data collection
- Any application needing efficient numerical buffers

**❌ Consider alternatives for:**

- Simple use cases where performance isn't critical
- Applications requiring dynamic resizing of the buffers
- Non-numeric data

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
