# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-02

### Added

- Initial public release.
- Formal stability policy and migration strategy (see [VERSIONING.md](VERSIONING.md)).

## [1.0.1] - 2026-05-05

### Fixed

- Exclude transient build artifacts from wheel distribution.
- **README**: Correct `RunningMeanSqBuffer` usage example.
- **README**: Fix version typo in citation metadata.

### Changed

- **README**: Clarify `OverwriteCircBuffer` thread safety.
- Add short library description and source link to `__init__.py`.
- Add module-level docstrings to all core components and utilities.

## [1.0.2] - 2026-05-06

### Fixed

- Corrected default buffer `dtype` hints and documentation. Previous versions incorrectly suggested `float32` instead of `float64`, which could lead to memory corruption, resulting in data corruption or segmentation faults when using `unchecked` methods.

## [1.0.3] - 2026-05-13

### Fixed

- **README**: Correct `determine_operation_focus` signature.
- Prevent file name collisions in multi-threaded workflows by appending unique IDs to temporary benchmark files created by `determine_operation_focus`.

## [1.1.0] - 2026-05-16

### Added

- **IntegratedGatedBuffer:** `append`, `extend`, and `extend_unchecked` now accept an `already_squared: bool = False` argument. When `False`, inputs are treated as linear amplitude and squared internally before storage. Callers with pre-squared values no longer need to `sqrt` before passing them in.

### Changed

- **README:** Refine `Features` section.
- **bench_utils.get_cpu_name:** Retrieves a more human-readable CPU name.

## [1.1.1] - 2026-05-23

### Fixed

- **README**: Correct DDR5 cold-cache throughput.

### Changed

- **README:** Refine library overview, and update relative performance multipliers.

## [1.1.2] - 2026-05-26

### Fixed

- Normalize Linux L3 cache size to bytes.

### Changed

- **README:** Refine library overview and `Features` section.

## [1.2.0] - 2026-07-31

### Fixed

- **RunningMeanBuffer:** Correct typing stubs to reflect that `__init__` supports the `recalc_threshold` argument.
- **RunningMeanSqBuffer** and **IntegratedGatedBuffer:** Correct the positional placement of the `dtype` argument in `__init__`.
- **Linux wheels:** Reduce binary size by removing unintended debug information from release builds:
  - `manylinux_2_24_x86_64`: 1.2 MB → 230 kB
  - `musllinux_1_2_x86_64`: 2.2 MB → 1.3 MB

### Changed

- **Type hints:**
  - Add a `py.typed` marker file (PEP 561) to formally expose the library's existing type hints to static analyzers like `mypy`.
  - Add strict type support conforming to `mypy --strict`.
  - Improve developer experience by adopting `TypeVar` and `Generic`.
  - Allow static type checkers to infer the generic type of buffer classes from the `dtype` argument.
  - Allow static type checkers to infer the generic type of views from their parent buffer object.
  - Relax strictness of benchmark buffer protocols.

- **Performance Optimizations:**
  - RunningMeanBuffer, RunningMeanSqBuffer, and IntegratedGatedBuffer:
    - Implement custom bit-checking for finite validation to handle non-finite (NaN/Inf) edge cases, primarily affecting Windows.
    - Eliminate function call overhead from `stdlib:isfinite`, enabling the Windows compiler (MSVC) to utilize SIMD registers and perform loop unrolling during finite-array checks.
  - IntegratedGatedBuffer:
    - Replace standard `fmax` and `fmaxf` calls with custom unchecked equivalents under the guaranteed absence of NaN values.
  - Use explicit conditional expressions and move select helper functions into C++ to ensure compilers reliably generate branchless conditional moves.
