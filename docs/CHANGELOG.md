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
