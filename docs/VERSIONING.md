# Versioning Strategy and Release Process

This document outlines the versioning strategy, release process, and compatibility guarantees for the NumCircBuf library.

## Table of Contents

- [Semantic Versioning](#semantic-versioning)
- [Version Number Format](#version-number-format)
- [Compatibility Guarantees](#compatibility-guarantees)
- [Deprecation Policy](#deprecation-policy)
- [Backward Compatibility](#backward-compatibility)
- [API Stability](#api-stability)

## Semantic Versioning

NumCircBuf follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) with the format `MAJOR.MINOR.PATCH`.

### Version Components

- **MAJOR**: Incremented for backward-incompatible changes
- **MINOR**: Incremented for backward-compatible new functionality
- **PATCH**: Incremented for backward-compatible bug fixes

### Examples

- `1.2.3` -> `2.0.0`: Major release with breaking changes
- `1.2.3` -> `1.3.0`: Minor release with new features
- `1.2.3` -> `1.2.4`: Patch release with bug fixes

## Version Number Format

```
MAJOR.MINOR.PATCH[.DEV[N]]
```

Where:

- `MAJOR`, `MINOR`, `PATCH`: Standard semantic versioning components
- `.DEV[N]`: Optional development suffix (e.g., `1.2.3.DEV1`)

## Compatibility Guarantees

### API Compatibility

| Version Change | API Compatibility      | Data Format Compatibility |
| -------------- | ---------------------- | ------------------------- |
| MAJOR increase | ❌ Breaking changes    | ❌ May change             |
| MINOR increase | ✅ Backward compatible | ✅ Preserved              |
| PATCH increase | ✅ Backward compatible | ✅ Preserved              |

### Supported Python Versions

- **Current support**: >= Python 3.9

### Supported NumPy Versions

- **Current support**: NumPy >= 2.0.0

## Deprecation Policy

### Deprecation Timeline

1. **Announcement**: Feature marked as deprecated in release notes.
2. **Light Warning Period**: at least 1 minor release with `DeprecationWarning`.
3. **Strong Warning Period**: at least 1 minor release with `FutureWarning`.
4. **Removal**: Feature removed in the next major version.

### Example deprecation timeline

- **1.87.0**: New feature introduced and old one marked as deprecated with `DeprecationWarning`.
- **1.88.0**: Stronger `FutureWarning` implemented.
- **2.0.0**: Feature removed.

## Backward Compatibility

### API Evolution Strategy

1. **Additive changes**: New features added without breaking existing functionality
2. **Deprecation first**: Features marked deprecated before removal
3. **Clear migration paths**: Documentation and tools for upgrading

## API Stability

### Stable APIs

The following APIs are considered stable and subject to backward compatibility guarantees:

- `OverwriteCircBuffer` class and all its methods
- `BlockingCircBuffer` class and all its methods
- `RunningMeanBuffer` class and all its methods
- `RunningMeanSqBuffer` class and all its methods
- `IntegratedGatedBuffer` class and all its methods
- All public methods and properties documented in type stubs

### Experimental APIs

Experimental APIs (if any) would be marked with appropriate warnings and may change without notice.

### Internal APIs

Internal APIs (prefixed with `_` or not documented) are not subject to compatibility guarantees and may change at any time.

### Cython and C++ APIs

The Cython and C++ layer APIs, including any public methods exposed at that level, are currently **experimental** and **not subject to backward compatibility guarantees**. They may change or be removed without notice. Users should rely only on the Python-level stable APIs listed above.

## Versioning Best Practices

### For Library Users

1. **Use version specifiers**: Pin to the current major version (e.g., `numcircbuf < 1.0.0`) to prevent breaking changes from being installed automatically.
2. **Test before upgrading**: Especially for minor/major version updates
3. **Monitor deprecation warnings**: Plan for upcoming changes
4. **Review changelogs**: Understand what's changing between versions

## Conclusion

NumCircBuf's versioning strategy ensures:

- **Predictability**: Users know what to expect from version numbers
- **Stability**: Backward compatibility for minor and patch releases
- **Transparency**: Clear documentation of changes and migration paths
- **Quality**: Thorough testing and release processes

By following semantic versioning and providing clear compatibility guarantees, NumCircBuf aims to provide a stable foundation for building reliable applications while allowing for innovation and improvement.

## Quick Reference

### Versioning Cheat Sheet

| Scenario                         | Version Change | Example       |
| -------------------------------- | -------------- | ------------- |
| Breaking API changes             | MAJOR          | 1.0.0 → 2.0.0 |
| New backward-compatible features | MINOR          | 1.0.0 → 1.1.0 |
| Bug fixes                        | PATCH          | 1.0.0 → 1.0.1 |
| Development builds               | DEV suffix     | 1.0.0.DEV1    |

### Dependency Versioning

**Recommended version specifiers:**

Assuming current version is 1.0.0 as an example:

```toml
# For stable production use (no updates at all)
numcircbuf = "==1.0.0"

# For safe updates (allows 1.1.0, 1.2.0, etc., but not 2.0.0)
numcircbuf = "~=1.0"

# For patch updates only (allows 1.0.1, 1.0.2, etc., but not 1.1.0)
numcircbuf = "~=1.0.0"

# For maximum compatibility (current major version)
numcircbuf = ">=1.0.0,<2.0.0"

# For development (latest stable)
numcircbuf = ">=1.0.0"
```

## Additional Resources

- **Project Overview**: [README.md](../README.md) - Installation, usage, and examples.
- **Change History**: [CHANGELOG.md](CHANGELOG.md) — Detailed list of changes per version.
- **Support**: [Support Policy](../README.md#support) — How to get help and report issues.

## Versioning FAQ

**Q: What happens to deprecated features?**
A: Deprecated features are removed in the next major release, there will be at least a 2-minor-release warning period (1 with `DeprecationWarning`, 1 with `FutureWarning`)
