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
Custom exceptions and warnings for the NumCircBuf library.

Defines a hierarchy of descriptive error and warning classes. All
exceptions inherit from the base `NumCircBufError`, and all warnings
inherit from `NumCircBufWarning`, allowing for granular error handling.
"""

from __future__ import annotations

from .constants import Limits

# --- Base Classes ---


class _NumCircBufBase:
    """Shared base for errors and warnings with structured fields."""

    def __init__(self, *, class_obj=None, obj=None, message=None):
        self.class_obj = class_obj
        self.obj = obj
        self.message = message
        if message is not None:
            super().__init__(message)
        else:
            super().__init__()


# Exception Base Classes


class NumCircBufError(_NumCircBufBase, Exception):
    """Base exception class for all NumCircBuf errors."""


class NumCircBufInitError(NumCircBufError):
    """Base exception class for all NumCircBuf initialization errors."""


class NumCircBufTypeError(NumCircBufError, TypeError):
    """Base exception class for all NumCircBuf Type errors."""


class NumCircBufValueError(NumCircBufError, ValueError):
    """Base exception class for all NumCircBuf Value errors."""


class NumCircBufIndexError(NumCircBufError, IndexError):
    """Base exception class for all NumCircBuf Index errors."""


class NumCircBufArithmeticError(NumCircBufError, ArithmeticError):
    """Base exception class for all NumCircBuf Arithmetic errors."""


class NumCircBufRuntimeError(NumCircBufError, RuntimeError):
    """Base exception class for all NumCircBuf Runtime errors."""


class NumCircBufOSError(NumCircBufError, OSError):
    """Base exception class for all NumCircBuf OS errors."""


class NumCircBufNotImplementedError(NumCircBufError, NotImplementedError):
    """Base exception class for all NumCircBuf NotImplemented errors."""


class BufferCapacityError(NumCircBufInitError):
    """Base exception class for invalid buffer capacity values."""


class ConfigurationError(NumCircBufInitError):
    """Base exception class for invalid configuration parameters."""


# Warning Base Classes


class NumCircBufWarning(_NumCircBufBase, Warning):
    """Base warning class for all NumCircBuf warnings."""


class NumCircBufRuntimeWarning(NumCircBufWarning, RuntimeWarning):
    """Base warning class for all NumCircBuf Runtime warnings."""


class _NumCircBufDeprecationBase(NumCircBufWarning):
    """Base warning class for all NumCircBuf deprecation warnings."""

    default_message: str = "Feature deprecated."

    def __init__(
        self,
        obj,
        feature,
        replacement,
        remove_in_version,
        message=None,
    ):
        self.feature = feature
        self.replacement = replacement
        self.remove_in_version = remove_in_version
        if message is None:
            message = self.default_message
        super().__init__(
            class_obj=obj.__class__,
            obj=obj,
            message=(
                f"\n{obj}: {message}\n"
                f"Feature: {feature}\n"
                f"Replacement: {replacement}\n"
                f"Planned removal: {remove_in_version}"
            ),
        )


# --- Exceptions ---


class BufferCapacityTypeError(BufferCapacityError, NumCircBufTypeError):
    """Exception raised for invalid buffer capacity types."""

    def __init__(
        self,
        class_obj,
        received_type,
        message="Buffer capacity/maxlen must be a python integer.",
    ):
        self.received_type = received_type
        self.valid_types = (int,)
        super().__init__(
            class_obj=class_obj,
            message=(
                f"\n{class_obj.__name__}:\n{message}\nGot: {received_type}"
            ),
        )


class BufferCapacityValueError(BufferCapacityError, NumCircBufValueError):
    """Exception raised for invalid buffer capacity values."""

    def __init__(
        self,
        overflow: bool,
        class_obj,
        received_value=None,
        max_maxlen: int = Limits.PY_SSIZE_T_MAX.value,
        message=None,
    ):
        self.overflow = overflow
        self.received_value = received_value
        self.max_maxlen = max_maxlen
        self.valid_values = {"min": 1, "max": max_maxlen}

        error_str = f"\n{class_obj.__name__}:\n"

        if overflow:
            message = message or "Buffer capacity/maxlen too large."
            error_str += f"{message}\nMax: {max_maxlen:_}\n"
        else:
            message = (
                message or "Buffer capacity/maxlen must be greater than 0."
            )
            error_str += f"{message}\n"

        error_str += f"Got: {received_value:_}"
        super().__init__(class_obj=class_obj, message=error_str)


class DataTypeError(
    NumCircBufInitError, NumCircBufValueError, NumCircBufTypeError
):
    """Exception raised for unsupported data types."""

    def __init__(
        self,
        class_obj,
        received_value,
        valid_values,
        message="Unsupported data type.",
    ):
        self.class_obj = class_obj
        self.received_value = received_value
        self.valid_values = valid_values
        super().__init__(
            class_obj=class_obj,
            message=(
                f"\n{class_obj.__name__}:\n{message}\n"
                f"Valid types: {valid_values}\n"
                f"Got:         {received_value}"
            ),
        )


class ConfigurationTypeError(ConfigurationError, NumCircBufTypeError):
    """Exception raised for invalid configuration parameter types."""

    def __init__(
        self,
        class_obj,
        parameter,
        received_type,
        valid_types,
        message="Invalid configuration parameter.",
    ):
        self.parameter = parameter
        self.received_type = received_type
        self.valid_types = valid_types

        if len(valid_types) == 1:
            valid_types = valid_types[0]

        super().__init__(
            class_obj=class_obj,
            message=(
                f"\n{class_obj.__name__}:\n{message}\n"
                f"Parameter:   {parameter}\n"
                f"Valid types: {valid_types}\n"
                f"Got:         {received_type}"
            ),
        )


class ConfigurationValueError(ConfigurationError, NumCircBufValueError):
    """Exception raised for invalid configuration parameters."""

    def __init__(
        self,
        class_obj,
        parameter,
        received_value,
        valid_values: tuple | dict,
        message="Invalid configuration parameter.",
    ):
        self.parameter = parameter
        self.received_value = received_value
        self.valid_values = valid_values

        if isinstance(received_value, str):
            received_value = f"'{received_value}'"

        super().__init__(
            class_obj=class_obj,
            message=(
                f"\n{class_obj.__name__}:\n{message}\n"
                f"Parameter:    {parameter}\n"
                f"Valid values: {valid_values}\n"
                f"Got:          {received_value}"
            ),
        )


class IndexOutOfBounds(NumCircBufIndexError):
    """Exception raised for invalid buffer indexing."""

    def __init__(self, obj, index, message="Index out of bounds."):
        self.index = index
        super().__init__(
            class_obj=obj.__class__,
            obj=obj,
            message=f"\n{obj}: {message} Index: {index}",
        )


class InvalidModification(NumCircBufTypeError):
    """Raised when attempting to modify a buffer invalidly."""

    def __init__(self, obj, recommendation):
        self.recommendation = recommendation
        super().__init__(
            class_obj=obj.__class__,
            obj=obj,
            message=(
                f"\n{obj} cannot be modified this way."
                f"\nRecommendation: {recommendation}"
            ),
        )


class UnsupportedOperation(NumCircBufArithmeticError):
    """Operation not supported for this buffer dtype."""

    def __init__(self, obj, func, func_str, message):
        self.func = func
        self.func_str = func_str
        super().__init__(class_obj=obj.__class__, obj=obj, message=message)


# --- Warnings ---


class DataSizeWarning(NumCircBufRuntimeWarning):
    """Warning for when data size exceeds buffer capacity."""

    def __init__(
        self,
        obj,
        data_size,
        maxlen,
        message="Data size exceeds total buffer capacity. "
        "Only last N elements will be stored.",
    ):
        self.data_size = data_size
        self.maxlen = maxlen
        super().__init__(
            class_obj=obj.__class__,
            obj=obj,
            message=(
                f"{obj}: {message} "
                f"Data size: {data_size:_}, Buffer maxlen: {maxlen:_}"
            ),
        )


class NumCircBufDeprecationWarning(
    _NumCircBufDeprecationBase, DeprecationWarning
):
    """Light Warning for deprecated features."""


class NumCircBufFutureWarning(_NumCircBufDeprecationBase, FutureWarning):
    """Strong Warning for features that will be removed in the near future."""

    default_message = "Feature about to be removed!"
