from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars._utils.deprecation import deprecated

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    if sys.version_info >= (3, 13):
        from warnings import deprecated
    else:
        from typing_extensions import deprecated  # noqa: TC004


__all__ = [
    "StringCache",
    "disable_string_cache",
    "enable_string_cache",
    "using_string_cache",
]


@deprecated("the `StringCache` is deprecated; it is no longer needed")
class StringCache(contextlib.ContextDecorator):
    """
    Context manager for enabling and disabling the global string cache.

    This class was historically useful to control when categorical mappings
    get dropped but no longer does anything.

    .. deprecated:: 1.32.0
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


@deprecated("the `StringCache` is deprecated; it is no longer needed")
def enable_string_cache() -> None:
    """
    Enable the global string cache.

    This function was historically useful to control when categorical mappings
    get dropped but no longer does anything.

    .. deprecated:: 1.32.0
    """


@deprecated("the `StringCache` is deprecated; it is no longer needed")
def disable_string_cache() -> bool:
    """
    Disable and clear the global string cache.

    This function was historically useful to control when categorical mappings
    get dropped but no longer does anything.

    .. deprecated:: 1.32.0
    """


@deprecated("the `StringCache` is deprecated; it is no longer needed")
def using_string_cache() -> bool:
    """
    Check whether the global string cache is enabled.

    This function was historically useful to control when categorical mappings
    get dropped but no longer does anything, simply always returning true.

    .. deprecated:: 1.32.0
    """
    return True
