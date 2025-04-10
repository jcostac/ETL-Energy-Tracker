import warnings
import functools

def deprecated(replacement=None, reason=None):
    """A decorator to mark methods or functions as deprecated with an optional replacement and reason."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated."
            if reason:
                message += f" Reason: {reason}"
            if replacement:
                message += f" Use {replacement} instead."
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
