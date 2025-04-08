import warnings
import functools

def deprecated(replacement=None):
    """A decorator to mark methods or functions as deprecated with an optional replacement."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated."
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
