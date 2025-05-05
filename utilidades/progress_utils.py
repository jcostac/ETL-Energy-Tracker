import threading
import time
from typing import Callable, TypeVar, Any
from functools import wraps
from tqdm import tqdm

T = TypeVar('T')

class ProgressIndicator:
    """
    A context manager and utility class for showing progress messages or bars during long-running operations.
    """
    def __init__(self, message: str = "Processing...", interval: float = 2.0, 
                 use_progress_bar: bool = False, total_steps: int = None):
        """
        Initialize the progress indicator.

        Args:
            message (str): Message to display periodically. Defaults to "Processing..."
            interval (float): Time in seconds between message displays. Defaults to 2.0
            use_progress_bar (bool): Whether to use a progress bar instead of message. Defaults to False
            total_steps (int): Total steps for progress bar (required if use_progress_bar=True)
        """
        self.message = message
        self.interval = interval
        self.is_running = False
        self.thread = None
        self.use_progress_bar = use_progress_bar
        self.total_steps = total_steps
        self.progress_bar = None
        
        if self.use_progress_bar and self.total_steps is None:
            raise ValueError("total_steps must be provided when using progress bar")

    def _show_progress(self):
        """Internal method to periodically show the progress message."""
        if self.use_progress_bar:
            self.progress_bar = tqdm(
                total=self.total_steps,
                desc=self.message,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            )
        else:
            while self.is_running:
                print(self.message, flush=True)
                time.sleep(self.interval)

    def update_progress(self, steps: int = 1):
        """
        Update the progress bar by the specified number of steps.
        Only applicable when using progress bar mode.

        Args:
            steps (int): Number of steps to increment. Defaults to 1
        """
        if self.use_progress_bar and self.progress_bar:
            self.progress_bar.update(steps)

    def __enter__(self):
        """Start the progress indicator when entering the context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress indicator when exiting the context."""
        self.stop()

    def start(self):
        """Start showing the progress indicator."""
        self.is_running = True
        if not self.use_progress_bar:
            self.thread = threading.Thread(target=self._show_progress)
            self.thread.daemon = True
            self.thread.start()
        else:
            self._show_progress()

    def stop(self):
        """Stop showing the progress indicator."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.progress_bar:
            self.progress_bar.close()
        if not self.use_progress_bar:
            print("Processing complete!")

# === DECORATOR ===
def with_progress(message: str = "Processing...", interval: float = 2.0,
                 use_progress_bar: bool = False, total_steps: int = None):
    """
    A decorator that shows a progress indicator while the decorated function runs.

    Args:
        message (str): Message to display periodically
        interval (float): Time in seconds between message displays
        use_progress_bar (bool): Whether to use a progress bar instead of message
        total_steps (int): Total steps for progress bar (required if use_progress_bar=True)

    Example:
        @with_progress("Processing data...", use_progress_bar=True, total_steps=100)
        def long_running_function(progress_indicator=None):
            for i in range(100):
                # do something
                if progress_indicator:
                    progress_indicator.update_progress()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with ProgressIndicator(message, interval, use_progress_bar, total_steps) as progress:
                # Add progress indicator to kwargs if the function accepts it
                if 'progress_indicator' in func.__code__.co_varnames:
                    kwargs['progress_indicator'] = progress
                return func(*args, **kwargs)
        return wrapper
    return decorator

# === RUN WITH PROGRESS FUNCTION ===
def run_with_progress(func: Callable[..., T], message: str = "Processing...", 
                     interval: float = 2.0, use_progress_bar: bool = False,
                     total_steps: int = None, *args, **kwargs) -> T:
    """
    Run a function with a progress indicator.

    Args:
        func (Callable): The function to run
        message (str): Message to display periodically
        interval (float): Time in seconds between message displays
        use_progress_bar (bool): Whether to use a progress bar instead of message
        total_steps (int): Total steps for progress bar (required if use_progress_bar=True)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Example:
        def process_data(items, progress_indicator=None):
            for item in items:
                # process item
                if progress_indicator:
                    progress_indicator.update_progress()
        
        result = run_with_progress(
            process_data, 
            "Processing data...", 
            use_progress_bar=True,
            total_steps=len(items),
            items=items
        )
    """
    with ProgressIndicator(message, interval, use_progress_bar, total_steps) as progress:
        if 'progress_indicator' in func.__code__.co_varnames:
            kwargs['progress_indicator'] = progress
        return func(*args, **kwargs) 