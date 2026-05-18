"""Async utilities for subagent spawning and parallel execution."""

import asyncio
import logging
from typing import List, Callable, Any

logger = logging.getLogger(__name__)


async def run_in_executor_no_blocking(func: Callable, *args) -> Any:
    """Run a blocking function in a thread pool without blocking the event loop.

    Args:
        func: The blocking function to run.
        *args: Arguments to pass to the function.

    Returns:
        The return value of the function.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


async def gather_with_errors(*tasks) -> List[Any]:
    """Gather tasks, returning only successful results and logging errors for failed tasks.

    Args:
        *tasks: Coroutines to execute.

    Returns:
        List of successful results. Exceptions are logged but not returned.
    """
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            task_name = tasks[i].get_name() if hasattr(tasks[i], "get_name") else f"task-{i}"
            logger.error(f"Task {task_name} failed with exception: {type(r).__name__}: {str(r)}")
        else:
            successful.append(r)
    return successful


def create_task_with_logging(coro, name: str) -> asyncio.Task:
    """Create an asyncio task with logging on failure.

    Args:
        coro: The coroutine to wrap in a task.
        name: A descriptive name for the task (used in logging).

    Returns:
        An asyncio.Task that will log errors on failure.
    """
    task = asyncio.create_task(coro)
    task.set_name(name)

    def log_task_exception(result):
        if isinstance(result, Exception):
            logger.error(f"Task '{name}' failed: {type(result).__name__}: {str(result)}")

    task.add_done_callback(lambda t: log_task_exception(t.exception()) if t.exception() else None)

    return task
