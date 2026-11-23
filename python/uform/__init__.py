"""Distributed training and inference engine for multi-modal perception.

File: python/uform/__init__.py
Author: Ash Vardanian
Date: February 21, 2023
"""

import asyncio

from uform.lib import *  # noqa: F403
from uform.lib import Status, __version__ as __version__


async def _settled(call):
    """Answers what `call.wait()` answers, without parking a thread on the call.

    Polls `call.status`, and while it is deferred, re-polls on the next turn of the running loop
    through a future `loop.call_soon` resolves, then calls `wait()`, which no longer blocks.
    """
    loop = asyncio.get_running_loop()
    while call.status == Status.DEFERRED:
        turn = loop.create_future()
        loop.call_soon(turn.set_result, None)
        await turn
    return call.wait()
