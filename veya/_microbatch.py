"""Bounded async collection of compatible HTTP work; inference stays on the owner's worker."""
import asyncio
import time
from dataclasses import dataclass


class MicrobatchUnavailable(RuntimeError):
    """The queue is full or its application is stopping."""


@dataclass
class _Job:
    key: str
    payload: object
    future: asyncio.Future
    enqueued: float


class Microbatcher:
    """Collect adjacent compatible jobs without letting cancelled callers grow the queue."""

    def __init__(self, execute, *, capacity, max_states, wait_seconds):
        self.execute = execute
        self.queue = asyncio.Queue(maxsize=capacity)
        self.max_states = max_states
        self.wait_seconds = wait_seconds
        self.closed = False
        self.active = []
        self.pending = None
        self.task = asyncio.create_task(self._work())

    async def submit(self, key, payload):
        if self.closed:
            raise MicrobatchUnavailable("server is stopping")
        future = asyncio.get_running_loop().create_future()
        job = _Job(key, payload, future, time.perf_counter())
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise MicrobatchUnavailable("server busy, try again later") from None
        # Cancellation only affects this request's future, never a shared model call.
        return await future

    async def _take(self):
        job = await self.queue.get()
        self.queue.task_done()
        return job

    @staticmethod
    def _finish(job, value=None, error=None):
        if not job.future.done():
            if error is None:
                job.future.set_result(value)
            else:
                job.future.set_exception(error)

    async def _work(self):
        try:
            while True:
                first = self.pending if self.pending is not None else await self._take()
                self.pending = None
                if first.future.cancelled():
                    continue
                self.active = [first]
                deadline = first.enqueued + self.wait_seconds
                while len(self.active) < self.max_states:
                    try:
                        job = self.queue.get_nowait()
                        self.queue.task_done()
                    except asyncio.QueueEmpty:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            job = await asyncio.wait_for(self._take(), remaining)
                        except asyncio.TimeoutError:
                            break
                    if job.future.cancelled():
                        continue
                    if job.key != first.key:
                        # A single pending item preserves FIFO across incompatible groups.
                        self.pending = job
                        break
                    self.active.append(job)
                self.active = [job for job in self.active if not job.future.cancelled()]
                if not self.active:
                    continue
                try:
                    outputs = await self.execute(self.active)
                    if len(outputs) != len(self.active):
                        raise RuntimeError("microbatch returned the wrong number of results")
                    for job, (result, error, infer_ms, queue_ms) in zip(self.active, outputs):
                        self._finish(job, (result, infer_ms, queue_ms), error)
                except Exception as exc:
                    for job in self.active:
                        self._finish(job, error=exc)
                self.active = []
        finally:
            self.closed = True
            error = MicrobatchUnavailable("server is stopping")
            for job in self.active:
                self._finish(job, error=error)
            if self.pending is not None:
                self._finish(self.pending, error=error)
            while not self.queue.empty():
                job = self.queue.get_nowait()
                self.queue.task_done()
                self._finish(job, error=error)

    async def close(self):
        self.closed = True
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
