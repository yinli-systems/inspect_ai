from logging import getLogger
from typing import Any, Callable

import anyio.to_thread
from typing_extensions import override

from inspect_ai._util.async_zip import AsyncZipReader
from inspect_ai._util.file import filesystem
from inspect_ai._util.log_layout import eval_log_name

from .._log import EvalLog, EvalSample, EvalSampleSummary, EvalSpec
from .recorder import Recorder

logger = getLogger(__name__)


async def write_local_snapshot(
    location: str, fsync: bool, writer: Callable[[], None]
) -> bool:
    """Run a blocking atomic local write in a worker thread.

    The worker thread keeps the blocking write/fsync from stalling the
    event loop; the writer's atomic temp-file + rename pattern (see
    :mod:`inspect_ai._util.atomic_write`) means an interrupted write can't
    corrupt the existing log on disk (#2949).

    Shared local-write policy for the file recorders. ``fsync=False`` marks
    the write as a skippable intermediate snapshot: a ``PermissionError``
    from the atomic rename (Windows file-in-use, e.g. the viewer reading
    the log) is logged and swallowed rather than failing the task — the
    previous valid log is untouched and the next flush retries naturally.
    Final writes (``fsync=True``) always raise.

    The worker thread is not abandoned on cancellation (the anyio default):
    callers may reuse resources the writer reads (e.g. the zip temp file)
    as soon as this returns, so the thread must have finished by then.

    Args:
        location: Destination path, used in the skip warning.
        fsync: Whether this is a durable final write (True) or a skippable
            intermediate snapshot (False). The writer itself is responsible
            for actually fsync'ing (or not) accordingly — snapshots skip
            fsync since physical writeback of a large log would stall the
            worker thread for seconds per flush, and their crash-durability
            matches the pre-atomic-write behaviour (none).
        writer: Blocking callable that performs the atomic write.

    Returns:
        True if the write happened, False if it was skipped.
    """
    try:
        await anyio.to_thread.run_sync(writer)
        return True
    except PermissionError as ex:
        # On Windows os.replace() needs DELETE access on the target,
        # denied while a reader holds the log open.
        if fsync:
            raise
        logger.warning(
            f"Skipped intermediate log write for {location} "
            f"(file in use by another program): {ex}"
        )
        return False


class LogSampleSnapshot:
    """An operation's parsed input, independent of later writes to its location.

    JSON inputs are already loaded in full. Index them once for bulk callers,
    without a process-wide path cache. Each lookup returns a private copy so
    scorers and attachment resolution cannot mutate another lookup's input.
    """

    def __init__(self, log: EvalLog, location: str) -> None:
        self.header = log.model_copy(update={"samples": None})
        self.location = location
        self._samples: dict[tuple[str, int], EvalSample] = {}
        for sample in log.samples or []:
            self._samples.setdefault((str(sample.id), sample.epoch), sample)
        self.keys = sorted(
            [(sample.id, sample.epoch) for sample in log.samples or []],
            key=lambda key: (
                key[1],
                key[0] if isinstance(key[0], str) else str(key[0]).zfill(20),
            ),
        )

    def read(self, id: str | int, epoch: int) -> EvalSample:
        sample = self._samples.get((str(id), epoch))
        if sample is None:
            raise IndexError(
                f"Sample id {id} for epoch {epoch} not found in log {self.location}"
            )
        return sample.model_copy(deep=True)


class FileRecorder(Recorder):
    def __init__(
        self, log_dir: str, suffix: str, fs_options: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self.log_dir = log_dir.rstrip("/\\")
        self.suffix = suffix

        # initialise filesystem
        self.fs = filesystem(log_dir, fs_options if fs_options is not None else {})
        self.fs.mkdir(self.log_dir, exist_ok=True)

    def is_local(self) -> bool:
        return self.fs.is_local()

    @override
    def is_writeable(self) -> bool:
        return self.fs.is_writeable(self.log_dir)

    @override
    @classmethod
    async def read_log_sample(
        cls,
        location: str,
        id: str | int | None = None,
        epoch: int = 1,
        uuid: str | None = None,
        exclude_fields: set[str] | None = None,
        reader: AsyncZipReader | None = None,
    ) -> EvalSample:
        if id is None and uuid is None:
            raise ValueError("You must specify an 'id' or 'uuid' to read")

        # read the current document for this operation
        eval_log = await cls._log_file_maybe_cached(location)

        # throw if no samples
        if not eval_log.samples:
            raise IndexError(f"No samples found in log {location}")

        # find the sample by id, matched in string form exactly as the .eval
        # reader's member name `samples/{id}_epoch_{epoch}.json` matches (so
        # 1 finds "1" but never "001"), else by uuid
        eval_sample: EvalSample | None = None
        if id is not None:
            eval_sample = next(
                (
                    sample
                    for sample in eval_log.samples
                    if str(sample.id) == str(id) and sample.epoch == epoch
                ),
                None,
            )
        if eval_sample is None and uuid:
            eval_sample = next(
                (sample for sample in eval_log.samples if sample.uuid == uuid), None
            )
        if eval_sample is None:
            if id is None:
                raise IndexError(
                    f"Sample with uuid '{uuid}' not found in log {location}"
                )
            raise IndexError(
                f"Sample id {id} for epoch {epoch} not found in log {location}"
            )
        else:
            return eval_sample

    @classmethod
    @override
    async def read_log_sample_summaries(cls, location: str) -> list[EvalSampleSummary]:
        # read the current document for this operation
        eval_log = await cls._log_file_maybe_cached(location)
        if not eval_log.samples:
            return []
        return [sample.summary() for sample in eval_log.samples]

    @classmethod
    async def _log_file_maybe_cached(cls, location: str) -> EvalLog:
        # Retain this private entrypoint for subclasses/tests, but never share
        # an unversioned parsed document across independent read operations.
        # Same path does not imply same version; bulk APIs own their parse.
        return await cls.read_log(location)

    def _log_file_key(self, eval: EvalSpec) -> str:
        return eval_log_name(
            task=eval.task,
            task_id=eval.task_id,
            created=eval.created,
            model=eval.model,
        )

    def _log_file_path(self, eval: EvalSpec) -> str:
        return f"{self.log_dir}{self.fs.sep}{self._log_file_key(eval)}{self.suffix}"
