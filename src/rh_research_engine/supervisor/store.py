from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .models import KNOWN_QUEUE_SCHEMA_VERSIONS, HypothesisQueue, QueueSchemaError


class HypothesisQueueStore:
    def __init__(self, path: Path = Path("research_state/hypotheses.json")) -> None:
        self.path = path

    def load(self, *, allow_missing: bool = False) -> HypothesisQueue:
        """Read the queue. Never writes -- migration happens in memory.

        Fails closed when the file is absent. The queue is the research plan,
        and an empty plan is a report that there is nothing to work on -- the
        same shape of lie as an empty knowledge base. `allow_missing=True` is
        for the two callers where absence is genuinely legitimate: adding the
        first hypothesis, and building a property graph from whatever inputs
        exist.
        """
        if not self.path.exists():
            if not allow_missing:
                raise QueueSchemaError(
                    f"no hypothesis queue at {self.path}. An absent research plan is "
                    "not an empty one: reporting zero hypotheses reads as 'nothing "
                    "outstanding' when the truth is that the plan could not be read. "
                    "Run `rhre supervisor add` to create it."
                )
            return HypothesisQueue()
        raw = self.path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        if isinstance(document, dict) and "schema_version" not in document:
            # Unreadable, not assumable. Guessing "probably current" is how a v1
            # file gets read as v2 and its `state` fields dropped as unknown keys
            # -- a silent downgrade of every record in the plan.
            raise QueueSchemaError(
                f"{self.path} declares no schema_version. Add one -- "
                f"{sorted(KNOWN_QUEUE_SCHEMA_VERSIONS)} are readable -- rather than "
                "leaving the reader to guess which shape the records are in."
            )
        queue = HypothesisQueue.model_validate(document)
        # Only this layer sees the bytes, so only it can record what they
        # hashed to. The model computes the two semantic hashes.
        if queue.migration is not None and queue.migration.source_file_sha256 is None:
            queue.migration.source_file_sha256 = hashlib.sha256(raw).hexdigest()
        return queue

    def save(self, queue: HypothesisQueue) -> None:
        """Write atomically.

        A partial write here is worse than no write: the queue is the research
        plan, and a truncated JSON file fails to load at all, so an interrupted
        save would take the plan with it. Writing to a temporary file in the
        same directory and replacing keeps the old contents intact until the new
        ones are complete on disk.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = queue.model_dump(mode="json")
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            # os.replace is atomic on POSIX and Windows alike.
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
