from __future__ import annotations

import json
from pathlib import Path

from .models import Claim, ExperimentResult

# Every read and write below pins encoding and newline explicitly.
#
# Without encoding, `read_text()` uses the platform codec -- cp1252 on Windows,
# UTF-8 on Linux -- so a hand-edited claims.json round-trips correctly on one
# platform and is silently mojibaked on the other. Without newline="", Python's
# own translation writes CRLF on Windows and LF elsewhere, so identical state
# hashes differently per machine.


class ResearchStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.claims_path = root / "claims.json"
        self.experiments_path = root / "experiments.jsonl"

    def load_claims(self) -> list[Claim]:
        if not self.claims_path.exists():
            return []
        payload = json.loads(self.claims_path.read_text(encoding="utf-8"))
        return [Claim.model_validate(item) for item in payload]

    def save_claims(self, claims: list[Claim]) -> None:
        payload = [c.model_dump(mode="json") for c in claims]
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.claims_path.write_text(text, encoding="utf-8", newline="")

    def append_experiment(self, result: ExperimentResult) -> None:
        with self.experiments_path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(result.model_dump_json() + "\n")

    def load_experiments(self) -> list[ExperimentResult]:
        if not self.experiments_path.exists():
            return []
        items: list[ExperimentResult] = []
        text = self.experiments_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                items.append(ExperimentResult.model_validate_json(line))
        return items

    def latest_experiment(self) -> ExperimentResult | None:
        items = self.load_experiments()
        return items[-1] if items else None

    def experiment_index(self, result: ExperimentResult) -> int:
        """1-based position of a result in the append-only log.

        This is the logical clock handed to DRE as ``observed_at``: monotonic,
        reproducible, and independent of wall-clock time.
        """
        items = self.load_experiments()
        for idx, item in enumerate(items, start=1):
            if item == result:
                return idx
        return len(items) + 1
