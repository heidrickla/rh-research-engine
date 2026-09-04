from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import PropertyGraph, PropertyKind


class PropertyGraphIntegrityError(ValueError):
    """Raised when the stored property graph cannot be trusted as loaded."""


class PropertyGraphStore:
    """Persistence for the property graph.

    The graph is derived state, but it is derived state that downstream tools
    query for *rigorous* properties -- so a hand-edited or corrupted file is a
    way to introduce a rigorous-looking claim that no extractor produced. It
    gets the same sidecar seal as durable memory.
    """

    def __init__(self, path: Path = Path("research_state/property_graph.json")) -> None:
        self.path = path

    @property
    def seal_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".sha256")

    def _verify_seal(self, raw: bytes) -> None:
        if not self.seal_path.exists():
            return
        expected = self.seal_path.read_text(encoding="utf-8").split()[0].strip()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise PropertyGraphIntegrityError(
                f"{self.path} does not match its seal ({self.seal_path.name}).\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "Rebuild it with `rhre properties build` rather than editing it by hand."
            )

    def seal(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.seal_path.write_text(digest + "\n", encoding="utf-8", newline="")
        return digest

    def load(self) -> PropertyGraph:
        if not self.path.exists():
            return PropertyGraph()
        raw = self.path.read_bytes()
        self._verify_seal(raw)
        try:
            return PropertyGraph.model_validate_json(raw.decode("utf-8"))
        except ValueError as exc:
            # PropertyRecord refuses statuses this package cannot establish, so
            # an injected `"status": "proved"` surfaces here rather than loading
            # as an authoritative record.
            raise PropertyGraphIntegrityError(f"{self.path} failed validation: {exc}") from exc

    def save(self, graph: PropertyGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = graph.model_dump(mode="json")
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.path.write_text(text, encoding="utf-8", newline="")
        self.seal()

    def query(
        self,
        *,
        object_id: str | None = None,
        kind: PropertyKind | None = None,
        rigorous_only: bool = False,
    ):
        graph = self.load()
        out = graph.properties
        if object_id is not None:
            out = [item for item in out if item.object_id == object_id]
        if kind is not None:
            out = [item for item in out if item.kind == kind]
        if rigorous_only:
            out = [item for item in out if item.is_rigorous]
        return sorted(out, key=lambda item: item.id)
