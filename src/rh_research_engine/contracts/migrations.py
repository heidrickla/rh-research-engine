"""Migration records and the rules a migration must obey.

A migration of authoritative state is the one operation where "it looked fine
afterwards" is not evidence. Every migration produces a manifest naming the
source and target, both content hashes, the record count, and anything it could
not translate -- and refuses to guess.

Two rules the plan is explicit about, both encoded here:

1. **A path move and a reformatting are separate migrations.** A move is
   correct exactly when the content hash is unchanged; if the bytes changed too,
   there is no way to tell a relocation from an edit. ``MoveMigration`` enforces
   equal hashes.
2. **Ambiguity is reported, never resolved.** A record that does not map is
   listed in ``ambiguous_records`` and the migration fails, because silently
   picking one reading is how a status becomes wrong without anyone deciding it.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MigrationError(RuntimeError):
    """A migration cannot proceed without guessing."""


class MigrationManifest(BaseModel):
    """The record of one migration, written alongside its result."""

    manifest_version: str = "1"
    migration_id: str
    description: str = ""
    source_path: str | None = None
    target_path: str | None = None
    source_schema: str
    target_schema: str
    source_hash: str
    target_hash: str
    record_count: int
    mapping_version: int = 1
    #: Records that could not be translated. A non-empty list fails the run.
    ambiguous_records: list[str] = Field(default_factory=list)
    #: Invariants asserted before and after, and what they held at.
    invariants: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def bytes_unchanged(self) -> bool:
        return self.source_hash == self.target_hash

    @model_validator(mode="after")
    def _ambiguity_is_fatal(self):
        if self.ambiguous_records:
            raise MigrationError(
                f"migration {self.migration_id!r} left "
                f"{len(self.ambiguous_records)} record(s) ambiguous: "
                f"{self.ambiguous_records[:5]}. Resolve them explicitly; a migration "
                "that guesses produces state nobody decided on."
            )
        return self


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def move_preserving_bytes(
    source: Path,
    target: Path,
    *,
    migration_id: str,
    archive: Path | None = None,
    description: str = "",
) -> MigrationManifest:
    """Relocate a file without touching a byte, and prove it.

    Reformatting during a move destroys the only evidence that the move was
    faithful. If the content needs canonicalising, that is a second migration
    with its own manifest and its own change of hash.
    """
    if not source.exists():
        raise MigrationError(f"{source} does not exist; nothing to move")
    if target.exists():
        raise MigrationError(
            f"{target} already exists. Refusing to overwrite during a move -- "
            "delete or archive it deliberately first."
        )
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()

    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archive)
        archived_hash = sha256_file(archive)
        if archived_hash != source_hash:
            raise MigrationError(
                f"archive copy at {archive} hashes {archived_hash}, not {source_hash}"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    target_hash = sha256_file(target)
    if target_hash != source_hash:
        raise MigrationError(
            f"move changed the bytes: {source_hash} -> {target_hash}. A relocation "
            "that edits content cannot be distinguished from an edit that relocates."
        )
    source.unlink()

    return MigrationManifest(
        migration_id=migration_id,
        description=description,
        source_path=str(source).replace("\\", "/"),
        target_path=str(target).replace("\\", "/"),
        source_schema="bytes",
        target_schema="bytes",
        source_hash=source_hash,
        target_hash=target_hash,
        record_count=0,
        notes=[
            "byte-for-byte relocation; content unchanged",
            *( [f"legacy bytes archived at {archive}".replace("\\", "/")] if archive else [] ),
        ],
    )


def write_manifest(manifest: MigrationManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump_json(indent=2)
    path.write_text(payload + "\n", encoding="utf-8", newline="")
    return path
