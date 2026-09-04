#!/usr/bin/env python3
"""Relocate durable memory into the authoritative-state layout, in stages.

STAGED, NOT DESTRUCTIVE. This copies the bytes to the new path, seals the copy,
writes a manifest, and **leaves the original in place**. The loader prefers the
new path while both exist and are identical. Retiring the old path is a
separate, later commit, so a rollback during the release is a file deletion
rather than a restore-from-history.

BYTES, NOT CONTENT. A relocation is correct exactly when the SHA-256 is
unchanged. Reformatting during a move destroys the only evidence that the move
was faithful -- a canonical-JSON normalisation is a different migration with its
own manifest and its own change of hash.

The manifest also records what must survive a *content* migration later: every
record ID, a per-record semantic hash, the dependency edges, the research
target, and the four no-go routes. Byte equality proves a file was copied; those
prove the research record survived.

Usage:
  python tools/migrate-knowledge-path.py            # perform the staged copy
  python tools/migrate-knowledge-path.py --check    # verify the staged state
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MIGRATION_ID = "knowledge-path-authoritative-v1"
MANIFEST_PATH = REPO / "docs" / "contracts" / "knowledge-path-migration.json"


def _invariants(path: Path) -> dict:
    from rh_research_engine.core.knowledge import KnowledgeBase, KnowledgeStatus

    kb = KnowledgeBase(path)
    items = kb.load()
    return {
        "record_count": len(items),
        "ids": sorted(item.id for item in items),
        "semantic_hashes": kb.semantic_hashes(),
        "dependency_edges": [list(edge) for edge in kb.dependency_edges()],
        "no_go_ids": sorted(
            i.id for i in items if i.status is KnowledgeStatus.FALSE_ROUTE
        ),
        "research_target_ids": sorted(
            i.id for i in items if i.status is KnowledgeStatus.RESEARCH_TARGET
        ),
        "dependency_errors": kb.validate_dependencies(),
    }


def _diff_invariants(before: dict, after: dict) -> list[str]:
    problems: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            for record in sorted(set(old) | set(new)):
                if old.get(record) != new.get(record):
                    problems.append(
                        f"  {key}: record {record} changed "
                        f"({str(old.get(record))[:16]} -> {str(new.get(record))[:16]})"
                    )
        elif isinstance(old, list) and isinstance(new, list):
            gone = [x for x in old if x not in new]
            added = [x for x in new if x not in old]
            problems.append(f"  {key} changed ({len(old)} -> {len(new)})")
            if gone:
                problems.append(f"      missing: {gone[:8]}")
            if added:
                problems.append(f"      new:     {added[:8]}")
        else:
            problems.append(f"  {key}: {old!r} -> {new!r}")
    return problems


def migrate(*, check_only: bool) -> int:
    from rh_research_engine.contracts.migrations import (
        MigrationManifest,
        write_manifest,
    )
    from rh_research_engine.core.knowledge import (
        CANONICAL_KNOWLEDGE_PATH,
        LEGACY_KNOWLEDGE_PATH,
    )

    legacy = REPO / LEGACY_KNOWLEDGE_PATH
    canonical = REPO / CANONICAL_KNOWLEDGE_PATH

    if check_only:
        if not canonical.exists():
            print(f"not migrated: {CANONICAL_KNOWLEDGE_PATH} does not exist", file=sys.stderr)
            return 1
        if not MANIFEST_PATH.exists():
            print(
                f"no migration manifest at {MANIFEST_PATH.relative_to(REPO)}. Without it "
                "there is nothing to check the canonical copy against.",
                file=sys.stderr,
            )
            return 1

        import json as _json

        manifest = _json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        recorded_hash = manifest["target_hash"]
        recorded_invariants = manifest["invariants"]

        canonical_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
        current = _invariants(canonical)

        # Compare against the MANIFEST, never against the canonical copy itself.
        #
        # The staged form of this check compared canonical to canonical once the
        # legacy path was gone -- vacuously true, so post-retirement it verified
        # nothing at all. The manifest is the only record of what the migration
        # was supposed to preserve, so it is the only defensible reference.
        problems: list[str] = []
        if canonical_hash != recorded_hash:
            problems.append(
                f"  canonical bytes differ from the manifest:\n"
                f"      manifest {recorded_hash}\n      actual   {canonical_hash}"
            )
        problems.extend(_diff_invariants(recorded_invariants, current))

        if legacy.exists():
            legacy_hash = hashlib.sha256(legacy.read_bytes()).hexdigest()
            if legacy_hash != canonical_hash:
                problems.append(
                    "  staged copies have DIVERGED -- reconcile before retiring:\n"
                    f"      legacy    {legacy_hash}\n      canonical {canonical_hash}"
                )
            state = "staged: both paths present"
        else:
            state = "retired: canonical only"

        if problems:
            print("knowledge relocation FAILED verification:", file=sys.stderr)
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"{state}, verified against {MANIFEST_PATH.name} ({canonical_hash[:16]}...)")
        print(f"  {current['record_count']} records, no-go {current['no_go_ids']}")
        return 0

    if not legacy.exists() and canonical.exists():
        print("already migrated and retired; legacy path is gone. Nothing to do.")
        return 0
    if not legacy.exists():
        print(f"{LEGACY_KNOWLEDGE_PATH} does not exist", file=sys.stderr)
        return 1

    before = _invariants(legacy)
    raw = legacy.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()

    if canonical.exists():
        if hashlib.sha256(canonical.read_bytes()).hexdigest() != source_hash:
            print(
                f"{CANONICAL_KNOWLEDGE_PATH} already exists with different bytes. "
                "Refusing to overwrite; reconcile deliberately.",
                file=sys.stderr,
            )
            return 1
        print("canonical copy already present and identical; re-verifying")
    else:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(raw)

    target_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
    if target_hash != source_hash:
        print(f"copy changed the bytes: {source_hash} -> {target_hash}", file=sys.stderr)
        return 1

    # Seal the destination. The legacy seal is left alone so the old path stays
    # loadable for rollback.
    seal_path = canonical.with_suffix(canonical.suffix + ".sha256")
    seal_path.write_text(target_hash + "\n", encoding="utf-8", newline="")

    after = _invariants(canonical)
    problems = _diff_invariants(before, after)
    if problems:
        print("invariants did not survive the copy:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    manifest = MigrationManifest(
        migration_id=MIGRATION_ID,
        description=(
            "Staged relocation of durable memory into the authoritative-state "
            "layout. Bytes unchanged; legacy path retained for rollback and "
            "retired in a later commit."
        ),
        source_path=str(LEGACY_KNOWLEDGE_PATH).replace("\\", "/"),
        target_path=str(CANONICAL_KNOWLEDGE_PATH).replace("\\", "/"),
        source_schema="math_knowledge/1",
        target_schema="math_knowledge/1",
        source_hash=source_hash,
        target_hash=target_hash,
        record_count=after["record_count"],
        invariants=after,
        notes=[
            "byte-for-byte copy; SHA-256 identical on both sides",
            "legacy path retained: rollback is a deletion, not a restore",
            "canonical JSON normalisation is a separate later migration",
            "the historical trailing '}]}' artifact was already removed in a6c47c1, "
            "so there is none left to preserve",
        ],
    )
    write_manifest(manifest, MANIFEST_PATH)

    print(f"staged {LEGACY_KNOWLEDGE_PATH} -> {CANONICAL_KNOWLEDGE_PATH}")
    print(f"  sha256 unchanged: {target_hash}")
    print(f"  {after['record_count']} records, no-go {after['no_go_ids']}, "
          f"target {after['research_target_ids']}")
    print(f"  manifest: {MANIFEST_PATH.relative_to(REPO)}")
    print("  legacy path retained for rollback")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return migrate(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
