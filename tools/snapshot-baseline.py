#!/usr/bin/env python3
"""Record the frozen baseline: state hashes, module inventory, versions.

WHY. Phase 1 migrates authoritative research state. `math_knowledge.json` is
the durable research record and is sealed; moving or reformatting it must be a
deliberate act with a verifiable before/after. This manifest is the "before".

The migration plan is explicit that a path move and a JSON normalization must
not happen in the same commit, so the manifest records exact bytes: a move is
correct when the content hash is unchanged, and a normalization is a separate,
separately reviewable change of hash.

Usage:
  python tools/snapshot-baseline.py            # refresh the current-state manifest
  python tools/snapshot-baseline.py --check    # fail if state drifted from it
  python tools/snapshot-baseline.py --frozen   # write the immutable df7016d baseline
  python tools/snapshot-baseline.py --since-baseline   # diff now against df7016d
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Regenerated every commit: what the authoritative state is *now*.
DEFAULT_OUT = REPO / "docs" / "contracts" / "current-state-manifest.json"

#: Written once from the tag and never regenerated: what it was at df7016d.
#:
#: These were one file, which made "baseline" a lie -- it moved with every
#: commit, so it could not answer "what has changed since the program started".
#: A frozen reference that drifts is not a reference.
FROZEN_OUT = REPO / "docs" / "contracts" / "frozen-baseline-df7016d.json"
STATE_DIR = REPO / "research_state"
PACK = REPO / "dre" / "model-packs" / "riemann-research"
BASELINE_TAG = "rh-lab-v1-baseline-df7016d"


class NotAGitRepository(RuntimeError):
    """The baseline is reconstructed from git history, which is absent."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO, encoding="utf-8"
    )
    if result.returncode == 0:
        return result.stdout.strip()

    detail = (result.stderr or "").strip().splitlines()
    if any("not a git repository" in line for line in detail):
        # The one case worth its own message: a CalledProcessError traceback
        # names an exit status and nothing a reader can act on.
        raise NotAGitRepository(
            "not a git repository. This gate rebuilds the frozen baseline "
            "from the tag, so it needs history -- an exported or unpacked "
            "copy cannot be checked against it."
        )
    # Any OTHER git failure keeps its original type -- a missing tag in a
    # shallow clone is the common one, and callers already degrade for it
    # deliberately. Reclassifying every git error as "no repository" broke that
    # fallback in CI once.
    raise subprocess.CalledProcessError(
        result.returncode, ["git", *args], result.stdout, result.stderr
    )


#: State that is rewritten by every run and is not authoritative.
#:
#: The baseline exists to catch AUTHORITATIVE state drifting unnoticed. The open
#: findings ledger is neither: `rhre patterns sweep` rewrites it every time, so
#: including it made a routine sweep fail the gate, and the only way through was
#: to delete the file -- which is a habit that would eventually delete something
#: that mattered.
#:
#: Narrow and named, never a pattern. `pattern_noise.json` stays tracked: a noise
#: rule is a judgement somebody made once, with a reason, and losing one is
#: losing research. The exclusions are reported in the manifest rather than
#: applied silently, because a gate that quietly stops watching a file is worse
#: than one that never watched it.
REGENERATED_PER_RUN = frozenset({"research_state/pattern_open.json"})


def _hash_tree(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if str(path.relative_to(REPO)).replace("\\", "/") in REGENERATED_PER_RUN:
            continue
        raw = path.read_bytes()
        key = str(path.relative_to(REPO)).replace("\\", "/")
        out[key] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "crlf": b"\r\n" in raw,
        }
    return out


def _knowledge_invariants(kb) -> dict:
    """Semantic invariants, in one shape shared by both manifests.

    The frozen and current manifests previously computed different field sets,
    so a field present on only one side surfaced as a change and printed the
    entire semantic-hash dict. A diff that is unreadable is a diff nobody reads.
    """
    from rh_research_engine.core.knowledge import KnowledgeStatus

    items = kb.load()
    return {
        "record_count": len(items),
        "ids": sorted(item.id for item in items),
        "semantic_hashes": kb.semantic_hashes(),
        "dependency_edges": [list(edge) for edge in kb.dependency_edges()],
        "no_go_ids": sorted(
            item.id for item in items if item.status is KnowledgeStatus.FALSE_ROUTE
        ),
        "research_target_ids": sorted(
            item.id for item in items if item.status is KnowledgeStatus.RESEARCH_TARGET
        ),
        "governance_ids": sorted(
            item.id for item in items if item.status is KnowledgeStatus.GOVERNANCE
        ),
        "dependency_errors": kb.validate_dependencies(),
    }


def _knowledge_facts() -> dict:
    from rh_research_engine.core.knowledge import KnowledgeBase, resolve_knowledge_path

    return _knowledge_invariants(KnowledgeBase(resolve_knowledge_path()))


def _source_inventory() -> dict:
    files = [
        p
        for p in sorted((REPO / "src").rglob("*.py"))
        if "__pycache__" not in p.parts
    ]
    total = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in files)
    return {"module_count": len(files), "source_lines": total}


def _git_show(ref: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, cwd=REPO
    )
    return result.stdout if result.returncode == 0 else None


def build_frozen_manifest(ref: str = BASELINE_TAG) -> dict:
    """Reconstruct the baseline from the tag, not from the working tree.

    Reading the working tree would record whatever is there today and label it
    "baseline", which is how the two manifests became one file in the first
    place.
    """
    import tempfile

    state: dict[str, dict] = {}
    for relative in (
        "research_state/claims.json",
        "research_state/experiments.jsonl",
        "research_state/math_knowledge.json",
        "research_state/math_knowledge.json.sha256",
    ):
        raw = _git_show(ref, relative)
        if raw is None:
            continue
        state[relative] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "crlf": b"\r\n" in raw,
        }

    knowledge_raw = _git_show(ref, "research_state/math_knowledge.json")
    invariants: dict = {}
    if knowledge_raw is not None:
        from rh_research_engine.core.knowledge import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "math_knowledge.json"
            target.write_bytes(knowledge_raw)
            invariants = _knowledge_invariants(KnowledgeBase(target))

    return {
        "manifest_version": "1",
        "manifest_kind": "frozen-baseline",
        "immutable": True,
        "baseline_ref": ref,
        "baseline_commit": _git("rev-list", "-n", "1", ref),
        "authoritative_state": state,
        "knowledge_invariants": invariants,
        "note": (
            "Reconstructed from the tag and never regenerated. Regenerating it "
            "would make it describe the present, which is what current-state-"
            "manifest.json is for."
        ),
    }


def build_manifest() -> dict:
    from rh_research_engine import __version__

    model_yaml = (PACK / "model.yaml").read_text(encoding="utf-8")
    pack_version = next(
        (line.split(":", 1)[1].strip() for line in model_yaml.splitlines() if line.strip().startswith("version:")),
        "unknown",
    )
    requires_engine = next(
        (line.split(":", 1)[1].strip() for line in model_yaml.splitlines() if line.strip().startswith("requires_engine:")),
        "unknown",
    )
    # The baseline is a fixed point, not "wherever HEAD is". Recording HEAD
    # would make the manifest drift with ordinary development and quietly stop
    # describing the thing Phase 1 migrates away from.
    try:
        baseline_commit = _git("rev-list", "-n", "1", BASELINE_TAG)
        baseline_ref = BASELINE_TAG
    except subprocess.CalledProcessError:
        baseline_commit = _git("rev-parse", "HEAD")
        baseline_ref = "HEAD (baseline tag not found)"
    return {
        "manifest_version": "1",
        "manifest_kind": "current-state",
        "baseline_ref": baseline_ref,
        "baseline_commit": baseline_commit,
        "package_version": __version__,
        "dre_pack": {
            "path": "dre/model-packs/riemann-research",
            "version": pack_version,
            "requires_engine": requires_engine,
        },
        "source": _source_inventory(),
        "authoritative_state": _hash_tree(STATE_DIR),
        # Named in the manifest so a reader can see WHAT the gate stopped
        # watching. A gate that quietly narrows what it looks at is worse than
        # one that never looked.
        "excluded_as_regenerated": sorted(REGENERATED_PER_RUN),
        "model_pack_files": _hash_tree(PACK),
        "knowledge_invariants": _knowledge_facts(),
    }


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


#: Why each invariant is checked. Printed with the failure, because the reader
#: of a failing migration gate is often not the person who wrote the gate.
_INVARIANT_RATIONALE = {
    "record_count": "the durable memory must not gain or lose entries in a path migration",
    "ids": "record IDs are referenced by dependencies and provenance; renaming one breaks both",
    "no_go_ids": "a lost no-go route lets a refuted line of attack be rediscovered as novel",
    "research_target_ids": "the research targets are what the frontier is measured against",
    "governance_ids": "governance records must stay excluded from property extraction",
    "dependency_errors": "a dangling dependency means the knowledge graph no longer resolves",
}


def _describe_invariant_drift(before: dict, after: dict) -> list[str]:
    """Say which invariant broke, by how much, and why it matters."""
    problems: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        why = _INVARIANT_RATIONALE.get(key, "")
        if isinstance(old, list) and isinstance(new, list):
            gone = [item for item in old if item not in new]
            added = [item for item in new if item not in old]
            problems.append(f"  knowledge invariant '{key}' changed ({len(old)} -> {len(new)})")
            if gone:
                problems.append(f"      missing: {gone}")
            if added:
                problems.append(f"      new:     {added}")
        elif isinstance(old, dict) or isinstance(new, dict):
            old_map, new_map = old or {}, new or {}
            changed = sorted(
                k for k in set(old_map) | set(new_map) if old_map.get(k) != new_map.get(k)
            )
            problems.append(
                f"  knowledge invariant '{key}' changed for {len(changed)} record(s): "
                f"{changed[:8]}{' ...' if len(changed) > 8 else ''}"
            )
        else:
            problems.append(f"  knowledge invariant '{key}' changed: {old!r} -> {new!r}")
        if why:
            problems.append(f"      why it matters: {why}")
    return problems


def main(argv: list[str]) -> int:
    try:
        return _run(argv)
    except NotAGitRepository as exc:
        print(f"authoritative state: CANNOT CHECK -- {exc}", file=sys.stderr)
        # Nonzero on purpose: "could not verify" is not "verified".
        return 2


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--frozen", action="store_true", help="Write the immutable baseline.")
    parser.add_argument("--force", action="store_true", help="Overwrite the frozen baseline.")
    parser.add_argument(
        "--since-baseline", action="store_true", help="Diff current state against df7016d."
    )
    args = parser.parse_args(argv)

    if args.frozen:
        if FROZEN_OUT.exists() and not args.force:
            print(
                f"{FROZEN_OUT.relative_to(REPO)} already exists and is immutable. "
                "Rewriting it would destroy the only record of where the program "
                "started; pass --force only if it was generated wrongly.",
                file=sys.stderr,
            )
            return 1
        frozen = build_frozen_manifest()
        FROZEN_OUT.parent.mkdir(parents=True, exist_ok=True)
        FROZEN_OUT.write_text(render(frozen), encoding="utf-8", newline="")
        print(f"wrote {FROZEN_OUT.relative_to(REPO)} (immutable)")
        print(f"  {frozen['baseline_ref']} -> {frozen['baseline_commit'][:7]}")
        print(f"  {frozen['knowledge_invariants'].get('record_count')} knowledge records")
        return 0

    if args.since_baseline:
        if not FROZEN_OUT.exists():
            print("no frozen baseline; run --frozen first", file=sys.stderr)
            return 1
        frozen = json.loads(FROZEN_OUT.read_text(encoding="utf-8"))
        current = build_manifest()
        problems = _describe_invariant_drift(
            frozen["knowledge_invariants"], current["knowledge_invariants"]
        )
        if not problems:
            print(f"knowledge invariants unchanged since {frozen['baseline_ref']}")
            return 0
        print(f"changes since {frozen['baseline_ref']}:")
        print("\n".join(problems))
        return 0

    manifest = build_manifest()

    if args.check:
        if not args.out.exists():
            print(f"no manifest at {args.out}", file=sys.stderr)
            return 1
        stored = json.loads(args.out.read_text(encoding="utf-8"))
        problems: list[str] = []
        # The commit and source inventory move with ordinary development; the
        # authoritative state and the knowledge invariants must not.
        if stored["authoritative_state"] != manifest["authoritative_state"]:
            for key in sorted(set(stored["authoritative_state"]) | set(manifest["authoritative_state"])):
                before = stored["authoritative_state"].get(key)
                after = manifest["authoritative_state"].get(key)
                if before == after:
                    continue
                if before is None:
                    problems.append(f"  state file ADDED: {key} ({after['bytes']} bytes)")
                elif after is None:
                    problems.append(f"  state file REMOVED: {key} (was {before['bytes']} bytes)")
                else:
                    problems.append(f"  state changed: {key}")
                    problems.append(
                        f"      bytes  {before['bytes']} -> {after['bytes']}"
                        f"   ({after['bytes'] - before['bytes']:+d})"
                    )
                    problems.append(
                        f"      sha256 {before['sha256'][:16]}... -> {after['sha256'][:16]}..."
                    )
                    if before["crlf"] != after["crlf"]:
                        problems.append(
                            f"      line endings changed: crlf={before['crlf']} -> {after['crlf']}"
                            "  (this alone changes every content hash)"
                        )
        problems.extend(_describe_invariant_drift(
            stored["knowledge_invariants"], manifest["knowledge_invariants"]
        ))
        if problems:
            print("authoritative state DRIFTED from the frozen baseline:", file=sys.stderr)
            print("\n".join(problems), file=sys.stderr)
            print(
                "\nA migration may legitimately change these -- but it must say so and "
                "regenerate the manifest in the same commit.",
                file=sys.stderr,
            )
            return 1
        print(
            f"authoritative state unchanged: {manifest['knowledge_invariants']['record_count']} "
            f"knowledge records, no-go {manifest['knowledge_invariants']['no_go_ids']}"
        )
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(manifest), encoding="utf-8", newline="")
    print(f"wrote {args.out}")
    print(f"  commit {manifest['baseline_commit'][:7]}, package v{manifest['package_version']}")
    print(
        f"  {manifest['source']['module_count']} modules, "
        f"{manifest['source']['source_lines']} source lines"
    )
    print(
        f"  {manifest['knowledge_invariants']['record_count']} knowledge records, "
        f"no-go {manifest['knowledge_invariants']['no_go_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
