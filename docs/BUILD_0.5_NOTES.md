# Build 0.5 — Durable RH Mathematical Memory

v0.5 makes the repository, rather than the chat transcript, the authoritative memory of the RH program.

## Added

- `research_state/math_knowledge.json` with 42 structured mathematical-memory entries.
- `docs/RH_MATHEMATICAL_MEMORY.md` containing the full human-readable research map.
- `KnowledgeBase` loader/search/dependency validation.
- CLI commands: `rhre knowledge list|show|search|validate`.
- Agent prompt requirement to consult durable memory before proposing or reviewing a route.
- Tests that ensure the knowledge graph loads, its dependencies resolve, and permanent no-go routes remain encoded.

## Epistemic design

The memory intentionally separates exact identities, established frameworks, internally derived statements awaiting independent verification, research targets, and false routes. This prevents the harness from turning a remembered derivation into an unearned theorem after context loss.
