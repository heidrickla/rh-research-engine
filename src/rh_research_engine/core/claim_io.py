from __future__ import annotations

import json
from pathlib import Path

from .models import Claim


def export_claim_packet(claim: Claim, path: Path) -> None:
    packet = {
        "claim": claim.model_dump(mode="json"),
        "review_questions": [
            "Is this equivalent to RH rather than progress toward RH?",
            "Which assumptions are strictly stronger than known theorems?",
            "Can a known no-go model falsify the reasoning?",
            "What explicit exponent on Theta would this prove?",
            "What deterministic computation or formal proof could validate it?",
        ],
    }
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
