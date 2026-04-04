"""
inject_violations.py — Phase 1: Produce violated data files for enforcer testing.

Injection A (week3):
    Multiply extracted_facts[].confidence × 100 in week3 extractions.
    → outputs/week3/extractions_violated.jsonl
    Expected runner result: CRITICAL on fact_confidence.range (max > 1.0)

Injection B (week5):
    Set event_type to "InvalidEventXYZ" on first 50 week5 events.
    → outputs/week5/events_violated.jsonl
    Expected runner result: FAIL on event_type.enum_conformance
"""

import json
import copy
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

WEEK3_SRC = ROOT / "outputs" / "week3" / "extractions.jsonl"
WEEK3_DST = ROOT / "outputs" / "week3" / "extractions_violated.jsonl"

WEEK5_SRC = ROOT / "outputs" / "week5" / "events.jsonl"
WEEK5_DST = ROOT / "outputs" / "week5" / "events_violated.jsonl"


def inject_week3_confidence_scale(src: Path, dst: Path) -> None:
    """Multiply all extracted_facts[].confidence values by 100."""
    records = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    violated = []
    for rec in records:
        r = copy.deepcopy(rec)
        facts = r.get("extracted_facts", [])
        for fact in facts:
            if "confidence" in fact and isinstance(fact["confidence"], (int, float)):
                fact["confidence"] = round(fact["confidence"] * 100, 4)
        r["_violation_injected"] = "fact_confidence scaled 0-1 -> 0-100"
        violated.append(r)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in violated:
            f.write(json.dumps(r) + "\n")

    injected_count = sum(
        len([fc for fc in r.get("extracted_facts", []) if "confidence" in fc])
        for r in violated
    )
    print(f"[inject_violations] Week3: {len(violated)} records written → {dst}")
    print(f"  confidence fields scaled: {injected_count}")


def inject_week5_invalid_event_type(src: Path, dst: Path, n: int = 50) -> None:
    """Replace event_type with 'InvalidEventXYZ' on first n records."""
    records = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    violated = []
    for i, rec in enumerate(records):
        r = copy.deepcopy(rec)
        if i < n:
            r["_original_event_type"] = r.get("event_type")
            r["event_type"] = "InvalidEventXYZ"
            r["_violation_injected"] = "event_type set to unregistered value"
        violated.append(r)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in violated:
            f.write(json.dumps(r) + "\n")

    print(f"[inject_violations] Week5: {len(violated)} records written → {dst}")
    print(f"  event_type violations injected: {min(n, len(violated))}")


if __name__ == "__main__":
    print("=== Phase 1: Violation Injection ===")
    inject_week3_confidence_scale(WEEK3_SRC, WEEK3_DST)
    inject_week5_invalid_event_type(WEEK5_SRC, WEEK5_DST)
    print("\nDone. Run ValidationRunner on violated files to verify detection.")
