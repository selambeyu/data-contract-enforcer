"""
Migration: src/outputs/week5/events.json   (single JSON array)
       --> outputs/week5/events.jsonl      (one event per line)

DEVIATIONS FIXED:
  1. File format: single JSON array -> .jsonl (one event per line)
  2. aggregate_id: non-UUID string (e.g. 'loan-pg-doc-001')
                   -> deterministic UUID5(namespace=DNS, name=aggregate_id_string)
  3. occurred_at: postgres timestamp format '2026-03-24 13:41:39.344203+00'
                  -> ISO 8601 with T separator '2026-03-24T13:41:39.344203+00:00'
  4. recorded_at: same postgres format -> ISO 8601
  5. schema_version: '1.0' (two-part) -> '1.0.0' (semver three-part)
  6. metadata.causation_id / correlation_id / user_id: null values preserved
     (contract marks these as optional, so null is valid — no fix needed)

NOT FIXED (documented deviations for DOMAIN_NOTES.md):
  - event_id: already UUID format (valid — no change)
  - event_type: already PascalCase (valid — no change)
  - sequence_number: monotonic per aggregate (valid — no change)
  - metadata.source_service: null in many records (optional field — acceptable)
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "src" / "outputs" / "week5" / "events.json"
DST = Path(__file__).parent.parent / "week5" / "events.jsonl"

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace DNS

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(str(s)))


def fix_aggregate_id(raw: str) -> str:
    """Return UUID5 if raw is not already a UUID."""
    if is_uuid(raw):
        return raw
    return str(uuid.uuid5(_NS, raw))


def fix_timestamp(raw: str) -> str:
    """
    Convert postgres-style timestamp to ISO 8601.
    '2026-03-24 13:41:39.344203+00' -> '2026-03-24T13:41:39.344203+00:00'
    """
    if raw is None:
        return raw
    s = str(raw).strip()
    # Replace space separator with T
    s = re.sub(r"^(\d{4}-\d{2}-\d{2}) ", r"\1T", s)
    # Normalise '+00' or '+0000' -> '+00:00'
    s = re.sub(r"\+00(?::?00)?$", "+00:00", s)
    s = re.sub(r"-00(?::?00)?$", "-00:00", s)
    # Validate by parsing
    try:
        datetime.fromisoformat(s)
    except ValueError:
        # Last resort: just return original untouched
        return raw
    return s


def fix_schema_version(raw: str) -> str:
    """'1.0' -> '1.0.0' (semver)."""
    if raw is None:
        return "1.0.0"
    parts = str(raw).split(".")
    while len(parts) < 3:
        parts.append("0")
    return ".".join(parts[:3])


def migrate(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    with src.open() as f:
        events = json.load(f)

    if not isinstance(events, list):
        raise ValueError(f"Expected a JSON array, got {type(events)}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Track aggregate_id mapping for traceability
    agg_id_map: dict[str, str] = {}

    with dst.open("w") as f_out:
        for event in events:
            raw_agg_id = event.get("aggregate_id", "")

            # Cache the mapping so the same original string always maps to the same UUID
            if raw_agg_id not in agg_id_map:
                agg_id_map[raw_agg_id] = fix_aggregate_id(raw_agg_id)

            canonical = dict(event)
            canonical["aggregate_id"] = agg_id_map[raw_agg_id]
            canonical["occurred_at"] = fix_timestamp(event.get("occurred_at"))
            canonical["recorded_at"] = fix_timestamp(event.get("recorded_at"))
            canonical["schema_version"] = fix_schema_version(event.get("schema_version"))

            canonical["_migration_meta"] = {
                "original_aggregate_id": raw_agg_id,
                "migrated_from": str(src),
                "migration_script": __file__,
                "deviations_fixed": [
                    "aggregate_id converted to UUID5 if not already UUID",
                    "occurred_at converted from postgres format to ISO 8601",
                    "recorded_at converted from postgres format to ISO 8601",
                    "schema_version normalised to semver (X.Y.Z)",
                    "file format converted from JSON array to JSONL",
                ],
            }

            f_out.write(json.dumps(canonical) + "\n")

    print(f"Migrated {len(events)} events: {src} -> {dst}")
    print(f"  unique aggregate_ids remapped: {sum(1 for k, v in agg_id_map.items() if k != v)}")


if __name__ == "__main__":
    migrate(SRC, DST)
    # Smoke checks
    with DST.open() as f:
        first = json.loads(f.readline())
    assert is_uuid(first["aggregate_id"]), f"aggregate_id not UUID: {first['aggregate_id']}"
    assert "T" in first["occurred_at"], f"occurred_at missing T: {first['occurred_at']}"
    assert "T" in first["recorded_at"], f"recorded_at missing T: {first['recorded_at']}"
    v = first["schema_version"].split(".")
    assert len(v) == 3, f"schema_version not semver: {first['schema_version']}"
    # Validate ISO 8601 parse
    datetime.fromisoformat(first["occurred_at"])
    datetime.fromisoformat(first["recorded_at"])
    print("Smoke checks passed.")
