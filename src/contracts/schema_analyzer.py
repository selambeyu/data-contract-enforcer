"""
SchemaEvolutionAnalyzer — Phase 3 (producer-side CI gate).

Diffs consecutive timestamped schema snapshots and classifies each change as:
  BREAKING  — consumers will fail (field removed, type narrowed, required added,
               enum value removed, format/pattern added to existing field)
  COMPATIBLE — consumers are unaffected (optional field added, enum extended,
               range widened, description changed)

The SchemaEvolutionAnalyzer runs on the PRODUCER side as a pre-emptive CI gate.
It catches breaking changes BEFORE they ship to consumers.
The ValidationRunner is the reactive layer that catches what the Analyzer missed.

CLI (spec-aligned):
    # Diff two most recent snapshots for a specific contract
    python src/contracts/schema_analyzer.py \
        --contract-id week3-document-refinery-extractions \
        --since "7 days ago" \
        --snapshots schema_snapshots \
        --output validation_reports/schema_evolution_week3.json

    # Diff all contracts (batch mode)
    python src/contracts/schema_analyzer.py \
        --all \
        --snapshots schema_snapshots \
        --output-dir validation_reports
"""

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import yaml


# ── Snapshot discovery ─────────────────────────────────────────────────────────

def discover_snapshots(snapshots_root: Path, contract_id: str) -> list[Path]:
    """
    Return all timestamped snapshot YAML files for a contract, sorted oldest-first.
    Looks in schema_snapshots/{contract_id}/*.yaml
    Also handles legacy v1/v2 layout by checking snapshots_root/v1/ and v2/.
    """
    contract_dir = snapshots_root / contract_id
    if contract_dir.exists():
        snaps = sorted(contract_dir.glob("*.yaml"))
        if snaps:
            return snaps

    # Legacy fallback: v1/v2 dirs
    v1 = snapshots_root / "v1" / f"{contract_id}.json"
    v2 = snapshots_root / "v2" / f"{contract_id}.json"
    if v1.exists() and v2.exists():
        return [v1, v2]

    return []


def load_snapshot(path: Path) -> dict:
    with open(path) as f:
        if path.suffix == ".yaml":
            return yaml.safe_load(f)
        return json.load(f)


def _parse_snapshot_time(snap: dict, path: Path) -> datetime:
    """Parse captured_at from snapshot dict, or fall back to file mtime."""
    ts_str = snap.get("captured_at") or snap.get("snapshot_version", "")
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def find_snapshots_since(
    snapshots_root: Path,
    contract_id: str,
    since_str: str | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Return (older_snapshot, newer_snapshot) pair.
    If since_str is provided (e.g. "7 days ago"), find the last snapshot
    before that cutoff and the most recent snapshot after it.
    Otherwise returns the two most recent snapshots.
    """
    paths = discover_snapshots(snapshots_root, contract_id)
    if len(paths) < 2:
        return None, None

    if since_str:
        cutoff = _parse_since(since_str)
        snaps_with_time = []
        for p in paths:
            snap = load_snapshot(p)
            t = _parse_snapshot_time(snap, p)
            snaps_with_time.append((t, p))
        snaps_with_time.sort(key=lambda x: x[0])

        before = [s for s in snaps_with_time if s[0] <= cutoff]
        after  = [s for s in snaps_with_time if s[0] > cutoff]
        if not before or not after:
            # Fall back to oldest vs newest
            return snaps_with_time[0][1], snaps_with_time[-1][1]
        return before[-1][1], after[-1][1]

    # Default: two most recent
    return paths[-2], paths[-1]


def _parse_since(since_str: str) -> datetime:
    """
    Parse human-readable 'since' string like '7 days ago', '2 weeks ago'.
    Returns a UTC datetime representing the cutoff.
    """
    since_str = since_str.strip().lower()
    now = datetime.now(timezone.utc)
    try:
        parts = since_str.split()
        n = int(parts[0])
        unit = parts[1].rstrip("s")  # days/day → day, weeks/week → week
        if unit in ("day",):
            return now - timedelta(days=n)
        if unit in ("week",):
            return now - timedelta(weeks=n)
        if unit in ("hour",):
            return now - timedelta(hours=n)
    except Exception:
        pass
    return now - timedelta(days=7)  # default: 7 days


# ── Change classification ──────────────────────────────────────────────────────

def _classify_field_change(old_clause: dict, new_clause: dict, field: str) -> list[dict]:
    """Return classified change records for a modified field."""
    changes = []

    # Type change
    if old_clause.get("type") != new_clause.get("type"):
        changes.append({
            "field": field,
            "change_type": "type_changed",
            "classification": "BREAKING",
            "old": old_clause.get("type"),
            "new": new_clause.get("type"),
            "reason": "Type change breaks all consumers that parse this field.",
        })

    # required: False→True = BREAKING, True→False = COMPATIBLE
    old_req = old_clause.get("required", False)
    new_req = new_clause.get("required", False)
    if old_req != new_req:
        classification = "BREAKING" if (not old_req and new_req) else "COMPATIBLE"
        changes.append({
            "field": field,
            "change_type": "required_changed",
            "classification": classification,
            "old": old_req,
            "new": new_req,
            "reason": (
                "Making a field required breaks producers that omit it."
                if classification == "BREAKING"
                else "Making a field optional is backwards-compatible."
            ),
        })

    # Enum changes
    old_enum = set(old_clause.get("enum", []))
    new_enum = set(new_clause.get("enum", []))
    if old_enum and new_enum:
        removed_values = old_enum - new_enum
        added_values   = new_enum - old_enum
        if removed_values:
            changes.append({
                "field": field,
                "change_type": "enum_values_removed",
                "classification": "BREAKING",
                "removed": sorted(removed_values),
                "reason": "Removing enum values breaks consumers that send/receive those values.",
            })
        if added_values:
            changes.append({
                "field": field,
                "change_type": "enum_values_added",
                "classification": "COMPATIBLE",
                "added": sorted(added_values),
                "reason": "Adding enum values is compatible if consumers handle unknown values.",
            })
    elif old_enum and not new_clause.get("enum"):
        changes.append({
            "field": field,
            "change_type": "enum_removed",
            "classification": "BREAKING",
            "old_enum": sorted(old_enum),
            "reason": "Removing enum constraint drops validation guarantee for all consumers.",
        })
    elif not old_enum and new_clause.get("enum"):
        changes.append({
            "field": field,
            "change_type": "enum_added",
            "classification": "BREAKING",
            "new_enum": sorted(new_clause.get("enum", [])),
            "reason": "Adding enum constraint to existing field breaks any values not in the enum.",
        })

    # format: added = BREAKING, removed = COMPATIBLE
    old_fmt = old_clause.get("format")
    new_fmt = new_clause.get("format")
    if old_fmt != new_fmt:
        classification = "BREAKING" if (not old_fmt and new_fmt) else "COMPATIBLE"
        changes.append({
            "field": field,
            "change_type": "format_changed",
            "classification": classification,
            "old": old_fmt,
            "new": new_fmt,
            "reason": (
                "Adding a format constraint breaks existing data that doesn't conform."
                if classification == "BREAKING"
                else "Removing a format constraint is generally compatible."
            ),
        })

    # minimum/maximum — narrowed = BREAKING, widened = COMPATIBLE
    for bound in ("minimum", "maximum"):
        old_val = old_clause.get(bound)
        new_val = new_clause.get(bound)
        if old_val is None and new_val is not None:
            changes.append({
                "field": field,
                "change_type": f"{bound}_added",
                "classification": "BREAKING",
                "new": new_val,
                "reason": f"Adding {bound} constraint breaks existing data outside that bound.",
            })
        elif old_val is not None and new_val is None:
            changes.append({
                "field": field,
                "change_type": f"{bound}_removed",
                "classification": "COMPATIBLE",
                "old": old_val,
                "reason": f"Removing {bound} constraint is backwards-compatible.",
            })
        elif old_val is not None and new_val is not None and old_val != new_val:
            if bound == "minimum":
                classification = "BREAKING" if new_val > old_val else "COMPATIBLE"
            else:
                classification = "BREAKING" if new_val < old_val else "COMPATIBLE"
            changes.append({
                "field": field,
                "change_type": f"{bound}_changed",
                "classification": classification,
                "old": old_val,
                "new": new_val,
                "reason": f"{'Narrowing' if classification == 'BREAKING' else 'Widening'} {bound} range.",
            })

    return changes


def diff_schemas(v1_schema: dict, v2_schema: dict) -> list[dict]:
    """Diff two schema dicts, return classified change records."""
    changes = []
    v1_fields = set(v1_schema.keys())
    v2_fields = set(v2_schema.keys())

    for field in v1_fields - v2_fields:
        clause = v1_schema[field]
        changes.append({
            "field": field,
            "change_type": "field_removed",
            "classification": "BREAKING",
            "was_required": clause.get("required", False),
            "was_type": clause.get("type"),
            "reason": "Removing a field breaks all consumers that read it.",
        })

    for field in v2_fields - v1_fields:
        clause = v2_schema[field]
        is_required = clause.get("required", False)
        changes.append({
            "field": field,
            "change_type": "field_added",
            "classification": "BREAKING" if is_required else "COMPATIBLE",
            "required": is_required,
            "type": clause.get("type"),
            "reason": (
                "Adding a required field breaks producers that don't include it."
                if is_required
                else "Adding an optional field is backwards-compatible."
            ),
        })

    for field in v1_fields & v2_fields:
        old_clause = v1_schema[field]
        new_clause = v2_schema[field]
        if old_clause != new_clause:
            changes.extend(_classify_field_change(old_clause, new_clause, field))

    return changes


# ── Migration checklist ────────────────────────────────────────────────────────

def build_migration_checklist(changes: list[dict]) -> list[str]:
    checklist = []
    breaking  = [c for c in changes if c["classification"] == "BREAKING"]
    compatible = [c for c in changes if c["classification"] == "COMPATIBLE"]

    if breaking:
        checklist.append("=== BREAKING CHANGES — require version bump and consumer migration ===")
        for c in breaking:
            field = c["field"]
            ct    = c["change_type"]
            if ct == "field_removed":
                checklist.append(
                    f"  [ ] REMOVE field '{field}': notify all registry subscribers to stop reading this field."
                )
            elif ct == "field_added" and c.get("required"):
                checklist.append(
                    f"  [ ] ADD required field '{field}' ({c.get('type')}): all producers must populate before deploying."
                )
            elif ct == "type_changed":
                checklist.append(
                    f"  [ ] TYPE CHANGE '{field}': {c.get('old')} → {c.get('new')} — update all read/write paths."
                )
            elif ct == "enum_values_removed":
                checklist.append(
                    f"  [ ] ENUM SHRINK '{field}': removed {c.get('removed')} — consumers must stop sending these values."
                )
            elif ct == "enum_added":
                checklist.append(
                    f"  [ ] ENUM ADDED to '{field}': values restricted to {c.get('new_enum')} — validate all producers."
                )
            elif ct == "required_changed" and c.get("new"):
                checklist.append(
                    f"  [ ] NOW REQUIRED '{field}': all producers must include this field before deploying."
                )
            elif "minimum" in ct or "maximum" in ct:
                bound = ct.split("_")[0]
                checklist.append(
                    f"  [ ] RANGE NARROWED '{field}' {bound}: audit existing data for out-of-range values."
                )
            else:
                checklist.append(f"  [ ] {ct.upper()} '{field}': {c.get('reason','')}")
        checklist.append("")

    if compatible:
        checklist.append("=== COMPATIBLE CHANGES — no consumer action required ===")
        for c in compatible:
            field = c["field"]
            ct    = c["change_type"]
            if ct == "field_added":
                checklist.append(
                    f"  [✓] NEW optional field '{field}' ({c.get('type')}) — consumers may adopt at leisure."
                )
            elif ct == "enum_values_added":
                checklist.append(
                    f"  [✓] ENUM EXTENDED '{field}': new values {c.get('added')} — consumers that handle unknown values are unaffected."
                )
            elif "removed" in ct:
                checklist.append(f"  [✓] CONSTRAINT RELAXED '{field}': {ct} — backwards-compatible.")
            else:
                checklist.append(f"  [✓] {ct.upper()} '{field}' — compatible change.")

    return checklist


# ── Rollback plan ──────────────────────────────────────────────────────────────

def build_rollback_plan(changes: list[dict], contract_id: str) -> list[str]:
    """
    Generate a rollback plan for BREAKING changes.
    In production: "revert to previous contract version and re-deploy".
    """
    breaking = [c for c in changes if c["classification"] == "BREAKING"]
    if not breaking:
        return ["No breaking changes — rollback not required."]

    plan = [
        f"=== ROLLBACK PLAN for {contract_id} ===",
        "If breaking changes cause downstream failures, execute in order:",
        "",
    ]
    plan.append("  1. Immediately pin all consumers to the previous contract version:")
    plan.append(f"     Set contract_version in subscriptions.yaml back to the prior version.")
    plan.append("")
    plan.append("  2. Revert the producer schema to the previous snapshot:")
    plan.append(f"     git revert the commit that introduced the breaking change.")
    plan.append(f"     Or restore from: schema_snapshots/{contract_id}/<prev_timestamp>.yaml")
    plan.append("")
    plan.append("  3. Notify all registered subscribers via their contact field:")
    for c in breaking[:5]:
        plan.append(f"     - Field '{c['field']}' ({c['change_type']}): {c.get('reason', '')}")
    plan.append("")
    plan.append(
        "  4. After rollback confirmed: re-introduce changes as COMPATIBLE "
        "(add optional fields, deprecate with alias before removal)."
    )
    return plan


# ── Diff runner ────────────────────────────────────────────────────────────────

def diff_snapshots(older_path: Path, newer_path: Path, contract_id: str = "") -> dict:
    older = load_snapshot(older_path)
    newer = load_snapshot(newer_path)
    cid = contract_id or older.get("contract_id", str(older_path.stem))

    changes    = diff_schemas(older.get("schema", {}), newer.get("schema", {}))
    breaking   = [c for c in changes if c["classification"] == "BREAKING"]
    compatible = [c for c in changes if c["classification"] == "COMPATIBLE"]
    checklist  = build_migration_checklist(changes)
    rollback   = build_rollback_plan(changes, cid)

    return {
        "contract_id":   cid,
        "diff_generated_at": datetime.now(timezone.utc).isoformat(),
        "older_snapshot": str(older_path),
        "newer_snapshot": str(newer_path),
        "older_captured_at": older.get("captured_at", ""),
        "newer_captured_at": newer.get("captured_at", ""),
        "summary": {
            "total_changes":   len(changes),
            "breaking":        len(breaking),
            "compatible":      len(compatible),
            "requires_version_bump": len(breaking) > 0,
            "compatibility_verdict": "BREAKING" if breaking else "COMPATIBLE",
        },
        "changes": changes,
        "migration_checklist": checklist,
        "rollback_plan": rollback,
    }


def print_diff_summary(result: dict) -> None:
    s      = result["summary"]
    cid    = result["contract_id"]
    verdict = "⚠️  BREAKING — version bump required" if s["requires_version_bump"] else "✓ COMPATIBLE"
    print(f"\n[schema_analyzer] {cid}")
    print(f"  Older snapshot : {result['older_snapshot']}")
    print(f"  Newer snapshot : {result['newer_snapshot']}")
    print(f"  Changes        : {s['total_changes']}  (breaking={s['breaking']}, compatible={s['compatible']})")
    print(f"  Verdict        : {verdict}")
    for line in result.get("migration_checklist", []):
        print(f"    {line}")
    if s["requires_version_bump"]:
        print(f"\n  Rollback Plan:")
        for line in result.get("rollback_plan", []):
            print(f"    {line}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diff schema snapshots and classify breaking vs compatible changes. "
            "Runs on the producer side as a CI gate."
        )
    )
    # Spec-aligned primary CLI
    parser.add_argument(
        "--contract-id",
        help="Contract ID to analyze (e.g. week3-document-refinery-extractions)",
    )
    parser.add_argument(
        "--since",
        default="7 days ago",
        help="Find snapshot before this cutoff vs most recent (e.g. '7 days ago')",
    )
    # Batch mode
    parser.add_argument(
        "--all",
        action="store_true",
        help="Diff all contracts found under --snapshots directory",
    )
    # Paths
    parser.add_argument(
        "--snapshots",
        default="schema_snapshots",
        help="Root dir of schema snapshots (default: schema_snapshots/)",
    )
    parser.add_argument(
        "--output",
        help="Output path for single diff JSON (used with --contract-id)",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_reports",
        help="Output dir for --all mode (default: validation_reports/)",
    )
    # Legacy direct-path arguments (kept for backwards compat)
    parser.add_argument("--v1", help="[legacy] Path to older snapshot file")
    parser.add_argument("--v2", help="[legacy] Path to newer snapshot file")

    args = parser.parse_args()
    snapshots_root = Path(args.snapshots)

    if args.v1 and args.v2:
        # Legacy mode: explicit file paths
        result = diff_snapshots(Path(args.v1), Path(args.v2))
        print_diff_summary(result)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[schema_analyzer] Written: {out}")
        return

    if args.all:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Discover all contract directories
        contract_ids = set()
        for p in snapshots_root.iterdir():
            if p.is_dir() and p.name not in ("v1", "v2", "diffs", "baselines.json"):
                contract_ids.add(p.name)
        # Also check legacy v1/v2
        v1_dir = snapshots_root / "v1"
        if v1_dir.exists():
            for f in v1_dir.glob("*.json"):
                contract_ids.add(f.stem)

        if not contract_ids:
            print("[schema_analyzer] No contracts found. Run the generator first.")
            return

        print(f"[schema_analyzer] Diffing {len(contract_ids)} contract(s) ...")
        for cid in sorted(contract_ids):
            older, newer = find_snapshots_since(snapshots_root, cid, args.since)
            if not older or not newer:
                print(f"  SKIP {cid} — need at least 2 snapshots (run generator twice).")
                continue
            result = diff_snapshots(older, newer, cid)
            out_path = output_dir / f"schema_evolution_{cid}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print_diff_summary(result)
            print(f"  Written: {out_path}")
        return

    if args.contract_id:
        older, newer = find_snapshots_since(snapshots_root, args.contract_id, args.since)
        if not older or not newer:
            print(
                f"[schema_analyzer] Not enough snapshots for '{args.contract_id}'. "
                f"Run the generator at least twice to generate two snapshots."
            )
            return
        result = diff_snapshots(older, newer, args.contract_id)
        print_diff_summary(result)
        out = Path(args.output) if args.output else (
            Path(args.output_dir) / f"schema_evolution_{args.contract_id}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[schema_analyzer] Written: {out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
