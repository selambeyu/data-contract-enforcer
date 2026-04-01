"""
ValidationRunner — executes all contract checks against a dataset snapshot.

Usage:
    uv run python src/contracts/runner.py \
        --contract generated_contracts/week3-document-refinery-extractions.yaml \
        --data outputs/week3/extractions.jsonl \
        --output validation_reports/week3_baseline.json

Check order (per spec):
    Structural first: required, type, enum, uuid_pattern, datetime_format
    Statistical second: range, statistical_drift

Rule: never crash — always produce a report even on broken input.
"""

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_contract(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_baselines(baselines_path: Path) -> dict:
    if baselines_path.exists():
        with open(baselines_path) as f:
            return json.load(f)
    return {}


def save_baselines(baselines: dict, baselines_path: Path) -> None:
    baselines_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2)


def snapshot_id(data_path: str | Path) -> str:
    """SHA-256 of first 64 KB of the data file."""
    with open(data_path, "rb") as f:
        return hashlib.sha256(f.read(65536)).hexdigest()


# ── DataFrame builder (mirrors generator flatten logic) ───────────────────────

_PREFERRED_EXPLODE_KEYS = ["extracted_facts", "nodes", "edges", "events", "items", "records"]


def _pick_explode_key(record: dict) -> str | None:
    for key in _PREFERRED_EXPLODE_KEYS:
        if key in record and isinstance(record[key], list):
            return key
    for k, v in record.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k
    return None


def flatten_for_validation(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    _PREFIX_MAP = {
        "extracted_facts": "fact_",
        "nodes": "node_",
        "edges": "edge_",
        "events": "event_",
        "items": "item_",
        "records": "record_",
    }
    explode_key = _pick_explode_key(records[0])
    prefix = _PREFIX_MAP.get(explode_key, explode_key.rstrip("s") + "_") if explode_key else ""

    rows = []
    for r in records:
        base = {
            k: v for k, v in r.items()
            if not isinstance(v, (list, dict)) and not k.startswith("_")
        }
        items = r.get(explode_key, []) if explode_key else []
        if items:
            for item in items:
                if isinstance(item, dict):
                    item_flat = {
                        f"{prefix}{k}": v
                        for k, v in item.items()
                        if not isinstance(v, (list, dict))
                    }
                    rows.append({**base, **item_flat})
                else:
                    rows.append({**base, f"{prefix}value": item})
        else:
            rows.append(base)
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    return df


# ── Result builder ────────────────────────────────────────────────────────────

def _result(
    check_id: str,
    column_name: str,
    check_type: str,
    status: str,
    actual_value: Any,
    expected: str,
    severity: str,
    records_failing: int = 0,
    sample_failing: list | None = None,
    message: str = "",
) -> dict:
    return {
        "check_id": check_id,
        "column_name": column_name,
        "check_type": check_type,
        "status": status,
        "actual_value": str(actual_value),
        "expected": expected,
        "severity": severity,
        "records_failing": records_failing,
        "sample_failing": sample_failing or [],
        "message": message,
    }


# ── Structural checks ─────────────────────────────────────────────────────────

def check_required_fields(df: pd.DataFrame, schema: dict) -> list[dict]:
    results = []
    for col, clause in schema.items():
        if not clause.get("required", False):
            continue
        if col not in df.columns:
            results.append(_result(
                check_id=f"{col}.required.present",
                column_name=col,
                check_type="required",
                status="CRITICAL",
                actual_value="column missing entirely",
                expected="column present with 0 nulls",
                severity="CRITICAL",
                records_failing=len(df),
                message=f"Required column '{col}' is missing from data.",
            ))
            continue
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            results.append(_result(
                check_id=f"{col}.required.no_nulls",
                column_name=col,
                check_type="required",
                status="CRITICAL",
                actual_value=f"{null_count} nulls ({null_count/len(df)*100:.1f}%)",
                expected="null_count = 0",
                severity="CRITICAL",
                records_failing=null_count,
                message=f"Required field '{col}' has {null_count} null values.",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.required.no_nulls",
                column_name=col,
                check_type="required",
                status="PASS",
                actual_value="0 nulls",
                expected="null_count = 0",
                severity="CRITICAL",
                message=f"Required field '{col}' has no nulls.",
            ))
    return results


def check_type_match(df: pd.DataFrame, schema: dict) -> list[dict]:
    BITOL_TO_PANDAS = {
        "number": ["float64", "float32", "int64", "int32"],
        "integer": ["int64", "int32", "int16", "int8"],
        "boolean": ["bool"],
        "string": ["object"],
    }
    results = []
    for col, clause in schema.items():
        expected_type = clause.get("type")
        if not expected_type or col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        allowed = BITOL_TO_PANDAS.get(expected_type, [])
        if allowed and actual_dtype not in allowed:
            results.append(_result(
                check_id=f"{col}.type_match",
                column_name=col,
                check_type="type",
                status="CRITICAL",
                actual_value=f"dtype={actual_dtype}",
                expected=f"type={expected_type} (one of {allowed})",
                severity="CRITICAL",
                message=f"Column '{col}' dtype is '{actual_dtype}', expected Bitol type '{expected_type}'.",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.type_match",
                column_name=col,
                check_type="type",
                status="PASS",
                actual_value=f"dtype={actual_dtype}",
                expected=f"type={expected_type}",
                severity="CRITICAL",
                message=f"Column '{col}' dtype matches contract type '{expected_type}'.",
            ))
    return results


def check_enum_conformance(df: pd.DataFrame, schema: dict) -> list[dict]:
    results = []
    for col, clause in schema.items():
        enum_vals = clause.get("enum")
        if not enum_vals or col not in df.columns:
            continue
        non_null = df[col].dropna()
        bad_mask = ~non_null.isin(enum_vals)
        bad_count = int(bad_mask.sum())
        bad_sample = list(non_null[bad_mask].unique()[:5])
        if bad_count > 0:
            results.append(_result(
                check_id=f"{col}.enum_conformance",
                column_name=col,
                check_type="enum",
                status="FAIL",
                actual_value=f"{bad_count} non-conforming values",
                expected=f"all values in {enum_vals}",
                severity="FAIL",
                records_failing=bad_count,
                sample_failing=[str(v) for v in bad_sample],
                message=f"Column '{col}' has {bad_count} values outside enum: {bad_sample}",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.enum_conformance",
                column_name=col,
                check_type="enum",
                status="PASS",
                actual_value="all values conforming",
                expected=f"all values in {enum_vals}",
                severity="FAIL",
                message=f"Column '{col}' all values within enum.",
            ))
    return results


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def check_uuid_pattern(df: pd.DataFrame, schema: dict) -> list[dict]:
    results = []
    for col, clause in schema.items():
        if clause.get("format") != "uuid" or col not in df.columns:
            continue
        non_null = df[col].dropna().astype(str)
        # Sample up to 100 if large dataset
        sample = non_null.sample(min(100, len(non_null)), random_state=42) if len(non_null) > 10000 else non_null
        bad_mask = ~sample.str.match(_UUID_RE)
        bad_count = int(bad_mask.sum())
        bad_sample = list(sample[bad_mask].unique()[:5])
        if bad_count > 0:
            results.append(_result(
                check_id=f"{col}.uuid_pattern",
                column_name=col,
                check_type="uuid_pattern",
                status="FAIL",
                actual_value=f"{bad_count} values fail UUID regex",
                expected="all values match ^[0-9a-f]{8}-...-[0-9a-f]{12}$",
                severity="FAIL",
                records_failing=bad_count,
                sample_failing=[str(v) for v in bad_sample],
                message=f"Column '{col}' has {bad_count} values that are not valid UUIDs.",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.uuid_pattern",
                column_name=col,
                check_type="uuid_pattern",
                status="PASS",
                actual_value="all values match UUID pattern",
                expected="UUID format",
                severity="FAIL",
                message=f"Column '{col}' all values are valid UUIDs.",
            ))
    return results


def check_datetime_format(df: pd.DataFrame, schema: dict) -> list[dict]:
    results = []
    for col, clause in schema.items():
        if clause.get("format") != "date-time" or col not in df.columns:
            continue
        non_null = df[col].dropna().astype(str)
        bad_vals = []
        for v in non_null:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                bad_vals.append(v)
        bad_count = len(bad_vals)
        if bad_count > 0:
            results.append(_result(
                check_id=f"{col}.datetime_format",
                column_name=col,
                check_type="datetime_format",
                status="FAIL",
                actual_value=f"{bad_count} unparseable values",
                expected="all values parseable as ISO 8601",
                severity="FAIL",
                records_failing=bad_count,
                sample_failing=bad_vals[:5],
                message=f"Column '{col}' has {bad_count} values that are not valid ISO 8601 datetimes.",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.datetime_format",
                column_name=col,
                check_type="datetime_format",
                status="PASS",
                actual_value="all values parse as ISO 8601",
                expected="ISO 8601 date-time",
                severity="FAIL",
                message=f"Column '{col}' all values are valid ISO 8601 datetimes.",
            ))
    return results


# ── Statistical checks ────────────────────────────────────────────────────────

def check_range(df: pd.DataFrame, schema: dict) -> list[dict]:
    results = []
    for col, clause in schema.items():
        has_min = "minimum" in clause
        has_max = "maximum" in clause
        if not (has_min or has_max) or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        data_min = float(non_null.min())
        data_max = float(non_null.max())
        failures = []

        if has_min and data_min < clause["minimum"]:
            failures.append(
                f"min={data_min:.4f} < contract minimum={clause['minimum']}"
            )
        if has_max and data_max > clause["maximum"]:
            failures.append(
                f"max={data_max:.4f} > contract maximum={clause['maximum']}"
            )

        if failures:
            bad_count = 0
            if has_min:
                bad_count += int((non_null < clause["minimum"]).sum())
            if has_max:
                bad_count += int((non_null > clause["maximum"]).sum())
            results.append(_result(
                check_id=f"{col}.range",
                column_name=col,
                check_type="range",
                status="CRITICAL",
                actual_value=f"min={data_min:.4f}, max={data_max:.4f}",
                expected=(
                    f"min>={clause.get('minimum', '-inf')}, "
                    f"max<={clause.get('maximum', '+inf')}"
                ),
                severity="CRITICAL",
                records_failing=bad_count,
                message=f"Range violation on '{col}': {'; '.join(failures)}. "
                        f"This is the check that catches a 0.0–1.0 → 0–100 scale change.",
            ))
        else:
            results.append(_result(
                check_id=f"{col}.range",
                column_name=col,
                check_type="range",
                status="PASS",
                actual_value=f"min={data_min:.4f}, max={data_max:.4f}",
                expected=(
                    f"min>={clause.get('minimum', '-inf')}, "
                    f"max<={clause.get('maximum', '+inf')}"
                ),
                severity="CRITICAL",
                message=f"Column '{col}' values within contract range.",
            ))
    return results


def check_statistical_drift(
    column: str,
    current_mean: float,
    current_std: float,
    baselines: dict,
) -> dict | None:
    """
    Returns WARN if z_score > 2, FAIL if z_score > 3, None if no baseline yet.
    Baseline is written after this run.
    """
    if column not in baselines:
        return None
    b = baselines[column]
    z_score = abs(current_mean - b["mean"]) / max(b["stddev"], 1e-9)
    if z_score > 3:
        return {
            "status": "FAIL",
            "z_score": round(z_score, 2),
            "message": f"{column} mean drifted {z_score:.1f} stddev from baseline",
        }
    elif z_score > 2:
        return {
            "status": "WARN",
            "z_score": round(z_score, 2),
            "message": f"{column} mean drifted {z_score:.1f} stddev from baseline (warning threshold)",
        }
    return {"status": "PASS", "z_score": round(z_score, 2), "message": "Within baseline."}


def run_statistical_drift_checks(
    df: pd.DataFrame, schema: dict, baselines: dict
) -> tuple[list[dict], dict]:
    """
    Returns (results, updated_baselines).
    Updates baselines with current stats for columns that have no baseline yet.
    """
    results = []
    updated = dict(baselines)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if len(non_null) < 2:
            continue
        current_mean = float(non_null.mean())
        current_std = float(non_null.std())

        drift = check_statistical_drift(col, current_mean, current_std, baselines)

        if drift is None:
            # No baseline yet — record current stats, skip check
            updated[col] = {"mean": current_mean, "stddev": current_std}
            results.append(_result(
                check_id=f"{col}.statistical_drift",
                column_name=col,
                check_type="statistical_drift",
                status="PASS",
                actual_value=f"mean={current_mean:.4f}, stddev={current_std:.4f}",
                expected="baseline established (first run)",
                severity="WARN",
                message=f"No baseline for '{col}' — baseline recorded for future runs.",
            ))
        else:
            status = drift["status"]
            severity = "FAIL" if status == "FAIL" else "WARN"
            results.append(_result(
                check_id=f"{col}.statistical_drift",
                column_name=col,
                check_type="statistical_drift",
                status=status,
                actual_value=f"mean={current_mean:.4f}, z_score={drift['z_score']}",
                expected=f"z_score <= 2 (baseline mean={baselines[col]['mean']:.4f})",
                severity=severity,
                records_failing=0,
                message=drift["message"],
            ))

    return results, updated


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    results: list[dict],
    contract_id: str,
    snap_id: str,
    data_path: str,
) -> dict:
    statuses = [r["status"] for r in results]
    return {
        "report_id": str(uuid.uuid4()),
        "contract_id": contract_id,
        "snapshot_id": snap_id,
        "data_path": str(data_path),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checks": len(results),
        "passed": statuses.count("PASS"),
        "failed": statuses.count("FAIL"),
        "warned": statuses.count("WARN"),
        "critical": statuses.count("CRITICAL"),
        "errored": statuses.count("ERROR"),
        "results": results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all contract checks against a dataset and produce a validation report."
    )
    parser.add_argument("--contract", required=True, help="Path to contract YAML file")
    parser.add_argument("--data", required=True, help="Path to source JSONL data file")
    parser.add_argument("--output", required=True, help="Path for output validation report JSON")
    parser.add_argument(
        "--baselines",
        default="schema_snapshots/baselines.json",
        help="Path to baselines JSON (default: schema_snapshots/baselines.json)",
    )
    args = parser.parse_args()

    all_results: list[dict] = []

    # ── Load inputs
    print(f"[runner] Loading contract: {args.contract}")
    contract = load_contract(args.contract)
    schema = contract.get("schema", {})
    contract_id = contract.get("id", Path(args.contract).stem)

    print(f"[runner] Loading data: {args.data}")
    try:
        records = load_jsonl(args.data)
        print(f"[runner] Loaded {len(records)} records.")
        df = flatten_for_validation(records)
        print(f"[runner] DataFrame shape: {df.shape}")
        data_ok = True
    except Exception as exc:
        print(f"[runner] ERROR loading data: {exc}")
        all_results.append(_result(
            check_id="data.load",
            column_name="__file__",
            check_type="load",
            status="ERROR",
            actual_value=str(exc),
            expected="valid JSONL file",
            severity="CRITICAL",
            message=f"Could not load data file: {exc}",
        ))
        data_ok = False

    if data_ok:
        snap = snapshot_id(args.data)

        # ── Structural checks (always run in this order)
        print("[runner] Running structural checks ...")
        all_results += check_required_fields(df, schema)
        all_results += check_type_match(df, schema)
        all_results += check_enum_conformance(df, schema)
        all_results += check_uuid_pattern(df, schema)
        all_results += check_datetime_format(df, schema)

        # ── Statistical checks
        print("[runner] Running statistical checks ...")
        all_results += check_range(df, schema)

        baselines_path = Path(args.baselines)
        baselines = load_baselines(baselines_path)
        drift_results, updated_baselines = run_statistical_drift_checks(df, schema, baselines)
        all_results += drift_results
        save_baselines(updated_baselines, baselines_path)
        print(f"[runner] Baselines saved to {baselines_path}")
    else:
        snap = "error"

    # ── Build report
    report = build_report(all_results, contract_id, snap, args.data)

    # ── Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Summary
    print(f"\n[runner] ── Validation Report ──────────────────────────")
    print(f"[runner] Contract : {contract_id}")
    print(f"[runner] Checks   : {report['total_checks']}")
    print(f"[runner] PASS     : {report['passed']}")
    print(f"[runner] WARN     : {report['warned']}")
    print(f"[runner] FAIL     : {report['failed']}")
    print(f"[runner] CRITICAL : {report['critical']}")
    print(f"[runner] ERROR    : {report['errored']}")
    print(f"[runner] Report   : {output_path}")

    # Print any non-PASS results for immediate visibility
    non_pass = [r for r in all_results if r["status"] != "PASS"]
    if non_pass:
        print(f"\n[runner] ── Issues Found ───────────────────────────────")
        for r in non_pass:
            print(f"  [{r['status']}] {r['check_id']}: {r['message']}")
    else:
        print("[runner] All checks passed.")


if __name__ == "__main__":
    main()
