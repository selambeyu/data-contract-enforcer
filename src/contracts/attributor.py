"""
ViolationAttributor — Phase 2.

Four-step attribution pipeline (spec-aligned):

  Step 1 — Registry blast radius query:
    Load contract_registry/subscriptions.yaml.
    Find all subscriptions where contract_id matches the failing contract AND
    breaking_fields contains the failing field.
    This is the DEFINITIVE subscriber list. Do NOT compute from lineage graph.

  Step 2 — Lineage traversal for enrichment:
    Use the Week 4 lineage graph to compute transitive contamination.
    BFS upstream from the source_path to find producer files.
    Annotate blast_radius with contamination_depth.
    This is ADDITIVE to the registry result, not a replacement.

  Step 3 — Git blame for cause attribution (Tier 1 only):
    For each upstream file identified via lineage traversal, run:
      git log --follow --since="14 days ago" -- {file_path}
    Rank by temporal proximity + keyword relevance.
    Confidence score formula: 1.0 − (days_since_commit × 0.1) − (lineage_hops × 0.2)
    Return top 5 candidates, never fewer than 1.

  Step 4 — Write violation_log/violations.jsonl:
    Spec-compliant schema with top-level check_id, blame_chain[], and blast_radius{}.

CLI:
    python src/contracts/attributor.py \
        --violation validation_reports/violated_week3.json \
        --registry contract_registry/subscriptions.yaml \
        --lineage outputs/week4/lineage_snapshots.jsonl \
        --output violation_log/violations.jsonl \
        --codebase-root /path/to/repo
"""

import argparse
import json
import subprocess
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Step 1: Registry blast radius ─────────────────────────────────────────────

def load_registry(registry_path: str | Path) -> list[dict]:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get("subscriptions", [])


def _normalize_breaking_fields(breaking_fields: list) -> list[str]:
    """
    Registry breaking_fields can be list of strings or list of {field, reason} dicts.
    Always return list of field name strings for matching.
    """
    result = []
    for bf in breaking_fields:
        if isinstance(bf, dict):
            result.append(bf.get("field", ""))
        else:
            result.append(str(bf))
    return [f for f in result if f]


def get_blast_radius(contract_id: str, failed_field: str, subscriptions: list[dict]) -> list[dict]:
    """
    Return subscriptions where contract_id matches AND breaking_fields contains
    a field that matches the failed_field (normalized, prefix-stripped).
    """
    # Normalize the failed field: strip explode prefix (fact_, node_, edge_, event_)
    normalized = failed_field
    for prefix in ("fact_", "node_", "edge_", "event_", "item_"):
        if failed_field.startswith(prefix):
            normalized = failed_field[len(prefix):]
            break

    affected = []
    for sub in subscriptions:
        if sub.get("contract_id") != contract_id:
            continue
        bf_names = _normalize_breaking_fields(sub.get("breaking_fields", []))
        # Match if any breaking field contains our field name as a substring
        matched = [b for b in bf_names if normalized in b or failed_field in b]
        if matched:
            sub_copy = dict(sub)
            sub_copy["_matched_breaking_fields"] = matched
            affected.append(sub_copy)
    return affected


# ── Step 2: Lineage traversal ─────────────────────────────────────────────────

def load_lineage_snapshot(lineage_path: str | Path) -> dict:
    """Load the latest snapshot from lineage_snapshots.jsonl."""
    lineage_path = Path(lineage_path)
    if not lineage_path.exists():
        return {}
    try:
        with open(lineage_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return {}
        return json.loads(lines[-1])
    except Exception as exc:
        print(f"[attributor] WARNING: Could not load lineage: {exc}")
        return {}


def _extract_file_path(node_id: str) -> str | None:
    """
    Extract file path from node_id.
    Formats seen: 'transformation:/abs/path/file.py:181', 'dataset:name'
    Returns the absolute path portion, or None if not a file node.
    """
    if not node_id:
        return None
    # Strip type prefix (transformation:, dataset:, etc.)
    parts = node_id.split(":", 1)
    if len(parts) < 2:
        return None
    path_part = parts[1]
    # Remove trailing line number (:181)
    path_part = path_part.rsplit(":", 1)[0] if path_part.count(":") >= 1 else path_part
    # Must be an absolute path to a real file
    if path_part.startswith("/") and Path(path_part).suffix:
        return path_part
    return None


def bfs_upstream_producers(
    source_data_path: str,
    snapshot: dict,
    max_depth: int = 4,
) -> list[dict]:
    """
    BFS upstream from nodes related to source_data_path.
    Traverses PRODUCES / WRITES / READS edges in reverse to find producer files.
    Returns list of {node_id, file_path, depth} for transformation nodes only.
    """
    if not snapshot:
        return []

    nodes_by_id = {n["node_id"]: n for n in snapshot.get("nodes", [])}
    edges = snapshot.get("edges", [])
    # Build reverse adjacency: target → list of sources (upstream producers)
    reverse_adj: dict[str, list[tuple[str, int]]] = {}
    for edge in edges:
        rel = edge.get("relationship", "")
        if rel in ("PRODUCES", "WRITES", "READS", "CONSUMES"):
            target = edge.get("target", "")
            source = edge.get("source", "")
            if target and source:
                reverse_adj.setdefault(target, []).append((source, int(edge.get("confidence", 1) * 100)))

    # Seed: find nodes whose node_id or file path matches our source file stem
    src_stem = Path(source_data_path).stem.lower()
    seed_nodes = [
        nid for nid in nodes_by_id
        if src_stem in nid.lower() or src_stem in nodes_by_id[nid].get("label", "").lower()
    ]

    # If no direct match, use all transformation nodes as seeds (broad sweep)
    if not seed_nodes:
        seed_nodes = [
            nid for nid in nodes_by_id
            if nid.startswith("transformation:")
        ][:10]  # cap

    visited = set()
    queue: deque[tuple[str, int]] = deque()
    for nid in seed_nodes:
        queue.append((nid, 0))
    producers = []

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)

        file_path = _extract_file_path(node_id)
        node_type = nodes_by_id.get(node_id, {}).get("type", "")
        if file_path and node_type in ("MODEL", "FILE", "PIPELINE", "SERVICE"):
            producers.append({
                "node_id": node_id,
                "file_path": file_path,
                "label": nodes_by_id.get(node_id, {}).get("label", ""),
                "depth": depth,
            })

        for upstream_id, _ in reverse_adj.get(node_id, []):
            if upstream_id not in visited:
                queue.append((upstream_id, depth + 1))

    # Deduplicate by file_path
    seen_paths: set[str] = set()
    unique_producers = []
    for p in producers:
        if p["file_path"] not in seen_paths:
            seen_paths.add(p["file_path"])
            unique_producers.append(p)

    return unique_producers[:5]  # cap at 5 per spec


# ── Step 3: Git blame ─────────────────────────────────────────────────────────

def _days_since(commit_date_str: str) -> float:
    """Parse git date string and return days elapsed."""
    try:
        # git --date=iso produces: "2026-04-01 21:14:27 +0300"
        # fromisoformat handles this in Python 3.11+
        ts = datetime.fromisoformat(commit_date_str.strip().replace(" +", "+").replace(" -", "-"))
        delta = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        return max(0.0, delta.total_seconds() / 86400)
    except Exception:
        return 30.0  # conservative default if unparseable


def _confidence_score(commit_date_str: str, lineage_hops: int) -> float:
    """
    Spec formula: base = 1.0 − (days_since_commit × 0.1)
    Reduce by 0.2 for each lineage hop between blamed file and failing column.
    Clamp to [0.0, 1.0].
    """
    days = _days_since(commit_date_str)
    score = 1.0 - (days * 0.1) - (lineage_hops * 0.2)
    return round(max(0.0, min(1.0, score)), 3)


_BLAME_KEYWORDS = [
    "confidence", "scale", "percent", "0-100", "0.0", "1.0",
    "type", "rename", "change", "fix", "update", "schema",
    "event_type", "enum", "format",
]


def _run_git_log(file_path: str, codebase_root: str) -> list[dict]:
    """Run git log --follow --since='14 days ago' on a file."""
    try:
        result = subprocess.run(
            [
                "git", "log",
                "--follow",
                "--since=14 days ago",
                "--format=%H|%an|%ae|%ai|%s",
                "--", file_path,
            ],
            capture_output=True,
            text=True,
            cwd=codebase_root,
            timeout=10,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append({
                    "commit_hash": parts[0],
                    "author": parts[2],  # use email (parts[2]) not name
                    "commit_timestamp": parts[3].strip(),
                    "commit_message": parts[4],
                    "file_path": file_path,
                })
        return commits
    except Exception:
        return []


def build_blame_chain(
    producers: list[dict],
    violated_field: str,
    codebase_root: str | None,
    tier: int,
) -> list[dict]:
    """
    Build ranked blame chain from upstream producer files.
    Returns list of {rank, file_path, commit_hash, author, commit_timestamp,
                     commit_message, confidence_score}
    Never returns more than 5 candidates.
    """
    if tier >= 3:
        return [{
            "rank": 1,
            "file_path": None,
            "commit_hash": None,
            "author": None,
            "commit_timestamp": None,
            "commit_message": None,
            "confidence_score": 0.0,
            "note": "Tier 3 external system — no git access. Attribution by contract version only.",
        }]

    if not codebase_root:
        return [{
            "rank": 1,
            "file_path": None,
            "commit_hash": None,
            "author": None,
            "commit_timestamp": None,
            "commit_message": None,
            "confidence_score": 0.0,
            "note": "No codebase_root provided — cannot run git blame. Pass --codebase-root.",
        }]

    all_commits = []
    for producer in producers:
        file_path = producer.get("file_path", "")
        depth = producer.get("depth", 0)
        commits = _run_git_log(file_path, codebase_root)
        for c in commits:
            c["_depth"] = depth
        all_commits.extend(commits)

    if not all_commits:
        # Fall back: run git log on migration scripts in the repo
        migrate_scripts = list(Path(codebase_root).glob("outputs/migrate/*.py"))[:3]
        for script in migrate_scripts:
            commits = _run_git_log(str(script), codebase_root)
            for c in commits:
                c["_depth"] = 1
            all_commits.extend(commits)

    if not all_commits:
        return [{
            "rank": 1,
            "file_path": None,
            "commit_hash": None,
            "author": None,
            "commit_timestamp": None,
            "commit_message": None,
            "confidence_score": 0.0,
            "note": f"No recent commits found in git history for upstream producers of '{violated_field}'.",
        }]

    # Score each commit
    scored = []
    for c in all_commits:
        msg_lower = c.get("commit_message", "").lower()
        keyword_hits = sum(1 for kw in _BLAME_KEYWORDS if kw in msg_lower)
        conf = _confidence_score(c.get("commit_timestamp", ""), c.get("_depth", 0))
        scored.append({**c, "_keyword_hits": keyword_hits, "_conf": conf})

    # Sort: keyword hits desc, then confidence desc
    scored.sort(key=lambda x: (x["_keyword_hits"], x["_conf"]), reverse=True)

    chain = []
    seen_hashes: set[str] = set()
    for i, c in enumerate(scored):
        h = c.get("commit_hash", "")
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        chain.append({
            "rank": len(chain) + 1,
            "file_path": c.get("file_path"),
            "commit_hash": c.get("commit_hash"),
            "author": c.get("author"),
            "commit_timestamp": c.get("commit_timestamp"),
            "commit_message": c.get("commit_message"),
            "confidence_score": c.get("_conf", 0.0),
        })
        if len(chain) >= 5:
            break

    return chain


# ── Contamination depth calculation ──────────────────────────────────────────

def _compute_contamination_depths(
    producer_contract_id: str,
    subscriber_ids: list[str],
    snapshot: dict,
) -> dict[str, int]:
    """
    For each subscriber_id, compute the minimum BFS hop distance from the
    producer node (identified by producer_contract_id) to that subscriber
    through the lineage graph.

    Returns {subscriber_id: depth} where depth >= 1.
    depth=1 means the subscriber is directly connected to the producer.
    depth=N means there are N-1 intermediate system hops between them.

    If the subscriber cannot be found in the lineage graph, defaults to 1
    (direct connection assumed — conservative estimate).

    contamination_depth lets downstream tooling quantify propagation severity:
    a violation at depth=1 is contained; at depth=3 it has crossed three
    system boundaries and is significantly harder to roll back.
    """
    if not snapshot or not subscriber_ids:
        return {sid: 1 for sid in subscriber_ids}

    nodes_by_id   = {n["node_id"]: n for n in snapshot.get("nodes", [])}
    edges          = snapshot.get("edges", [])

    # Build forward adjacency (producer → consumers)
    forward_adj: dict[str, list[str]] = {}
    for edge in edges:
        rel = edge.get("relationship", "")
        if rel in ("PRODUCES", "WRITES", "CONSUMES", "READS"):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src and tgt:
                forward_adj.setdefault(src, []).append(tgt)

    # Find seed nodes whose label or node_id contains the producer contract ID
    # Strip hyphens for fuzzy matching (week3-document-refinery → week3 document refinery)
    id_stem = producer_contract_id.lower().replace("-", "").replace("_", "")
    seed_nodes = [
        nid for nid, n in nodes_by_id.items()
        if (
            id_stem in nid.lower().replace("-", "").replace("_", "")
            or id_stem in n.get("label", "").lower().replace("-", "").replace("_", "")
        )
    ]
    if not seed_nodes:
        # No node found for producer — return default depth=1 for all
        return {sid: 1 for sid in subscriber_ids}

    # BFS forward from producer seeds; record depth at which each node_id is reached
    visited: dict[str, int] = {}   # node_id → depth
    queue: deque[tuple[str, int]] = deque()
    for nid in seed_nodes:
        queue.append((nid, 0))

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited:
            continue
        visited[node_id] = depth
        for nxt in forward_adj.get(node_id, []):
            if nxt not in visited:
                queue.append((nxt, depth + 1))

    # Map each subscriber_id to its reached depth (default 1 if not found)
    result: dict[str, int] = {}
    for sid in subscriber_ids:
        if sid in visited:
            # depth=0 means it is the producer itself; consumers start at depth>=1
            result[sid] = max(1, visited[sid])
        else:
            # Subscriber not found in graph — try partial label match
            matched_depth = None
            sid_stem = sid.lower().replace("-", "").replace("_", "")
            for nid, depth in visited.items():
                nid_stem = nid.lower().replace("-", "").replace("_", "")
                lbl_stem = nodes_by_id.get(nid, {}).get("label", "").lower().replace("-", "").replace("_", "")
                if sid_stem in nid_stem or sid_stem in lbl_stem:
                    matched_depth = max(1, depth)
                    break
            result[sid] = matched_depth if matched_depth is not None else 1

    return result


# ── Step 4: Build spec-compliant violation entry ───────────────────────────────

def build_violation_entry(
    violation_report: dict,
    subscriptions: list[dict],
    snapshot: dict,
    codebase_root: str | None = None,
) -> list[dict]:
    """
    Build violation log entries — one per failing check — in spec-compliant schema:
    {
      violation_id, check_id, contract_id, detected_at,
      blame_chain: [{rank, file_path, commit_hash, author, commit_timestamp,
                     commit_message, confidence_score}],
      blast_radius: {affected_nodes, affected_pipelines, estimated_records}
    }
    """
    contract_id = violation_report.get("contract_id", "")
    source_path = violation_report.get("data_path", violation_report.get("source_path", ""))
    node_labels = {
        n["node_id"]: n.get("label", n["node_id"])
        for n in snapshot.get("nodes", [])
    }

    all_results = violation_report.get("results", violation_report.get("issues", []))
    failing = [
        r for r in all_results
        if r.get("status") not in ("PASS", "pass")
    ]

    entries = []
    for issue in failing[:5]:  # cap at 5 issues per report
        field_name = issue.get("column_name", issue.get("field", ""))
        check_id = issue.get("check_id", f"{contract_id}.{field_name}.check")
        severity = issue.get("status", issue.get("severity", "UNKNOWN"))
        records_failing = issue.get("records_failing", 0)

        # ── Step 1: registry blast radius for this specific field
        affected_subs = get_blast_radius(contract_id, field_name, subscriptions)

        affected_nodes = []
        affected_pipelines = []
        for sub in affected_subs:
            sid = sub.get("subscriber_id", "")
            label = node_labels.get(sid, sid)
            tier = sub.get("tier", 1)
            bf_raw = sub.get("breaking_fields", [])
            bf_names = _normalize_breaking_fields(bf_raw)
            affected_nodes.append({
                "node_id": sid,
                "label": label,
                "tier": tier,
                "fields_consumed": sub.get("fields_consumed", []),
                "matched_breaking_fields": sub.get("_matched_breaking_fields", []),
                "contact": sub.get("contact", ""),
                "validation_mode": sub.get("validation_mode", "AUDIT"),
            })
            # Pipeline name = subscriber_id (it IS the downstream pipeline)
            affected_pipelines.append(f"{sid}-pipeline")

        # ── Step 2: BFS lineage enrichment for upstream producers
        producers = bfs_upstream_producers(source_path, snapshot, max_depth=4)

        # ── Step 3: git blame (Tier 1 only)
        # Use tier of first directly affected subscriber (conservative)
        tier = affected_subs[0].get("tier", 1) if affected_subs else 1
        chain = build_blame_chain(producers, field_name, codebase_root, tier)

        # ── Compute contamination_depth per affected node ──────────────────────
        # contamination_depth = min BFS hops from the violating producer to each
        # subscriber in the lineage graph.  depth=1 means directly connected;
        # depth=N means the corruption must traverse N inter-system hops before
        # reaching that subscriber.  depth=0 is reserved for the producer itself.
        contamination_map = _compute_contamination_depths(
            contract_id, [n["node_id"] for n in affected_nodes], snapshot
        )

        # Annotate each affected_node with its contamination_depth
        for node in affected_nodes:
            node["contamination_depth"] = contamination_map.get(node["node_id"], 1)

        # Summary metric: max contamination depth across all affected subscribers
        max_depth_observed = max(
            (n["contamination_depth"] for n in affected_nodes), default=0
        )

        entries.append({
            "violation_id": str(uuid.uuid4()),
            "check_id": check_id,
            "contract_id": contract_id,
            "source_path": source_path,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "failing_check": {
                "check_type": issue.get("check_type", ""),
                "actual_value": issue.get("actual_value", ""),
                "expected": issue.get("expected", ""),
                "message": issue.get("message", ""),
                "records_failing": records_failing,
            },
            "blame_chain": chain,
            "blast_radius": {
                "affected_nodes": affected_nodes,
                "affected_pipelines": list(set(affected_pipelines)),
                "estimated_records": records_failing,
                # contamination_depth quantifies propagation severity:
                # higher depth = corruption travels further before detection
                "max_contamination_depth": max_depth_observed,
            },
        })

    return entries


def write_violation_log(entries: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute violations to producers and compute blast radius."
    )
    parser.add_argument("--violation", required=True, help="Path to validation_reports/*.json")
    parser.add_argument("--registry", default="contract_registry/subscriptions.yaml")
    parser.add_argument("--lineage", required=True, help="Path to lineage_snapshots.jsonl")
    parser.add_argument("--output", default="violation_log/violations.jsonl")
    parser.add_argument(
        "--codebase-root",
        default=None,
        help="Git repo root for git blame (default: auto-detect from lineage snapshot)",
    )
    args = parser.parse_args()

    print(f"[attributor] Loading violation report: {args.violation}")
    with open(args.violation) as f:
        violation_report = json.load(f)

    all_results = violation_report.get("results", violation_report.get("issues", []))
    failing = [r for r in all_results if r.get("status") not in ("PASS", "pass")]
    if not failing:
        print("[attributor] No issues found in report — nothing to attribute.")
        return

    print(f"[attributor] {len(failing)} failing check(s) to attribute.")

    print(f"[attributor] Loading registry: {args.registry}")
    subscriptions = load_registry(args.registry)

    print(f"[attributor] Loading lineage: {args.lineage}")
    snapshot = load_lineage_snapshot(args.lineage)
    print(f"[attributor] Loaded {len(snapshot.get('nodes', []))} nodes, "
          f"{len(snapshot.get('edges', []))} edges.")

    # Auto-detect codebase root from snapshot if not provided
    codebase_root = args.codebase_root or snapshot.get("codebase_root")
    if codebase_root:
        print(f"[attributor] Codebase root: {codebase_root}")
    else:
        print("[attributor] WARNING: No codebase_root — git blame disabled.")

    entries = build_violation_entry(
        violation_report=violation_report,
        subscriptions=subscriptions,
        snapshot=snapshot,
        codebase_root=codebase_root,
    )

    write_violation_log(entries, args.output)

    print(f"\n[attributor] ── Attribution Report ───────────────────────")
    print(f"[attributor] Contract    : {violation_report.get('contract_id', '')}")
    print(f"[attributor] Violations  : {len(entries)}")
    for entry in entries:
        print(f"\n  ▸ check_id    : {entry['check_id']}")
        print(f"    severity    : {entry['severity']}")
        br = entry["blast_radius"]
        print(f"    blast radius: {len(br['affected_nodes'])} subscriber(s), "
              f"~{br['estimated_records']} records, "
              f"max_contamination_depth={br.get('max_contamination_depth', 'n/a')}")
        for node in br["affected_nodes"]:
            print(f"      → {node['node_id']} (tier={node['tier']}, "
                  f"mode={node['validation_mode']}, "
                  f"contamination_depth={node.get('contamination_depth', 'n/a')})")
        chain = entry["blame_chain"]
        print(f"    blame chain : {len(chain)} candidate(s)")
        for bc in chain[:3]:
            if bc.get("commit_hash"):
                print(f"      #{bc['rank']} [{bc['confidence_score']:.2f}] "
                      f"{bc['commit_hash'][:12]} — {bc['commit_message'][:60]}")
            elif bc.get("note"):
                print(f"      note: {bc['note']}")

    print(f"\n[attributor] Written to: {args.output}")


if __name__ == "__main__":
    main()
