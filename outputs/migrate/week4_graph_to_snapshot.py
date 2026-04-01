"""
Migration: src/outputs/week4/lineage_graph.jsonl  (actually a single JSON object)
       --> outputs/week4/lineage_snapshots.jsonl   (one snapshot per line)

DEVIATIONS FIXED:
  1. File format: single JSON object -> .jsonl with one snapshot per line
  2. snapshot_id: missing -> uuid4 (generated once, stable per migration run)
  3. git_commit: missing -> read from Week 4 repo's git log (or fallback to 40-char zeros)
  4. captured_at: missing -> file mtime (fallback: now)
  5. nodes[].node_id: was 'id' field -> renamed to 'node_id'
  6. nodes[].type: was domain-specific ('transformation', 'dataset', '__unresolved__')
                   -> mapped to canonical enum FILE|TABLE|SERVICE|MODEL|PIPELINE|EXTERNAL
  7. edges[].relationship: was field named 'type' -> renamed to 'relationship'
  8. edges[].confidence: missing -> set to 1.0 (all edges are asserted, not inferred)
  9. edges[].source/target: already present, kept as-is
 10. codebase_root: missing -> derived from common path prefix of node source_file values
"""

import json
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

SRC = Path(__file__).parent.parent.parent / "src" / "outputs" / "week4" / "lineage_graph.json"
DST = Path(__file__).parent.parent / "week4" / "lineage_snapshots.jsonl"

WEEK4_REPO = Path("/Users/melkam/Documents/10 Academy/Week4/ol-data-platform")

# Node type mapping: source types -> canonical enum
# FILE|TABLE|SERVICE|MODEL|PIPELINE|EXTERNAL
NODE_TYPE_MAP = {
    "transformation": "MODEL",
    "dataset": "TABLE",
    "table": "TABLE",
    "view": "TABLE",
    "model": "MODEL",
    "file": "FILE",
    "service": "SERVICE",
    "pipeline": "PIPELINE",
    "external": "EXTERNAL",
    "__unresolved__": "EXTERNAL",
}

# Edge relationship mapping: source types -> canonical enum
# IMPORTS|CALLS|READS|WRITES|PRODUCES|CONSUMES
EDGE_REL_MAP = {
    "CONSUMES": "CONSUMES",
    "PRODUCES": "PRODUCES",
    "IMPORTS": "IMPORTS",
    "CALLS": "CALLS",
    "READS": "READS",
    "WRITES": "WRITES",
}


def get_git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        commit = result.stdout.strip()
        if len(commit) == 40:
            return commit
    except Exception:
        pass
    return "0" * 40


def get_file_mtime(path: Path) -> str:
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def infer_codebase_root(nodes: list) -> str:
    """Find the common directory prefix of all source_file paths."""
    paths = [
        n.get("source_file") or n.get("evidence", {}).get("path", "")
        for n in nodes
        if n.get("source_file") or n.get("evidence", {}).get("path")
    ]
    if not paths:
        return str(WEEK4_REPO)
    # Find longest common prefix
    parts_list = [Path(p).parts for p in paths if p]
    if not parts_list:
        return str(WEEK4_REPO)
    common = list(parts_list[0])
    for parts in parts_list[1:]:
        new_common = []
        for a, b in zip(common, parts):
            if a == b:
                new_common.append(a)
            else:
                break
        common = new_common
    return str(Path(*common)) if common else str(WEEK4_REPO)


def map_node_type(raw_type: str) -> str:
    """Map raw node type string to canonical enum value."""
    # Extract the prefix before ':' (e.g. 'transformation:/path' -> 'transformation')
    prefix = raw_type.split(":")[0].lower() if raw_type else "external"
    return NODE_TYPE_MAP.get(prefix, "EXTERNAL")


def canonical_node(raw_node: dict) -> dict:
    raw_type = raw_node.get("type", "external")
    node_id = raw_node.get("id", str(uuid.uuid4()))
    label = (
        raw_node.get("label")
        or Path(raw_node.get("source_file", node_id)).name
        or node_id
    )
    return {
        "node_id": node_id,
        "type": map_node_type(raw_type),
        "label": label,
        "metadata": {
            "source_file": raw_node.get("source_file") or raw_node.get("evidence", {}).get("path"),
            "line_range": raw_node.get("line_range"),
            "transformation_type": raw_node.get("transformation_type"),
            "original_type": raw_type,
        },
    }


def canonical_edge(raw_edge: dict) -> dict:
    raw_rel = raw_edge.get("type", "PRODUCES")
    relationship = EDGE_REL_MAP.get(raw_rel.upper(), "PRODUCES")
    return {
        "source": raw_edge["source"],
        "target": raw_edge["target"],
        "relationship": relationship,
        "confidence": 1.0,
        "_migration_meta": {
            "original_type": raw_rel,
        },
    }


def migrate(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    # lineage_graph.jsonl is actually a single JSON object despite the extension
    with src.open() as f:
        raw = json.load(f)

    raw_nodes = raw.get("nodes", [])
    raw_edges = raw.get("edges", [])

    git_commit = get_git_commit(WEEK4_REPO)
    captured_at = get_file_mtime(src)
    codebase_root = infer_codebase_root(raw_nodes)

    nodes = [canonical_node(n) for n in raw_nodes]
    edges = [canonical_edge(e) for e in raw_edges]

    # Report node type distribution
    type_dist = Counter(n["type"] for n in nodes)
    print(f"Node type distribution: {dict(type_dist)}")

    snapshot = {
        "snapshot_id": str(uuid.uuid4()),
        "codebase_root": codebase_root,
        "git_commit": git_commit,
        "nodes": nodes,
        "edges": edges,
        "captured_at": captured_at,
        "_migration_meta": {
            "source_schema_version": raw.get("schema_version"),
            "source_graph_type": raw.get("graph_type"),
            "migrated_from": str(src),
            "migration_script": __file__,
            "deviations_fixed": [
                "snapshot_id added (uuid4)",
                "git_commit added from Week 4 repo HEAD",
                "captured_at added from file mtime",
                "nodes[].node_id renamed from nodes[].id",
                "nodes[].type mapped to canonical enum (FILE|TABLE|SERVICE|MODEL|PIPELINE|EXTERNAL)",
                "edges[].relationship renamed from edges[].type",
                "edges[].confidence added (1.0 — all edges are asserted)",
                "codebase_root derived from common source_file prefix",
                "file format converted from single JSON to JSONL (one snapshot per line)",
            ],
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f_out:
        f_out.write(json.dumps(snapshot) + "\n")

    print(f"Migrated 1 snapshot: {src} -> {dst}")
    print(f"  nodes: {len(nodes)}, edges: {len(edges)}, git_commit: {git_commit[:8]}...")


if __name__ == "__main__":
    migrate(SRC, DST)
    # Smoke checks
    with DST.open() as f:
        snap = json.loads(f.readline())
    assert len(snap["snapshot_id"]) == 36, "snapshot_id not UUID"
    assert len(snap["git_commit"]) == 40, "git_commit not 40 chars"
    assert isinstance(snap["nodes"], list) and snap["nodes"], "nodes empty"
    assert isinstance(snap["edges"], list) and snap["edges"], "edges empty"
    assert "relationship" in snap["edges"][0], "edges[].relationship missing"
    assert "confidence" in snap["edges"][0], "edges[].confidence missing"
    assert snap["edges"][0]["relationship"] in {"IMPORTS", "CALLS", "READS", "WRITES", "PRODUCES", "CONSUMES"}
    print("Smoke checks passed.")
