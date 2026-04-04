"""
PDFReportGenerator — generates enforcer_report/report_{date}.pdf
from enforcer_report/report_data.json.

Spec requirement:
  enforcer_report/report_{date}.pdf  — auto-generated, readable by
  someone who has never heard of a data contract.

Sections (matching the spec exactly):
  1. Data Health Score + one-sentence narrative
  2. Violations this week — count by severity + plain-language top 3
  3. Schema changes detected — compatibility verdict per contract
  4. AI system risk assessment — drift, prompt schema, output violation rate
  5. Recommended actions — top 3 prioritised

CLI:
    python src/contracts/pdf_report.py \
        --input enforcer_report/report_data.json \
        --output enforcer_report/
"""

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1a2e")
C_ACCENT  = colors.HexColor("#0f3460")
C_BLUE    = colors.HexColor("#16213e")
C_GREEN   = colors.HexColor("#1a7a4a")
C_YELLOW  = colors.HexColor("#b58900")
C_RED     = colors.HexColor("#c0392b")
C_ORANGE  = colors.HexColor("#e07b00")
C_LIGHT   = colors.HexColor("#f8f9fa")
C_BORDER  = colors.HexColor("#dee2e6")
C_TEXT    = colors.HexColor("#212529")
C_MUTED   = colors.HexColor("#6c757d")

W, H = A4


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=C_DARK,
            spaceAfter=4,
            leading=26,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_MUTED,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=C_ACCENT,
            spaceBefore=14,
            spaceAfter=4,
            leading=16,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_TEXT,
            leading=13,
            spaceAfter=4,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_TEXT,
            leading=13,
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=8,
            textColor=C_MUTED,
            leading=11,
        ),
        "mono": ParagraphStyle(
            "mono",
            fontName="Courier",
            fontSize=8,
            textColor=C_TEXT,
            leading=11,
            spaceAfter=2,
        ),
        "center": ParagraphStyle(
            "center",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_TEXT,
            alignment=TA_CENTER,
            leading=12,
        ),
    }


def _hr(elements, color=C_BORDER, thickness=0.5):
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=thickness, color=color))
    elements.append(Spacer(1, 6))


def _section(elements, title, s):
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(title.upper(), s["section"]))
    _hr(elements, C_ACCENT, 1)


def _grade_color(grade: str) -> colors.Color:
    return {
        "A": C_GREEN, "B": colors.HexColor("#2d7d2d"),
        "C": C_YELLOW, "D": C_ORANGE, "F": C_RED,
    }.get(grade, C_MUTED)


def _severity_color(sev: str) -> colors.Color:
    return {
        "CRITICAL": C_RED, "HIGH": C_ORANGE,
        "MEDIUM": C_YELLOW, "LOW": C_MUTED,
        "PASS": C_GREEN, "WARN": C_ORANGE, "FAIL": C_RED,
    }.get(sev.upper(), C_MUTED)


def _status_badge(text: str) -> str:
    """Return coloured inline text for a status word."""
    c = _severity_color(text)
    hex_color = c.hexval() if hasattr(c, 'hexval') else "#666666"
    return f'<font color="{hex_color}"><b>{text}</b></font>'


# ── Section builders ───────────────────────────────────────────────────────────

def build_cover(elements, data: dict, s: dict):
    generated_at = data.get("generated_at", "")
    try:
        dt = datetime.fromisoformat(generated_at).strftime("%d %B %Y, %H:%M UTC")
    except Exception:
        dt = generated_at

    score  = data.get("data_health_score", 0)
    grade  = data.get("health_grade", "?")
    gc     = _grade_color(grade)
    gc_hex = gc.hexval() if hasattr(gc, 'hexval') else "#333333"

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("DATA CONTRACT ENFORCER", s["title"]))
    elements.append(Paragraph("Enforcer Report — Weekly Data Quality Summary", s["subtitle"]))
    elements.append(Spacer(1, 0.3 * cm))

    # Score banner table
    score_table = Table(
        [[
            Paragraph(f'<font size="36" color="{gc_hex}"><b>{score}</b></font><font size="14" color="#6c757d">/100</font>', s["center"]),
            Paragraph(f'<font size="48" color="{gc_hex}"><b>{grade}</b></font>', s["center"]),
            Paragraph(
                f'<b>Generated</b><br/>{dt}<br/><br/>'
                f'<b>LLM Backend</b><br/>{data.get("llm_backend", "heuristic").upper()}',
                s["center"]
            ),
        ]],
        colWidths=[5 * cm, 3 * cm, 8 * cm],
        rowHeights=[2.5 * cm],
    )
    score_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C_LIGHT),
        ("BOX",         (0, 0), (-1, -1), 1, C_BORDER),
        ("INNERGRID",   (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 0.4 * cm))

    # Narrative
    narrative = data.get("health_narrative", "")
    elements.append(Paragraph(narrative, s["body"]))

    # Totals strip
    totals = data.get("totals", {})
    strip_data = [[
        Paragraph(f'<b>{totals.get("contracts_covered","—")}</b><br/><font size="7">Contracts</font>', s["center"]),
        Paragraph(f'<b>{totals.get("total_checks_baseline","—")}</b><br/><font size="7">Checks Run</font>', s["center"]),
        Paragraph(f'<b>{totals.get("total_violations_logged","—")}</b><br/><font size="7">Violations Logged</font>', s["center"]),
        Paragraph(f'<b>{totals.get("total_subscriptions","—")}</b><br/><font size="7">Subscriptions</font>', s["center"]),
        Paragraph(f'<b>{totals.get("breaking_schema_changes","—")}</b><br/><font size="7">Breaking Changes</font>', s["center"]),
    ]]
    strip = Table(strip_data, colWidths=[3.2 * cm] * 5, rowHeights=[1.2 * cm])
    strip.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C_ACCENT),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.white),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(strip)


def build_violations(elements, data: dict, s: dict):
    _section(elements, "1. Violations This Week", s)

    v = data.get("violations_this_week", {})
    crit  = v.get("critical_count", 0)
    high  = v.get("high_count", 0)
    total = v.get("total_logged", 0)

    # Severity counts
    sev_data = [
        [
            Paragraph(_status_badge("CRITICAL"), s["center"]),
            Paragraph(_status_badge("HIGH"), s["center"]),
            Paragraph(f'<b>{total}</b>', s["center"]),
        ],
        [
            Paragraph(str(crit), s["center"]),
            Paragraph(str(high), s["center"]),
            Paragraph("Total logged", s["center"]),
        ],
    ]
    sev_table = Table(sev_data, colWidths=[5.3 * cm, 5.3 * cm, 5.3 * cm], rowHeights=[0.7 * cm, 0.6 * cm])
    sev_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(sev_table)
    elements.append(Spacer(1, 8))

    # Plain-English descriptions
    elements.append(Paragraph("Top violations — plain-English impact summary:", s["body_bold"]))
    for i, desc in enumerate(v.get("plain_english", []), 1):
        # Wrap long text
        wrapped = textwrap.fill(desc, 110)
        elements.append(Paragraph(f"{i}.&nbsp;&nbsp;{wrapped}", s["body"]))


def build_schema_changes(elements, data: dict, s: dict):
    _section(elements, "2. Schema Changes Detected", s)

    sc = data.get("schema_changes_detected", {})
    n_diffs    = sc.get("total_diffs_run", 0)
    n_breaking = sc.get("breaking_changes", 0)

    elements.append(Paragraph(
        f"Schema evolution analysis ran across <b>{n_diffs}</b> contract(s). "
        f"Breaking changes detected: <b>{n_breaking}</b>.",
        s["body"]
    ))
    elements.append(Spacer(1, 4))

    summaries = sc.get("summaries", [])
    if not summaries:
        elements.append(Paragraph("No schema diff data available.", s["small"]))
        return

    header = [
        Paragraph("<b>Contract</b>", s["body_bold"]),
        Paragraph("<b>Verdict</b>", s["body_bold"]),
        Paragraph("<b>Breaking</b>", s["body_bold"]),
        Paragraph("<b>Compatible</b>", s["body_bold"]),
        Paragraph("<b>Action</b>", s["body_bold"]),
    ]
    rows = [header]
    for diff in summaries:
        verdict = diff.get("verdict", "UNKNOWN")
        v_color = C_GREEN if verdict == "COMPATIBLE" else C_RED
        v_hex   = v_color.hexval() if hasattr(v_color, 'hexval') else "#333"
        rows.append([
            Paragraph(diff.get("contract_id", ""), s["mono"]),
            Paragraph(f'<font color="{v_hex}"><b>{verdict}</b></font>', s["body"]),
            Paragraph(str(diff.get("breaking", 0)), s["center"]),
            Paragraph(str(diff.get("compatible", 0)), s["center"]),
            Paragraph(diff.get("action_required", ""), s["small"]),
        ])

    col_widths = [5.5 * cm, 2.5 * cm, 1.8 * cm, 2 * cm, 4 * cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)


def build_ai_risk(elements, data: dict, s: dict):
    _section(elements, "3. AI System Risk Assessment", s)

    ai = data.get("ai_system_risk_assessment", {})
    if not ai:
        elements.append(Paragraph("No AI extension data available.", s["small"]))
        return

    label_map = {
        "embedding_drift":         "Embedding Drift Detection",
        "prompt_schema_validation":"Prompt Input Schema Validation",
        "llm_output_violation_rate":"LLM Output Schema Violation Rate",
    }

    rows = []
    for key, label in label_map.items():
        entry = ai.get(key, {})
        status  = entry.get("status", "UNKNOWN")
        message = entry.get("message", "No data.")
        sc = _severity_color(status)
        sc_hex = sc.hexval() if hasattr(sc, 'hexval') else "#333"
        rows.append([
            Paragraph(f"<b>{label}</b>", s["body_bold"]),
            Paragraph(f'<font color="{sc_hex}"><b>{status}</b></font>', s["body"]),
            Paragraph(message, s["small"]),
        ])

    t = Table(rows, colWidths=[4.5 * cm, 1.8 * cm, 9.5 * cm])
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT, colors.white]),
    ]))
    elements.append(t)


def build_recommendations(elements, data: dict, s: dict):
    _section(elements, "4. Recommended Actions", s)

    recs = data.get("top_recommendations", [])
    if not recs:
        elements.append(Paragraph("No recommendations available.", s["small"]))
        return

    elements.append(Paragraph(
        "Ordered by risk reduction value. Address CRITICAL violations before HIGH.",
        s["small"]
    ))
    elements.append(Spacer(1, 6))

    priority_colors = [C_RED, C_ORANGE, C_YELLOW]
    priority_labels = ["P1 — URGENT", "P2 — HIGH", "P3 — MEDIUM"]

    for i, rec in enumerate(recs[:3]):
        pc     = priority_colors[i] if i < len(priority_colors) else C_MUTED
        pc_hex = pc.hexval() if hasattr(pc, 'hexval') else "#666"
        label  = priority_labels[i] if i < len(priority_labels) else f"P{i+1}"
        row = [[
            Paragraph(f'<font color="{pc_hex}"><b>{label}</b></font>', s["body_bold"]),
            Paragraph(rec, s["body"]),
        ]]
        t = Table(row, colWidths=[3 * cm, 12.8 * cm])
        t.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND",    (0, 0), (0, -1), C_LIGHT),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 4))


def build_contract_table(elements, data: dict, s: dict):
    _section(elements, "5. Contract Summary", s)

    rows_data = data.get("contract_summary_table", [])
    if not rows_data:
        elements.append(Paragraph("No contract data.", s["small"]))
        return

    header = [
        Paragraph("<b>Contract</b>", s["body_bold"]),
        Paragraph("<b>Checks</b>", s["body_bold"]),
        Paragraph("<b>Passed</b>", s["body_bold"]),
        Paragraph("<b>Critical</b>", s["body_bold"]),
        Paragraph("<b>High</b>", s["body_bold"]),
        Paragraph("<b>Subs</b>", s["body_bold"]),
        Paragraph("<b>Enforce</b>", s["body_bold"]),
        Paragraph("<b>Mode</b>", s["body_bold"]),
    ]
    rows = [header]
    for r in rows_data:
        crit_val = r.get("critical", 0)
        high_val = r.get("high", "—")
        crit_str = f'<font color="#c0392b"><b>{crit_val}</b></font>' if crit_val and crit_val != "—" and int(str(crit_val)) > 0 else str(crit_val)
        rows.append([
            Paragraph(r.get("contract_id", ""), s["mono"]),
            Paragraph(str(r.get("total_checks", "—")), s["center"]),
            Paragraph(str(r.get("passed", "—")), s["center"]),
            Paragraph(crit_str, s["center"]),
            Paragraph(str(high_val), s["center"]),
            Paragraph(str(r.get("subscriber_count", "—")), s["center"]),
            Paragraph(str(r.get("enforce_mode_subs", "—")), s["center"]),
            Paragraph(str(r.get("enforcement_mode", "AUDIT")), s["center"]),
        ])

    col_widths = [5.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm, 1.2 * cm, 1.5 * cm, 2 * cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_LIGHT]),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)


def build_footer_note(elements, data: dict, s: dict):
    elements.append(Spacer(1, 0.8 * cm))
    _hr(elements)
    deductions = data.get("score_deductions", [])
    if deductions:
        elements.append(Paragraph("<b>Score Deductions:</b>", s["body_bold"]))
        for d in deductions:
            elements.append(Paragraph(f"• {d}", s["small"]))
        elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Generated by Data Contract Enforcer — Week 7 TRP Challenge. "
        "Enforcement is always at the consumer boundary. "
        "Contracts in ENFORCE mode will block the pipeline on any CRITICAL or HIGH violation.",
        s["small"]
    ))


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_pdf(input_path: Path, output_dir: Path) -> Path:
    with open(input_path) as f:
        data = json.load(f)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_path = output_dir / f"report_{date_str}.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Data Contract Enforcer Report",
        author="Data Contract Enforcer",
        subject="Weekly Data Quality Summary",
    )

    s = _styles()
    elements = []

    build_cover(elements, data, s)
    _hr(elements, C_ACCENT, 1.5)

    build_violations(elements, data, s)
    build_schema_changes(elements, data, s)
    build_ai_risk(elements, data, s)
    build_recommendations(elements, data, s)
    build_contract_table(elements, data, s)
    build_footer_note(elements, data, s)

    doc.build(elements)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate PDF enforcer report from report_data.json.")
    parser.add_argument("--input",  default="enforcer_report/report_data.json")
    parser.add_argument("--output", default="enforcer_report/")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)

    if not input_path.exists():
        print(f"[pdf_report] ERROR: {input_path} not found. Run report_generator.py first.")
        raise SystemExit(1)

    print(f"[pdf_report] Reading {input_path} ...")
    output_path = generate_pdf(input_path, output_dir)
    print(f"[pdf_report] Written: {output_path}")


if __name__ == "__main__":
    main()
