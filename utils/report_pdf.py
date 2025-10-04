
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime

# ========= Helper styles =========
def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#111827"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#374151"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#111827"),
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontSize=9.2,
            leading=12,
            textColor=colors.HexColor("#4B5563"),
        ),
        "chip_ok": ParagraphStyle("chip_ok", parent=base["BodyText"], textColor=colors.HexColor("#166534"), fontSize=10.5, leading=14),
        "chip_warn": ParagraphStyle("chip_warn", parent=base["BodyText"], textColor=colors.HexColor("#92400E"), fontSize=10.5, leading=14),
        "chip_bad": ParagraphStyle("chip_bad", parent=base["BodyText"], textColor=colors.HexColor("#991B1B"), fontSize=10.5, leading=14),
    }
    return styles

def _badge(text, kind="ok"):
    s = _styles()
    if kind == "ok":
        return Paragraph(f"✅ {text}", s["chip_ok"])
    if kind == "warn":
        return Paragraph(f"🟡 {text}", s["chip_warn"])
    return Paragraph(f"🔴 {text}", s["chip_bad"])

def _dp_color(dp):
    if dp is None:
        return None
    dp = abs(dp)
    if dp <= 0.05:
        return colors.HexColor("#E7F6EC")  # green-50
    if dp <= 0.10:
        return colors.HexColor("#FEF3C7")  # amber-100
    return colors.HexColor("#FEE2E2")      # red-100

def _eo_color(eo):
    if eo is None:
        return None
    eo = abs(eo)
    if eo <= 0.05:
        return colors.HexColor("#E7F6EC")
    if eo <= 0.10:
        return colors.HexColor("#FEF3C7")
    return colors.HexColor("#FEE2E2")

def _di_badge(di):
    if di is None:
        return _badge("DI —", "warn")
    return _badge(f"DI {di:.2f} " + ("(cumple)" if di >= 0.80 else "(no cumple)"),
                  "ok" if di >= 0.80 else "bad")

def _fmt(v, pct=False):
    if v is None:
        return "—"
    try:
        if pct:
            return f"{v:.0%}"
        return f"{v:.2f}"
    except Exception:
        return str(v)

# ========= Main builder =========
def build_fairness_pdf(
    out_path,
    meta,
    metrics_by_var,
    metrics_by_stage,
    global_warnings,
    top_factors=None,
    chart_paths=None
):
    '''
    Pretty HR-friendly PDF.
    - meta: dict con {dataset_name, rows, cols, sensitive_cols, thresholds...}
    - metrics_by_var: lista de dicts por variable: {col, di, dp_gap, eo_gap, groups:{g:{n,rate}}, warn}
    - metrics_by_stage: lista por etapa (opcional)
    - global_warnings: lista de strings
    - top_factors: lista de bullets (Top 3 explicaciones RR.HH.)
    - chart_paths: rutas a imágenes (opcional)
    '''
    styles = _styles()
    top_factors = top_factors or []
    chart_paths = chart_paths or []

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.6*cm,
        rightMargin=1.6*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm
    )
    story = []

    # ===== Cover =====
    title = Paragraph("📊 FairAudit — Informe de Equidad", styles["title"])
    sub = Paragraph(
        f"Dataset: <b>{meta.get('dataset_name','—')}</b> &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"Filas: <b>{meta.get('rows','—')}</b> &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"Columnas: <b>{meta.get('cols','—')}</b>",
        styles["subtitle"]
    )
    datep = Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles["small"])
    story += [title, sub, datep, Spacer(1, 0.6*cm)]
    story += [Paragraph("Este informe resume brevemente los indicadores de equidad y las principales señales para RR.HH. en lenguaje claro.", styles["body"])]
    story += [Spacer(1, 0.6*cm)]

    # ===== Section 1: Resumen por variable =====
    story += [Paragraph("1) Resumen de métricas por variable", styles["h2"])]
    header = ["Variable", "DI (80%)", "DP (gap)", "EO (gap)", "Grupos (n y tasa)"]
    rows = [header]

    for item in metrics_by_var:
        col = item.get("col","—")
        di = item.get("di")
        dp = item.get("dp_gap")
        eo = item.get("eo_gap")
        groups = item.get("groups", {})

        # groups text
        parts = []
        for g, v in groups.items():
            n = v.get("n")
            r = v.get("rate")
            parts.append(f"{g}: n={n}, {_fmt(r, pct=True)}")
        gtext = Paragraph(", ".join(parts) if parts else "—", styles["small"])

        rows.append([
            Paragraph(f"<b>{col}</b>", styles["body"]),
            _di_badge(di),
            Paragraph(_fmt(dp), styles["body"]),
            Paragraph(_fmt(eo), styles["body"]),
            gtext
        ])

    tbl = Table(rows, colWidths=[4.0*cm, 3.0*cm, 3.0*cm, 3.0*cm, 6.0*cm])
    # color DP/EO cells by value
    style_cmds = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2D6AE3")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D1D5DB")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]

    # dynamic backgrounds for DP/EO columns
    for ridx, item in enumerate(metrics_by_var, start=1):
        dp = item.get("dp_gap")
        eo = item.get("eo_gap")
        c_dp = _dp_color(dp)
        c_eo = _eo_color(eo)
        if c_dp:
            style_cmds.append(("BACKGROUND", (2, ridx), (2, ridx), c_dp))
        if c_eo:
            style_cmds.append(("BACKGROUND", (3, ridx), (3, ridx), c_eo))

    tbl.setStyle(TableStyle(style_cmds))
    story += [tbl, Spacer(1, 0.5*cm)]

    # ===== Section 2: Top 3 factores =====
    story += [Paragraph("2) Factores que más influyen (Top 3)", styles["h2"])]
    if top_factors:
        for b in top_factors:
            story += [Paragraph(b, styles["body"])]
    else:
        story += [Paragraph("No se identificaron factores principales con tamaño de muestra suficiente.", styles["small"])]
    story += [Spacer(1, 0.4*cm)]

    # ===== Section 3: Advertencias =====
    story += [Paragraph("3) Advertencias / Calidad de datos", styles["h2"])]
    if global_warnings:
        for w in global_warnings:
            story += [Paragraph("⚠️ " + w, styles["small"])]
    else:
        story += [Paragraph("No se detectaron advertencias relevantes.", styles["small"])]
    story += [Spacer(1, 0.4*cm)]

    # ===== (Optional) charts =====
    if chart_paths:
        story += [Paragraph("Anexos visuales", styles["h2"])]
        # Images could be added here if provided
        # from reportlab.platypus import Image
        # for p in chart_paths:
        #     story += [Image(p, width=16*cm, height=9*cm), Spacer(1, 0.2*cm)]

    # ===== Section 4: Detalle por etapa (si existe) =====
    if metrics_by_stage:
        story += [PageBreak(), Paragraph("4) Detalle por etapa", styles["h2"])]
        header2 = ["Etapa", "Variable", "DI (80%)", "Notas"]
        rows2 = [header2]
        for it in metrics_by_stage:
            sname = it.get("stage","—")
            col = it.get("col","—")
            di = it.get("di")
            warn = it.get("warn")
            note = warn or "—"
            rows2.append([Paragraph(sname, styles["body"]), Paragraph(col, styles["body"]), _di_badge(di), Paragraph(note, styles["small"])])

        tbl2 = Table(rows2, colWidths=[4*cm, 6*cm, 3.2*cm, 6.3*cm])
        tbl2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2D6AE3")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D1D5DB")),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story += [tbl2]

    # Footer note
    story += [Spacer(1, 0.6*cm), Paragraph("Generado con FairAudit · Este informe es orientativo y depende de la calidad de los datos.", styles["small"])]

    doc.build(story)
    return out_path
