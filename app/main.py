# -*- coding: utf-8 -*-
import os
import io
import json
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ========================
# Configuración de la App
# ========================
st.set_page_config(
    page_title="AuditiAHR - MVP Seguro",
    page_icon="🛡️",
    layout="wide"
)

ANON_API_URL = os.getenv("ANON_API_URL", "").strip()  # p.ej. https://auditiahr-func.azurewebsites.net/api/anon-upload
MAX_PREVIEW_ROWS = int(os.getenv("MAX_PREVIEW_ROWS", "30"))

# Estado
if "anon_df" not in st.session_state:
    st.session_state.anon_df = None
if "download_url" not in st.session_state:
    st.session_state.download_url = None
if "server_payload" not in st.session_state:
    st.session_state.server_payload = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None


# ================
# Utilidades HTTP
# ================
def post_to_azure_function(file_bytes: bytes, filename: str) -> dict:
    """
    Envía el archivo a la Azure Function que anonimiza y guarda en Blob Storage.
    Respuesta esperada (flexible):
      a) {"download_url": "<sas-url>", "preview": <json|csv-string|array-de-objetos>}
      b) {"blob_url" | "sas_url": "<url>"}
    """
    files = {"file": (filename, io.BytesIO(file_bytes))}
    data = {"return_preview": "true", "max_preview_rows": str(MAX_PREVIEW_ROWS)}
    resp = requests.post(ANON_API_URL, files=files, data=data, timeout=90)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def read_head_from_url(url: str, nrows: int = 30) -> Optional[pd.DataFrame]:
    try:
        lower = url.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(url, nrows=nrows)
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(url, nrows=nrows)
        # fallback csv
        return pd.read_csv(url, nrows=nrows)
    except Exception:
        return None


def payload_to_preview(payload: dict) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Intenta construir un DataFrame de vista previa anonimizado y recuperar un enlace de descarga.
    """
    download_url = payload.get("download_url") or payload.get("blob_url") or payload.get("sas_url")

    # Caso 1: preview ya viene en la respuesta
    if "preview" in payload and payload["preview"] is not None:
        pv = payload["preview"]
        try:
            if isinstance(pv, list):
                df = pd.DataFrame(pv)
            elif isinstance(pv, str):
                # probamos JSON y si no CSV
                try:
                    df = pd.read_json(io.StringIO(pv))
                except Exception:
                    df = pd.read_csv(io.StringIO(pv))
            else:
                df = pd.DataFrame(pv)
            return df, download_url
        except Exception:
            pass

    # Caso 2: leer primeras filas desde la URL SAS
    if download_url:
        df = read_head_from_url(download_url, nrows=MAX_PREVIEW_ROWS)
        return df, download_url

    return None, download_url


# =======================
# Validaciones de calidad
# =======================
def summarize_types(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = df.dtypes.astype(str).rename("dtype")
    nulls = df.isnull().sum().rename("nulls")
    null_pct = (df.isnull().mean() * 100).round(2).rename("null_pct")
    uniques = df.nunique(dropna=True).rename("unique_vals")
    out = pd.concat([dtypes, nulls, null_pct, uniques], axis=1)
    return out.reset_index(names=["column"])


def quick_outliers(df: pd.DataFrame, max_cols: int = 20) -> pd.DataFrame:
    """
    Marca outliers por IQR para columnas numéricas (solo conteo y % aprox).
    """
    nums = df.select_dtypes(include=[np.number]).copy()
    cols = nums.columns[:max_cols]
    rows = []
    for c in cols:
        s = nums[c].dropna()
        if s.empty:
            rows.append({"column": c, "outliers": 0, "out_pct": 0.0})
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        cnt = int(mask.sum())
        pct = round((cnt / len(s)) * 100, 2)
        rows.append({"column": c, "outliers": cnt, "out_pct": pct})
    return pd.DataFrame(rows)


# ==============
# Fairness básica
# ==============
def positive_rate(series: pd.Series, positive_values: List[Any]) -> float:
    if series.isna().all():
        return np.nan
    if len(positive_values) == 0:
        # Si no indicó valores positivos, asumimos 1/True/"yes"
        positive_values = [1, True, "1", "yes", "si", "sí", "apto", "positivo"]
    return round((series.astype(str).isin([str(v) for v in positive_values]).mean()) * 100, 2)


def disparate_impact(df: pd.DataFrame, sens_attr: str, outcome_col: str, positive_values: List[Any]) -> pd.DataFrame:
    """
    Calcula tasa positiva por grupo y DI = tasa_grupo / tasa_referencia.
    Referencia = grupo con mayor tamaño (o primero si igual).
    """
    tmp = df[[sens_attr, outcome_col]].dropna()
    counts = tmp[sens_attr].value_counts(dropna=False)
    ref_group = counts.index[0] if not counts.empty else None
    rows = []
    for g, gdf in tmp.groupby(sens_attr):
        pr = positive_rate(gdf[outcome_col], positive_values)
        rows.append({"group": g, "rows": len(gdf), "positive_rate_%": pr})
    out = pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)
    if ref_group is not None and not out.empty:
        ref_pr = float(out.loc[out["group"] == ref_group, "positive_rate_%"].values[0])
        out["DI_vs_ref"] = out["positive_rate_%"].apply(lambda x: round((x / ref_pr), 3) if ref_pr and not np.isnan(x) else np.nan)
        out.attrs["ref_group"] = ref_group
        out.attrs["ref_rate"] = ref_pr
    return out


# ==============================
# Generación de reporte (HTML)
# ==============================
def html_report(
    dataset_name: str,
    validation_summary: pd.DataFrame,
    outliers_summary: pd.DataFrame,
    fairness_table: Optional[pd.DataFrame],
    download_url: Optional[str]
) -> str:
    def df_to_html(df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "<em>Sin datos</em>"
        return df.to_html(index=False, escape=True)

    parts = [
        "<html><head><meta charset='utf-8'><title>AuditiAHR - Reporte</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:6px;font-size:13px} h2{margin-top:28px}</style>",
        "</head><body>",
        f"<h1>Reporte - {dataset_name or 'dataset'}</h1>",
        "<p>Este reporte se generó sobre el dataset <strong>anonimizado</strong> procesado en Azure.</p>",
        "<h2>Validación: Esquema / Nulos / Únicos</h2>",
        df_to_html(validation_summary),
        "<h2>Outliers (resumen IQR)</h2>",
        df_to_html(outliers_summary),
    ]
    if fairness_table is not None:
        ref_group = fairness_table.attrs.get("ref_group")
        ref_rate = fairness_table.attrs.get("ref_rate")
        hdr = "<h2>Fairness (tasa positiva por grupo & Disparate Impact)</h2>"
        sub = f"<p>Grupo de referencia: <strong>{ref_group}</strong> (tasa {ref_rate}%). Regla 80% ≈ DI ≥ 0.8.</p>" if ref_group is not None else ""
        parts += [hdr, sub, df_to_html(fairness_table)]

    if download_url:
        parts += [f"<h2>Descargas</h2><p>CSV anonimizado (Blob SAS): <a href='{download_url}' target='_blank'>{download_url}</a></p>"]

    parts += ["</body></html>"]
    return "\n".join(parts)


# =================
# UI - Navegación
# =================
st.sidebar.title("AuditiAHR")
section = st.sidebar.radio(
    "Secciones",
    ["1) Cargar & Anonimizar (Azure)", "2) Validación", "3) Fairness", "4) Descargas & Reporte"],
    index=0
)


# =========================================
# 1) Cargar & Anonimizar (siempre en Azure)
# =========================================
if section.startswith("1"):
    st.title("🛡️ Cargar & Anonimizar (Azure)")

    if not ANON_API_URL:
        st.error("Falta configurar **ANON_API_URL** en el entorno/secrets. Ej: https://<tu-func>.azurewebsites.net/api/anon-upload")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        file = st.file_uploader("Subí el archivo (.csv / .xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    with col2:
        st.info(
            "El archivo se envía directo a Azure Function → se anonimiza en el servidor → se guarda cifrado en Blob.\n\n"
            "Esta app **no** muestra nunca el archivo crudo.",
            icon="ℹ️"
        )

    if file is not None:
        # Leemos bytes (no mostramos preview cruda)
        raw_bytes = file.read()
        st.session_state.dataset_name = file.name

        with st.spinner("Procesando en Azure (anonimización y guardado en Blob)…"):
            try:
                payload = post_to_azure_function(raw_bytes, file.name)
            except requests.HTTPError as e:
                st.error(f"Error HTTP desde Azure Function: {e.response.status_code}\n\n{e.response.text}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"No se pudo contactar a la Azure Function: {e}")
                st.stop()

        # Construimos preview solo con datos anonimizados
        anon_preview, download_url = payload_to_preview(payload)

        # Guardamos estado
        st.session_state.server_payload = payload
        st.session_state.download_url = download_url

        if anon_preview is not None and len(anon_preview.columns) > 0:
            st.session_state.anon_df = anon_preview.copy()
            st.success("✅ Procesado y almacenado. Mostrando vista previa **anonimizada**.")
            st.dataframe(st.session_state.anon_df, use_container_width=True, hide_index=True)
        else:
            st.session_state.anon_df = None
            st.warning(
                "Se procesó en Azure, pero no se pudo generar una vista previa segura desde la respuesta. "
                "Si tu Function publica una URL SAS, la encontrarás en la sección **Descargas**."
            )

        with st.expander("Respuesta técnica de la Function (para auditoría)"):
            st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")

    else:
        st.info("Subí un archivo para comenzar.")


# ==================
# 2) Validación base
# ==================
elif section.startswith("2"):
    st.title("🧪 Validación (dataset anonimizado)")
    if st.session_state.anon_df is None:
        st.warning("Primero cargá y procesá un archivo en la sección 1).")
        st.stop()

    df = st.session_state.anon_df

    st.subheader("Resumen de columnas")
    vdf = summarize_types(df)
    st.dataframe(vdf, use_container_width=True, hide_index=True)

    st.subheader("Duplicados")
    dup_count = int(df.duplicated().sum())
    st.write(f"Filas duplicadas: **{dup_count}** ({round(100*dup_count/len(df),2) if len(df)>0 else 0}%)")

    st.subheader("Outliers (IQR, numéricos)")
    outdf = quick_outliers(df)
    st.dataframe(outdf, use_container_width=True, hide_index=True)


# ===============
# 3) Fairness MVP
# ===============
elif section.startswith("3"):
    st.title("⚖️ Fairness (sobre dataset anonimizado)")
    if st.session_state.anon_df is None:
        st.warning("Primero cargá y procesá un archivo en la sección 1).")
        st.stop()

    df = st.session_state.anon_df
    cols = list(df.columns)

    with st.form("fairness"):
        c1, c2 = st.columns(2)
        with c1:
            sens_attr = st.selectbox("Atributo sensible (ej. genero, edad_band, provincia)", options=["(ninguno)"] + cols)
        with c2:
            outcome_col = st.selectbox("Variable de outcome (target/pred)", options=["(ninguno)"] + cols)

        positive_hint = st.text_input(
            "Valores considerados 'positivos' (separados por coma) — ej: 1,si,apto,true",
            value="1,true,si,sí,apto"
        )
        submitted = st.form_submit_button("Calcular fairness")

    if submitted:
        if sens_attr == "(ninguno)" or outcome_col == "(ninguno)":
            st.warning("Seleccioná un atributo sensible y una variable de outcome.")
        else:
            pos_vals = [x.strip() for x in positive_hint.split(",") if x.strip()]
            table = disparate_impact(df, sens_attr, outcome_col, pos_vals)
            if table.empty:
                st.info("No hay datos suficientes para calcular fairness.")
            else:
                ref_group = table.attrs.get("ref_group")
                ref_rate = table.attrs.get("ref_rate")
                if ref_group is not None:
                    st.caption(f"Grupo de referencia: **{ref_group}** (tasa positiva {ref_rate}%). Regla 80% ≈ DI ≥ 0.8.")
                st.dataframe(table, use_container_width=True, hide_index=True)


# ==========================
# 4) Descargas & Reporte
# ==========================
elif section.startswith("4"):
    st.title("📦 Descargas & Reporte")
    df = st.session_state.anon_df
    download_url = st.session_state.download_url

    colA, colB = st.columns([2, 1])
    with colA:
        if df is not None and len(df.columns) > 0:
            st.subheader("CSV anonimizado (primeras filas)")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No hay vista previa disponible. Generala cargando un archivo en la sección 1).")

    with colB:
        if download_url:
            st.link_button("⬇️ Descargar desde Blob (SAS)", url=download_url)
        else:
            st.write("No se recibió `download_url` desde la Function.")

    st.divider()
    st.subheader("Generar reporte HTML")
    # Armamos insumos
    vdf = summarize_types(df) if df is not None else pd.DataFrame()
    outdf = quick_outliers(df) if df is not None else pd.DataFrame()

    # Si el usuario ya corrió fairness, no lo retuvimos; permitimos re-calcular rápido opcional
    fairness_html_note = st.caption("Opcional: volvés a calcular fairness acá para incluirlo en el reporte.")
    cols = list(df.columns) if df is not None else []
    c1, c2 = st.columns(2)
    sens_attr = c1.selectbox("Atributo sensible (opcional)", options=["(ninguno)"] + cols, index=0)
    outcome_col = c2.selectbox("Outcome (opcional)", options=["(ninguno)"] + cols, index=0)
    pos_vals = [x.strip() for x in st.text_input("Valores 'positivos' (coma)", "1,true,si,sí,apto").split(",") if x.strip()]

    fair_table = None
    if df is not None and sens_attr != "(ninguno)" and outcome_col != "(ninguno)":
        fair_table = disparate_impact(df, sens_attr, outcome_col, pos_vals)

    dataset_name = st.session_state.dataset_name or "dataset"
    html = html_report(dataset_name, vdf, outdf, fair_table, download_url)
    st.download_button(
        "📝 Descargar reporte HTML",
        data=html.encode("utf-8"),
        file_name=f"{dataset_name}-reporte.html",
        mime="text/html"
    )

# =======================
# Footer / pequeñas notas
# =======================
st.sidebar.markdown("---")
st.sidebar.caption(
    "Esta app **siempre** procesa los archivos en Azure Function (servidor). "
    "La vista previa que ves arriba proviene de datos ya **anonimizados**."
)
