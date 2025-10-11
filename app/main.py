import sys, os, io, base64, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st, pandas as pd, numpy as np
import plotly.express as px
import requests

# ==== Config y módulos de tu proyecto ====
from core import io, fairness, validate, explain, heuristics, nlg
from core.config import load_config
from utils.utils_column_roles import suggest_roles, small_sample_warning
from utils.report_pdf import build_fairness_pdf

# --------------------------------------------------------------------------------------
#  Helpers: llamar a Azure Function SIEMPRE y obtener un DataFrame ANONIMIZADO
# --------------------------------------------------------------------------------------
def _read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    # lee csv robusto en mem (utf-8 con BOM ok)
    bio = io.BytesIO(csv_bytes) if hasattr(io, "BytesIO") else None
    if bio is None:
        import io as _pyio
        bio = _pyio.BytesIO(csv_bytes)
    return pd.read_csv(bio)

def _df_from_function_response(resp: requests.Response) -> pd.DataFrame:
    ctype = resp.headers.get("Content-Type", "")
    # CSV directo
    if "text/csv" in ctype or "application/octet-stream" in ctype:
        return _read_csv_bytes(resp.content)
    # JSON con csv_b64 o preview
    data = resp.json()
    if "csv_b64" in data and data["csv_b64"]:
        csv_bytes = base64.b64decode(data["csv_b64"])
        return _read_csv_bytes(csv_bytes)
    if "preview" in data and isinstance(data["preview"], list):
        # fallback: solo preview (limitado). Lo avisamos claro al usuario.
        df = pd.DataFrame(data["preview"])
        st.warning("Se recibió solo una **muestra** (preview) desde Azure (no CSV completo). El análisis puede ser parcial.")
        return df
    raise ValueError("La Function no devolvió CSV ni 'csv_b64'/'preview' válidos.")

def process_in_azure(file_obj, filename: str) -> pd.DataFrame:
    """
    Envía SIEMPRE el archivo a la Azure Function definida en ANON_API_URL.
    - Espera CSV anonimizado como respuesta (ideal) o JSON con csv_b64 / preview.
    - Si la Function requiere clave, lee ANON_API_KEY de secrets/env.
    """
    # 1) Endpoint y key
    api_url = os.getenv("ANON_API_URL") or st.secrets.get("ANON_API_URL", None)
    if not api_url:
        st.error("Falta configurar **ANON_API_URL** en el entorno/secrets. Ej: https://<tu-func>.azurewebsites.net/api/anon-upload")
        st.stop()
    api_key = os.getenv("ANON_API_KEY") or st.secrets.get("ANON_API_KEY", None)

    # 2) Bytes
    file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if not file_bytes:
        raise ValueError("El archivo está vacío o no se pudo leer.")

    # 3) POST -> Azure
    params = {"filename": filename or f"upload_{int(time.time())}.xlsx"}
    headers = {"Content-Type": "application/octet-stream"}
    if api_key:
        # Para authLevel=function: se puede pasar ?code= o header x-functions-key
        # Preferimos header si el host lo acepta:
        headers["x-functions-key"] = api_key

    r = requests.post(api_url, params=params, headers=headers, data=file_bytes, timeout=120)
    if r.status_code != 200:
        # Intenta leer mensaje de error
        try:
            msg = r.text
        except Exception:
            msg = f"status={r.status_code}"
        raise RuntimeError(f"Azure Function respondió error: {msg}")

    # 4) Construir DataFrame anonimizado
    df = _df_from_function_response(r)
    if df is None or df.empty:
        raise ValueError("La Function devolvió un dataset vacío.")
    return df

# --------------------------------------------------------------------------------------
#  UI & Lógica de tu app (igual que tu main, reemplazando la carga local por Azure)
# --------------------------------------------------------------------------------------
st.set_page_config("AuditIA", layout="wide")
cfg = load_config()
TH = {
    "DI": cfg["THRESHOLD_DI"],
    "DP_GREEN": cfg["THRESHOLD_DP_GREEN"],
    "DP_YELLOW": cfg["THRESHOLD_DP_YELLOW"],
    "EO_GREEN": cfg["THRESHOLD_EO_GREEN"],
    "EO_YELLOW": cfg["THRESHOLD_EO_YELLOW"]
}

# Carpeta para exportaciones
Path(f"{cfg['APP_PROJECT_DIR']}/demo").mkdir(parents=True, exist_ok=True)

# ===== Modo RR.HH. (simple) + estilos =====
st.title("📊 AuditIA")
st.caption("Auditoría de Equidad")
hr_mode = st.toggle("👩‍💼 Modo RR.HH.", value=True, help="Oculta detalles técnicos y deja solo lo esencial.")
st.markdown("""
<style>
h1, h2, h3 { margin-bottom: .4rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

def kpi_status_box(di, dp, eo, th, low_sample=False):
    if low_sample:
        st.warning("⚠️ Muestras chicas en al menos un grupo. Tomar con cautela.")
    if di is None:
        st.info("DI: —")
    elif di >= th["DI"]:
        st.success(f"✅ Cumple la 'regla del 80%'. DI={di:.2f}")
    else:
        st.error(f"❌ No cumple la 'regla del 80%'. DI={di:.2f}")
    if dp is not None:
        if abs(dp) <= th["DP_GREEN"]:
            st.success(f"✅ Tasas similares (DP={dp:.2f}).")
        elif abs(dp) <= th["DP_YELLOW"]:
            st.warning(f"🟡 Diferencia moderada (DP={dp:.2f}).")
        else:
            st.error(f"🔴 Brecha grande (DP={dp:.2f}).")
    if eo is not None:
        if abs(eo) <= th["EO_GREEN"]:
            st.success(f"✅ Oportunidad similar entre calificados (EO={eo:.2f}).")
        elif abs(eo) <= th["EO_YELLOW"]:
            st.warning(f"🟡 Brecha moderada entre calificados (EO={eo:.2f}).")
        else:
            st.error(f"🔴 Brecha importante entre calificados (EO={eo:.2f}).")

def hr_explanations(df, outcome_col, features, max_items=5):
    bullets = []
    for feat in features:
        if feat == outcome_col or feat not in df.columns:
            continue
        s = df[feat]
        try:
            if s.dtype.kind in "biufc" and s.nunique(dropna=True) > 10:
                q1 = s.quantile(0.25); q4 = s.quantile(0.75)
                low = df[s <= q1]; high = df[s >= q4]
                if len(low) >= 15 and len(high) >= 15:
                    r_low = float(low[outcome_col].mean()); r_high = float(high[outcome_col].mean())
                    gap = (r_high - r_low) * 100
                    direction = "aumenta" if gap > 0 else "reduce"
                    bullets.append(f"• Valores altos de **{feat}** {direction} la probabilidad de avanzar en ~**{abs(gap):.1f} p.p.** (Q4={r_high:.2f} vs Q1={r_low:.2f}).")
            else:
                grp = df[[feat, outcome_col]].dropna()
                if grp.empty: continue
                rates = grp.groupby(feat)[outcome_col].mean().sort_values()
                counts = grp.groupby(feat)[outcome_col].count()
                valid = counts[counts >= 15].index
                rates = rates.loc[rates.index.intersection(valid)]
                if len(rates) >= 2:
                    worst = rates.index[0]; best = rates.index[-1]
                    r_w = float(rates.iloc[0]); r_b = float(rates.iloc[-1])
                    gap = (r_b - r_w) * 100
                    bullets.append(f"• En **{feat}**, **{best}** avanza ~**{r_b:.2f}** vs **{worst}** ~**{r_w:.2f}** (gap ~**{gap:.1f} p.p.**).")
        except Exception:
            continue
        if len(bullets) >= max_items: break
    return bullets or ["• No se encontraron patrones claros con tamaños de muestra suficientes."]

with st.expander("¿Qué necesito?", expanded=False):
    st.markdown("""
**Archivos**: CSV/Excel.  
**Columnas**:
- **Resultado final** (0 = no avanza, 1 = avanza).
- (Opcional) **¿Realmente calificado?** (para EO).
- **Datos personales** (Género, Edad, País). Se detectan **automáticamente**.
**Privacidad**: el archivo se **anonimiza en Azure** antes de cualquier análisis.
""")

st.header("1) Cargar datos")
mode = st.radio("Elegí una opción:", ["Un archivo con etapas", "Varios archivos (uno por etapa)"], horizontal=True)
df = None
f = None

# === SIEMPRE CLOUD ===
if mode == "Un archivo con etapas":
    f = st.file_uploader("Subí tu archivo (CSV/XLSX)", type=["csv","xlsx"])
    if f:
        with st.spinner("Anonimizando en Azure..."):
            df = process_in_azure(f, f.name)
        st.success("✅ Procesado en Azure. Datos recibidos ya **anonimizados**.")
        # (si querés mostrar muestra técnica, destildá el HR mode)
        if not hr_mode:
            st.dataframe(df.head(8), use_container_width=True)
else:
    files = st.file_uploader("Subí uno o más archivos (CSV/XLSX)", type=["csv","xlsx"], accept_multiple_files=True)
    if files:
        opts = ["Detectar automáticamente","preselection","interview","offer","hire"]
        labels = []
        for ff in files:
            lab = st.selectbox(f"Etapa para **{ff.name}**", options=opts, key=f"name_{ff.name}")
            labels.append(None if lab=="Detectar automáticamente" else lab)

        # Procesar cada archivo en Azure y concatenar
        dfs = []
        with st.spinner("Anonimizando todos los archivos en Azure..."):
            for ff, lab in zip(files, labels):
                dfi = process_in_azure(ff, ff.name)
                if lab:
                    dfi["stage"] = lab
                dfs.append(dfi)
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            st.success("✅ Archivos procesados en Azure y combinados.")
            if not hr_mode:
                st.dataframe(df.head(8), use_container_width=True)

# >>> Paso 2 en adelante: tu flujo original
if df is not None and not df.empty:
    st.header("2) Confirmar columnas")
    mapping, _ = io.smart_map_columns(df)
    cols = list(df.columns)

    c1, c2 = st.columns(2)
    with c1:
        outcome_col = st.selectbox("Resultado final (obligatorio)", ["(elegir)"] + cols,
            index=(cols.index(mapping.get("outcome"))+1) if mapping.get("outcome") in cols else 0)
        y_true_col = st.selectbox("¿Realmente calificado? (opcional)", ["(ninguna)"] + cols,
            index=(cols.index(mapping.get("qualified"))+1) if mapping.get("qualified") in cols else 0)
    with c2:
        stage_col = st.selectbox("Etapa (opcional)", ["(ninguna)"] + cols,
            index=(cols.index(mapping.get("stage"))+1) if mapping.get("stage") in cols else 0)
        # quitamos opción de anonimizar local para evitar confusiones:
        st.caption("🔒 La anonimización ya se aplicó en Azure antes del análisis.")

    if "edad" in df.columns:
        with st.expander("Opcional: agrupar EDAD en rangos (recomendado)", expanded=True):
            modo_age = st.radio("Cómo agrupar edad:", ["Automático (de 10 años)", "Personalizado (elegir cortes)", "Sin agrupar"], index=0, horizontal=True)
            edad_min, edad_max = int(df["edad"].min()), int(df["edad"].max())
            st.caption(f"Rango observado de edad: **{edad_min}–{edad_max}**")
            if modo_age == "Automático (de 10 años)":
                start = (edad_min // 10) * 10
                stops = list(range(start, ((edad_max // 10) + 1) * 10 + 1, 10))
                bins = stops
                labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                df["edad_rango"] = pd.cut(df["edad"], bins=bins, right=False, labels=labels, include_lowest=True)
                st.success(f"Se creó 'edad_rango' con bandas: {', '.join(labels)}")
            elif modo_age == "Personalizado (elegir cortes)":
                defecto = f"{max(0, edad_min//10*10)},{(edad_min//10*10)+10},{(edad_min//10*10)+20},{(edad_min//10*10)+30}"
                cortes_txt = st.text_input("Cortes (enteros, asc):", value=defecto)
                try:
                    cortes = sorted(set(int(x.strip()) for x in cortes_txt.split(",") if x.strip() != ""))
                    if len(cortes) >= 2:
                        bins = cortes
                        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                        df["edad_rango"] = pd.cut(df["edad"], bins=bins, right=False, labels=labels, include_lowest=True)
                        st.success(f"Se creó 'edad_rango' con bandas: {', '.join(labels)}")
                except Exception as e:
                    st.error(f"No se pudo crear bandas personalizadas: {e}")
                    if "edad_rango" in df.columns: df.drop(columns=["edad_rango"], inplace=True)
            else:
                if "edad_rango" in df.columns: df.drop(columns=["edad_rango"], inplace=True)
                st.info("Se analizará 'edad' tal cual.")

    ready = (df is not None and not df.empty and 'outcome_col' in locals() and outcome_col != "(elegir)")
    if not ready:
        st.info("Elegí **Resultado final** para continuar.")

    if ready:
        st.header("3) Resultados")
        roles = heuristics.infer_roles(df)

        candidatos = []
        for k in ["gender","age","nationality"]:
            c = mapping.get(k)
            if c and c in df.columns and c not in candidatos:
                candidatos.append(c)

        bloquear = {outcome_col}
        if y_true_col and y_true_col != "(ninguna)":
            bloquear.add(y_true_col)
        if stage_col and stage_col != "(ninguna)":
            bloquear.add(stage_col)

        for c in df.columns:
            if c in bloquear:
                continue
            if roles.get(c) in {"id","notes","stage"}:
                continue
            nunq = df[c].nunique(dropna=True)
            if 1 < nunq <= 10 and c not in candidatos:
                candidatos.append(c)

        if "edad_rango" in df.columns:
            if "edad" in candidatos: candidatos.remove("edad")
            if "edad_rango" not in candidatos: candidatos.insert(0, "edad_rango")

        if not candidatos:
            st.warning("No se detectaron columnas sensibles categóricas (p. ej., Género, Rango de Edad, País).")
        else:
            candidatos_sel = st.multiselect(
                "Datos personales a comparar (detectados automáticamente)",
                options=candidatos,
                default=candidatos,
                help="Podés destildar los que no quieras analizar. Si no tocás nada, se usan todos."
            )
            cols_a_usar = candidatos_sel if candidatos_sel else candidatos

            metrics_by_var = []
            metrics_by_stage = []
            global_warnings = []

            tabs = st.tabs([str(c) for c in cols_a_usar])
            for colname, tab in zip(cols_a_usar, tabs):
                with tab:
                    st.subheader(f"Comparación por **{colname}**")

                    di, rates, ns = fairness.disparate_impact(df, colname, outcome_col)
                    dp, _, _      = fairness.demographic_parity(df, colname, outcome_col)
                    eo = None
                    if y_true_col and y_true_col != "(ninguna)":
                        eo, _, _ = fairness.equal_opportunity(df, colname, outcome_col, y_true_col)

                    warn_msg = small_sample_warning(df, colname, outcome_col, min_n=30)
                    if warn_msg:
                        st.warning(warn_msg)
                        global_warnings.append(warn_msg)

                    kpi_status_box(di, dp, eo, TH, low_sample=bool(warn_msg))

                    if rates:
                        data = []
                        for g, r in rates.items():
                            n = ns.get(g, 0)
                            se = (max(r*(1-r), 1e-9) / max(n, 1))**0.5
                            ci95 = 1.96 * se
                            data.append({"Grupo": str(g), "Tasa de avance": r, "n": n, "ci95": ci95})
                        dfr = pd.DataFrame(data)
                        fig = px.bar(dfr, x="Grupo", y="Tasa de avance", text="Tasa de avance", range_y=[0, 1], color="Grupo")
                        fig.update_traces(textposition="outside", texttemplate="%{y:.2f}")
                        if not hr_mode:
                            fig.update_traces(error_y=dict(type="data", array=dfr["ci95"], visible=True))
                        fig.add_hline(y=0.8*dfr["Tasa de avance"].max(), line_dash="dash", annotation_text="80%")
                        fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=10, b=10),
                                          annotations=[dict(x=i, y=0, text=f"n={n}", xanchor="center", yanchor="top", showarrow=False, yshift=-20)
                                                       for i, n in enumerate(dfr["n"])])
                        st.plotly_chart(fig, use_container_width=True, key=f"rates_{colname}")

                    groups_var = {}
                    if rates:
                        for g, r in rates.items():
                            n_g = ns.get(g, 0)
                            groups_var[str(g)] = {"n": int(n_g), "rate": float(r)}
                    metrics_by_var.append({
                        "col": str(colname),
                        "di": None if di is None else float(di),
                        "dp_gap": None if dp is None else float(dp),
                        "eo_gap": None if eo is None else float(eo),
                        "groups": groups_var,
                        "warn": warn_msg if 'warn_msg' in locals() else None,
                    })

                    if stage_col and stage_col != "(ninguna)" and stage_col in df.columns:
                        st.markdown("**Por etapa**")
                        etapas = sorted(df[stage_col].dropna().astype(str).unique())
                        stage_tabs = st.tabs(etapas)
                        for sname, stab in zip(etapas, stage_tabs):
                            with stab:
                                sdf = df[df[stage_col].astype(str) == sname]
                                di_s, rates_s, ns_s = fairness.disparate_impact(sdf, colname, outcome_col)

                                warn_stage = small_sample_warning(sdf, colname, outcome_col, min_n=30)
                                if warn_stage:
                                    st.warning(warn_stage)
                                    global_warnings.append(f"[{sname}] {warn_stage}")

                                if rates_s:
                                    data = []
                                    for g, r in rates_s.items():
                                        n = ns_s.get(g, 0)
                                        se = (max(r*(1-r), 1e-9) / max(n, 1))**0.5
                                        ci95 = 1.96 * se
                                        data.append({"Grupo": str(g), "Tasa de avance": r, "n": n, "ci95": ci95})
                                    dt = pd.DataFrame(data)
                                    fig2 = px.bar(dt, x="Grupo", y="Tasa de avance", text="Tasa de avance", range_y=[0, 1], color="Grupo")
                                    fig2.update_traces(textposition="outside", texttemplate="%{y:.2f}")
                                    if not hr_mode:
                                        fig2.update_traces(error_y=dict(type="data", array=dt["ci95"], visible=True))
                                    fig2.add_hline(y=0.8*dt["Tasa de avance"].max(), line_dash="dash", annotation_text="80%")
                                    fig2.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=10, b=10),
                                                      annotations=[dict(x=i, y=0, text=f"n={n}", xanchor="center", yanchor="top", showarrow=False, yshift=-20)
                                                                   for i, n in enumerate(dt["n"])])
                                    st.plotly_chart(fig2, use_container_width=True, key=f"rates_{colname}_{sname}")

                                metrics_by_stage.append({
                                    "stage": str(sname),
                                    "col": str(colname),
                                    "di": None if di_s is None else float(di_s),
                                    "dp_gap": None,
                                    "eo_gap": None,
                                    "warn": warn_stage if 'warn_stage' in locals() else None,
                                })

        st.divider()
        st.header("4) Factores que más influyen (global)")
        auto_features, roles = heuristics.suggest_features(
            df, outcome_col=outcome_col,
            y_true_col=y_true_col if y_true_col!="(ninguna)" else None
        )
        auto_features = [c for c in auto_features if df[c].nunique(dropna=True) >= 2]

        if not hr_mode:
            with st.expander("Columnas excluidas (ID/Etapa/PII/Notas)"):
                excluded = [c for c,r in roles.items() if r in {"id","stage","notes"}]
                st.write(", ".join(excluded) if excluded else "—")

        if st.button("🧠 Factores que más influyen"):
            if not auto_features:
                st.warning("No hay variables aptas para explicar (todas tienen un solo valor o fueron excluidas).")
            else:
                rrhh_bullets = hr_explanations(df, outcome_col=outcome_col, features=auto_features, max_items=5)
                st.subheader("En pocas palabras")
                for b in rrhh_bullets:
                    st.info(b)

                if not hr_mode:
                    try:
                        imp, _, bullets = explain.train_and_explain(df, y_col=outcome_col, feature_cols=auto_features)
                        st.subheader("Detalle técnico")
                        for b in bullets:
                            st.caption(b)
                        if imp:
                            imp_df = pd.DataFrame(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15],
                                                  columns=["Variable","Importancia relativa"])
                            figi = px.bar(imp_df.sort_values("Importancia relativa"), x="Importancia relativa", y="Variable",
                                          orientation="h", text="Importancia relativa", color="Importancia relativa", color_continuous_scale="Blues")
                            st.plotly_chart(figi, use_container_width=True, key="imp_global")
                    except ValueError as e:
                        st.warning(str(e))
                    except Exception as e:
                        st.error(f"No se pudo generar la explicación técnica: {e}")

        with st.expander("ℹ️ ¿Qué significan estos indicadores?"):
            st.markdown("""
- **Regla del 80% (DI)**: compara la tasa de avance de cada grupo contra el de referencia. Si es < 0.80, alarma.
- **Diferencia de tasas (DP)**: diferencia absoluta de probabilidades entre grupos (0 = igual).
- **Igualdad de oportunidades (EO)**: diferencia entre grupos **solo entre los realmente calificados**.
- **n por grupo**: cuántas personas hay en cada grupo. Si es bajo, las conclusiones pueden ser inestables.
""")

        # ----- 5) Descargas -----
        st.header("5) Descargas")
        from datetime import datetime
        csv_name = f"dataset_preparado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar dataset preparado (CSV)",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
            help="Descarga el dataset ya procesado (anonimización, edad_rango, merges, etc.)."
        )

        st.subheader("Informe PDF")
        if st.button("📄 Generar informe para RR.HH. (PDF)"):
            meta = {
                "dataset_name": getattr(f, 'name', 'dataset') if 'f' in locals() else 'dataset',
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "sensitive_cols": cols_a_usar,
                "thresholds": {"di_threshold": 0.8, "alpha": 0.05}
            }
            out_path = f"{cfg['APP_PROJECT_DIR']}/demo/reporte_completo_v622.pdf"

            build_fairness_pdf(
                out_path=out_path,
                meta=meta,
                metrics_by_var=metrics_by_var,
                metrics_by_stage=metrics_by_stage,
                global_warnings=global_warnings,
                top_factors=[],
                chart_paths=[]
            )

            with open(out_path, "rb") as fpdf:
                st.download_button("⬇️ Descargar informe RR.HH. (PDF)", fpdf, file_name="reporte_completo_v622.pdf")
            st.success(f"PDF generado en: {out_path}")
