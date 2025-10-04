# utils_column_roles.py
# Heurísticas simples para sugerir roles de columnas y advertir por tamaños muestrales.
import re
import pandas as pd

KW = {
    "id": [r"\b(id|dni|legajo|employee[_\s]*id|candidate[_\s]*id|user[_\s]*id)\b"],
    "stage": [r"\b(stage|etapa|paso|fase)\b"],
    "outcome": [r"\b(resultado|outcome|label|target|aprobado|rechazado|status|estado)\b"],
    "qualified": [r"\b(qualified|calificado|apto|eligible|y_true|ground[_\s]*truth)\b"],
    "pii": [r"\b(email|mail|correo|telefono|tel|phone|movil|cel|documento)\b"],
    "sensitive": [r"\b(genero|sexo|gender|edad|age|nacionalidad|country|origen)\b"],
}

def _score(name: str, series: pd.Series) -> dict:
    """Calcula puntajes por rol en base al nombre y al contenido."""
    n_unique = series.dropna().nunique()
    is_binary = n_unique <= 2
    s = str(name).strip().lower()
    sc = {"id":0, "stage":0, "outcome":0, "qualified":0, "pii":0, "sensitive":0}

    # coincidencias por nombre (regex)
    for k, patterns in KW.items():
        for pat in patterns:
            if re.search(pat, s):
                sc[k] += 2  # peso por nombre

    # afinadores por contenido
    if is_binary:
        # si es binaria y "outcome" aparece por nombre → subí puntaje outcome
        if sc["outcome"] > 0:
            sc["outcome"] += 2
        # si es binaria y "stage" aparece → bajá stage (para no confundir con outcome)
        if sc["stage"] > 0:
            sc["stage"] -= 1

    # id suele tener muchos únicos y tipo string/número sin patrón binario
    total = len(series)
    if n_unique > max(50, int(total * 0.5)):  # heurística simple
        sc["id"] += 1

    # si parece teléfono/mail, reforzá PII y restá stage/outcome
    if sc["pii"] > 0:
        sc["stage"] -= 1
        sc["outcome"] -= 1

    # “qualified/y_true” suele ser binaria; si no lo es, penalizamos un poco
    if sc["qualified"] > 0 and not is_binary:
        sc["qualified"] -= 1

    return sc

def suggest_roles(df: pd.DataFrame) -> dict:
    """
    Devuelve dict {columna: {'best':'outcome|stage|id|...','scores':{...}}}
    No aplica cambios, solo sugiere.
    """
    suggestions = {}
    for col in df.columns:
        sc = _score(col, df[col])
        ordered = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        best, _ = ordered[0]
        # mini regla: si 'stage' y 'outcome' compiten y la col es binaria, preferí outcome
        n_unique = df[col].dropna().nunique()
        is_binary = n_unique <= 2
        if best in ("stage", "outcome"):
            # si outcome y stage empatan, forzamos outcome si binaria
            top_score = ordered[0][1]
            second = ordered[1] if len(ordered) > 1 else None
            if second and second[1] == top_score and set([best, second[0]]) == set(["stage", "outcome"]) and is_binary:
                best = "outcome"
        suggestions[col] = {"best": best, "scores": sc}
    return suggestions

def small_sample_warning(df: pd.DataFrame, col_sensitive: str, col_outcome: str, min_n: int = 30) -> str | None:
    """
    Retorna un mensaje de advertencia si algún grupo en col_sensitive tiene menos de min_n filas.
    """
    if col_sensitive not in df.columns or col_outcome not in df.columns:
        return None
    grp = df[[col_sensitive, col_outcome]].dropna()
    if grp.empty:
        return None
    c = grp.groupby(col_sensitive)[col_outcome].count().rename("n")
    bad = c[c < min_n]
    if not bad.empty:
        grupos = ", ".join([f"{g} (n={int(n)})" for g, n in bad.items()])
        return f"⚠️ Muestras chicas en '{col_sensitive}': {grupos}. Interpretar con cautela."
    return None
