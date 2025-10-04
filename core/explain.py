
import pandas as pd, numpy as np, shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _prep_X(df, cols):
    if not cols:
        raise ValueError("No hay variables de entrada para explicar. Agregá al menos una columna no excluida.")
    X = df[cols].copy()
    keep = [c for c in X.columns if X[c].nunique(dropna=True) >= 2]
    if not keep:
        raise ValueError("Todas las variables tienen un solo valor o están vacías.")
    X = X[keep]
    X = pd.get_dummies(X, drop_first=False, dtype=float)
    if X.shape[1] == 0:
        raise ValueError("No quedaron variables tras la preparación (one-hot).")
    X = X.fillna(0.0)
    return X

def train_and_explain(df, y_col, feature_cols, random_state=42):
    y = df[y_col].astype(float)
    if set(y.dropna().unique()) - {0.0, 1.0}:
        try:
            y = y.astype(int)
        except Exception:
            raise ValueError("La columna de resultado debe ser binaria (0/1).")
    if y.nunique(dropna=True) < 2:
        raise ValueError("El resultado final tiene una sola clase. Necesitamos 0 y 1 para explicar.")

    X = _prep_X(df, feature_cols)
    scaler = StandardScaler(with_mean=False); Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state)
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique()>1 else None)
    except ValueError:
        Xtr, ytr = Xs, y

    model.fit(Xtr, ytr)

    try:
        explainer = shap.LinearExplainer(model, Xs, feature_dependence="independent")
        sv = explainer.shap_values(Xs)
        sv = sv if isinstance(sv, np.ndarray) else np.array(sv)
    except Exception:
        sv = shap.Explainer(model, Xs)(Xs).values

    mean_abs = (np.abs(sv)).mean(axis=0)
    mean_signed = sv.mean(axis=0)

    oh_cols = list(pd.DataFrame(X).columns)
    agg_abs, agg_signed = {}, {}
    for oh, a, s in zip(oh_cols, mean_abs, mean_signed):
        base = oh.split("_")[0]
        agg_abs[base] = agg_abs.get(base, 0.0) + float(a)
        agg_signed[base] = agg_signed.get(base, 0.0) + float(s)
    total = sum(agg_abs.values()) or 1.0
    norm_imp = {k: v/total for k, v in agg_abs.items()}
    directions = {k: ("+" if v>0 else ("-" if v<0 else "0")) for k, v in agg_signed.items()}

    top5 = sorted(norm_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    bullets = []
    for feat, _ in top5:
        sign = directions.get(feat, "0")
        if sign == "+": bullets.append(f"• **{feat}** suele **aumentar** las chances de avanzar.")
        elif sign == "-": bullets.append(f"• **{feat}** suele **reducir** las chances de avanzar.")
        else: bullets.append(f"• **{feat}** tiene **influencia neutra** en promedio.")

    return norm_imp, directions, bullets
