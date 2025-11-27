import numpy as np
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)


def _clean_series(s):
    """Convierte a float y descarta NaN para evitar problemas en Fairlearn."""
    return pd.to_numeric(s, errors="coerce")


def disparate_impact(df, group_col, outcome_col):
    """
    Disparate Impact según Fairlearn (ratio de tasas de selección).
    Devuelve:
      - di: float o None
      - rates: dict grupo -> tasa de avance
      - ns: dict grupo -> n de observaciones
    """
    if group_col not in df.columns or outcome_col not in df.columns:
        return None, {}, {}

    y = _clean_series(df[outcome_col])
    g = df[group_col]

    # Métrica base: selection_rate (tasa de avance)
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y,
        y_pred=y,  # aquí outcome es ya 0/1
        sensitive_features=g,
    )

    rates = mf.by_group.to_dict()
    ns = df.groupby(group_col)[outcome_col].count().to_dict()

    # Fairlearn provee ratio() = min / max
    try:
        di = float(mf.ratio())
    except Exception:
        di = None

    return di, rates, ns


def demographic_parity(df, group_col, outcome_col):
    """
    Demographic Parity Difference: diferencia máxima de tasas de selección.
    """
    if group_col not in df.columns or outcome_col not in df.columns:
        return None, {}, {}

    y = _clean_series(df[outcome_col])
    g = df[group_col]

    # Usamos MetricFrame para conseguir tasas por grupo y diferencia
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y,
        y_pred=y,
        sensitive_features=g,
    )

    rates = mf.by_group.to_dict()
    ns = df.groupby(group_col)[outcome_col].count().to_dict()

    try:
        dp = float(mf.difference())
    except Exception:
        dp = None

    return dp, rates, ns


def equal_opportunity(df, group_col, outcome_col, y_true_col):
    """
    Equal Opportunity Difference: diferencia en TPR entre grupos.
    Se basa en true_positive_rate de Fairlearn.

    Devuelve:
      - eo_diff: float o None
      - tpr_by_group: dict grupo -> TPR
      - n_pos: dict grupo -> n de positivos reales
    """
    if (
        y_true_col is None
        or y_true_col not in df.columns
        or outcome_col not in df.columns
        or group_col not in df.columns
    ):
        return None, {}, {}

    y_true = _clean_series(df[y_true_col])
    y_pred = _clean_series(df[outcome_col])
    g = df[group_col]

    mf = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=g,
    )

    tpr_by_group = mf.by_group.to_dict()

    # n de positivos reales por grupo (para advertencias de tamaño muestral)
    n_pos = {}
    for group_value, subdf in df.groupby(group_col):
        pos = subdf[subdf[y_true_col] == 1]
        n_pos[group_value] = int(len(pos))

    try:
        eo_diff = float(mf.difference())
    except Exception:
        eo_diff = None

    return eo_diff, tpr_by_group, n_pos


def equalized_odds(df, group_col, outcome_col, y_true_col):
    """
    Equalized Odds Difference (métrica complementaria).
    No se muestra necesariamente en UI, pero
    se implementa según lo descripto en el PDF.
    """
    if (
        y_true_col is None
        or y_true_col not in df.columns
        or outcome_col not in df.columns
        or group_col not in df.columns
    ):
        return None

    y_true = _clean_series(df[y_true_col])
    y_pred = _clean_series(df[outcome_col])
    g = df[group_col]

    try:
        eod = float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=g,
            )
        )
    except Exception:
        eod = None

    return eod
