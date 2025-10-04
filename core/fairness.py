
import numpy as np
def disparate_impact(df, group_col, outcome_col):
    rates = df.groupby(group_col)[outcome_col].mean().dropna()
    if len(rates) < 2: return None, rates.to_dict(), df.groupby(group_col)[outcome_col].count().to_dict()
    piv = rates.idxmax()
    if rates[piv] == 0: return None, rates.to_dict(), df.groupby(group_col)[outcome_col].count().to_dict()
    di = float(rates.min()/rates[piv])
    return di, rates.to_dict(), df.groupby(group_col)[outcome_col].count().to_dict()

def demographic_parity(df, group_col, outcome_col):
    rates = df.groupby(group_col)[outcome_col].mean().dropna()
    if len(rates) < 2: return None, rates.to_dict(), df.groupby(group_col)[outcome_col].count().to_dict()
    dp = float(rates.max()-rates.min())
    return dp, rates.to_dict(), df.groupby(group_col)[outcome_col].count().to_dict()

def equal_opportunity(df, group_col, outcome_col, y_true_col):
    if y_true_col is None or y_true_col not in df.columns: return None, {}, {}
    tpr = {}; npos = {}
    for g, gdf in df.groupby(group_col):
        pos = gdf[gdf[y_true_col]==1]
        npos[g] = len(pos)
        tpr[g] = float(pos[outcome_col].mean()) if len(pos)>0 else float('nan')
    vals = [v for v in tpr.values() if v==v]
    if len(vals) < 2: return None, tpr, npos
    eo = float(max(vals)-min(vals))
    return eo, tpr, npos
