
import re, pandas as pd
try:
    from rapidfuzz import fuzz; HAVE_RAPIDFUZZ=True
except: HAVE_RAPIDFUZZ=False
try:
    from unidecode import unidecode; HAVE_UNIDECODE=True
except: HAVE_UNIDECODE=False
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?\d[\d\s\-\(\)]{6,}$")
_UUID  = re.compile(r"^[a-f0-9]{8}\-?[a-f0-9]{4}\-?[a-f0-9]{4}\-?[a-f0-9]{4}\-?[a-f0-9]{12}$", re.I)
_HEX   = re.compile(r"^[a-f0-9]{12,}$", re.I)
_HINTS_ID = ["id","id_candidato","candidate id","identificador","document","documento","dni","ssn","passport","email","mail","telefono","phone","uuid","hash"]
_HINTS_STAGE = ["stage","etapa","fase","proceso","instancia","paso","pipeline stage","step","phase","preseleccion","preselección","screening","entrevista","interview","oferta","offer","contratacion","contratación","hire"]
_HINTS_NOTES = ["notas","notes","observaciones","comments","comentarios"]
def _norm(s):
    s=str(s) if s is not None else ""; s=s.strip().lower()
    try: s=unidecode(s)
    except: pass
    return " ".join(re.findall(r"[a-z0-9]+", s))
def _fuzzy(a,b):
    a=_norm(a); b=_norm(b)
    if not a or not b: return 0.0
    try: return max(fuzz.partial_ratio(a,b),fuzz.token_set_ratio(a,b),fuzz.QRatio(a,b))/100.0
    except: 
        aset,bset=set(a.split()),set(b.split())
        inter=len(aset&bset); union=len(aset|bset) or 1
        return max(inter/union, 1.0 if (a in b or b in a) else 0.0)
def infer_roles(df: pd.DataFrame):
    roles={}
    for c in df.columns:
        role="other"; cname=str(c)
        s=df[c].dropna().astype(str).str.strip()
        if max((_fuzzy(cname,h) for h in _HINTS_NOTES), default=0.0)>=0.7: role="notes"
        elif max((_fuzzy(cname,h) for h in _HINTS_STAGE), default=0.0)>=0.6: role="stage"
        elif max((_fuzzy(cname,h) for h in _HINTS_ID), default=0.0)>=0.6: role="id"
        else:
            if not s.empty:
                sample=s.sample(min(len(s),200), random_state=0)
                if (sample.str.match(_EMAIL)).mean()>0.3 or (sample.str.match(_PHONE)).mean()>0.3: role="id"
                else:
                    n=len(s); uniq=s.nunique(dropna=True)
                    if n>=50 and uniq/n>0.98: role="id"
        roles[cname]=role
    return roles
def suggest_features(df, outcome_col=None, y_true_col=None):
    roles=infer_roles(df)
    blocked={"id","stage","notes"}
    blocked_by={"outcome","resultado_final","resultado","label","target","qualified","y_true","ground truth","gt"}
    feats=[]
    for c in df.columns:
        if roles.get(str(c)) in blocked: continue
        if any(_fuzzy(str(c),b)>=0.7 for b in blocked_by): continue
        if outcome_col and _fuzzy(str(c), outcome_col)>=0.95: continue
        if y_true_col and _fuzzy(str(c), y_true_col)>=0.95: continue
        feats.append(str(c))
    return feats, roles
