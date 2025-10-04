
import os, logging, re, pandas as pd
from .config import load_config
try:
    from rapidfuzz import fuzz; HAVE_RAPIDFUZZ=True
except: HAVE_RAPIDFUZZ=False
try:
    from unidecode import unidecode; HAVE_UNIDECODE=True
except: HAVE_UNIDECODE=False
cfg = load_config()
os.makedirs(os.path.dirname(cfg["APP_LOG_PATH"]), exist_ok=True)
logging.basicConfig(filename=cfg["APP_LOG_PATH"], level=logging.INFO, format="%(asctime)s %(message)s")
TARGETS = {
    "outcome":{"syns":["outcome","result","resultado","label","target","seleccion","selección","decision","decisión","contratado","aprobado","avanza"],"required":True},
    "stage":{"syns":["stage","etapa","fase","proceso","paso","instancia","pipeline_stage"],"required":False},
    "qualified":{"syns":["qualified","ground truth","gt","label_true","y_true","calificado","apto","positivo_real"],"required":False},
    "gender":{"syns":["gender","sexo","genero","género","sex","m/f","masculino","femenino"]},
    "age":{"syns":["age","edad","age_group","rango_edad","franja_etaria","intervalo_edad","años"]},
    "nationality":{"syns":["nacionalidad","nationality","pais","país","origen","country"]}
}
_WORD_RE = re.compile(r"[a-z0-9]+")
def _norm(s):
    if s is None: return ""
    s=str(s).strip().lower()
    s=unidecode(s) if HAVE_UNIDECODE else s
    return " ".join(_WORD_RE.findall(s))
def _score(a,b):
    a=_norm(a); b=_norm(b)
    if not a or not b: return 0.0
    if HAVE_RAPIDFUZZ:
        return max(fuzz.partial_ratio(a,b),fuzz.token_set_ratio(a,b),fuzz.QRatio(a,b))/100.0
    aset,bset=set(a.split()), set(b.split())
    inter=len(aset&bset); union=len(aset|bset) or 1
    return max(inter/union, 1.0 if (a in b or b in a) else 0.0)
def smart_map_columns(df: pd.DataFrame, targets=TARGETS, min_score=0.55):
    mapping={}
    for tgt, meta in targets.items():
        syns=[tgt]+meta.get("syns",[])
        best=None; sc=0.0
        for c in df.columns:
            s=max((_score(c,s) for s in syns), default=0.0)
            if s>sc: sc=s; best=c
        mapping[tgt]=best if (best and sc>=min_score) else None
    return mapping, []
def read_any(file):
    name=getattr(file,"name","")
    if name.endswith(".xlsx"): df=pd.read_excel(file)
    else: df=pd.read_csv(file)
    return df, *smart_map_columns(df)
def merge_files_with_stages(files_and_labels):
    frames=[]
    for f, label in files_and_labels:
        if getattr(f,"name","").endswith(".xlsx"): tmp=pd.read_excel(f)
        else: tmp=pd.read_csv(f)
        if label: tmp["stage"]=label
        frames.append(tmp)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)
def anonymize(df, columns=("email","phone","dni","document","ssn","id","Email","Telefono")):
    out=df.copy()
    import hashlib, pandas as pd
    for c in columns:
        if c in out.columns:
            out[c]=out[c].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:10] if pd.notna(x) else x)
    return out
def export_dataset(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); df.to_csv(path, index=False)
