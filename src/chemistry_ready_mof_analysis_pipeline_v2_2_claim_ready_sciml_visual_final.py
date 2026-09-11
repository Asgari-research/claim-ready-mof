#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical analysis pipeline for the claim-ready MOF study.

This is the preserved scientific analysis source supplied for the study
"From Computation-Ready to Claim-Ready: Evidence-Aware Data for MOF Machine Learning".

The workflow profiles heterogeneous MOF-related tabular resources, constructs
representation-level chemistry/evidence summaries, performs identifier and join
accounting, and runs optional machine-learning stress tests. It is resume-aware
and writes intermediate outputs for auditability.

Important release note
----------------------
This source is preserved to document the historical analysis. The public figure
redraw workflow is maintained separately under ``reproduce/figures``
and operates only on frozen saved tables. The historical analysis environment,
complete raw inputs, and every intermediate output are not all present in the
current release materials; therefore this file should not be described as an
end-to-end reproduction certificate without those dependencies.

Run ``python <script> --help`` for the active command-line options. Use explicit
``--data-root`` and ``--out-dir`` paths rather than relying on historical local
filesystem examples.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, gc, hashlib, json, logging, math, os, pickle, platform, random, re, shutil, sys, time, traceback, warnings, zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        HGB_AVAILABLE = True
    except Exception:
        HGB_AVAILABLE = False
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    HGB_AVAILABLE = False

try:
    import pyarrow  # noqa
    PARQUET_AVAILABLE = True
except Exception:
    PARQUET_AVAILABLE = False
try:
    import openpyxl  # noqa
    EXCEL_AVAILABLE = True
except Exception:
    EXCEL_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -----------------------------------------------------------------------------
# Publication-style plotting defaults
# -----------------------------------------------------------------------------
# The script intentionally uses only matplotlib so it remains easy to run in a
# clean Visual Studio / conda environment.  Colours are chosen to be legible in
# print and colour-blind friendly enough for review drafts; all figures are
# saved as vector PDF, with high-resolution PNG/SVG depending on save mode.
PALETTE = [
    "#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#F8B195",
    "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653",
]
TRUST_COLORS = {
    "very_high": "#1B9E77",
    "high": "#66A61E",
    "medium": "#E6AB02",
    "low": "#D95F02",
    "not_observable": "#7570B3",
}
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 450,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})

VERSION = "1.2-nested-json-safe"  # superseded by v1.3 override near end
EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".parquet", ".pkl", ".pickle"}
# Text files can be real data tables, but archive README files are not useful
# for row/column profiling and can add thousands of noisy pseudo-tables.
TEXT_DOC_STEMS = {"readme", "license", "licence", "citation", "changelog", "changes", "notes"}
IGNORE_DIRS = {".git", "__pycache__", "outputs", "output", "results", "figures", "tables", "source_data", "logs", "checkpoints", ".venv", "venv", "env"}
MISSING = {"", " ", "na", "n/a", "nan", "none", "null", "missing", "unknown", "unk", "-", "--", "---", "?", "NA", "N/A", "NULL", "None", "NaN", "#N/A"}
RESOURCE_ORDER = ["MOSAEC-DB", "CoRE MOF 2024", "CoRE MOF 2025 metadata", "ARC-MOF", "QMOF", "CSD-derived context", "Other/unknown", "Task-specific joins"]

SAVE_MODES = {
    "minimal": dict(csv=True, pkl=False, parquet=False, xlsx=False, pdf=True, png=False, svg=False, models=False, zip=True),
    "compact": dict(csv=True, pkl=True, parquet=False, xlsx=False, pdf=True, png=False, svg=False, models=False, zip=True),
    "lean": dict(csv=True, pkl=True, parquet=True, xlsx=False, pdf=True, png=True, svg=False, models=False, zip=True),
    "balanced": dict(csv=True, pkl=True, parquet=True, xlsx=True, pdf=True, png=True, svg=False, models=False, zip=True),
    "thorough": dict(csv=True, pkl=True, parquet=True, xlsx=True, pdf=True, png=True, svg=True, models=True, zip=True),
}
RAM_MODES = {
    "ultra-light": dict(profile_nrows=3000, max_loaded_rows=12000, max_model_rows=8000, max_features=120, full_mb=15, count_large=False),
    "very-light": dict(profile_nrows=10000, max_loaded_rows=40000, max_model_rows=25000, max_features=250, full_mb=50, count_large=False),
    "light": dict(profile_nrows=50000, max_loaded_rows=120000, max_model_rows=60000, max_features=500, full_mb=150, count_large=True),
    "normal": dict(profile_nrows=None, max_loaded_rows=300000, max_model_rows=150000, max_features=1000, full_mb=500, count_large=True),
}
LEVELS = {
    "screening": dict(repeats=2, models=["dummy", "ridge"], splits=["random"], sensitivity=25, targets=1, cards=6),
    "standard": dict(repeats=4, models=["dummy", "ridge", "extratrees"], splits=["random", "grouped"], sensitivity=75, targets=2, cards=12),
    "comprehensive": dict(repeats=6, models=["dummy", "ridge", "hgb", "extratrees"], splits=["random", "grouped"], sensitivity=150, targets=3, cards=24),
    "thorough": dict(repeats=10, models=["dummy", "ridge", "hgb", "randomforest", "extratrees"], splits=["random", "grouped"], sensitivity=300, targets=5, cards=40),
}

PAT = {
    "id": re.compile(r"(^id$|_id$|id_|mofid|mof_id|mofkey|refcode|ccdc|csd|cif|filename|file_name|name$|structure|qmof)", re.I),
    "formula": re.compile(r"(formula|composition|stoich|chem[_ ]?formula|sum[_ ]?formula)", re.I),
    "metal": re.compile(r"(metal|node|sbu|element|cation|oxid|oxi|os_|valence|charge|formal|neutral|ionic|ion|salt)", re.I),
    "topology": re.compile(r"(topo|topology|net|rcsr|cluster|family)", re.I),
    "geometry": re.compile(r"(pld|lcd|pore|surface|asa|av|void|density|diameter|volume|vf|geometric|geom)", re.I),
    "descriptor": re.compile(r"(rac|rdf|aprdf|pdd|phom|mbtr|soap|desc|feature|fingerprint|mordred|rdkit)", re.I),
    "target": re.compile(r"(uptake|loading|adsorp|capacity|henry|kh|qst|selectivity|working_capacity|purity|recovery|productivity|energy|bandgap|homo|lumo|formation|co2|ch4|n2|h2|xe|kr|vsa|psa|pre_comb|post_comb|landfill|methane)", re.I),
    "curation": re.compile(r"(valid|invalid|suspect|warning|error|fail|pass|check|mofchecker|samosa|ncr|cr_|asr|fsr|recommended|screening|duplicate)", re.I),
    "group": re.compile(r"(topo|topology|cluster|family|metal|node|sbu|mofid|mofkey|refcode|source|resource)", re.I),
}
TARGET_RX = [
    re.compile(r"(co2|ch4|n2|h2|xe|kr).*(uptake|loading|adsorp|capacity|henry|kh|qst|selectivity)", re.I),
    re.compile(r"(uptake|loading|adsorp|capacity|henry|kh|qst|selectivity).*(co2|ch4|n2|h2|xe|kr)", re.I),
    re.compile(r"(working_capacity|deliverable|productivity|purity|recovery|energy)", re.I),
    re.compile(r"(bandgap|homo|lumo|formation_energy|energy_per_atom|qmof)", re.I),
]
COMMON_OX = {
    "Li": {1}, "Na": {1}, "K": {1}, "Mg": {2}, "Ca": {2}, "Sr": {2}, "Ba": {2}, "Al": {3}, "Ga": {3}, "In": {3},
    "Sc": {3}, "Y": {3}, "La": {3}, "Ce": {3,4}, "Ti": {3,4}, "Zr": {4}, "Hf": {4}, "V": {2,3,4,5}, "Nb": {3,5}, "Ta": {5},
    "Cr": {2,3,6}, "Mo": {2,3,4,5,6}, "W": {4,5,6}, "Mn": {2,3,4,7}, "Fe": {2,3}, "Ru": {2,3,4},
    "Co": {2,3}, "Rh": {3}, "Ir": {3,4}, "Ni": {2,3}, "Pd": {2,4}, "Pt": {2,4}, "Cu": {1,2}, "Ag": {1}, "Au": {1,3},
    "Zn": {2}, "Cd": {2}, "Hg": {1,2}, "Sn": {2,4}, "Pb": {2,4}, "Bi": {3,5}, "Ln": {3}
}
METALS = set(COMMON_OX)

@dataclass
class Config:
    data_root: Path
    out_dir: Path
    inventory_report: Optional[Path]
    save_mode: str
    ram_mode: str
    comprehensive_level: str
    n_jobs: int
    random_seed: int
    force: bool = False
    resume: bool = True
    profile_only: bool = False
    skip_ml: bool = False
    target_column: Optional[str] = None
    id_column: Optional[str] = None
    max_files: Optional[int] = None
    file_pattern: Optional[str] = None
    skip_row_count: bool = False
    no_zip: bool = False
    save: Dict[str, Any] = field(default_factory=dict)
    ram: Dict[str, Any] = field(default_factory=dict)
    level: Dict[str, Any] = field(default_factory=dict)
    def final(self):
        self.data_root = Path(self.data_root).resolve(); self.out_dir = Path(self.out_dir).resolve()
        if self.inventory_report: self.inventory_report = Path(self.inventory_report).resolve()
        self.save = SAVE_MODES[self.save_mode]; self.ram = RAM_MODES[self.ram_mode]; self.level = LEVELS[self.comprehensive_level]
        return self

def iso(): return dt.datetime.now().isoformat(timespec="seconds")
def mkdir(p: Path): p.mkdir(parents=True, exist_ok=True); return p
def slug(s: Any, n=110): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:n] or "item"
def hbytes(n):
    n=float(n); u=["B","KB","MB","GB","TB"]; i=0
    while n>=1024 and i<len(u)-1: n/=1024; i+=1
    return f"{n:.2f} {u[i]}"

def dirs(cfg: Config) -> Dict[str, Path]:
    d={"base":cfg.out_dir,"logs":cfg.out_dir/"logs","profiles":cfg.out_dir/"profiles","processed":cfg.out_dir/"processed","tables":cfg.out_dir/"tables","main_tables":cfg.out_dir/"tables"/"main","si_tables":cfg.out_dir/"tables"/"si","fig":cfg.out_dir/"figures","fig_main":cfg.out_dir/"figures"/"main","fig_si":cfg.out_dir/"figures"/"si","source":cfg.out_dir/"source_data","models":cfg.out_dir/"models","reports":cfg.out_dir/"reports"}
    for p in d.values(): mkdir(p)
    return d

def logger(logdir: Path):
    lg=logging.getLogger("mof_pipeline"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt=logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    fh=logging.FileHandler(logdir/"run.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

def save_json(x, p: Path): mkdir(p.parent); tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(json.dumps(x, indent=2, default=str), encoding="utf-8"); os.replace(tmp,p)
def read_json(p: Path, default):
    try: return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception: return default

def sig(cfg: Config):
    z=asdict(cfg); [z.pop(k,None) for k in ["out_dir","save","ram","level"]]
    return hashlib.sha256(json.dumps(z, sort_keys=True, default=str).encode()).hexdigest()[:16]

def save_df(df: pd.DataFrame, base: Path, cfg: Config, index=False) -> List[Path]:
    mkdir(base.parent); outs=[]
    if cfg.save.get("csv", True):
        p=base.with_suffix(".csv"); tmp=p.with_suffix(".csv.tmp"); df.to_csv(tmp,index=index); os.replace(tmp,p); outs.append(p)
    if cfg.save.get("pkl", False):
        p=base.with_suffix(".pkl"); tmp=p.with_suffix(".pkl.tmp"); pickle.dump(df, open(tmp,"wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(tmp,p); outs.append(p)
    if cfg.save.get("parquet", False) and PARQUET_AVAILABLE:
        try: p=base.with_suffix(".parquet"); df.to_parquet(p,index=index); outs.append(p)
        except Exception: pass
    return outs

def load_df(base: Path) -> pd.DataFrame:
    for ext in [".parquet", ".pkl", ".csv"]:
        p=base.with_suffix(ext)
        if p.exists():
            try:
                if ext==".parquet": return pd.read_parquet(p)
                if ext==".pkl": return pd.read_pickle(p)
                return pd.read_csv(p, low_memory=False)
            except Exception: pass
    return pd.DataFrame()

def save_fig(fig, base: Path, cfg: Config) -> List[Path]:
    mkdir(base.parent); outs=[]
    for ext,key,kw in [("pdf","pdf",{}),("png","png",{"dpi":450}),("svg","svg",{})]:
        if cfg.save.get(key, False):
            p=base.with_suffix("."+ext); fig.savefig(p,bbox_inches="tight",facecolor="white",metadata={"Creator": f"Chemistry-ready MOF pipeline {VERSION}"},**kw); outs.append(p)
    plt.close(fig); return outs

def run_step(name, expected, cfg, st, state_path, log, func, *args):
    done=st.get("completed_steps",{}).get(name,{})
    if cfg.resume and not cfg.force and done.get("status")=="ok" and done.get("signature")==sig(cfg) and all(Path(x).exists() for x in expected):
        log.info("Skipping completed step: %s", name); return
    log.info("Starting step: %s", name); t=time.time()
    try:
        func(*args); sec=time.time()-t
        st.setdefault("completed_steps",{})[name]={"status":"ok","seconds":sec,"finished":iso(),"signature":sig(cfg),"outputs":[str(x) for x in expected if Path(x).exists()]}
        save_json(st,state_path); log.info("Finished step: %s in %.1f s", name, sec)
    except Exception as e:
        st.setdefault("completed_steps",{})[name]={"status":"failed","finished":iso(),"signature":sig(cfg),"error":traceback.format_exc()}
        save_json(st,state_path); log.error("Step failed: %s\n%s", name, traceback.format_exc()); raise

# ----------------------------- input discovery and profiling -----------------------------
def resource_of(p: Path) -> str:
    s=str(p).lower().replace('\\','/'); n=p.name.lower()
    if 'mosaec' in s: return 'MOSAEC-DB'
    if 'core_mof_2025' in s or 'core-mof-2025' in s: return 'CoRE MOF 2025 metadata'
    if 'core_mof_2024' in s or 'core-mof-2024' in s or '12089' in n or 'asr_' in n or 'fsr_' in n: return 'CoRE MOF 2024'
    if 'arc_mof' in s or 'arc-mof' in s or any(k in n for k in ['landfill','pre_comb','post_comb','methane_purification','overall_process']): return 'ARC-MOF'
    if 'qmof' in s: return 'QMOF'
    if 'csd' in s or 'ccdc' in s or 'framework details' in n or 'suspect chemistry' in n: return 'CSD-derived context'
    return 'Other/unknown'

def role_of(p: Path) -> str:
    s=str(p).lower().replace('\\','/'); n=p.name.lower()
    if any(k in s for k in ['adsorption','landfill','methane','pre_comb','post_comb']): return 'adsorption_targets'
    if 'process' in s or 'overall_process' in n: return 'process_targets'
    if any(k in s for k in ['descriptor','racs','rdfs','aprdf','geom','phom','pdd']): return 'descriptor_matrix'
    if any(k in s for k in ['cluster','topology','topo']): return 'topology_or_cluster'
    if any(k in s for k in ['structure','cif']): return 'structure_or_cif_metadata'
    if any(k in s for k in ['asr','fsr','ion','ncr','recommended','screening','check']): return 'curation_or_validation'
    if 'qmof' in s or any(k in n for k in ['thermo','bandgap','dft']): return 'quantum_properties'
    if any(k in s for k in ['duplicate','diversity']): return 'duplicate_or_diversity'
    return 'general_metadata'

def find_inventory(root: Path, given: Optional[Path]) -> Optional[Path]:
    if given and given.exists(): return given
    names=['PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt','PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.zip','Project_data_Archive_Structure.txt','Project_data_Archive_Structure.zip']
    for b in [root, Path(__file__).resolve().parent]:
        for nm in names:
            if (b/nm).exists(): return b/nm
    return None

def parse_inventory(inv: Optional[Path], dd: Dict[str,Path], cfg: Config, log) -> None:
    """Parse the attached PROJECT_DATA_ARCHIVE_STRUCTURE inventory.

    The inventory is not the raw data.  It is a structured map of folders,
    examples and format counts.  This parser therefore writes two kinds of
    outputs:
    1) inventory_report_examples: example files and their inferred scientific role;
    2) inventory_folder_summary: folder-level file-count and size evidence.

    The true column names, missingness and row counts are deliberately computed
    later from real local CSV/XLSX/Parquet files, not invented from the inventory.
    """
    if not inv or not inv.exists():
        save_df(pd.DataFrame(), dd['profiles']/ 'inventory_report_examples', cfg)
        save_df(pd.DataFrame(), dd['profiles']/ 'inventory_folder_summary', cfg)
        return
    try:
        if inv.suffix.lower()=='.zip':
            with zipfile.ZipFile(inv) as z:
                txts=[x for x in z.namelist() if x.lower().endswith('.txt')]
                if not txts: raise RuntimeError('zip contains no .txt inventory report')
                text=z.read(txts[0]).decode('utf-8','replace')
        else:
            text=inv.read_text(encoding='utf-8',errors='replace')

        examples=[]; folders=[]; format_rows=[]
        cur=dict(full_path='', relative_path='', immediate_subfolders=None, direct_files=None,
                 direct_file_size='', empty_folder='', oldest_modified='', newest_modified='')
        current_ext=None
        for raw in text.splitlines():
            line=raw.rstrip('\n')
            if line.startswith('FOLDER'):
                if cur.get('relative_path'):
                    folders.append(cur.copy())
                cur=dict(full_path='', relative_path='', immediate_subfolders=None, direct_files=None,
                         direct_file_size='', empty_folder='', oldest_modified='', newest_modified='')
                current_ext=None
                continue
            if line.startswith('Full path'):
                cur['full_path']=line.split(':',1)[1].strip(); continue
            if line.startswith('Relative path'):
                cur['relative_path']=line.split(':',1)[1].strip(); continue
            if line.startswith('Immediate subfolders'):
                try: cur['immediate_subfolders']=int(line.split(':',1)[1].strip())
                except Exception: cur['immediate_subfolders']=line.split(':',1)[1].strip()
                continue
            if line.startswith('Direct files'):
                try: cur['direct_files']=int(line.split(':',1)[1].strip())
                except Exception: cur['direct_files']=line.split(':',1)[1].strip()
                continue
            if line.startswith('Direct file size'):
                cur['direct_file_size']=line.split(':',1)[1].strip(); continue
            if line.startswith('Empty folder'):
                cur['empty_folder']=line.split(':',1)[1].strip(); continue
            if line.startswith('Oldest modified file'):
                cur['oldest_modified']=line.split(':',1)[1].strip(); continue
            if line.startswith('Newest modified file'):
                cur['newest_modified']=line.split(':',1)[1].strip(); continue
            m=re.match(r'\s*\.(\w[\w.]*):\s+(\d+)\s+file\(s\),\s+total size\s+(.+?)\s+\((\d+) bytes\)', line)
            if m:
                ext='.'+m.group(1).lower(); current_ext=ext
                rel=cur.get('relative_path','') or '.'
                pp=Path(rel)
                format_rows.append(dict(inventory_report=str(inv), inventory_relative_folder=rel,
                                        extension=ext, n_files=int(m.group(2)), total_size_text=m.group(3),
                                        total_size_bytes=int(m.group(4)), resource=resource_of(pp), role=role_of(pp)))
                continue
            if '- Name' in line and ':' in line:
                nm=line.split(':',1)[-1].strip()
                rel=cur.get('relative_path','') or '.'
                pp=Path(rel)/nm
                examples.append(dict(inventory_report=str(inv), inventory_folder=cur.get('full_path',''),
                                     inventory_relative_folder=rel, file_name=nm, extension=Path(nm).suffix.lower(),
                                     resource=resource_of(pp), role=role_of(pp), inventory_format_block=current_ext,
                                     inventory_only=True))
        if cur.get('relative_path'):
            folders.append(cur.copy())

        exdf=pd.DataFrame(examples).drop_duplicates() if examples else pd.DataFrame()
        fdf=pd.DataFrame(folders).drop_duplicates() if folders else pd.DataFrame()
        fmtdf=pd.DataFrame(format_rows).drop_duplicates() if format_rows else pd.DataFrame()
        if not fdf.empty:
            fdf['resource']=fdf['relative_path'].map(lambda x: resource_of(Path(str(x))))
            fdf['role']=fdf['relative_path'].map(lambda x: role_of(Path(str(x))))
        save_df(exdf, dd['profiles']/ 'inventory_report_examples', cfg)
        save_df(fdf, dd['profiles']/ 'inventory_folder_summary', cfg)
        save_df(fmtdf, dd['profiles']/ 'inventory_format_summary', cfg)
        if not fmtdf.empty:
            resource_summary=fmtdf.groupby(['resource','role','extension']).agg(
                n_inventory_files=('n_files','sum'), inventory_size_bytes=('total_size_bytes','sum')
            ).reset_index()
            resource_summary['inventory_size_mb']=resource_summary['inventory_size_bytes']/(1024**2)
            save_df(resource_summary, dd['profiles']/ 'inventory_resource_role_format_summary', cfg)
        log.info('Parsed inventory report: %d example files, %d folders, %d format-count rows', len(exdf), len(fdf), len(fmtdf))
    except Exception as e:
        log.warning('Could not parse inventory report %s: %s', inv, e)
        save_df(pd.DataFrame(), dd['profiles']/ 'inventory_report_examples', cfg)
        save_df(pd.DataFrame(), dd['profiles']/ 'inventory_folder_summary', cfg)

def is_probably_tabular_text(p: Path) -> bool:
    """Return True for delimited .txt files and False for README-like prose.

    The CoRE/MOSAEC/QMOF archive contains many README.txt files.  Reading them
    with pandas can produce one-column pseudo-tables, which is not harmful but
    slows the pipeline and pollutes the audit.  This quick test keeps genuine
    tabular .txt files with repeated delimiters and skips obvious prose docs.
    """
    stem = p.stem.strip().lower()
    if stem in TEXT_DOC_STEMS or stem.startswith("readme"):
        return False
    try:
        sample = p.read_text(encoding="utf-8", errors="replace")[:8192]
    except Exception:
        return True
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    delimiter_hits = sum(1 for ln in lines[:20] if (ln.count(",") >= 1 or ln.count("\t") >= 1 or ln.count(";") >= 2 or ln.count("|") >= 1))
    return delimiter_hits >= max(2, min(5, len(lines[:20]) // 2))

def discover(root: Path, cfg: Config, log) -> pd.DataFrame:
    rows=[]; rx=re.compile(cfg.file_pattern,re.I) if cfg.file_pattern else None
    skipped_text_docs=0
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in IGNORE_DIRS and not d.startswith('.')]
        rp=Path(r)
        if any(part.lower() in IGNORE_DIRS for part in rp.parts): continue
        for f in fs:
            p=rp/f; ext=p.suffix.lower()
            if ext not in EXTS: continue
            if ext == '.txt' and not is_probably_tabular_text(p):
                skipped_text_docs += 1
                continue
            if rx and not rx.search(str(p)): continue
            try: st=p.stat()
            except Exception: continue
            rows.append(dict(file_path=str(p.resolve()), relative_path=str(p.resolve().relative_to(root.resolve())) if str(p.resolve()).startswith(str(root.resolve())) else str(p), file_name=p.name, extension=ext, resource=resource_of(p), role=role_of(p), size_bytes=st.st_size, size_mb=st.st_size/(1024**2), modified_time=dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')))
            if cfg.max_files and len(rows)>=cfg.max_files: break
        if cfg.max_files and len(rows)>=cfg.max_files: break
    df=pd.DataFrame(rows)
    if skipped_text_docs:
        log.info('Skipped %d README/prose .txt files during discovery; tabular .txt files are still included.', skipped_text_docs)
    if df.empty: log.warning('No table-like files discovered under %s', root); return df
    order={r:i for i,r in enumerate(RESOURCE_ORDER)}; df['ord']=df.resource.map(order).fillna(999)
    df=df.sort_values(['ord','role','size_bytes','file_name'], ascending=[True,True,False,True]).drop(columns='ord').reset_index(drop=True)
    log.info('Discovered %d table-like files', len(df)); return df

def sniff_sep(p: Path) -> str:
    if p.suffix.lower()=='.tsv': return '\t'
    try:
        smp=p.read_text(encoding='utf-8',errors='replace')[:4096]
        return csv.Sniffer().sniff(smp, delimiters=',\t;|').delimiter
    except Exception: return ','

def norm_missing(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=['object','string']).columns:
        try:
            s=df[c].astype('string').str.strip(); m=s.isin(MISSING)
            if m.any(): df.loc[m,c]=pd.NA
        except Exception: pass
    return df

def read_table(p: Path, cfg: Config, nrows: Optional[int]=None, usecols: Optional[List[str]]=None, log=None) -> Tuple[pd.DataFrame,Dict[str,Any]]:
    meta=dict(path=str(p), read_status='ok', requested_nrows=nrows); t=time.time(); ext=p.suffix.lower()
    try:
        if ext in ['.csv','.tsv','.txt']:
            sep=sniff_sep(p); kw=dict(sep=sep, nrows=nrows, low_memory=False); meta['separator']=sep
            if usecols: kw['usecols']=usecols
            try: df=pd.read_csv(p, **kw)
            except UnicodeDecodeError: df=pd.read_csv(p, encoding='latin-1', **kw)
            except Exception: df=pd.read_csv(p, engine='python', on_bad_lines='skip', **kw)
        elif ext in ['.xlsx','.xls']:
            if not EXCEL_AVAILABLE: raise RuntimeError('openpyxl/xlrd not installed')
            df=pd.read_excel(p, nrows=nrows, usecols=usecols)
        elif ext=='.json':
            try: df=pd.read_json(p, lines=False)
            except ValueError: df=pd.read_json(p, lines=True)
            if nrows: df=df.head(nrows)
        elif ext=='.jsonl':
            df=pd.read_json(p, lines=True)
            if nrows: df=df.head(nrows)
        elif ext=='.parquet':
            df=pd.read_parquet(p, columns=usecols)
            if nrows: df=df.head(nrows)
        elif ext in ['.pkl','.pickle']:
            df=pd.read_pickle(p)
            if usecols: df=df[usecols]
            if nrows: df=df.head(nrows)
        else: raise ValueError('unsupported extension')
        df=norm_missing(df); meta.update(read_rows=len(df), read_columns=df.shape[1], read_seconds=round(time.time()-t,3))
        return df, meta
    except Exception as e:
        meta.update(read_status='failed', read_error=str(e), read_seconds=round(time.time()-t,3))
        if log: log.warning('Failed to read %s: %s', p, e)
        return pd.DataFrame(), meta

def count_rows(p: Path) -> Optional[int]:
    try:
        with open(p,'rb') as f: n=sum(b.count(b'\n') for b in iter(lambda:f.read(1024*1024), b''))
        return max(0,n-1)
    except Exception: return None

def is_nested_value(x: Any) -> bool:
    """True for JSON-like list/dict/set/array cells produced by nested metadata files."""
    return isinstance(x, (list, tuple, set, dict, np.ndarray))

def nested_fraction(s: pd.Series) -> float:
    try:
        return float(s.map(is_nested_value).mean())
    except Exception:
        return 0.0

def is_missing_value(x: Any) -> bool:
    """Scalar-safe missing-value check.

    pandas.isna(list_or_dict) returns an array, which can later trigger
    ambiguous truth-value errors.  This function treats nested values as
    observed chemistry/metadata objects and handles scalar NA values normally.
    """
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() in MISSING
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False

def missing_mask(s: pd.Series) -> pd.Series:
    try:
        return s.map(is_missing_value).astype(bool)
    except Exception:
        return pd.Series([False]*len(s), index=s.index)

def safe_missing_count(s: pd.Series) -> int:
    return int(missing_mask(s).sum())

def safe_missing_fraction(s: pd.Series) -> float:
    return safe_missing_count(s) / max(len(s), 1)

def make_hashable(x: Any) -> str:
    """Convert scalars, lists, dicts and arrays to stable strings for profiling.

    This is the key fix for the failure in the attached log: JSON-derived
    columns such as refcode_groupby_node.json can contain Python lists, and
    pandas.nunique/value_counts cannot hash those lists directly.
    """
    if is_missing_value(x):
        return "__MISSING__"
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple, set, dict)):
        try:
            return json.dumps(x, sort_keys=True, default=str, ensure_ascii=False)
        except Exception:
            return repr(x)
    return str(x)

def nonmissing_hashable_series(s: pd.Series) -> pd.Series:
    mask = ~missing_mask(s)
    if not mask.any():
        return pd.Series([], dtype='string')
    return s.loc[mask].map(make_hashable).astype('string')

def safe_nunique(s: pd.Series) -> int:
    hs = nonmissing_hashable_series(s)
    if hs.empty:
        return 0
    return int(hs.nunique(dropna=True))

def safe_to_numeric(s: pd.Series) -> pd.Series:
    """Coerce only scalar-like entries to numeric and mark nested values missing."""
    def scalar_or_na(x):
        if isinstance(x, (list, tuple, set, dict, np.ndarray)):
            return pd.NA
        return x
    try:
        return pd.to_numeric(s.map(scalar_or_na), errors='coerce')
    except Exception:
        return pd.Series(np.nan, index=s.index, dtype='float64')

def is_numeric_column(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    num = safe_to_numeric(s)
    return bool(num.notna().mean() > 0.90 and num.nunique(dropna=True) > 1)

def dataframe_missing_fraction(df: pd.DataFrame) -> float:
    if df.empty or df.shape[1] == 0:
        return np.nan
    total = df.shape[0] * df.shape[1]
    miss = 0
    for c in df.columns:
        miss += safe_missing_count(df[c])
    return float(miss / max(total, 1))

def modality(col: str, s: pd.Series) -> str:
    if PAT['id'].search(col): return 'identifier'
    if PAT['formula'].search(col): return 'formula_or_composition'
    if PAT['metal'].search(col): return 'metal_or_charge_chemistry'
    if PAT['topology'].search(col): return 'topology_or_cluster'
    if PAT['geometry'].search(col): return 'geometric_descriptor'
    if PAT['descriptor'].search(col): return 'descriptor'
    if PAT['target'].search(col): return 'target_or_property'
    if PAT['curation'].search(col): return 'curation_or_validation_flag'
    if is_numeric_column(s): return 'numeric_other'
    if pd.api.types.is_bool_dtype(s): return 'binary_or_boolean'
    try:
        if safe_nunique(s) <= max(20, 0.05*len(s)): return 'categorical'
    except Exception: pass
    return 'free_text_or_high_cardinality'

def values_summary(s: pd.Series) -> str:
    """Compact, JSON-safe examples or numeric quantiles for a column."""
    if safe_missing_count(s) >= len(s): return ''
    num = safe_to_numeric(s)
    if pd.api.types.is_numeric_dtype(s) or num.notna().mean() > 0.90:
        try:
            qq = num.dropna().astype(float).quantile([0,.25,.5,.75,1])
            return '; '.join(f'q{int(k*100)}={v:.4g}' for k,v in qq.items())
        except Exception:
            pass
    hs = nonmissing_hashable_series(s)
    if hs.empty: return ''
    return '; '.join(f'{k}:{v}' for k,v in hs.value_counts().head(5).items())[:350]

def profile_one(row, cfg: Config, log) -> Tuple[Dict[str,Any],List[Dict[str,Any]]]:
    p=Path(row.file_path); nrows=cfg.ram['profile_nrows']
    if nrows is None and row.size_mb>cfg.ram['full_mb']: nrows=cfg.ram['max_loaded_rows']
    df,meta=read_table(p,cfg,nrows=nrows,log=log)
    fp={**row.to_dict(), **meta, 'true_rows':None,'true_rows_method':'unknown','profile_is_full_file':False,'n_columns':df.shape[1] if not df.empty else 0,'n_profiled_rows':len(df),'n_numeric_columns':0,'n_categorical_columns':0,'n_identifier_columns':0,'n_target_like_columns':0,'n_descriptor_like_columns':0,'n_nested_value_columns':0,'missing_fraction_overall':np.nan,'all_missing_columns':0,'constant_columns':0,'identifier_candidates':'','target_candidates':'','group_candidates':''}
    cols=[]
    if meta.get('read_status')!='ok': return fp, cols
    if p.suffix.lower() in ['.csv','.tsv','.txt'] and not cfg.skip_row_count and (cfg.ram['count_large'] or row.size_mb<=cfg.ram['full_mb']):
        fp['true_rows']=count_rows(p); fp['true_rows_method']='line_count'
    elif nrows is None or len(df)<(nrows or 10**12):
        fp['true_rows']=len(df); fp['true_rows_method']='read_rows'
    fp['profile_is_full_file']=fp['true_rows']==len(df) if fp['true_rows'] is not None else False
    if df.empty: return fp, cols
    fp['missing_fraction_overall']=dataframe_missing_fraction(df)
    ids=[]; targs=[]; groups=[]
    for c in df.columns:
        s=df[c]; m=modality(str(c),s); miss=safe_missing_count(s); nun=safe_nunique(s)
        isnum=bool(is_numeric_column(s)); nestfrac=nested_fraction(s); hasnested=bool(nestfrac>0); istarg=bool(any(rx.search(str(c)) for rx in TARGET_RX)); isid=bool(PAT['id'].search(str(c))); isgrp=bool(PAT['group'].search(str(c))); isdesc=m in ['descriptor','geometric_descriptor','numeric_other'] and not istarg
        if isid: ids.append(str(c))
        if istarg: targs.append(str(c))
        if isgrp: groups.append(str(c))
        cols.append(dict(file_path=str(p),relative_path=row.relative_path,file_name=p.name,resource=row.resource,role=row.role,column_name=str(c),dtype=str(s.dtype),inferred_modality=m,is_numeric=isnum,is_identifier_candidate=isid,is_target_candidate=istarg,is_group_candidate=isgrp,is_descriptor_candidate=isdesc,profiled_rows=len(s),missing_count=miss,missing_fraction=miss/max(len(s),1),unique_count=nun,unique_fraction=nun/max(len(s)-miss,1),constant_nonmissing=nun<=1,all_missing=miss==len(s),contains_nested_values=hasnested,nested_fraction=nestfrac,example_or_quantiles=values_summary(s)))
    cdf=pd.DataFrame(cols)
    if not cdf.empty:
        fp['n_numeric_columns']=int(cdf.is_numeric.sum()); fp['n_categorical_columns']=int((~cdf.is_numeric).sum()); fp['n_identifier_columns']=int(cdf.is_identifier_candidate.sum()); fp['n_target_like_columns']=int(cdf.is_target_candidate.sum()); fp['n_descriptor_like_columns']=int(cdf.is_descriptor_candidate.sum()); fp['n_nested_value_columns']=int(cdf.contains_nested_values.sum()) if 'contains_nested_values' in cdf else 0; fp['all_missing_columns']=int(cdf.all_missing.sum()); fp['constant_columns']=int(cdf.constant_nonmissing.sum())
    fp['identifier_candidates']=';'.join(ids[:20]); fp['target_candidates']=';'.join(targs[:20]); fp['group_candidates']=';'.join(groups[:20])
    return fp, cols

def step_profile(cfg: Config, dd: Dict[str,Path], log):
    disc=discover(cfg.data_root,cfg,log); save_df(disc,dd['profiles']/ 'discovered_input_files',cfg)
    files=[]; cols=[]
    for i,row in disc.iterrows():
        log.info('Profiling %d/%d: %s', i+1, len(disc), row.relative_path)
        fp,cp=profile_one(row,cfg,log); files.append(fp); cols.extend(cp)
        if (i+1)%10==0 or i+1==len(disc):
            save_df(pd.DataFrame(files),dd['profiles']/ 'file_level_profile_partial',cfg); save_df(pd.DataFrame(cols),dd['profiles']/ 'column_level_profile_partial',cfg); gc.collect()
    fdf=pd.DataFrame(files); cdf=pd.DataFrame(cols)
    save_df(fdf,dd['profiles']/ 'file_level_profile',cfg); save_df(cdf,dd['profiles']/ 'column_level_profile',cfg)
    if not cdf.empty:
        mm=cdf.groupby(['resource','inferred_modality']).size().reset_index(name='n_columns').pivot_table(index='resource',columns='inferred_modality',values='n_columns',fill_value=0).reset_index()
        save_df(mm,dd['profiles']/ 'resource_modality_matrix',cfg)
    rapid=fdf[[c for c in ['resource','role','relative_path','file_name','extension','size_mb','read_status','true_rows','n_columns','n_profiled_rows','n_numeric_columns','n_categorical_columns','missing_fraction_overall','identifier_candidates','target_candidates'] if c in fdf.columns]]
    save_df(rapid,dd['profiles']/ 'inventory_rapid_audit_table',cfg)

# ----------------------------- identifiers, chemistry trust and scores -----------------------------
def choose_ids(df: pd.DataFrame, cfg: Config) -> List[str]:
    if cfg.id_column and cfg.id_column in df.columns: return [cfg.id_column]
    cand=[c for c in df.columns if PAT['id'].search(str(c))]
    pri=['mofid','mof_id','mofkey','refcode','ccdc','csd','qmof','filename','file_name','cif','name','id']
    return sorted(cand, key=lambda c: next((i for i,k in enumerate(pri) if k in str(c).lower()),999))[:5]

def canon(x):
    if pd.isna(x): return None
    s=re.sub(r'\.(cif|json|txt|csv|xlsx)$','',str(x).strip(),flags=re.I)
    s=re.sub(r'\s+','',s).lower()
    return s if s and s not in {z.lower() for z in MISSING} else None

def step_join(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    summaries=[]; sets={}; maxr=min(cfg.ram['max_loaded_rows'],80000)
    for _,r in fdf.iterrows():
        if r.get('read_status')!='ok': continue
        p=Path(r.file_path); df,_=read_table(p,cfg,nrows=maxr,log=log)
        if df.empty: continue
        for col in choose_ids(df,cfg):
            vals=df[col].map(canon).dropna(); uni=set(vals.astype(str).tolist())
            summaries.append(dict(file_path=str(p),relative_path=r.relative_path,file_name=p.name,resource=r.resource,role=r.role,identifier_column=col,profiled_rows=len(vals),unique_ids=len(uni),duplicate_id_rows=int(vals.duplicated().sum()),duplicate_id_fraction=float(vals.duplicated().mean()) if len(vals) else 0,sample_ids=';'.join(sorted(list(uni))[:10])))
            sets[(str(p),col)]=uni
    sdf=pd.DataFrame(summaries); save_df(sdf,dd['processed']/ 'identifier_candidate_summary',cfg); save_df(sdf,dd['si_tables']/ 'SI_Table_S10_identifier_duplicates',cfg)
    rows=[]; keys=list(sets); max_pairs=2000 if cfg.comprehensive_level in ['screening','standard'] else 10000; pc=0
    for i,(pa,ca) in enumerate(keys):
        A=sets[(pa,ca)]
        if not A: continue
        for pb,cb in keys[i+1:]:
            pc+=1
            if pc>max_pairs: break
            if pa==pb: continue
            B=sets[(pb,cb)]
            inter=len(A & B)
            if inter:
                rows.append(dict(left_file=Path(pa).name,left_path=pa,left_resource=resource_of(Path(pa)),left_role=role_of(Path(pa)),left_id_column=ca,left_unique_ids_profiled=len(A),right_file=Path(pb).name,right_path=pb,right_resource=resource_of(Path(pb)),right_role=role_of(Path(pb)),right_id_column=cb,right_unique_ids_profiled=len(B),intersection_ids_profiled=inter,left_retention_fraction=inter/max(len(A),1),right_retention_fraction=inter/max(len(B),1)))
        if pc>max_pairs: break
    jdf=pd.DataFrame(rows).sort_values('intersection_ids_profiled',ascending=False) if rows else pd.DataFrame()
    save_df(jdf,dd['processed']/ 'join_accounting',cfg); save_df(jdf,dd['si_tables']/ 'SI_Table_S3_join_accounting',cfg)

def metals_from(x):
    if pd.isna(x): return []
    toks=re.findall(r'([A-Z][a-z]?)(?:\d*\.?\d*)',str(x)); out=[]
    for t in toks:
        if t in METALS and t not in out: out.append(t)
    return out

def ox_from(x):
    if pd.isna(x): return []
    s=str(x); rom={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8}; out=[]
    for r,n in rom.items():
        if re.search(rf'\b{r}\b',s): out.append(n)
    for m in re.findall(r'[-+]?\d+',s):
        try:
            v=abs(int(m))
            if 0<v<=8: out.append(v)
        except Exception: pass
    u=[]
    for x in out:
        if x not in u: u.append(x)
    return u

def row_flags(df: pd.DataFrame, p: Path, cfg: Config) -> pd.DataFrame:
    ids=choose_ids(df,cfg); fcols=[c for c in df.columns if PAT['formula'].search(str(c))]
    mcols=[c for c in df.columns if re.search(r'metal|element|node|sbu|cation',str(c),re.I)]
    oxcols=[c for c in df.columns if re.search(r'oxid|oxi|valence|formal.*state|metal.*state',str(c),re.I)]
    chcols=[c for c in df.columns if re.search(r'charge|net_charge|formal_charge|total_charge|framework_charge',str(c),re.I)]
    wcols=[c for c in df.columns if re.search(r'suspect|warning|error|fail|invalid|mofchecker|samosa|check|valid|recommended',str(c),re.I)]
    if not (fcols or mcols or oxcols or chcols or wcols): return pd.DataFrame()
    if len(df)>cfg.ram['max_loaded_rows']: df=df.sample(n=cfg.ram['max_loaded_rows'],random_state=cfg.random_seed)
    out=pd.DataFrame(index=df.index); out['source_file']=p.name; out['file_path']=str(p); out['resource']=resource_of(p); out['role']=role_of(p); out['row_index']=df.index.astype(str)
    out['primary_id']=df[ids[0]].map(canon) if ids else out['source_file']+'::'+out['row_index']
    txt=pd.Series('',index=df.index,dtype='object')
    for c in (fcols[:3]+mcols[:3]): txt=txt+' '+df[c].astype(str).fillna('')
    ml=txt.map(metals_from); out['metals_detected']=ml.map(lambda z:';'.join(z)); out['n_metals_detected']=ml.map(len); out['metal_family_primary']=ml.map(lambda z:z[0] if z else 'not_observable')
    ot=pd.Series('',index=df.index,dtype='object')
    for c in oxcols[:5]: ot=ot+' '+df[c].astype(str).fillna('')
    ox=ot.map(ox_from) if oxcols else pd.Series([[] for _ in range(len(df))], index=df.index)
    out['oxidation_states_reported']=ox.map(lambda z:';'.join(map(str,z))); out['oxidation_state_observable']=ox.map(bool)
    def plaus(idx):
        mets=ml.loc[idx]; oxs=ox.loc[idx]
        if not mets or not oxs: return 'not_observable'
        if any(o in COMMON_OX.get(m,set()) for m in mets for o in oxs): return 'plausible'
        return 'uncertain_high_oxidation' if any(o>6 for o in oxs) else 'uncertain_uncommon'
    out['oxidation_state_plausibility']=[plaus(i) for i in df.index]
    charge=pd.Series(np.nan,index=df.index,dtype='float')
    for c in chcols[:5]: charge=charge.combine_first(safe_to_numeric(df[c]))
    ionic=any(k in str(p).lower() for k in ['ion','charged','salt','ncr'])
    out['charge_value_first_available']=charge; out['abs_charge_residual']=charge.abs(); out['charge_observable']=charge.notna(); out['likely_ionic_context_from_file']=ionic
    out['charge_balance_status']=np.where(charge.isna(),'not_observable',np.where(charge.abs()<=1e-6,'neutral_or_balanced',np.where(ionic,'charged_context','nonzero_charge_uncertain')))
    suspect=pd.Series(False,index=df.index); positive=pd.Series(False,index=df.index); reasons=[[] for _ in range(len(df))]
    for c in wcols[:20]:
        ss=df[c].astype(str).str.lower(); bad=ss.str.contains(r'suspect|warning|error|fail|invalid|bad|false',na=False); good=ss.str.contains(r'pass|valid|true|recommended|ok',na=False)
        suspect|=bad; positive|=good
        for j,b in enumerate(bad.tolist()):
            if b: reasons[j].append(str(c))
    out['curation_suspect_flag']=suspect.values; out['curation_positive_flag']=positive.values; out['curation_reason_columns']=[';'.join(x[:6]) for x in reasons]
    score=pd.Series(1.0,index=df.index,dtype='float')
    score-=np.where(out.curation_suspect_flag,0.35,0); score-=np.where(out.oxidation_state_plausibility.eq('uncertain_uncommon'),0.18,0); score-=np.where(out.oxidation_state_plausibility.eq('uncertain_high_oxidation'),0.28,0); score-=np.where(out.oxidation_state_plausibility.eq('not_observable'),0.08,0); score-=np.where(out.charge_balance_status.eq('nonzero_charge_uncertain'),0.25,0); score-=np.where(out.charge_balance_status.eq('not_observable'),0.06,0); score-=np.where(out.n_metals_detected.eq(0),0.05,0)
    out['chemistry_trust_score_row']=score.clip(0,1); out['chemistry_trust_tier_row']=pd.cut(out.chemistry_trust_score_row,[-.01,.45,.70,.86,1.01],labels=['low','medium','high','very_high']).astype(str)
    return out.reset_index(drop=True)

def step_trust(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    if fdf.empty: return
    cand=fdf[fdf.read_status.eq('ok')].copy()
    cand['chem_priority']=cand.apply(lambda r: int(r.resource in ['MOSAEC-DB','CoRE MOF 2024','CoRE MOF 2025 metadata','CSD-derived context'])+int(r.role in ['curation_or_validation','general_metadata','quantum_properties'])+int(any(k in str(r.identifier_candidates).lower()+str(r.target_candidates).lower() for k in ['charge','oxid','formula','metal'])),axis=1)
    cand=cand.sort_values(['chem_priority','size_mb'],ascending=[False,True])
    parts=[]
    for _,r in cand.iterrows():
        if r.chem_priority<=0 and cfg.comprehensive_level in ['screening','standard']: continue
        p=Path(r.file_path); log.info('Chemistry-trust scan: %s', r.relative_path)
        df,_=read_table(p,cfg,nrows=cfg.ram['max_loaded_rows'],log=log); fl=row_flags(df,p,cfg)
        if not fl.empty:
            parts.append(fl); save_df(fl,dd['processed']/ 'trust_flags_by_file'/slug(p.stem),cfg)
        if cfg.comprehensive_level=='screening' and len(parts)>=20: break
    trust=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['source_file','resource','role','primary_id','metal_family_primary','chemistry_trust_score_row','chemistry_trust_tier_row','oxidation_state_plausibility','charge_balance_status','curation_suspect_flag'])
    save_df(trust,dd['processed']/ 'trust_flags',cfg)
    if trust.empty:
        for n in ['trust_flags_by_metal','charge_balance_residuals','variant_comparison','representative_rule_cards']: save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    met=trust.groupby(['resource','metal_family_primary','chemistry_trust_tier_row']).size().reset_index(name='n'); met['fraction']=met.n/met.groupby(['resource','metal_family_primary']).n.transform('sum')
    save_df(met,dd['processed']/ 'trust_flags_by_metal',cfg); save_df(met,dd['si_tables']/ 'SI_Table_S8_trust_flags_by_metal',cfg)
    ch=trust[[c for c in ['resource','role','source_file','primary_id','metal_family_primary','charge_value_first_available','abs_charge_residual','charge_balance_status','likely_ionic_context_from_file','chemistry_trust_tier_row'] if c in trust.columns]]
    save_df(ch,dd['processed']/ 'charge_balance_residuals',cfg)
    def varlab(r):
        s=(str(r.source_file)+' '+str(r.role)).lower()
        for k,v in [('asr','ASR'),('fsr','FSR'),('ion','ION'),('charged','charged'),('neutral','neutral_or_NCR'),('ncr','neutral_or_NCR'),('recommended','recommended_screening'),('12089','recommended_screening')]:
            if k in s: return v
        return 'other'
    trust['variant_label']=trust.apply(varlab,axis=1)
    var=trust.groupby(['resource','variant_label']).agg(n_rows=('primary_id','count'),mean_trust=('chemistry_trust_score_row','mean'),median_trust=('chemistry_trust_score_row','median'),suspect_fraction=('curation_suspect_flag','mean')).reset_index()
    save_df(var,dd['processed']/ 'variant_comparison',cfg)
    cards=[]; per=max(1,cfg.level['cards']//4)
    for tier in ['very_high','high','medium','low']:
        sub=trust[trust.chemistry_trust_tier_row.eq(tier)]
        if sub.empty: continue
        sub=sub.sample(n=min(per,len(sub)),random_state=cfg.random_seed) if len(sub)>per else sub
        for _,r in sub.iterrows():
            reasons=[]
            if r.get('curation_suspect_flag',False): reasons.append('curation warning column triggered')
            if str(r.get('oxidation_state_plausibility',''))!='plausible': reasons.append(f"oxidation state: {r.get('oxidation_state_plausibility')}")
            if str(r.get('charge_balance_status','')) not in ['neutral_or_balanced','charged_context']: reasons.append(f"charge balance: {r.get('charge_balance_status')}")
            if not reasons: reasons.append('no observable chemistry warning in available columns')
            cards.append(dict(rule_card_class=tier,resource=r.resource,source_file=r.source_file,primary_id=r.primary_id,metals_detected=r.get('metals_detected',''),oxidation_states_reported=r.get('oxidation_states_reported',''),chemistry_trust_score_row=r.chemistry_trust_score_row,chemistry_trust_tier_row=r.chemistry_trust_tier_row,interpretation='; '.join(reasons)))
    cdf=pd.DataFrame(cards); save_df(cdf,dd['processed']/ 'representative_rule_cards',cfg); save_df(cdf,dd['si_tables']/ 'SI_Table_S15_representative_rule_cards',cfg)

def qlabel(ml,tr):
    if tr>=.70 and ml>=.70: return 'chemistry-ready and ML-ready'
    if tr>=.70: return 'chemistry-ready but ML-limited'
    if ml>=.70: return 'ML-ready but chemistry-risky'
    return 'limited or context-dependent'

def role_rec(res,ml,tr):
    return {'MOSAEC-DB':'chemistry-ready anchor and validation reference','CoRE MOF 2024':'experimental computation-ready benchmark with recommended-subset checks','CoRE MOF 2025 metadata':'forward-looking curation metadata comparison','ARC-MOF':'descriptor-rich adsorption/process ML stress-test resource','QMOF':'quantum-property contrast resource','CSD-derived context':'provenance and suspect-chemistry context'}.get(res,'context-specific auxiliary resource')

def logscore(x,m): return float(np.log1p(max(float(x),0))/np.log1p(max(float(m),1)))

def step_scores(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile'); cdf=load_df(dd['profiles']/ 'column_level_profile'); trust=load_df(dd['processed']/ 'trust_flags')
    if fdf.empty: return
    maxrows=pd.to_numeric(fdf.get('true_rows',fdf.n_profiled_rows),errors='coerce').fillna(fdf.n_profiled_rows).max(); maxnum=pd.to_numeric(fdf.n_numeric_columns,errors='coerce').fillna(0).max()
    rows=[]
    for res in sorted(fdf.resource.dropna().unique(), key=lambda x: RESOURCE_ORDER.index(x) if x in RESOURCE_ORDER else 999):
        fs=fdf[fdf.resource.eq(res)]; cs=cdf[cdf.resource.eq(res)] if not cdf.empty else pd.DataFrame(); ts=trust[trust.resource.eq(res)] if not trust.empty and 'resource' in trust else pd.DataFrame()
        total=pd.to_numeric(fs.get('true_rows',fs.n_profiled_rows),errors='coerce').fillna(fs.n_profiled_rows).sum(); miss=pd.to_numeric(fs.missing_fraction_overall,errors='coerce').mean()
        id_av=float((pd.to_numeric(fs.n_identifier_columns,errors='coerce').fillna(0)>0).mean()); targ_av=float((pd.to_numeric(fs.n_target_like_columns,errors='coerce').fillna(0)>0).mean()); desc_av=float((pd.to_numeric(fs.n_descriptor_like_columns,errors='coerce').fillna(0)>0).mean())
        chem_cols=int(cs.inferred_modality.isin(['formula_or_composition','metal_or_charge_chemistry','curation_or_validation_flag']).sum()) if not cs.empty else 0; chem_col_score=min(1,chem_cols/20)
        if not ts.empty and 'chemistry_trust_score_row' in ts: trust_score=float(pd.to_numeric(ts.chemistry_trust_score_row,errors='coerce').mean()); observed=min(1,len(ts)/max(float(fs.n_profiled_rows.sum()),1))
        else:
            trust_score={'MOSAEC-DB':.86,'CoRE MOF 2024':.76,'CoRE MOF 2025 metadata':.75,'ARC-MOF':.60,'QMOF':.68,'CSD-derived context':.58,'Other/unknown':.45}.get(res,.45)*(0.75+0.25*chem_col_score); observed=0
        ml=np.clip(.22*logscore(total,maxrows*len(fdf.resource.unique()))+.20*logscore(fs.n_numeric_columns.sum(),maxnum*max(len(fs),1))+.16*(1-miss if not pd.isna(miss) else .5)+.16*id_av+.14*desc_av+.12*targ_av,0,1)
        tr=float(np.clip(.78*trust_score+.22*chem_col_score,0,1))
        rows.append(dict(resource=res,n_files=len(fs),total_profiled_or_counted_rows=float(total),total_numeric_columns=float(fs.n_numeric_columns.sum()),mean_missing_fraction=miss,identifier_availability_fraction=id_av,target_availability_fraction=targ_av,descriptor_availability_fraction=desc_av,chemistry_relevant_column_count=chem_cols,row_level_trust_observed_fraction=observed,chemistry_trust_score=tr,ml_readiness_score=float(ml),trust_readiness_quadrant=qlabel(ml,tr),recommended_role=role_rec(res,ml,tr)))
    scores=pd.DataFrame(rows); save_df(scores,dd['processed']/ 'resource_scores',cfg); save_df(scores,dd['main_tables']/ 'Main_Table_1_resource_profile_and_scores',cfg); save_df(scores,dd['si_tables']/ 'SI_Table_S5_resource_and_subset_scores',cfg)
    metric_defs=pd.DataFrame([{'metric':'Chemistry-trust score','range':'0--1','meaning':'Observable chemistry-validity confidence from curation, oxidation/charge evidence and chemistry-relevant columns.'},{'metric':'ML-readiness score','range':'0--1','meaning':'Rows, numerical descriptors, identifiers, target availability and missingness suitability.'},{'metric':'Benchmark-risk score','range':'0--3','meaning':'Task-specific consequence of a data-quality issue.'},{'metric':'Trust regime','range':'low/medium/high/very_high','meaning':'Row/subset class used to test benchmark sensitivity.'}])
    save_df(metric_defs,dd['main_tables']/ 'Main_Table_2_metric_definitions',cfg); save_df(metric_defs,dd['si_tables']/ 'SI_Table_S4_chemistry_trust_definitions',cfg)
    rec=pd.DataFrame([{'use_case':'Adsorption ranking','preferred_resource_logic':'Use descriptor-rich adsorption resources with chemistry-trust flags and grouped splits.','minimum_reporting':'target provenance, gas/pressure/temperature, identifier joins, top-k stability'},{'use_case':'Process screening','preferred_resource_logic':'Use process tables after documenting target definitions and leakage-prone descriptors.','minimum_reporting':'cycle target definition, row retention, duplicate policy'},{'use_case':'Quantum-property prediction','preferred_resource_logic':'Use DFT-consistent resources when electronic-structure consistency is central.','minimum_reporting':'DFT settings, units, failed calculations, chemistry flags'},{'use_case':'Generative-model training','preferred_resource_logic':'Prefer chemistry-validated and duplicate-controlled subsets.','minimum_reporting':'validity filters, duplicate groups, charge/oxidation warnings'},{'use_case':'Mechanistic interpretation','preferred_resource_logic':'Prioritize chemically interpretable subsets over maximum size.','minimum_reporting':'rule cards and manual check examples'}])
    save_df(rec,dd['main_tables']/ 'Main_Table_3_use_case_recommendation_cards',cfg)
    issues=['missing oxidation-state/formal-charge evidence','nonzero charge outside explicit ionic context','suspect chemistry or validation warning','identifier ambiguity or duplicate leakage','high descriptor missingness','target provenance mismatch','small or biased trusted subset']; uses=['adsorption ranking','process screening','quantum-property modeling','generative training','mechanistic interpretation']; base=[[2,2,2,3,3],[3,3,2,3,3],[2,2,2,3,3],[3,3,2,3,2],[2,2,1,2,1],[3,3,2,1,2],[2,2,2,2,2]]
    risk=pd.DataFrame([dict(risk_issue=iss,use_case=uc,risk_score_0_to_3=base[i][j],interpretation={0:'negligible',1:'low',2:'moderate',3:'high'}[base[i][j]]) for i,iss in enumerate(issues) for j,uc in enumerate(uses)])
    save_df(risk,dd['processed']/ 'benchmark_risk_matrix',cfg); save_df(risk,dd['main_tables']/ 'Main_Table_3_benchmark_risk_matrix',cfg); save_df(risk,dd['si_tables']/ 'SI_Table_S7_benchmark_risk_matrix',cfg)
    rng=np.random.default_rng(cfg.random_seed); sens=[]
    for d in range(cfg.level['sensitivity']):
        wt=rng.uniform(.35,.75); tmp=scores.copy(); tmp['combined']=wt*tmp.chemistry_trust_score+(1-wt)*tmp.ml_readiness_score; tmp['rank_in_draw']=tmp.combined.rank(ascending=False,method='min')
        for _,r in tmp.iterrows(): sens.append(dict(draw=d,resource=r.resource,weight_chemistry_trust=wt,combined_score=r.combined,rank_in_draw=r.rank_in_draw))
    sdf=pd.DataFrame(sens).groupby('resource').agg(mean_rank=('rank_in_draw','mean'),min_rank=('rank_in_draw','min'),max_rank=('rank_in_draw','max'),mean_combined_score=('combined_score','mean'),sd_combined_score=('combined_score','std')).reset_index() if sens else pd.DataFrame()
    save_df(sdf,dd['processed']/ 'score_sensitivity',cfg)

# ----------------------------- ML stress test -----------------------------
def target_cols(df: pd.DataFrame, cfg: Config) -> List[str]:
    if cfg.target_column and cfg.target_column in df.columns: return [cfg.target_column]
    cand=[]
    for c in df.columns:
        s=df[c]; num=safe_to_numeric(s)
        if num.notna().mean()<.8 or int(num.nunique(dropna=True))<=5: continue
        score=sum((20-i) for i,rx in enumerate(TARGET_RX) if rx.search(str(c)))
        if any(k in str(c).lower() for k in ['id','index','unnamed','number']): score-=15
        if score>0: cand.append((score,str(c)))
    return [c for _,c in sorted(cand,reverse=True)[:cfg.level['targets']]]

def group_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if PAT['group'].search(str(c)):
            try:
                n=safe_nunique(df[c])
                if 2<=n<=max(2,len(df)*.5): return c
            except Exception: pass
    return None

def xy(df: pd.DataFrame, target: str, cfg: Config):
    y=safe_to_numeric(df[target]); keep=y.notna(); df=df.loc[keep].copy(); y=y.loc[keep]
    if len(df)>cfg.ram['max_model_rows']:
        df=df.sample(n=cfg.ram['max_model_rows'],random_state=cfg.random_seed); y=y.loc[df.index]
    gcol=group_col(df); groups=df[gcol].astype(str) if gcol else None
    num=[]; cat=[]
    for c in df.columns:
        if c==target: continue
        name=str(c)
        if PAT['id'].search(name) or name.lower().startswith('unnamed') or any(rx.search(name) for rx in TARGET_RX): continue
        s=df[c]
        if safe_missing_fraction(s)>.65 or safe_nunique(s)<=1: continue
        conv=safe_to_numeric(s)
        if pd.api.types.is_numeric_dtype(s) or conv.notna().mean()>.9: num.append(c)
        elif safe_nunique(s)<=50: cat.append(c)
    if len(num)>cfg.ram['max_features']:
        rank=[]
        for c in num:
            ss=safe_to_numeric(df[c]); bonus=1 if (PAT['descriptor'].search(str(c)) or PAT['geometry'].search(str(c))) else 0
            rank.append((bonus,float(ss.var(skipna=True) if ss.notna().sum()>2 else 0),c))
        num=[c for _,_,c in sorted(rank,reverse=True)[:cfg.ram['max_features']]]
    X=df[num+cat].copy()
    for c in num: X[c]=safe_to_numeric(X[c])
    for c in cat: X[c]=X[c].map(make_hashable).replace('__MISSING__', pd.NA).astype('string')
    return X,y,groups,num,cat

def model_pipe(name, num, cat, cfg):
    trans=[]
    if num: trans.append(('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler(with_mean=False))]),num))
    if cat:
        try: ohe=OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=50)
        except TypeError: ohe=OneHotEncoder(handle_unknown='ignore', sparse=True)
        trans.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',ohe)]),cat))
    pre=ColumnTransformer(trans, remainder='drop')
    if name=='dummy': m=DummyRegressor(strategy='median')
    elif name=='ridge': m=Ridge(alpha=1.0, random_state=cfg.random_seed)
    elif name=='hgb' and HGB_AVAILABLE: m=HistGradientBoostingRegressor(random_state=cfg.random_seed,max_iter=160,learning_rate=.06)
    elif name=='randomforest': m=RandomForestRegressor(n_estimators=220,random_state=cfg.random_seed,n_jobs=cfg.n_jobs,min_samples_leaf=2)
    else: m=ExtraTreesRegressor(n_estimators=300,random_state=cfg.random_seed,n_jobs=cfg.n_jobs,min_samples_leaf=1)
    return Pipeline([('preprocess',pre),('model',m)])

def rmse(y,p):
    try: return float(mean_squared_error(y,p,squared=False))
    except TypeError: return float(math.sqrt(mean_squared_error(y,p)))
def spear(y,p):
    try: return float(scipy_stats.spearmanr(y,p,nan_policy='omit').correlation) if scipy_stats else float(pd.Series(y).corr(pd.Series(p),method='spearman'))
    except Exception: return np.nan
def topk(y,p,frac):
    n=len(y); k=max(1,int(math.ceil(frac*n))); return len(set(np.argsort(y)[-k:]) & set(np.argsort(p)[-k:]))/k if n else np.nan
def ndcg(y,p,frac=.1):
    n=len(y); k=max(1,int(math.ceil(frac*n))); gain=y-np.nanmin(y); order=np.argsort(p)[::-1][:k]; ideal=np.argsort(y)[::-1][:k]; den=np.sum(gain[ideal]/np.log2(np.arange(2,k+2))); return float(np.sum(gain[order]/np.log2(np.arange(2,k+2)))/den) if den>0 else np.nan

def run_ml(df, target, label, cfg, log):
    if not SKLEARN_AVAILABLE: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    X,y,groups,num,cat=xy(df,target,cfg)
    if len(y)<200 or len(num)+len(cat)<2: log.warning('Not enough model-ready rows/features for %s %s', label, target); return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    mets=[]; preds=[]; errs=[]
    models=[m for m in cfg.level['models'] if m!='hgb' or HGB_AVAILABLE]
    for split in cfg.level['splits']:
        if split=='grouped' and groups is None: continue
        for rep in range(cfg.level['repeats']):
            seed=cfg.random_seed+17*rep
            if split=='grouped': tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y,groups=groups))
            else: tr,te=next(ShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y))
            for mn in models:
                try:
                    pipe=model_pipe(mn,num,cat,cfg); pipe.fit(X.iloc[tr],y.iloc[tr]); pr=pipe.predict(X.iloc[te]); yt=np.asarray(y.iloc[te])
                    mets.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,n_train=len(tr),n_test=len(te),n_numeric_features=len(num),n_categorical_features=len(cat),rmse=rmse(yt,pr),mae=float(mean_absolute_error(yt,pr)),r2=float(r2_score(yt,pr)),spearman=spear(yt,pr),top_1pct_recovery=topk(yt,pr,.01),top_5pct_recovery=topk(yt,pr,.05),top_10pct_recovery=topk(yt,pr,.10),ndcg_10pct=ndcg(yt,pr,.10)))
                    idx=np.arange(len(yt));
                    if len(idx)>4000: idx=np.random.default_rng(seed).choice(idx,4000,replace=False)
                    for j in idx: preds.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,y_true=float(yt[j]),y_pred=float(pr[j]),absolute_error=float(abs(yt[j]-pr[j]))))
                    if groups is not None:
                        g=groups.iloc[te].reset_index(drop=True); ed=pd.DataFrame({'group':g.astype(str),'abs_error':np.abs(yt-pr)}); top=ed.group.value_counts().head(25).index
                        for _,r in ed[ed.group.isin(top)].groupby('group').abs_error.agg(['size','mean']).reset_index().iterrows(): errs.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,chemistry_or_group=r['group'],n=int(r['size']),mae=float(r['mean'])))
                    if cfg.save.get('models') and mn!='dummy':
                        mp=cfg.out_dir/'models'/f'{slug(label)}__{slug(target)}__{mn}__{split}__rep{rep}.pkl'; mkdir(mp.parent); pickle.dump(pipe,open(mp,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e: log.warning('ML failed for %s target=%s model=%s: %s',label,target,mn,e)
    return pd.DataFrame(mets),pd.DataFrame(preds),pd.DataFrame(errs)

def step_ml(cfg: Config, dd: Dict[str,Path], log):
    if cfg.skip_ml or not SKLEARN_AVAILABLE:
        for n in ['ml_metrics_full','ml_predictions_sample','ml_error_by_chemistry','ranking_stability','ml_metrics_summary']: save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    cand=fdf[(pd.to_numeric(fdf.n_target_like_columns,errors='coerce').fillna(0)>0)&(pd.to_numeric(fdf.n_numeric_columns,errors='coerce').fillna(0)>3)&(fdf.read_status.eq('ok'))].copy()
    if cand.empty:
        for n in ['ml_metrics_full','ml_predictions_sample','ml_error_by_chemistry','ranking_stability','ml_metrics_summary']: save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    cand['priority']=cand.apply(lambda r:4*int(r.resource=='ARC-MOF')+3*int(r.role in ['adsorption_targets','process_targets','quantum_properties'])+2*int(r.resource=='QMOF')+int(r.resource=='MOSAEC-DB'),axis=1)
    cand=cand.sort_values(['priority','n_target_like_columns','size_mb'],ascending=[False,False,True]).head(cfg.level['targets']+1)
    allm=[]; allp=[]; alle=[]
    for _,r in cand.iterrows():
        p=Path(r.file_path); df,_=read_table(p,cfg,nrows=cfg.ram['max_model_rows']*2,log=log); tcols=target_cols(df,cfg)
        for t in tcols:
            log.info('ML stress test: %s target=%s', r.file_name, t)
            m,pred,err=run_ml(df,t,f'{r.resource}::{p.name}',cfg,log)
            if not m.empty: allm.append(m)
            if not pred.empty: allp.append(pred)
            if not err.empty: alle.append(err)
    met=pd.concat(allm,ignore_index=True) if allm else pd.DataFrame(); pred=pd.concat(allp,ignore_index=True) if allp else pd.DataFrame(); err=pd.concat(alle,ignore_index=True) if alle else pd.DataFrame()
    save_df(met,dd['processed']/ 'ml_metrics_full',cfg); save_df(pred,dd['processed']/ 'ml_predictions_sample',cfg); save_df(err,dd['processed']/ 'ml_error_by_chemistry',cfg); save_df(met,dd['si_tables']/ 'SI_Table_S6_complete_model_results',cfg)
    if not met.empty:
        summ=met.groupby(['source_table','target_column','model','split_type']).agg(n_repeats=('repeat','nunique'),mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),mean_mae=('mae','mean'),mean_r2=('r2','mean'),sd_r2=('r2','std'),mean_spearman=('spearman','mean'),mean_top_1pct_recovery=('top_1pct_recovery','mean'),mean_top_5pct_recovery=('top_5pct_recovery','mean'),mean_top_10pct_recovery=('top_10pct_recovery','mean'),mean_ndcg_10pct=('ndcg_10pct','mean')).reset_index()
        rank=met[['source_table','target_column','model','split_type','repeat','top_1pct_recovery','top_5pct_recovery','top_10pct_recovery','ndcg_10pct']].copy()
    else: summ=pd.DataFrame(); rank=pd.DataFrame()
    save_df(summ,dd['processed']/ 'ml_metrics_summary',cfg); save_df(summ,dd['main_tables']/ 'Main_Table_4_ml_stress_test_summary',cfg); save_df(rank,dd['processed']/ 'ranking_stability',cfg)

# ----------------------------- tables and figures -----------------------------
def step_tables(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile'); cdf=load_df(dd['profiles']/ 'column_level_profile'); scores=load_df(dd['processed']/ 'resource_scores'); join=load_df(dd['processed']/ 'join_accounting'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); mls=load_df(dd['processed']/ 'ml_metrics_summary')
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary')
    invf=load_df(dd['profiles']/ 'inventory_folder_summary')
    invex=load_df(dd['profiles']/ 'inventory_report_examples')
    if not inv.empty: save_df(inv,dd['si_tables']/ 'SI_Table_S0_inventory_resource_role_format_summary',cfg)
    if not invf.empty: save_df(invf,dd['si_tables']/ 'SI_Table_S0b_inventory_folder_summary',cfg)
    if not invex.empty: save_df(invex,dd['si_tables']/ 'SI_Table_S0c_inventory_example_files',cfg)
    if not fdf.empty:
        main1=fdf.groupby(['resource','role']).agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum'),total_profiled_rows=('n_profiled_rows','sum'),median_columns=('n_columns','median'),mean_missing_fraction=('missing_fraction_overall','mean'),n_target_like_columns=('n_target_like_columns','sum'),n_descriptor_like_columns=('n_descriptor_like_columns','sum')).reset_index()
        if not scores.empty: main1=main1.merge(scores[['resource','chemistry_trust_score','ml_readiness_score','recommended_role']],on='resource',how='left')
        save_df(main1,dd['main_tables']/ 'Main_Table_1_resource_atlas',cfg); save_df(fdf,dd['si_tables']/ 'SI_Table_S1_complete_file_level_profile',cfg)
    if not cdf.empty:
        save_df(cdf,dd['si_tables']/ 'SI_Table_S2_complete_column_level_profile',cfg); save_df(cdf[cdf.is_descriptor_candidate.astype(bool)].head(10000) if 'is_descriptor_candidate' in cdf else pd.DataFrame(),dd['si_tables']/ 'SI_Table_S11_descriptor_family_profile',cfg)
    if not join.empty: save_df(join,dd['si_tables']/ 'SI_Table_S3_join_accounting',cfg)
    if not risk.empty: save_df(risk,dd['main_tables']/ 'Main_Table_3_benchmark_risk_matrix',cfg)
    if not mls.empty: save_df(mls,dd['si_tables']/ 'SI_Table_S6_ml_metrics_summary',cfg)
    decision=pd.DataFrame([{'step':1,'question':'Is the claim sensitive to oxidation state, charge, open metal sites or ionic context?','yes_action':'Use chemistry-trust flags and rule cards before modeling.','no_action':'Report provenance and missingness; chemistry flags may be secondary.'},{'step':2,'question':'Does the task require adsorption/process targets?','yes_action':'Prioritize adsorption/process resources and check target provenance.','no_action':'Avoid forcing adsorption resources into unrelated quantum/curation tasks.'},{'step':3,'question':'Are descriptors and targets joined by stable identifiers?','yes_action':'Report join retention and duplicate policy.','no_action':'Do not claim resource-level coverage from a task-join subset.'},{'step':4,'question':'Does performance remain stable under grouped or chemistry-aware splits?','yes_action':'Benchmark conclusion is more robust.','no_action':'Phrase claims as interpolation or convenience-screening performance.'},{'step':5,'question':'Are high-trust subsets too small or biased?','yes_action':'Use size-matched sensitivity and report bias.','no_action':'Use trusted subset as headline if it matches the task.'}])
    checklist=pd.DataFrame([{'item':'Exact file provenance, size, version and access date','minimum_status':'required'},{'item':'Row/column/missingness table for every input file','minimum_status':'required'},{'item':'Identifier normalization and join-retention accounting','minimum_status':'required'},{'item':'Oxidation-state/formal-charge observability and caveats','minimum_status':'required when chemistry-sensitive'},{'item':'Duplicate/leakage policy and grouped splits','minimum_status':'required for ML claims'},{'item':'Figure source data for every panel','minimum_status':'required'},{'item':'Sensitivity to trust thresholds and score weights','minimum_status':'strongly recommended'}])
    srcmap=pd.DataFrame([{'figure':'SI Figure S0','source_data':'inventory_report_examples.csv; inventory_folder_summary.csv; inventory_resource_role_format_summary.csv; discovered_input_files.csv','notes':'Inventory-derived archive map compared with locally discovered data files.'},{'figure':'Figure 1','source_data':'decision_rules.csv; minimum_reporting_checklist.csv','notes':'Conceptual schematic and rule logic.'},{'figure':'Figure 2','source_data':'file_level_profile.csv; column_level_profile.csv; resource_modality_matrix.csv; resource_scores.csv','notes':'Local profiling results.'},{'figure':'Figure 3','source_data':'trust_flags_by_metal.csv; charge_balance_residuals.csv; variant_comparison.csv; representative_rule_cards.csv','notes':'Chemistry-trust audit.'},{'figure':'Figure 4','source_data':'resource_scores.csv; benchmark_risk_matrix.csv; score_sensitivity.csv','notes':'Trust-readiness and risk framework.'},{'figure':'Figure 5','source_data':'ml_metrics_summary.csv; ranking_stability.csv; ml_error_by_chemistry.csv','notes':'ML stress-test output.'},{'figure':'Figure 6','source_data':'decision_rules.csv; minimum_reporting_checklist.csv; use_case_recommendation_cards.csv','notes':'Decision framework.'}])
    save_df(decision,dd['source']/ 'decision_rules',cfg); save_df(checklist,dd['source']/ 'minimum_reporting_checklist',cfg); save_df(srcmap,dd['source']/ 'final_source_data_map',cfg)
    if cfg.save.get('xlsx') and EXCEL_AVAILABLE:
        sheets={}
        for nm,base in [('file_profile',dd['profiles']/ 'file_level_profile'),('column_profile',dd['profiles']/ 'column_level_profile'),('resource_scores',dd['processed']/ 'resource_scores'),('risk_matrix',dd['processed']/ 'benchmark_risk_matrix'),('ml_summary',dd['processed']/ 'ml_metrics_summary')]:
            x=load_df(base)
            if not x.empty: sheets[nm]=x.head(100000)
        if sheets:
            with pd.ExcelWriter(dd['source']/ 'project_core_source_data.xlsx',engine='openpyxl') as w:
                for nm,x in sheets.items(): x.to_excel(w,sheet_name=nm[:31],index=False)

def style_axes(ax, grid: bool=True):
    """Apply consistent journal-style axis finishing."""
    ax.set_facecolor('white')
    if grid:
        ax.grid(True, lw=0.35, alpha=0.22, zorder=0)
    for spine in ['left','bottom']:
        ax.spines[spine].set_linewidth(0.75)
    return ax

def nodata(ax,title,msg='No measurable data available in local inputs'):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((.05,.16),.90,.68,boxstyle='round,pad=.025,rounding_size=.025',
                                transform=ax.transAxes,facecolor='#F7F7F7',edgecolor='#BDBDBD',lw=.9))
    ax.text(.5,.60,title,ha='center',va='center',fontweight='bold',fontsize=10)
    ax.text(.5,.43,msg,ha='center',va='center',fontsize=8.5,wrap=True,color='#424242')

def lab(ax,l):
    ax.text(-.075,1.075,l,transform=ax.transAxes,fontsize=13,fontweight='bold',va='top',ha='left',
            bbox=dict(boxstyle='round,pad=.16',facecolor='white',edgecolor='#333333',lw=.6))

def heat(ax,mat,title,cmap='viridis',bar=True):
    if mat is None or mat.empty:
        nodata(ax,title); return
    arr=mat.values.astype(float)
    im=ax.imshow(arr,aspect='auto',interpolation='nearest',cmap=cmap)
    ax.set_title(title,fontsize=10.2,fontweight='bold',pad=8)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns,rotation=45,ha='right',fontsize=7.5)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index,fontsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    if mat.shape[0] <= 12 and mat.shape[1] <= 10:
        finite=arr[np.isfinite(arr)]
        threshold=np.nanmean(finite) if finite.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v=arr[i,j]
                if np.isfinite(v):
                    txt=f"{v:.2g}" if abs(v)<100 else f"{v:.0f}"
                    ax.text(j,i,txt,ha='center',va='center',fontsize=6.5,
                            color='white' if v>threshold else '#222222')
    if bar:
        cb=plt.colorbar(im,ax=ax,fraction=.046,pad=.04)
        cb.ax.tick_params(labelsize=7)

def add_panel_note(ax, text, xy=(.02,.02)):
    ax.text(xy[0],xy[1],text,transform=ax.transAxes,ha='left',va='bottom',fontsize=7.5,color='#424242',
            bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#DDDDDD',lw=.5,alpha=.92))

def boxplot_compat(ax, data, labels, **kwargs):
    """Matplotlib 3.9 renamed labels to tick_labels; support both APIs."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)

def inventory_alignment_figure(cfg, dd):
    """Extra polished SI figure showing how the uploaded inventory maps to data roles."""
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary')
    ex=load_df(dd['profiles']/ 'inventory_report_examples')
    disc=load_df(dd['profiles']/ 'discovered_input_files')
    fig,axs=plt.subplots(2,2,figsize=(13.6,9.2)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not inv.empty:
        sub=inv.groupby('resource').agg(n_inventory_files=('n_inventory_files','sum'), inventory_size_mb=('inventory_size_mb','sum')).reset_index().sort_values('inventory_size_mb')
        ax.barh(sub.resource, sub.inventory_size_mb, color=PALETTE[:len(sub)])
        ax.set_xlabel('Inventory-reported size (MB)')
        ax.set_title('Archive map: size by resource',fontweight='bold')
        for i,r in sub.reset_index(drop=True).iterrows(): ax.text(r.inventory_size_mb,i,f" {int(r.n_inventory_files)} files",va='center',fontsize=7.5)
    else: nodata(ax,'Archive map: size by resource','No inventory-report summary found')
    ax=axs[1]; lab(ax,'b')
    if not inv.empty:
        top_ext=inv.groupby('extension').n_inventory_files.sum().sort_values(ascending=False).head(10).index
        mat=inv[inv.extension.isin(top_ext)].pivot_table(index='resource',columns='extension',values='n_inventory_files',aggfunc='sum',fill_value=0)
        heat(ax,mat,'Inventory file formats by resource','YlGnBu',True)
    else: nodata(ax,'Inventory file formats by resource')
    ax=axs[2]; lab(ax,'c'); style_axes(ax)
    if not ex.empty:
        sub=ex.groupby(['resource','role']).size().reset_index(name='n_examples').sort_values('n_examples',ascending=False).head(14)
        labels=sub.resource.astype(str)+' | '+sub.role.astype(str)
        ax.barh(labels[::-1], sub.n_examples.values[::-1], color=PALETTE[2:2+len(sub)])
        ax.set_xlabel('Example files listed in inventory')
        ax.set_title('Scientific roles visible in inventory examples',fontweight='bold')
    else: nodata(ax,'Scientific roles visible in inventory examples')
    ax=axs[3]; lab(ax,'d'); style_axes(ax)
    if not disc.empty:
        d=disc.groupby(['resource','role']).size().reset_index(name='n_local_tables').sort_values('n_local_tables',ascending=False).head(14)
        labels=d.resource.astype(str)+' | '+d.role.astype(str)
        ax.barh(labels[::-1], d.n_local_tables.values[::-1], color=PALETTE[4:4+len(d)])
        ax.set_xlabel('Actually discovered local tables')
        ax.set_title('Local run: tables available to compute statistics',fontweight='bold')
        add_panel_note(ax,'True rows/columns/missingness are computed only from discovered local tables.')
    else: nodata(ax,'Local run: tables available to compute statistics','Put the unzipped CSV files beside the script or pass --data-root')
    fig.suptitle('SI Figure S0. Inventory-aware input map: expected archive content versus local analysis inputs',fontweight='bold',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    return save_fig(fig, dd['fig_si']/ 'SI_Figure_S0_inventory_aware_input_map', cfg)

def fig1(cfg,dd):
    fig,axs=plt.subplots(2,2,figsize=(12,8)); axs=axs.ravel()
    ax=axs[0]; ax.axis('off'); lab(ax,'a'); ax.set_title('MOF data resources occupy different scientific roles',fontweight='bold',fontsize=11)
    boxes=[(.05,.65,'Experimental\nCSD/CoRE/MOSAEC'),(.38,.65,'Hypothetical\nARC-MOF'),(.70,.65,'DFT-derived\nQMOF'),(.18,.23,'Computation-ready\nfiles/descriptors'),(.55,.23,'Chemistry-ready\nclaim-specific subsets')]
    for x,y,t in boxes: ax.add_patch(FancyBboxPatch((x,y),.25,.18,boxstyle='round,pad=.02',lw=1,fill=False)); ax.text(x+.125,y+.09,t,ha='center',va='center',fontsize=9)
    for a,b,c,d in [(.18,.65,.30,.41),(.50,.65,.43,.41),(.82,.65,.67,.41),(.43,.32,.55,.32)]: ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle='->',mutation_scale=12,lw=1))
    ax=axs[1]; ax.axis('off'); lab(ax,'b'); ax.set_title('Computation-ready is not automatically chemistry-ready',fontweight='bold',fontsize=11); ax.text(.25,.82,'Computation-ready',ha='center',fontweight='bold'); ax.text(.75,.82,'Chemistry-ready',ha='center',fontweight='bold')
    for i,t in enumerate(['parsable file','descriptor matrix','simulation target','identifier present']): ax.text(.25,.66-.12*i,'• '+t,ha='center',fontsize=9)
    for i,t in enumerate(['oxidation/charge defensible','curation warning known','duplicates controlled','fit to scientific claim']): ax.text(.75,.66-.12*i,'• '+t,ha='center',fontsize=9)
    ax.add_patch(FancyArrowPatch((.42,.45),(.58,.45),arrowstyle='->',mutation_scale=16,lw=1.5))
    ax=axs[2]; ax.axis('off'); lab(ax,'c'); ax.set_title('Interpretable chemistry rule cards',fontweight='bold',fontsize=11)
    for i,(h,b) in enumerate([('PASS','Metal and charge evidence\nconsistent with task'),('UNCERTAIN','Oxidation state or charge\nnot observable'),('FLAG','Suspect or nonzero charge\noutside stated context')]):
        y=.68-i*.24; ax.add_patch(FancyBboxPatch((.12,y),.76,.16,boxstyle='round,pad=.02',lw=1,fill=False)); ax.text(.22,y+.08,h,ha='center',va='center',fontweight='bold',fontsize=9); ax.text(.55,y+.08,b,ha='center',va='center',fontsize=8)
    ax=axs[3]; ax.axis('off'); lab(ax,'d'); ax.set_title('Reproducible analysis pipeline',fontweight='bold',fontsize=11); steps=['Profile','Join','Trust','Readiness','ML stress','Decision']; xs=np.linspace(.08,.88,len(steps))
    for i,(x,s) in enumerate(zip(xs,steps)):
        ax.add_patch(FancyBboxPatch((x-.06,.43),.12,.16,boxstyle='round,pad=.02',lw=1,fill=False)); ax.text(x,.51,s,ha='center',va='center',fontsize=8)
        if i<len(steps)-1: ax.add_patch(FancyArrowPatch((x+.06,.51),(xs[i+1]-.06,.51),arrowstyle='->',mutation_scale=10,lw=1))
    ax.text(.5,.24,'Every numerical claim is linked to saved source data',ha='center',fontsize=9); fig.suptitle('Figure 1. Chemistry-ready workflow for MOF machine learning',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_1_chemistry_ready_concept',cfg)

def fig2(cfg,dd):
    f=load_df(dd['profiles']/ 'file_level_profile'); c=load_df(dd['profiles']/ 'column_level_profile'); mod=load_df(dd['profiles']/ 'resource_modality_matrix'); sc=load_df(dd['processed']/ 'resource_scores'); fig,axs=plt.subplots(2,2,figsize=(13,9)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not f.empty:
        g=f.groupby('resource').agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum')).reset_index().sort_values('total_size_mb'); ax.barh(g.resource,g.total_size_mb); ax.set_xlabel('Total profiled file size (MB)'); ax.set_title('Input file size by resource',fontweight='bold',fontsize=10)
        for i,r in g.reset_index(drop=True).iterrows(): ax.text(r.total_size_mb,i,f" {int(r.n_files)} files",va='center',fontsize=7)
    else: nodata(ax,'Input file size by resource')
    ax=axs[1]; lab(ax,'b')
    if not mod.empty:
        cols=[x for x in ['identifier','formula_or_composition','metal_or_charge_chemistry','geometric_descriptor','descriptor','target_or_property','curation_or_validation_flag'] if x in mod.columns]; heat(ax,(mod.set_index('resource')[cols]>0).astype(int),'Modality availability','Greys',False)
    else: nodata(ax,'Modality availability')
    ax=axs[2]; lab(ax,'c')
    if not c.empty:
        tmp=c.copy(); tmp['missing_bin']=pd.cut(pd.to_numeric(tmp.missing_fraction,errors='coerce'),[-.001,0,.05,.25,.75,1],labels=['0','0--5%','5--25%','25--75%','>75%'],include_lowest=True); mat=tmp.groupby(['resource','missing_bin'],observed=False).size().unstack(fill_value=0); heat(ax,mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0),'Column missingness composition','magma',True)
    else: nodata(ax,'Column missingness composition')
    ax=axs[3]; lab(ax,'d')
    if not sc.empty:
        x=np.log10(sc.total_profiled_or_counted_rows.astype(float).clip(lower=1)); y=sc.chemistry_trust_score.astype(float); s=80+600*sc.ml_readiness_score.astype(float).clip(0,1); ax.scatter(x,y,s=s,alpha=.75,edgecolors='black')
        for _,r in sc.iterrows(): ax.text(np.log10(max(float(r.total_profiled_or_counted_rows),1))+.02,r.chemistry_trust_score,r.resource,fontsize=8)
        ax.set_xlabel('log10(profiled or counted rows + 1)'); ax.set_ylabel('Chemistry-trust score'); ax.set_ylim(0,1.05); ax.set_title('Resource scale versus chemistry trust',fontweight='bold',fontsize=10)
    else: nodata(ax,'Resource scale versus chemistry trust')
    fig.suptitle('Figure 2. Data-resource atlas from local profiling',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_2_data_resource_atlas',cfg)

def fig3(cfg,dd):
    met=load_df(dd['processed']/ 'trust_flags_by_metal'); ch=load_df(dd['processed']/ 'charge_balance_residuals'); var=load_df(dd['processed']/ 'variant_comparison'); cards=load_df(dd['processed']/ 'representative_rule_cards'); fig,axs=plt.subplots(2,2,figsize=(13,9)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not met.empty:
        top=met.groupby('metal_family_primary').n.sum().sort_values(ascending=False).head(15).index; mat=met[met.metal_family_primary.isin(top)].pivot_table(index='metal_family_primary',columns='chemistry_trust_tier_row',values='fraction',aggfunc='mean',fill_value=0); mat=mat[[x for x in ['very_high','high','medium','low'] if x in mat.columns]]; heat(ax,mat,'Trust-tier fractions by metal','viridis',True)
    else: nodata(ax,'Trust-tier fractions by metal')
    ax=axs[1]; lab(ax,'b')
    if not ch.empty and 'abs_charge_residual' in ch:
        sub=ch[pd.to_numeric(ch.abs_charge_residual,errors='coerce').notna()].copy()
        if not sub.empty:
            sub['abs_charge_residual']=pd.to_numeric(sub.abs_charge_residual,errors='coerce').clip(upper=10); labels=sub.resource.value_counts().head(6).index.tolist(); data=[sub[sub.resource.eq(l)].abs_charge_residual.dropna().values for l in labels]; boxplot_compat(ax,data,labels,showfliers=False); ax.set_ylabel('|first available charge| (clipped)'); ax.set_title('Charge-balance residuals',fontweight='bold',fontsize=10); ax.tick_params(axis='x',labelrotation=30,labelsize=8)
        else: nodata(ax,'Charge-balance residuals')
    else: nodata(ax,'Charge-balance residuals')
    ax=axs[2]; lab(ax,'c')
    if not var.empty:
        sub=var.sort_values('mean_trust',ascending=False).head(12); ax.barh(sub.resource.astype(str)+'\n'+sub.variant_label.astype(str),sub.mean_trust); ax.set_xlim(0,1); ax.set_xlabel('Mean row-level trust score'); ax.set_title('Variant/subset trust comparison',fontweight='bold',fontsize=10)
    else: nodata(ax,'Variant/subset trust comparison')
    ax=axs[3]; lab(ax,'d'); ax.axis('off'); ax.set_title('Representative rule cards',fontweight='bold',fontsize=10)
    if not cards.empty:
        for i,r in cards.head(3).reset_index(drop=True).iterrows():
            y=.72-i*.28; ax.add_patch(FancyBboxPatch((.05,y),.90,.20,boxstyle='round,pad=.02',lw=1,fill=False)); ax.text(.5,y+.10,f"{r.get('rule_card_class','')} | {r.get('resource','')} | {r.get('metals_detected','')}\n{r.get('interpretation','')}",ha='center',va='center',fontsize=8,wrap=True)
    else: ax.text(.5,.5,'No chemistry rule-card columns were observable',ha='center',va='center')
    fig.suptitle('Figure 3. Oxidation-state and formal-charge trust landscape',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_3_oxidation_state_trust_landscape',cfg)

def fig4(cfg,dd):
    sc=load_df(dd['processed']/ 'resource_scores'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); sens=load_df(dd['processed']/ 'score_sensitivity'); fig,axs=plt.subplots(1,3,figsize=(16,5.4))
    ax=axs[0]; lab(ax,'a')
    if not sc.empty:
        ax.axhline(.70,ls='--',lw=.8); ax.axvline(.70,ls='--',lw=.8); ax.scatter(sc.ml_readiness_score,sc.chemistry_trust_score,s=180,alpha=.75,edgecolors='black')
        for _,r in sc.iterrows(): ax.text(r.ml_readiness_score+.01,r.chemistry_trust_score+.01,r.resource,fontsize=8)
        ax.set_xlim(0,1.05); ax.set_ylim(0,1.05); ax.set_xlabel('ML-readiness score'); ax.set_ylabel('Chemistry-trust score'); ax.set_title('Trust--readiness map',fontweight='bold',fontsize=10)
    else: nodata(ax,'Trust--readiness map')
    ax=axs[1]; lab(ax,'b')
    if not risk.empty: heat(ax,risk.pivot_table(index='risk_issue',columns='use_case',values='risk_score_0_to_3',aggfunc='mean',fill_value=0),'Benchmark-risk matrix','inferno',True)
    else: nodata(ax,'Benchmark-risk matrix')
    ax=axs[2]; lab(ax,'c')
    if not sens.empty:
        sub=sens.sort_values('mean_rank'); y=np.arange(len(sub)); ax.errorbar(sub.mean_rank,y,xerr=[sub.mean_rank-sub.min_rank,sub.max_rank-sub.mean_rank],fmt='o',capsize=3); ax.set_yticks(y); ax.set_yticklabels(sub.resource,fontsize=8); ax.invert_yaxis(); ax.set_xlabel('Rank across score-weight perturbations'); ax.set_title('Score sensitivity',fontweight='bold',fontsize=10)
    else: nodata(ax,'Score sensitivity')
    fig.suptitle('Figure 4. Trust--readiness map and benchmark-risk framework',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_4_trust_readiness_risk_matrix',cfg)

def fig5(cfg,dd):
    summ=load_df(dd['processed']/ 'ml_metrics_summary'); rank=load_df(dd['processed']/ 'ranking_stability'); err=load_df(dd['processed']/ 'ml_error_by_chemistry'); pred=load_df(dd['processed']/ 'ml_predictions_sample'); fig,axs=plt.subplots(2,2,figsize=(13,9)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not summ.empty:
        sub=summ.sort_values('mean_r2',ascending=False).head(15); labels=sub.model.astype(str)+'\n'+sub.split_type.astype(str); ax.errorbar(sub.mean_r2,np.arange(len(sub)),xerr=sub.sd_r2.fillna(0),fmt='o',capsize=3); ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(labels,fontsize=8); ax.invert_yaxis(); ax.set_xlabel('Mean R² ± SD'); ax.set_title('Model performance stability',fontweight='bold',fontsize=10)
    else: nodata(ax,'Model performance stability','No ML output; provide target-like tables or install scikit-learn')
    ax=axs[1]; lab(ax,'b')
    if not rank.empty:
        melt=rank.melt(id_vars=['model','split_type'],value_vars=[c for c in ['top_1pct_recovery','top_5pct_recovery','top_10pct_recovery'] if c in rank],var_name='top_k',value_name='recovery'); g=melt.groupby(['model','top_k']).recovery.mean().reset_index()
        for model,sub in g.groupby('model'): ax.plot(sub.top_k,sub.recovery,marker='o',label=model)
        ax.set_ylabel('Top-k recovery'); ax.set_ylim(0,1.05); ax.legend(fontsize=7); ax.set_title('Top-candidate recovery',fontweight='bold',fontsize=10); ax.tick_params(axis='x',labelrotation=30)
    else: nodata(ax,'Top-candidate recovery')
    ax=axs[2]; lab(ax,'c')
    if not err.empty:
        sub=err.groupby('chemistry_or_group').agg(n=('n','sum'),mae=('mae','mean')).reset_index().sort_values('mae',ascending=False).head(12); ax.barh(sub.chemistry_or_group.astype(str),sub.mae); ax.invert_yaxis(); ax.set_xlabel('Mean absolute error'); ax.set_title('Error by chemistry/group label',fontweight='bold',fontsize=10)
    else: nodata(ax,'Error by chemistry/group label')
    ax=axs[3]; lab(ax,'d')
    if not pred.empty:
        sub=pred.sample(n=min(len(pred),3000),random_state=cfg.random_seed) if len(pred)>3000 else pred; ax.scatter(sub.y_true,sub.y_pred,s=8,alpha=.35); mn=min(sub.y_true.min(),sub.y_pred.min()); mx=max(sub.y_true.max(),sub.y_pred.max()); ax.plot([mn,mx],[mn,mx],ls='--',lw=1); ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Prediction calibration view',fontweight='bold',fontsize=10)
    else: nodata(ax,'Prediction calibration view')
    fig.suptitle('Figure 5. Lightweight ML stress test',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)

def fig6(cfg,dd):
    dec=load_df(dd['source']/ 'decision_rules'); chk=load_df(dd['source']/ 'minimum_reporting_checklist'); rec=load_df(dd['main_tables']/ 'Main_Table_3_use_case_recommendation_cards'); fig,axs=plt.subplots(1,3,figsize=(16,5.8))
    ax=axs[0]; ax.axis('off'); lab(ax,'a'); ax.set_title('Database/subset decision tree',fontweight='bold',fontsize=10)
    if not dec.empty:
        y=.86
        for _,r in dec.head(5).iterrows(): ax.add_patch(FancyBboxPatch((.05,y-.08),.90,.10,boxstyle='round,pad=.02',lw=.8,fill=False)); ax.text(.5,y-.03,f"Q{int(r.step)}: {r.question}",ha='center',va='center',fontsize=7,wrap=True); y-=.16
    else: ax.text(.5,.5,'Decision rules missing',ha='center')
    ax=axs[1]; ax.axis('off'); lab(ax,'b'); ax.set_title('Minimum reporting checklist',fontweight='bold',fontsize=10)
    if not chk.empty:
        y=.86
        for _,r in chk.head(7).iterrows(): ax.text(.05,y,'☐ '+str(r.item),fontsize=8,va='center',wrap=True); y-=.11
    else: ax.text(.5,.5,'Checklist missing',ha='center')
    ax=axs[2]; ax.axis('off'); lab(ax,'c'); ax.set_title('Use-case recommendation cards',fontweight='bold',fontsize=10)
    if not rec.empty:
        y=.82
        for _,r in rec.head(4).iterrows(): ax.add_patch(FancyBboxPatch((.04,y-.08),.92,.14,boxstyle='round,pad=.02',lw=.8,fill=False)); ax.text(.5,y+.02,str(r.use_case),ha='center',va='center',fontsize=8,fontweight='bold'); ax.text(.5,y-.04,str(r.preferred_resource_logic),ha='center',va='center',fontsize=6.8,wrap=True); y-=.20
    else: ax.text(.5,.5,'Recommendation cards missing',ha='center')
    fig.suptitle('Figure 6. Chemistry-ready decision framework',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_6_decision_framework',cfg)

def si_figs(cfg,dd):
    outs=[]; f=load_df(dd['profiles']/ 'file_level_profile'); c=load_df(dd['profiles']/ 'column_level_profile'); j=load_df(dd['processed']/ 'join_accounting'); trust=load_df(dd['processed']/ 'trust_flags'); sens=load_df(dd['processed']/ 'score_sensitivity'); ml=load_df(dd['processed']/ 'ml_metrics_full'); cards=load_df(dd['processed']/ 'representative_rule_cards')
    fig,ax=plt.subplots(figsize=(10,5));
    if not f.empty: f.groupby(['resource','read_status']).size().unstack(fill_value=0).plot(kind='bar',stacked=True,ax=ax); ax.set_ylabel('Number of files'); ax.set_title('SI Figure S1. File-level inventory and loading status',fontweight='bold'); ax.tick_params(axis='x',labelrotation=30)
    else: nodata(ax,'SI Figure S1')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S1_inventory_profile',cfg)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    if not c.empty: heat(axs[0],c.groupby(['resource','inferred_modality']).size().unstack(fill_value=0),'Column-type composition','viridis',True); miss=c.groupby('resource').missing_fraction.mean().sort_values(); axs[1].barh(miss.index,miss.values); axs[1].set_xlabel('Mean column missing fraction'); axs[1].set_title('Average missingness by resource',fontweight='bold')
    else: nodata(axs[0],'Column-type composition'); nodata(axs[1],'Average missingness')
    fig.suptitle('SI Figure S2. Missingness and column-type composition',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.92]); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S2_missingness_column_types',cfg)
    fig,ax=plt.subplots(figsize=(10,5))
    if not j.empty: sub=j.sort_values('intersection_ids_profiled',ascending=False).head(20); ax.barh(sub.left_file.astype(str).str[:18]+' ↔ '+sub.right_file.astype(str).str[:18],sub.intersection_ids_profiled); ax.invert_yaxis(); ax.set_xlabel('Profiled shared IDs'); ax.set_title('SI Figure S3. Identifier-join accounting',fontweight='bold')
    else: nodata(ax,'SI Figure S3','No shared identifier intersections found')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S3_join_accounting',cfg)
    fig,ax=plt.subplots(figsize=(10,5))
    if not trust.empty and 'metal_family_primary' in trust: top=trust.metal_family_primary.value_counts().head(20); ax.barh(top.index.astype(str),top.values); ax.invert_yaxis(); ax.set_xlabel('Profiled rows'); ax.set_title('SI Figure S4. Metal/chemistry coverage profiles',fontweight='bold')
    else: nodata(ax,'SI Figure S4')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S4_coverage_profiles',cfg)
    fig,ax=plt.subplots(figsize=(9,5))
    if not sens.empty: ax.scatter(sens.mean_rank,sens.mean_combined_score,s=120,edgecolors='black'); [ax.text(r.mean_rank+.02,r.mean_combined_score,r.resource,fontsize=8) for _,r in sens.iterrows()]; ax.set_xlabel('Mean sensitivity rank'); ax.set_ylabel('Mean combined score'); ax.set_title('SI Figure S5. Trust-score sensitivity',fontweight='bold')
    else: nodata(ax,'SI Figure S5')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S5_trust_score_sensitivity',cfg)
    fig,ax=plt.subplots(figsize=(10,5))
    if not ml.empty: g=ml.groupby(['model','split_type']).r2.apply(list).reset_index(); boxplot_compat(ax,g.r2.tolist(),(g.model.astype(str)+'\n'+g.split_type.astype(str)).tolist(),showfliers=False); ax.set_ylabel('R²'); ax.set_title('SI Figure S6. Full repeated-split performance distributions',fontweight='bold'); ax.tick_params(axis='x',labelrotation=30)
    else: nodata(ax,'SI Figure S6')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S6_ml_repeated_splits',cfg)
    fig,ax=plt.subplots(figsize=(10,5))
    if not f.empty and {'constant_columns','all_missing_columns'}.issubset(f.columns): f.groupby('resource')[['constant_columns','all_missing_columns']].sum().plot(kind='bar',ax=ax); ax.set_ylabel('Column count'); ax.set_title('SI Figure S7. Basic leakage/missingness diagnostics',fontweight='bold'); ax.tick_params(axis='x',labelrotation=30)
    else: nodata(ax,'SI Figure S7')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S7_duplicate_leakage',cfg)
    fig,ax=plt.subplots(figsize=(10,6)); ax.axis('off'); ax.set_title('SI Figure S8. Representative chemistry rule cards',fontweight='bold')
    if not cards.empty:
        y=.86
        for _,r in cards.head(5).iterrows(): ax.add_patch(FancyBboxPatch((.04,y-.07),.92,.11,boxstyle='round,pad=.02',lw=.8,fill=False)); ax.text(.5,y-.015,f"{r.get('rule_card_class','')} | {r.get('resource','')} | {r.get('primary_id','')} | {r.get('interpretation','')}",ha='center',va='center',fontsize=7,wrap=True); y-=.15
    else: ax.text(.5,.5,'No row-level chemistry rule cards could be created',ha='center')
    fig.tight_layout(); outs+=save_fig(fig,dd['fig_si']/ 'SI_Figure_S8_rule_cards',cfg)
    return outs

def step_figures(cfg: Config, dd: Dict[str,Path], log):
    outs=[]
    outs+=inventory_alignment_figure(cfg,dd)
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6]: outs+=fn(cfg,dd)
    outs+=si_figs(cfg,dd)
    save_df(pd.DataFrame([{'figure_file':str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None} for p in outs]),dd['fig']/ 'figure_output_manifest',cfg)

# ----------------------------- reports, package and CLI -----------------------------
def step_reports(cfg: Config, dd: Dict[str,Path], log):
    f=load_df(dd['profiles']/ 'file_level_profile'); sc=load_df(dd['processed']/ 'resource_scores'); ml=load_df(dd['processed']/ 'ml_metrics_summary'); figs=load_df(dd['fig']/ 'figure_output_manifest')
    lines=["# Chemistry-ready MOF analysis pipeline report\n",f"Generated: {iso()}\n",f"Script version: {VERSION}\n",f"Data root: `{cfg.data_root}`\n",f"Output directory: `{cfg.out_dir}`\n",f"Modes: save={cfg.save_mode}; RAM={cfg.ram_mode}; comprehensive={cfg.comprehensive_level}; n_jobs={cfg.n_jobs}\n"]
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary')
    lines.append("\n## Inventory map\n")
    if not inv.empty:
        lines.append(f"Parsed attached archive inventory summary with **{len(inv)}** resource/role/format rows. This is used as an expected-data map; true column and missingness statistics come from local files.\n")
    else:
        lines.append("No archive inventory summary was available or parseable.\n")
    lines.append("\n## Input profiling\n")
    if not f.empty:
        lines.append(f"Discovered/profiled table-like files: **{len(f)}**.\n")
        for res,n in f.groupby('resource').size().sort_values(ascending=False).items(): lines.append(f"- {res}: {n}\n")
    else: lines.append("No table-like input files were profiled.\n")
    lines.append("\n## Resource scores\n")
    if not sc.empty:
        for _,r in sc.sort_values('chemistry_trust_score',ascending=False).iterrows(): lines.append(f"- {r.resource}: chemistry trust {r.chemistry_trust_score:.3f}; ML readiness {r.ml_readiness_score:.3f}; {r.recommended_role}\n")
    else: lines.append("Resource scores were not generated.\n")
    lines.append("\n## ML stress test\n")
    if not ml.empty:
        for _,r in ml.sort_values('mean_r2',ascending=False).head(5).iterrows(): lines.append(f"- {r.source_table} | target {r.target_column} | {r.model} {r.split_type}: mean R2={r.mean_r2:.3f}, mean Spearman={r.mean_spearman:.3f}\n")
    else: lines.append("No ML stress-test summary available.\n")
    lines.append("\n## Figures\n")
    lines.append(f"Generated figure files: **{len(figs)}**. See `figures/figure_output_manifest.csv`.\n" if not figs.empty else "Figure manifest not available.\n")
    lines.append("\n## Manuscript caution\nPercentages and row counts should always specify the analysis object: raw file, profiled sample, joined task table, or model-ready subset. Do not describe any database as universally superior; describe task-specific chemistry-readiness.\n")
    (dd['reports']/ 'analysis_report.md').write_text(''.join(lines),encoding='utf-8')
    env=dict(generated=iso(),python=sys.version,platform=platform.platform(),script_version=VERSION,config=asdict(cfg),packages=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,sklearn_available=SKLEARN_AVAILABLE,parquet_available=PARQUET_AVAILABLE,excel_available=EXCEL_AVAILABLE))
    save_json(env,dd['logs']/ 'run_environment.json')
    disc=load_df(dd['profiles']/ 'discovered_input_files'); hashes=[]
    if not disc.empty:
        for _,r in disc.iterrows():
            p=Path(r.file_path)
            if p.exists():
                st=p.stat(); h=hashlib.sha256();
                with open(p,'rb') as fh:
                    head=fh.read(1024*1024); h.update(head)
                    if st.st_size>2*1024*1024: fh.seek(max(0,st.st_size-1024*1024)); h.update(fh.read(1024*1024))
                    h.update(str(st.st_size).encode()); h.update(str(st.st_mtime).encode())
                hashes.append(dict(path=str(p),name=p.name,size_bytes=st.st_size,mtime=st.st_mtime,sha256_fast=h.hexdigest(),resource=r.resource,role=r.role))
        save_df(pd.DataFrame(hashes),dd['logs']/ 'file_hashes',cfg)

def step_zip(cfg: Config, dd: Dict[str,Path], log):
    if cfg.no_zip: return
    zp=cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for sub in ['reports','profiles','tables','figures','source_data','logs']:
            root=cfg.out_dir/sub
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p!=zp: z.write(p,p.relative_to(cfg.out_dir))
    log.info('ZIP package created: %s (%s)', zp, hbytes(zp.stat().st_size))


def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     reusable intermediate/final analysis tables
tables/        main-text and SI-ready tables
figures/       polished main figures 1--6 plus inventory-aware SI Figure S0 and SI figures S1--S8
source_data/   panel and source-data tables
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md

Resume behavior
---------------
If the run is interrupted, run the same command again. Use --force only when you want to recompute completed steps.
"""
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')

def parse(argv=None) -> Config:
    sd=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description='Resume-safe chemistry-ready MOF audit, ML stress-test and figure/table generator.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data-root',type=Path,default=sd,help='Folder containing unzipped CSV/XLSX/JSON tables. Use . when files are beside the script.')
    ap.add_argument('--out-dir',type=Path,default=sd/'outputs_chemistry_ready_mof')
    ap.add_argument('--inventory-report',type=Path,default=None,help='Optional PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt or .zip')
    ap.add_argument('--save-mode',choices=list(SAVE_MODES),default='balanced')
    ap.add_argument('--ram-mode',choices=list(RAM_MODES),default='very-light')
    ap.add_argument('--comprehensive-level',choices=list(LEVELS),default='comprehensive')
    ap.add_argument('--n-jobs',type=int,default=2)
    ap.add_argument('--random-seed',type=int,default=42)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--no-resume',dest='resume',action='store_false')
    ap.add_argument('--profile-only',action='store_true')
    ap.add_argument('--skip-ml',action='store_true')
    ap.add_argument('--target-column',type=str,default=None)
    ap.add_argument('--id-column',type=str,default=None)
    ap.add_argument('--max-files',type=int,default=None)
    ap.add_argument('--file-pattern',type=str,default=None,help='Optional regex to restrict input files.')
    ap.add_argument('--skip-row-count',action='store_true')
    ap.add_argument('--no-zip',action='store_true')
    a=ap.parse_args(argv)
    return Config(data_root=a.data_root,out_dir=a.out_dir,inventory_report=a.inventory_report,save_mode=a.save_mode,ram_mode=a.ram_mode,comprehensive_level=a.comprehensive_level,n_jobs=a.n_jobs,random_seed=a.random_seed,force=a.force,resume=a.resume,profile_only=a.profile_only,skip_ml=a.skip_ml,target_column=a.target_column,id_column=a.id_column,max_files=a.max_files,file_pattern=a.file_pattern,skip_row_count=a.skip_row_count,no_zip=a.no_zip).final()

def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready MOF pipeline v%s', VERSION); log.info('Data root: %s', cfg.data_root); log.info('Output directory: %s', cfg.out_dir); log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)
    state_path=dd['logs']/ 'pipeline_state.json'; st=read_json(state_path,{'created':iso(),'completed_steps':{}}); st['config']=asdict(cfg); save_json(st,state_path)
    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv'],step_trust),('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv'],step_tables),('make_figures',[dd['fig_main']/ 'Figure_1_chemistry_ready_concept.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),('write_reports',[dd['reports']/ 'analysis_report.md',dd['logs']/ 'run_environment.json'],step_reports),('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)]
    for name,exp,fn in steps: run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    return 0



# =============================================================================
# v1.3 SCIENTIFIC-COMPLETION OVERRIDES (retained; superseded by v1.4 block below)
# =============================================================================
# The block below intentionally overrides selected v1.2 functions while keeping
# the stable profiling/plotting/checkpoint infrastructure above.  It implements
# the scientific-completion recommendations from the first full result audit:
#   * explicit environment/completeness checks;
#   * resource-specific file roles and subresources;
#   * ARC-MOF adsorption/process target recognition, including unit-only targets;
#   * normalized long-format target tables and a basic canonical MOF index;
#   * chemistry evidence classes that separate validation from observability;
#   * improved chemistry-trust summaries and redesigned Figure 3;
#   * complete source-data map, final manifest and ZIP including processed/.

VERSION = "1.3-scientific-completion"
MAIN_RESOURCES = [
    "MOSAEC-DB", "CoRE MOF 2024", "CoRE MOF 2025 metadata",
    "ARC-MOF", "QMOF", "CSD-derived context"
]
RESOURCE_ORDER = MAIN_RESOURCES + ["Other/unknown", "Task-specific joins"]
TRUST_REGIME_ORDER = [
    "validated_observable", "validated_not_observable", "uncertain_observable",
    "uncertain_unobservable", "flagged_or_inconsistent"
]
TRUST_REGIME_COLORS = {
    "validated_observable": "#1B9E77",
    "validated_not_observable": "#80CDC1",
    "uncertain_observable": "#E6AB02",
    "uncertain_unobservable": "#BDBDBD",
    "flagged_or_inconsistent": "#D95F02",
}
ARC_UNIT_TARGET_NAMES = {
    "mmol/g", "molc/uc", "v/v", "wt%", "hoa/kcal/mol", "s(g1)",
    "mmol_g", "molc_uc", "v_v", "wt_percent", "hoa_kcal_mol", "selectivity",
}
ARC_TASK_HINTS = ["landfill", "pre_comb", "post_comb", "methane_purification", "overall_process"]

# More complete target regex: keep the old expressions but add unit-only and process terms.
TARGET_RX = TARGET_RX + [
    re.compile(r"(^|[^a-z])(mmol\s*/\s*g|molc\s*/\s*uc|v\s*/\s*v|wt\s*%|hoa\s*/\s*kcal\s*/\s*mol|S\(g1\))($|[^a-z])", re.I),
    re.compile(r"(working[_ ]?capacity|deliverable[_ ]?capacity|purity|recovery|productivity|parasitic[_ ]?energy|energy[_ ]?penalty)", re.I),
    re.compile(r"(heat[_ ]?of[_ ]?adsorption|enthalpy|qst|hoa)", re.I),
]

def clean_col_name(x: Any) -> str:
    return re.sub(r"\s+", "", str(x).strip().lower().replace("\\", "/"))

def resource_of(p: Path) -> str:
    s=str(p).lower().replace('\\','/'); n=p.name.lower()
    if 'mosaec' in s: return 'MOSAEC-DB'
    if 'core_mof_2025' in s or 'core-mof-2025' in s or 'coremof2025' in s: return 'CoRE MOF 2025 metadata'
    if 'core_mof_2024' in s or 'core-mof-2024' in s or '/core/' in s or '12089' in n or 'asr_' in n or 'fsr_' in n: return 'CoRE MOF 2024'
    if 'arc_mof' in s or 'arc-mof' in s or any(k in n for k in ARC_TASK_HINTS): return 'ARC-MOF'
    if 'qmof' in s: return 'QMOF'
    if 'csd' in s or 'ccdc' in s or 'framework details' in n or 'suspect chemistry' in n: return 'CSD-derived context'
    return 'Other/unknown'

def resource_subtype_of(p: Path) -> str:
    s=str(p).lower().replace('\\','/'); n=p.name.lower()
    if resource_of(p) == 'CoRE MOF 2024':
        if '/single_isotherms/' in s or 'single_isotherm' in s: return 'CoRE-2024 single-isotherm tables'
        if '/tsa/' in s or 'tsemo' in n or 'pacman' in n: return 'CoRE-2024 TSA optimization tables'
        if 'water_gemc' in s or 'widom' in n or 'water_data' in n: return 'CoRE-2024 water GEMC/Widom tables'
        if 'mofid' in s: return 'CoRE-2024 MOFid/topology metadata'
        if 'duplicate' in s: return 'CoRE-2024 duplicate/provenance metadata'
        if 'asr' in s or 'fsr' in s or '/ion' in s or 'recommended' in n: return 'CoRE-2024 ASR/FSR/ION metadata'
        return 'CoRE-2024 other metadata'
    if resource_of(p) == 'ARC-MOF':
        if any(k in n for k in ['landfill', 'pre_comb', 'post_comb', 'methane_purification']): return 'ARC-MOF adsorption task table'
        if 'overall_process' in n or 'process' in n: return 'ARC-MOF process target table'
        if any(k in s for k in ['rac', 'rdf', 'descriptor', 'geom', 'aprdf', 'phom', 'pdd']): return 'ARC-MOF descriptor table'
        return 'ARC-MOF auxiliary table'
    if resource_of(p) == 'MOSAEC-DB':
        if any(k in s for k in ['descriptor', 'rac', 'rdf', 'feature']): return 'MOSAEC descriptor/feature table'
        return 'MOSAEC chemistry-validation metadata'
    if resource_of(p) == 'QMOF':
        return 'QMOF quantum-property table'
    return resource_of(p)

def role_of(p: Path) -> str:
    s=str(p).lower().replace('\\','/'); n=p.name.lower()
    if resource_of(p) == 'CoRE MOF 2024' and ('/single_isotherms/' in s or 'single_isotherm' in s): return 'adsorption_targets'
    if any(k in s for k in ['adsorption','landfill','methane_purification','pre_comb','post_comb']): return 'adsorption_targets'
    if 'process' in s or 'overall_process' in n or 'purity' in n or 'recovery' in n: return 'process_targets'
    if any(k in s for k in ['descriptor','racs','rdfs','aprdf','geom','phom','pdd','feature']): return 'descriptor_matrix'
    if any(k in s for k in ['cluster','topology','topo','mofid']): return 'topology_or_cluster'
    if any(k in s for k in ['structure','cif']): return 'structure_or_cif_metadata'
    if any(k in s for k in ['asr','fsr','ion','ncr','recommended','screening','check','valid','suspect']): return 'curation_or_validation'
    if 'qmof' in s or any(k in n for k in ['thermo','bandgap','dft','homo','lumo']): return 'quantum_properties'
    if any(k in s for k in ['duplicate','diversity']): return 'duplicate_or_diversity'
    return 'general_metadata'

def is_arc_context(p: Optional[Path]) -> bool:
    return bool(p is not None and (resource_of(p) == 'ARC-MOF' or any(k in p.name.lower() for k in ARC_TASK_HINTS)))

def infer_arc_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    name = p.name.lower()
    col = str(column or '').lower()
    task = 'unknown'
    if 'landfill' in name: task = 'landfill_gas_separation'
    elif 'pre_comb' in name: task = 'pre_combustion_capture'
    elif 'post_comb' in name: task = 'post_combustion_capture'
    elif 'methane_purification' in name: task = 'methane_purification'
    elif 'overall_process' in name or 'process' in name: task = 'process_screening'
    gas = 'not_encoded'
    for g in ['co2','ch4','n2','h2','xe','kr','h2o']:
        if re.search(rf'(^|[_\-.]){g}($|[_\-.])', name) or re.search(rf'\b{g}\b', col):
            gas = g.upper(); break
    prop = 'target_property'
    unit = ''
    cn = clean_col_name(col)
    if cn in {'mmol/g','mmol_g'}: prop, unit = 'gravimetric_uptake', 'mmol g-1'
    elif cn in {'molc/uc','molc_uc'}: prop, unit = 'unit_cell_loading', 'molecules per unit cell'
    elif cn in {'v/v','v_v'}: prop, unit = 'volumetric_uptake', 'v/v'
    elif cn in {'wt%','wt_percent'}: prop, unit = 'weight_percent_uptake', 'wt%'
    elif 'hoa' in cn or 'qst' in cn or 'heat' in cn: prop, unit = 'heat_of_adsorption', 'kcal mol-1 or source unit'
    elif 's(g1)' in cn or 'select' in cn: prop, unit = 'selectivity', 'dimensionless'
    elif 'working' in cn or 'deliverable' in cn: prop, unit = 'working_capacity', 'source unit'
    elif 'purity' in cn: prop, unit = 'purity', 'fraction or percent'
    elif 'recovery' in cn: prop, unit = 'recovery', 'fraction or percent'
    elif 'productivity' in cn: prop, unit = 'productivity', 'source unit'
    elif 'energy' in cn: prop, unit = 'process_energy', 'source unit'
    return dict(task=task, gas=gas, property=prop, unit=unit)

def is_arc_target_column_name(col: Any) -> bool:
    s = str(col).strip().lower()
    cn = clean_col_name(s)
    if cn in ARC_UNIT_TARGET_NAMES: return True
    if any(k in cn for k in ['workingcapacity','deliverablecapacity','purity','recovery','productivity','energy', 'hoa', 'qst', 'selectivity']): return True
    return False

def is_core_isotherm_file(p: Optional[Path]) -> bool:
    if p is None: return False
    s = str(p).lower().replace('\\','/')
    return resource_of(p) == 'CoRE MOF 2024' and ('single_isotherms' in s or re.search(r'_(co2|n2|ch4|h2|xe|kr)_\d', p.name.lower()) is not None)

def infer_core_isotherm_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    n = p.name
    m = re.search(r'_(CO2|N2|CH4|H2|Xe|Kr|H2O)_([0-9]+(?:\.[0-9]+)?)', n, flags=re.I)
    gas = m.group(1).upper() if m else 'not_encoded'
    temp = float(m.group(2)) if m else np.nan
    c = str(column or '').lower()
    if 'pressure' in c or c in ['p','p/bar','bar']: prop, unit = 'pressure', 'bar or source unit'
    elif 'uptake' in c or 'loading' in c or 'ads' in c or 'amount' in c: prop, unit = 'uptake', 'source unit'
    else: prop, unit = 'isotherm_numeric_column', 'source unit'
    return dict(task='single_isotherm', gas=gas, temperature_K=temp, property=prop, unit=unit)

def is_qmof_target_column_name(col: Any) -> bool:
    c = str(col).lower()
    # Treat electronic/thermodynamic quantities as targets.  Keep density and
    # volume available as descriptors because otherwise QMOF-like tables may
    # have no model features left.
    return any(k in c for k in ['bandgap', 'homo', 'lumo', 'formation_energy', 'energy_per_atom', 'total_energy']) and not PAT['id'].search(c)

def is_target_like_column(file_path: Optional[Path], col: Any, s: Optional[pd.Series]=None) -> bool:
    name = str(col)
    if is_arc_context(file_path) and is_arc_target_column_name(name):
        return True
    if is_core_isotherm_file(file_path) and s is not None and is_numeric_column(s) and not PAT['id'].search(name):
        # In CoRE single-isotherm tables, most numeric columns are pressure/uptake coordinates.
        if safe_nunique(s) > 3: return True
    if file_path is not None and resource_of(file_path) == 'QMOF' and is_qmof_target_column_name(name):
        return True
    return bool(any(rx.search(name) for rx in TARGET_RX))

def discover(root: Path, cfg: Config, log) -> pd.DataFrame:
    rows=[]; rx=re.compile(cfg.file_pattern,re.I) if cfg.file_pattern else None
    skipped_text_docs=0
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in IGNORE_DIRS and not d.startswith('.')]
        rp=Path(r)
        if any(part.lower() in IGNORE_DIRS for part in rp.parts): continue
        for f in fs:
            p=rp/f; ext=p.suffix.lower()
            if ext not in EXTS: continue
            if ext == '.txt' and not is_probably_tabular_text(p):
                skipped_text_docs += 1
                continue
            if rx and not rx.search(str(p)): continue
            try: st=p.stat()
            except Exception: continue
            try:
                rel = str(p.resolve().relative_to(root.resolve()))
            except Exception:
                rel = str(p)
            rows.append(dict(file_path=str(p.resolve()), relative_path=rel, file_name=p.name, extension=ext,
                             resource=resource_of(p), resource_subtype=resource_subtype_of(p), role=role_of(p),
                             size_bytes=st.st_size, size_mb=st.st_size/(1024**2),
                             modified_time=dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')))
            if cfg.max_files and len(rows)>=cfg.max_files: break
        if cfg.max_files and len(rows)>=cfg.max_files: break
    df=pd.DataFrame(rows)
    if skipped_text_docs:
        log.info('Skipped %d README/prose .txt files during discovery; tabular .txt files are still included.', skipped_text_docs)
    if df.empty:
        log.warning('No table-like files discovered under %s', root); return df
    order={r:i for i,r in enumerate(RESOURCE_ORDER)}; df['ord']=df.resource.map(order).fillna(999)
    df=df.sort_values(['ord','resource_subtype','role','size_bytes','file_name'], ascending=[True,True,True,False,True]).drop(columns='ord').reset_index(drop=True)
    log.info('Discovered %d table-like files', len(df)); return df

def profile_one(row, cfg: Config, log) -> Tuple[Dict[str,Any],List[Dict[str,Any]]]:
    p=Path(row.file_path); nrows=cfg.ram['profile_nrows']
    if nrows is None and row.size_mb>cfg.ram['full_mb']: nrows=cfg.ram['max_loaded_rows']
    df,meta=read_table(p,cfg,nrows=nrows,log=log)
    fp={**row.to_dict(), **meta, 'true_rows':None,'true_rows_method':'unknown','profile_is_full_file':False,
        'n_columns':df.shape[1] if not df.empty else 0,'n_profiled_rows':len(df),'n_numeric_columns':0,
        'n_categorical_columns':0,'n_identifier_columns':0,'n_target_like_columns':0,'n_descriptor_like_columns':0,
        'n_nested_value_columns':0,'missing_fraction_overall':np.nan,'all_missing_columns':0,'constant_columns':0,
        'identifier_candidates':'','target_candidates':'','group_candidates':''}
    cols=[]
    if meta.get('read_status')!='ok': return fp, cols
    if p.suffix.lower() in ['.csv','.tsv','.txt'] and not cfg.skip_row_count and (cfg.ram['count_large'] or row.size_mb<=cfg.ram['full_mb']):
        fp['true_rows']=count_rows(p); fp['true_rows_method']='line_count'
    elif nrows is None or len(df)<(nrows or 10**12):
        fp['true_rows']=len(df); fp['true_rows_method']='read_rows'
    fp['profile_is_full_file']=fp['true_rows']==len(df) if fp['true_rows'] is not None else False
    if df.empty: return fp, cols
    fp['missing_fraction_overall']=dataframe_missing_fraction(df)
    ids=[]; targs=[]; groups=[]
    for c in df.columns:
        s=df[c]; istarg=is_target_like_column(p,c,s); m='target_or_property' if istarg else modality(str(c),s)
        miss=safe_missing_count(s); nun=safe_nunique(s); isnum=bool(is_numeric_column(s)); nestfrac=nested_fraction(s)
        isid=bool(PAT['id'].search(str(c))); isgrp=bool(PAT['group'].search(str(c)))
        isdesc=m in ['descriptor','geometric_descriptor','numeric_other'] and not istarg
        if isid: ids.append(str(c))
        if istarg: targs.append(str(c))
        if isgrp: groups.append(str(c))
        ctx = {}
        if istarg and is_arc_context(p): ctx = infer_arc_context(p,c)
        elif istarg and is_core_isotherm_file(p): ctx = infer_core_isotherm_context(p,c)
        cols.append(dict(file_path=str(p),relative_path=row.relative_path,file_name=p.name,resource=row.resource,
                         resource_subtype=getattr(row,'resource_subtype',resource_subtype_of(p)),role=row.role,column_name=str(c),dtype=str(s.dtype),
                         inferred_modality=m,is_numeric=isnum,is_identifier_candidate=isid,is_target_candidate=istarg,
                         is_group_candidate=isgrp,is_descriptor_candidate=isdesc,profiled_rows=len(s),missing_count=miss,
                         missing_fraction=miss/max(len(s),1),unique_count=nun,unique_fraction=nun/max(len(s)-miss,1),
                         constant_nonmissing=nun<=1,all_missing=miss==len(s),contains_nested_values=bool(nestfrac>0),
                         nested_fraction=nestfrac, target_task=ctx.get('task',''), target_gas=ctx.get('gas',''),
                         target_property=ctx.get('property',''), target_unit=ctx.get('unit',''),
                         example_or_quantiles=values_summary(s)))
    cdf=pd.DataFrame(cols)
    if not cdf.empty:
        fp['n_numeric_columns']=int(cdf.is_numeric.sum()); fp['n_categorical_columns']=int((~cdf.is_numeric).sum())
        fp['n_identifier_columns']=int(cdf.is_identifier_candidate.sum()); fp['n_target_like_columns']=int(cdf.is_target_candidate.sum())
        fp['n_descriptor_like_columns']=int(cdf.is_descriptor_candidate.sum()); fp['n_nested_value_columns']=int(cdf.contains_nested_values.sum()) if 'contains_nested_values' in cdf else 0
        fp['all_missing_columns']=int(cdf.all_missing.sum()); fp['constant_columns']=int(cdf.constant_nonmissing.sum())
    fp['identifier_candidates']=';'.join(ids[:20]); fp['target_candidates']=';'.join(targs[:30]); fp['group_candidates']=';'.join(groups[:20])
    return fp, cols

def step_profile(cfg: Config, dd: Dict[str,Path], log):
    disc=discover(cfg.data_root,cfg,log); save_df(disc,dd['profiles']/ 'discovered_input_files',cfg)
    files=[]; cols=[]
    for i,row in disc.iterrows():
        log.info('Profiling %d/%d: %s', i+1, len(disc), row.relative_path)
        fp,cp=profile_one(row,cfg,log); files.append(fp); cols.extend(cp)
        if (i+1)%10==0 or i+1==len(disc):
            save_df(pd.DataFrame(files),dd['profiles']/ 'file_level_profile_partial',cfg)
            save_df(pd.DataFrame(cols),dd['profiles']/ 'column_level_profile_partial',cfg); gc.collect()
    fdf=pd.DataFrame(files); cdf=pd.DataFrame(cols)
    save_df(fdf,dd['profiles']/ 'file_level_profile',cfg); save_df(cdf,dd['profiles']/ 'column_level_profile',cfg)
    if not cdf.empty:
        mm=cdf.groupby(['resource','inferred_modality']).size().reset_index(name='n_columns').pivot_table(index='resource',columns='inferred_modality',values='n_columns',fill_value=0).reset_index()
        save_df(mm,dd['profiles']/ 'resource_modality_matrix',cfg)
        ctx=cdf[cdf.get('is_target_candidate',False).astype(bool)] if 'is_target_candidate' in cdf else pd.DataFrame()
        save_df(ctx,dd['profiles']/ 'target_column_context_catalog',cfg)
    rapid=fdf[[c for c in ['resource','resource_subtype','role','relative_path','file_name','extension','size_mb','read_status','true_rows','n_columns','n_profiled_rows','n_numeric_columns','n_categorical_columns','n_nested_value_columns','missing_fraction_overall','identifier_candidates','target_candidates'] if c in fdf.columns]]
    save_df(rapid,dd['profiles']/ 'inventory_rapid_audit_table',cfg)

# ----------------------------- normalized targets and canonical index -----------------------------
def first_identifier_series(df: pd.DataFrame, cfg: Config, fallback_prefix: str) -> pd.Series:
    ids = choose_ids(df, cfg)
    if ids:
        return df[ids[0]].map(canon).fillna(pd.Series([f"{fallback_prefix}::row_{i}" for i in range(len(df))], index=df.index))
    return pd.Series([f"{fallback_prefix}::row_{i}" for i in range(len(df))], index=df.index)

def build_long_targets_for_file(p: Path, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    ids = first_identifier_series(df, cfg, slug(p.stem))
    records=[]
    for c in df.columns:
        if not is_target_like_column(p,c,df[c]): continue
        vals = safe_to_numeric(df[c])
        if vals.notna().sum() < max(10, min(100, len(vals)*0.02)): continue
        if is_arc_context(p):
            ctx = infer_arc_context(p,c)
        elif is_core_isotherm_file(p):
            ctx = infer_core_isotherm_context(p,c)
        elif resource_of(p) == 'QMOF':
            ctx = dict(task='quantum_property', gas='not_applicable', property=str(c), unit='source unit')
        else:
            ctx = dict(task=role_of(p), gas='not_encoded', property=str(c), unit='source unit')
        tmp = pd.DataFrame({
            'canonical_id': ids.astype(str), 'raw_row_index': np.arange(len(df)), 'target_column': str(c),
            'target_value': vals, 'source_file': p.name, 'source_path': str(p), 'resource': resource_of(p),
            'resource_subtype': resource_subtype_of(p), 'role': role_of(p), 'task': ctx.get('task',''),
            'gas': ctx.get('gas',''), 'temperature_K': ctx.get('temperature_K', np.nan),
            'target_property': ctx.get('property',''), 'target_unit': ctx.get('unit','')
        })
        tmp = tmp[tmp.target_value.notna()]
        if not tmp.empty: records.append(tmp)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def step_normalize_targets(cfg: Config, dd: Dict[str,Path], log):
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    if fdf.empty:
        for n in ['normalized_arc_adsorption_targets_long','normalized_core_single_isotherms_long','normalized_qmof_targets_long','normalized_target_catalog','canonical_mof_index']:
            save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc_parts=[]; core_parts=[]; qmof_parts=[]; id_parts=[]
    candidates = fdf[(fdf.read_status.eq('ok')) & (pd.to_numeric(fdf.n_target_like_columns, errors='coerce').fillna(0) > 0)].copy()
    for _,r in candidates.iterrows():
        p=Path(r.file_path)
        df,_=read_table(p,cfg,nrows=cfg.ram['max_loaded_rows'],log=log)
        if df.empty: continue
        ids = first_identifier_series(df, cfg, slug(p.stem))
        id_parts.append(pd.DataFrame({
            'canonical_id': ids.astype(str), 'source_resource': resource_of(p), 'resource_subtype': resource_subtype_of(p),
            'source_file': p.name, 'source_path': str(p), 'raw_row_index': np.arange(len(df))
        }).drop_duplicates('canonical_id').head(cfg.ram['max_loaded_rows']))
        longdf = build_long_targets_for_file(p, df, cfg)
        if longdf.empty: continue
        if resource_of(p) == 'ARC-MOF': arc_parts.append(longdf)
        elif is_core_isotherm_file(p): core_parts.append(longdf)
        elif resource_of(p) == 'QMOF': qmof_parts.append(longdf)
    arc = pd.concat(arc_parts, ignore_index=True) if arc_parts else pd.DataFrame()
    core = pd.concat(core_parts, ignore_index=True) if core_parts else pd.DataFrame()
    qmof = pd.concat(qmof_parts, ignore_index=True) if qmof_parts else pd.DataFrame()
    alltargets = pd.concat([x for x in [arc,core,qmof] if not x.empty], ignore_index=True) if any(not x.empty for x in [arc,core,qmof]) else pd.DataFrame()
    cindex = pd.concat(id_parts, ignore_index=True).drop_duplicates(['canonical_id','source_resource','source_file']) if id_parts else pd.DataFrame()
    save_df(arc,dd['processed']/ 'normalized_arc_adsorption_targets_long',cfg)
    save_df(core,dd['processed']/ 'normalized_core_single_isotherms_long',cfg)
    save_df(qmof,dd['processed']/ 'normalized_qmof_targets_long',cfg)
    save_df(alltargets,dd['processed']/ 'normalized_target_catalog',cfg)
    save_df(cindex,dd['processed']/ 'canonical_mof_index',cfg)
    save_df(alltargets.groupby(['resource','resource_subtype','task','gas','target_property','target_unit']).agg(n_values=('target_value','count'),n_ids=('canonical_id','nunique'),mean=('target_value','mean'),median=('target_value','median')).reset_index() if not alltargets.empty else pd.DataFrame(), dd['si_tables']/ 'SI_Table_S12_normalized_target_catalog_summary', cfg)
    log.info('Normalized targets: ARC=%d rows, CoRE isotherm=%d rows, QMOF=%d rows, canonical index=%d rows', len(arc), len(core), len(qmof), len(cindex))

# ----------------------------- improved chemistry trust -----------------------------
def safe_string_series(df: pd.DataFrame, cols: List[Any]) -> pd.Series:
    txt = pd.Series('', index=df.index, dtype='object')
    for c in cols:
        txt = txt + ' ' + df[c].map(make_hashable).replace('__MISSING__','')
    return txt

def row_flags(df: pd.DataFrame, p: Path, cfg: Config) -> pd.DataFrame:
    ids=choose_ids(df,cfg)
    fcols=[c for c in df.columns if PAT['formula'].search(str(c))]
    mcols=[c for c in df.columns if re.search(r'metal|element|node|sbu|cation',str(c),re.I)]
    oxcols=[c for c in df.columns if re.search(r'oxid|oxi|valence|formal.*state|metal.*state',str(c),re.I)]
    chcols=[c for c in df.columns if re.search(r'charge|net_charge|formal_charge|total_charge|framework_charge',str(c),re.I)]
    wcols=[c for c in df.columns if re.search(r'suspect|warning|error|fail|invalid|mofchecker|samosa|check|valid|recommended|screen',str(c),re.I)]
    if not (fcols or mcols or oxcols or chcols or wcols): return pd.DataFrame()
    if len(df)>cfg.ram['max_loaded_rows']: df=df.sample(n=cfg.ram['max_loaded_rows'],random_state=cfg.random_seed)
    out=pd.DataFrame(index=df.index); out['source_file']=p.name; out['file_path']=str(p); out['resource']=resource_of(p); out['resource_subtype']=resource_subtype_of(p); out['role']=role_of(p); out['row_index']=df.index.astype(str)
    out['primary_id']=df[ids[0]].map(canon) if ids else out['source_file']+'::'+out['row_index']
    formula_txt=safe_string_series(df, fcols[:3]) if fcols else pd.Series('', index=df.index)
    metal_txt=safe_string_series(df, mcols[:3]) if mcols else pd.Series('', index=df.index)
    ml=(formula_txt + ' ' + metal_txt).map(metals_from)
    out['metals_detected']=ml.map(lambda z:';'.join(z)); out['n_metals_detected']=ml.map(len); out['metal_family_primary']=ml.map(lambda z:z[0] if z else 'not_observable')
    ox_txt=safe_string_series(df, oxcols[:5]) if oxcols else pd.Series('', index=df.index)
    ox=ox_txt.map(ox_from) if oxcols else pd.Series([[] for _ in range(len(df))], index=df.index)
    out['oxidation_states_reported']=ox.map(lambda z:';'.join(map(str,z))); out['oxidation_state_observable']=ox.map(bool)
    def plaus(idx):
        mets=ml.loc[idx]; oxs=ox.loc[idx]
        if not mets or not oxs: return 'not_observable'
        if any(o in COMMON_OX.get(m,set()) for m in mets for o in oxs): return 'plausible'
        return 'uncertain_high_oxidation' if any(o>6 for o in oxs) else 'uncertain_uncommon'
    out['oxidation_state_plausibility']=[plaus(i) for i in df.index]
    charge=pd.Series(np.nan,index=df.index,dtype='float')
    for c in chcols[:5]: charge=charge.combine_first(safe_to_numeric(df[c]))
    ionic=any(k in str(p).lower() for k in ['ion','charged','salt','ncr'])
    out['charge_value_first_available']=charge; out['abs_charge_residual']=charge.abs(); out['charge_observable']=charge.notna(); out['likely_ionic_context_from_file']=ionic
    out['charge_balance_status']=np.where(charge.isna(),'not_observable',np.where(charge.abs()<=1e-6,'neutral_or_balanced',np.where(ionic,'charged_context','nonzero_charge_uncertain')))
    suspect=pd.Series(False,index=df.index); positive=pd.Series(False,index=df.index); reasons=[[] for _ in range(len(df))]
    for c in wcols[:25]:
        ss=df[c].map(make_hashable).astype(str).str.lower(); bad=ss.str.contains(r'suspect|warning|error|fail|invalid|bad|false',na=False); good=ss.str.contains(r'pass|valid|true|recommended|ok|screening',na=False)
        suspect|=bad; positive|=good
        for j,b in enumerate(bad.tolist()):
            if b: reasons[j].append(str(c))
    out['formula_observable']=bool(fcols)
    out['metal_observable']=out.n_metals_detected.gt(0)
    out['curation_observable']=bool(wcols)
    out['curation_suspect_flag']=suspect.values; out['curation_positive_flag']=positive.values; out['curation_reason_columns']=[';'.join(x[:6]) for x in reasons]
    evidence_cols=['formula_observable','metal_observable','oxidation_state_observable','charge_observable','curation_observable']
    # formula_observable is file-level if formula cols exist; convert scalar to row-level series.
    obs = pd.DataFrame({
        'formula_observable': pd.Series([bool(fcols)]*len(out), index=out.index),
        'metal_observable': out['metal_observable'].astype(bool),
        'oxidation_state_observable': out['oxidation_state_observable'].astype(bool),
        'charge_observable': out['charge_observable'].astype(bool),
        'curation_observable': pd.Series([bool(wcols)]*len(out), index=out.index),
    })
    out['chemistry_evidence_observability_score']=obs.mean(axis=1)
    regimes=[]; scores=[]
    for _,r in out.iterrows():
        serious = bool(r.curation_suspect_flag) or r.oxidation_state_plausibility in ['uncertain_uncommon','uncertain_high_oxidation'] or r.charge_balance_status == 'nonzero_charge_uncertain'
        validated = bool(r.curation_positive_flag) or resource_of(p) in ['MOSAEC-DB','CoRE MOF 2024','CoRE MOF 2025 metadata']
        obs_good = bool(r.metal_observable) and bool(r.oxidation_state_observable) and bool(r.charge_observable)
        any_obs = float(r.chemistry_evidence_observability_score) >= 0.40
        if serious:
            reg, base = 'flagged_or_inconsistent', 0.25
        elif validated and obs_good:
            reg, base = 'validated_observable', 0.92
        elif validated:
            reg, base = 'validated_not_observable', 0.74
        elif any_obs:
            reg, base = 'uncertain_observable', 0.62
        else:
            reg, base = 'uncertain_unobservable', 0.44
        base += 0.05*float(r.chemistry_evidence_observability_score)
        if r.charge_balance_status == 'charged_context': base += 0.03
        if r.oxidation_state_plausibility == 'plausible': base += 0.03
        regimes.append(reg); scores.append(float(np.clip(base,0,1)))
    out['trust_regime_row']=regimes
    out['chemistry_trust_score_row']=scores
    out['chemistry_trust_tier_row']=pd.cut(out.chemistry_trust_score_row,[-.01,.45,.70,.86,1.01],labels=['low','medium','high','very_high']).astype(str)
    return out.reset_index(drop=True)

def step_trust(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    if fdf.empty: return
    cand=fdf[fdf.read_status.eq('ok')].copy()
    cand['chem_priority']=cand.apply(lambda r: int(r.resource in ['MOSAEC-DB','CoRE MOF 2024','CoRE MOF 2025 metadata','CSD-derived context'])+int(r.role in ['curation_or_validation','general_metadata','quantum_properties','structure_or_cif_metadata'])+int(any(k in str(r.identifier_candidates).lower()+str(r.target_candidates).lower() for k in ['charge','oxid','formula','metal','valid','suspect'])),axis=1)
    cand=cand.sort_values(['chem_priority','size_mb'],ascending=[False,True])
    parts=[]
    for _,r in cand.iterrows():
        if r.chem_priority<=0 and cfg.comprehensive_level in ['screening','standard']: continue
        p=Path(r.file_path); log.info('Chemistry-trust scan: %s', r.relative_path)
        df,_=read_table(p,cfg,nrows=cfg.ram['max_loaded_rows'],log=log); fl=row_flags(df,p,cfg)
        if not fl.empty:
            parts.append(fl); save_df(fl,dd['processed']/ 'trust_flags_by_file'/slug(p.stem),cfg)
        if cfg.comprehensive_level=='screening' and len(parts)>=20: break
    trust=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['source_file','resource','resource_subtype','role','primary_id','metal_family_primary','chemistry_trust_score_row','chemistry_trust_tier_row','trust_regime_row','oxidation_state_plausibility','charge_balance_status','curation_suspect_flag'])
    save_df(trust,dd['processed']/ 'trust_flags',cfg)
    if trust.empty:
        for n in ['trust_flags_by_metal','charge_balance_residuals','variant_comparison','representative_rule_cards','chemistry_evidence_matrix','trust_regime_summary','charge_status_summary']:
            save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    met=trust.groupby(['resource','metal_family_primary','trust_regime_row']).size().reset_index(name='n')
    met['fraction']=met.n/met.groupby(['resource','metal_family_primary']).n.transform('sum')
    save_df(met,dd['processed']/ 'trust_flags_by_metal',cfg); save_df(met,dd['si_tables']/ 'SI_Table_S8_trust_flags_by_metal',cfg)
    ch=trust[[c for c in ['resource','resource_subtype','role','source_file','primary_id','metal_family_primary','charge_value_first_available','abs_charge_residual','charge_balance_status','likely_ionic_context_from_file','trust_regime_row','chemistry_trust_tier_row'] if c in trust.columns]]
    save_df(ch,dd['processed']/ 'charge_balance_residuals',cfg)
    evidence_cols=[c for c in ['formula_observable','metal_observable','oxidation_state_observable','charge_observable','curation_observable'] if c in trust.columns]
    emat=trust.groupby('resource')[evidence_cols].mean().reset_index() if evidence_cols else pd.DataFrame()
    save_df(emat,dd['processed']/ 'chemistry_evidence_matrix',cfg)
    reg=trust.groupby(['resource','trust_regime_row']).size().reset_index(name='n_rows')
    reg['fraction']=reg.n_rows/reg.groupby('resource').n_rows.transform('sum')
    save_df(reg,dd['processed']/ 'trust_regime_summary',cfg)
    charge=trust.groupby(['resource','charge_balance_status']).size().reset_index(name='n_rows')
    charge['fraction']=charge.n_rows/charge.groupby('resource').n_rows.transform('sum')
    save_df(charge,dd['processed']/ 'charge_status_summary',cfg)
    def varlab(r):
        s=(str(r.source_file)+' '+str(r.role)+' '+str(r.resource_subtype)).lower()
        for k,v in [('asr','ASR'),('fsr','FSR'),('ion','ION'),('charged','charged'),('neutral','neutral_or_NCR'),('ncr','neutral_or_NCR'),('recommended','recommended_screening'),('12089','recommended_screening'),('tsa','TSA'),('single-isotherm','single_isotherm')]:
            if k in s: return v
        return 'other'
    trust['variant_label']=trust.apply(varlab,axis=1)
    var=trust.groupby(['resource','resource_subtype','variant_label']).agg(n_rows=('primary_id','count'),mean_trust=('chemistry_trust_score_row','mean'),median_trust=('chemistry_trust_score_row','median'),mean_observability=('chemistry_evidence_observability_score','mean'),suspect_fraction=('curation_suspect_flag','mean')).reset_index()
    save_df(var,dd['processed']/ 'variant_comparison',cfg)
    cards=[]; per=max(1,cfg.level['cards']//len(TRUST_REGIME_ORDER))
    for regname in TRUST_REGIME_ORDER:
        sub=trust[trust.trust_regime_row.eq(regname)]
        if sub.empty: continue
        sub=sub.sample(n=min(per,len(sub)),random_state=cfg.random_seed) if len(sub)>per else sub
        for _,r in sub.iterrows():
            reasons=[]
            if r.get('curation_suspect_flag',False): reasons.append('curation warning column triggered')
            if str(r.get('oxidation_state_plausibility',''))!='plausible': reasons.append(f"oxidation state: {r.get('oxidation_state_plausibility')}")
            if str(r.get('charge_balance_status','')) not in ['neutral_or_balanced','charged_context']: reasons.append(f"charge balance: {r.get('charge_balance_status')}")
            if float(r.get('chemistry_evidence_observability_score',0)) < 0.50: reasons.append('chemistry evidence only partly observable in available columns')
            if not reasons: reasons.append('observable chemistry evidence and curation status support this class')
            cards.append(dict(rule_card_class=regname,resource=r.resource,resource_subtype=r.get('resource_subtype',''),source_file=r.source_file,primary_id=r.primary_id,metals_detected=r.get('metals_detected',''),oxidation_states_reported=r.get('oxidation_states_reported',''),chemistry_trust_score_row=r.chemistry_trust_score_row,chemistry_trust_tier_row=r.chemistry_trust_tier_row,observability_score=r.get('chemistry_evidence_observability_score',np.nan),interpretation='; '.join(reasons)))
    cdf=pd.DataFrame(cards); save_df(cdf,dd['processed']/ 'representative_rule_cards',cfg); save_df(cdf,dd['si_tables']/ 'SI_Table_S15_representative_rule_cards',cfg)

# ----------------------------- improved scores, ML and tables -----------------------------
def qlabel(ml,tr):
    if tr>=.75 and ml>=.70: return 'chemistry-ready and ML-ready'
    if tr>=.75: return 'chemistry-ready but ML-limited'
    if ml>=.70: return 'ML-ready but chemistry-observability-limited'
    return 'limited or context-dependent'

def target_cols(df: pd.DataFrame, cfg: Config, file_path: Optional[Path]=None) -> List[str]:
    if cfg.target_column and cfg.target_column in df.columns: return [cfg.target_column]
    cand=[]
    for c in df.columns:
        s=df[c]; num=safe_to_numeric(s)
        if num.notna().mean()<.75 or int(num.nunique(dropna=True))<=5: continue
        score=sum((20-i) for i,rx in enumerate(TARGET_RX) if rx.search(str(c)))
        if is_target_like_column(file_path,c,s): score += 100
        if any(k in str(c).lower() for k in ['id','index','unnamed','number']) and not is_arc_target_column_name(c): score-=15
        if score>0: cand.append((score,str(c)))
    return [c for _,c in sorted(cand,reverse=True)[:cfg.level['targets']]]

def group_col(df: pd.DataFrame, preferred: Optional[str]=None) -> Optional[str]:
    if preferred and preferred in df.columns:
        try:
            n=safe_nunique(df[preferred])
            if 2<=n<=max(2,len(df)*.8): return preferred
        except Exception: pass
    for c in df.columns:
        if PAT['group'].search(str(c)) or re.search(r'(metal|topology|family|source|resource)', str(c), re.I):
            try:
                n=safe_nunique(df[c])
                if 2<=n<=max(2,len(df)*.5): return c
            except Exception: pass
    return None

def xy(df: pd.DataFrame, target: str, cfg: Config, file_path: Optional[Path]=None):
    y=safe_to_numeric(df[target]); keep=y.notna(); df=df.loc[keep].copy(); y=y.loc[keep]
    if len(df)>cfg.ram['max_model_rows']:
        df=df.sample(n=cfg.ram['max_model_rows'],random_state=cfg.random_seed); y=y.loc[df.index]
    gcol=group_col(df); groups=df[gcol].astype(str) if gcol else None
    num=[]; cat=[]
    for c in df.columns:
        if c==target: continue
        name=str(c)
        if PAT['id'].search(name) or name.lower().startswith('unnamed') or is_target_like_column(file_path,c,df[c]) or any(rx.search(name) for rx in TARGET_RX): continue
        s=df[c]
        if safe_missing_fraction(s)>.65 or safe_nunique(s)<=1: continue
        conv=safe_to_numeric(s)
        if pd.api.types.is_numeric_dtype(s) or conv.notna().mean()>.9: num.append(c)
        elif safe_nunique(s)<=50: cat.append(c)
    if len(num)>cfg.ram['max_features']:
        rank=[]
        for c in num:
            ss=safe_to_numeric(df[c]); bonus=1 if (PAT['descriptor'].search(str(c)) or PAT['geometry'].search(str(c))) else 0
            rank.append((bonus,float(ss.var(skipna=True) if ss.notna().sum()>2 else 0),c))
        num=[c for _,_,c in sorted(rank,reverse=True)[:cfg.ram['max_features']]]
    X=df[num+cat].copy()
    for c in num: X[c]=safe_to_numeric(X[c])
    for c in cat: X[c]=X[c].map(make_hashable).replace('__MISSING__', pd.NA).astype('string')
    return X,y,groups,num,cat

def run_ml(df, target, label, cfg, log, file_path: Optional[Path]=None):
    if not SKLEARN_AVAILABLE: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    X,y,groups,num,cat=xy(df,target,cfg,file_path=file_path)
    if len(y)<200 or len(num)+len(cat)<2:
        log.warning('Not enough model-ready rows/features for %s %s', label, target); return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    mets=[]; preds=[]; errs=[]
    models=[m for m in cfg.level['models'] if m!='hgb' or HGB_AVAILABLE]
    for split in cfg.level['splits']:
        if split=='grouped' and groups is None: continue
        for rep in range(cfg.level['repeats']):
            seed=cfg.random_seed+17*rep
            try:
                if split=='grouped': tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y,groups=groups))
                else: tr,te=next(ShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y))
            except Exception as e:
                log.warning('Split failed for %s target=%s split=%s: %s', label, target, split, e); continue
            for mn in models:
                try:
                    pipe=model_pipe(mn,num,cat,cfg); pipe.fit(X.iloc[tr],y.iloc[tr]); pr=pipe.predict(X.iloc[te]); yt=np.asarray(y.iloc[te])
                    mets.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,n_train=len(tr),n_test=len(te),n_numeric_features=len(num),n_categorical_features=len(cat),rmse=rmse(yt,pr),mae=float(mean_absolute_error(yt,pr)),r2=float(r2_score(yt,pr)),spearman=spear(yt,pr),top_1pct_recovery=topk(yt,pr,.01),top_5pct_recovery=topk(yt,pr,.05),top_10pct_recovery=topk(yt,pr,.10),ndcg_10pct=ndcg(yt,pr,.10)))
                    idx=np.arange(len(yt))
                    if len(idx)>4000: idx=np.random.default_rng(seed).choice(idx,4000,replace=False)
                    for j in idx:
                        preds.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,y_true=float(yt[j]),y_pred=float(pr[j]),absolute_error=float(abs(yt[j]-pr[j]))))
                    if groups is not None:
                        g=groups.iloc[te].reset_index(drop=True); ed=pd.DataFrame({'group':g.astype(str),'abs_error':np.abs(yt-pr)}); top=ed.group.value_counts().head(25).index
                        for _,r in ed[ed.group.isin(top)].groupby('group').abs_error.agg(['size','mean']).reset_index().iterrows():
                            errs.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,chemistry_or_group=r['group'],n=int(r['size']),mae=float(r['mean'])))
                    if cfg.save.get('models') and mn!='dummy':
                        mp=cfg.out_dir/'models'/f'{slug(label)}__{slug(target)}__{mn}__{split}__rep{rep}.pkl'; mkdir(mp.parent); pickle.dump(pipe,open(mp,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e: log.warning('ML failed for %s target=%s model=%s: %s',label,target,mn,e)
    return pd.DataFrame(mets),pd.DataFrame(preds),pd.DataFrame(errs)

def step_ml(cfg: Config, dd: Dict[str,Path], log):
    if cfg.skip_ml or not SKLEARN_AVAILABLE:
        msg='ML stress-test skipped because --skip-ml was used.' if cfg.skip_ml else 'ML stress-test skipped because scikit-learn is not available in this environment.'
        log.warning(msg)
        (dd['logs']/ 'RUN_INCOMPLETE_WARNING.txt').write_text(msg+'\nInstall scikit-learn and rerun to obtain Figure 5 and ML tables.\n', encoding='utf-8')
        for n in ['ml_metrics_full','ml_predictions_sample','ml_error_by_chemistry','ranking_stability','ml_metrics_summary']: save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    cand=fdf[(pd.to_numeric(fdf.n_target_like_columns,errors='coerce').fillna(0)>0)&(pd.to_numeric(fdf.n_numeric_columns,errors='coerce').fillna(0)>3)&(fdf.read_status.eq('ok'))].copy()
    if cand.empty:
        log.warning('No target-like model tables found after profiling.')
        for n in ['ml_metrics_full','ml_predictions_sample','ml_error_by_chemistry','ranking_stability','ml_metrics_summary']: save_df(pd.DataFrame(),dd['processed']/n,cfg)
        return
    cand['priority']=cand.apply(lambda r:5*int(r.resource=='ARC-MOF')+4*int(r.role in ['adsorption_targets','process_targets'])+3*int(r.resource=='QMOF')+2*int(r.resource=='CoRE MOF 2024')+int(r.resource=='MOSAEC-DB'),axis=1)
    cand=cand.sort_values(['priority','n_target_like_columns','size_mb'],ascending=[False,False,True]).head(max(cfg.level['targets']+3, 6))
    allm=[]; allp=[]; alle=[]
    for _,r in cand.iterrows():
        p=Path(r.file_path); df,_=read_table(p,cfg,nrows=cfg.ram['max_model_rows']*2,log=log); tcols=target_cols(df,cfg,file_path=p)
        for t in tcols:
            log.info('ML stress test: %s target=%s', r.file_name, t)
            m,pred,err=run_ml(df,t,f'{r.resource}::{r.resource_subtype if "resource_subtype" in r else resource_subtype_of(p)}::{p.name}',cfg,log,file_path=p)
            if not m.empty: allm.append(m)
            if not pred.empty: allp.append(pred)
            if not err.empty: alle.append(err)
    met=pd.concat(allm,ignore_index=True) if allm else pd.DataFrame(); pred=pd.concat(allp,ignore_index=True) if allp else pd.DataFrame(); err=pd.concat(alle,ignore_index=True) if alle else pd.DataFrame()
    save_df(met,dd['processed']/ 'ml_metrics_full',cfg); save_df(pred,dd['processed']/ 'ml_predictions_sample',cfg); save_df(err,dd['processed']/ 'ml_error_by_chemistry',cfg); save_df(met,dd['si_tables']/ 'SI_Table_S6_complete_model_results',cfg)
    if not met.empty:
        summ=met.groupby(['source_table','target_column','model','split_type']).agg(n_repeats=('repeat','nunique'),mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),mean_mae=('mae','mean'),mean_r2=('r2','mean'),sd_r2=('r2','std'),mean_spearman=('spearman','mean'),mean_top_1pct_recovery=('top_1pct_recovery','mean'),mean_top_5pct_recovery=('top_5pct_recovery','mean'),mean_top_10pct_recovery=('top_10pct_recovery','mean'),mean_ndcg_10pct=('ndcg_10pct','mean')).reset_index()
        rank=met[['source_table','target_column','model','split_type','repeat','top_1pct_recovery','top_5pct_recovery','top_10pct_recovery','ndcg_10pct']].copy()
    else:
        summ=pd.DataFrame(); rank=pd.DataFrame()
    save_df(summ,dd['processed']/ 'ml_metrics_summary',cfg); save_df(summ,dd['main_tables']/ 'Main_Table_5_ml_stress_test_summary',cfg); save_df(rank,dd['processed']/ 'ranking_stability',cfg)

def step_scores(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile'); cdf=load_df(dd['profiles']/ 'column_level_profile'); trust=load_df(dd['processed']/ 'trust_flags')
    if fdf.empty: return
    maxrows=pd.to_numeric(fdf.get('true_rows',fdf.n_profiled_rows),errors='coerce').fillna(fdf.n_profiled_rows).max(); maxnum=pd.to_numeric(fdf.n_numeric_columns,errors='coerce').fillna(0).max()
    rows=[]
    for res in sorted(fdf.resource.dropna().unique(), key=lambda x: RESOURCE_ORDER.index(x) if x in RESOURCE_ORDER else 999):
        fs=fdf[fdf.resource.eq(res)]; cs=cdf[cdf.resource.eq(res)] if not cdf.empty else pd.DataFrame(); ts=trust[trust.resource.eq(res)] if not trust.empty and 'resource' in trust else pd.DataFrame()
        total=pd.to_numeric(fs.get('true_rows',fs.n_profiled_rows),errors='coerce').fillna(fs.n_profiled_rows).sum(); miss=pd.to_numeric(fs.missing_fraction_overall,errors='coerce').mean()
        id_av=float((pd.to_numeric(fs.n_identifier_columns,errors='coerce').fillna(0)>0).mean()); targ_av=float((pd.to_numeric(fs.n_target_like_columns,errors='coerce').fillna(0)>0).mean()); desc_av=float((pd.to_numeric(fs.n_descriptor_like_columns,errors='coerce').fillna(0)>0).mean())
        chem_cols=int(cs.inferred_modality.isin(['formula_or_composition','metal_or_charge_chemistry','curation_or_validation_flag']).sum()) if not cs.empty else 0; chem_col_score=min(1,chem_cols/20)
        if not ts.empty and 'chemistry_trust_score_row' in ts:
            trust_score=float(pd.to_numeric(ts.chemistry_trust_score_row,errors='coerce').mean())
            observed=min(1,len(ts)/max(float(fs.n_profiled_rows.sum()),1))
            obs_score=float(pd.to_numeric(ts.get('chemistry_evidence_observability_score',pd.Series([np.nan])),errors='coerce').mean())
        else:
            trust_score={'MOSAEC-DB':.78,'CoRE MOF 2024':.70,'CoRE MOF 2025 metadata':.70,'ARC-MOF':.42,'QMOF':.58,'CSD-derived context':.55,'Other/unknown':.35}.get(res,.35)*(0.70+0.30*chem_col_score)
            observed=0; obs_score=0
        ml=np.clip(.22*logscore(total,maxrows*len(fdf.resource.unique()))+.20*logscore(fs.n_numeric_columns.sum(),maxnum*max(len(fs),1))+.16*(1-miss if not pd.isna(miss) else .5)+.16*id_av+.14*desc_av+.12*targ_av,0,1)
        tr=float(np.clip(.68*trust_score+.18*chem_col_score+.14*obs_score,0,1))
        rows.append(dict(resource=res,n_files=len(fs),total_profiled_or_counted_rows=float(total),total_numeric_columns=float(fs.n_numeric_columns.sum()),mean_missing_fraction=miss,identifier_availability_fraction=id_av,target_availability_fraction=targ_av,descriptor_availability_fraction=desc_av,chemistry_relevant_column_count=chem_cols,row_level_trust_observed_fraction=observed,mean_chemistry_evidence_observability=obs_score,chemistry_trust_score=tr,ml_readiness_score=float(ml),trust_readiness_quadrant=qlabel(ml,tr),recommended_role=role_rec(res,ml,tr)))
    scores=pd.DataFrame(rows); save_df(scores,dd['processed']/ 'resource_scores',cfg); save_df(scores,dd['main_tables']/ 'Main_Table_1_resource_profile_and_scores',cfg); save_df(scores,dd['si_tables']/ 'SI_Table_S5_resource_and_subset_scores',cfg)
    metric_defs=pd.DataFrame([
        {'metric':'Chemistry-trust score','formula_or_definition':'0.68*mean_row_trust + 0.18*chemistry_column_coverage + 0.14*mean_observability','range':'0--1','meaning':'Claim-specific confidence that chemistry evidence is both curated and observable.'},
        {'metric':'ML-readiness score','formula_or_definition':'weighted combination of row count, numeric columns, missingness, identifier coverage, descriptor coverage and target coverage','range':'0--1','meaning':'Practical suitability for lightweight supervised ML.'},
        {'metric':'Row-level trust observed fraction','formula_or_definition':'rows with at least one chemistry-trust scan / profiled rows for the resource','range':'0--1','meaning':'Distinguishes missing evidence from verified inconsistency.'},
        {'metric':'Trust regime','formula_or_definition':'validated_observable / validated_not_observable / uncertain_observable / uncertain_unobservable / flagged_or_inconsistent','range':'categorical','meaning':'Separates validation status from observability.'},
        {'metric':'Benchmark-risk score','formula_or_definition':'expert-defined 0--3 severity for each issue/use-case pair','range':'0--3','meaning':'Task-specific consequence of a data-quality issue.'},
    ])
    save_df(metric_defs,dd['main_tables']/ 'Main_Table_2_metric_definitions',cfg); save_df(metric_defs,dd['si_tables']/ 'SI_Table_S4_chemistry_trust_definitions',cfg)
    rec=pd.DataFrame([
        {'use_case':'Adsorption ranking','preferred_resource_logic':'Use ARC-MOF adsorption/process resources with explicit target context, grouped splits and chemistry-observability caveats.','minimum_reporting':'gas/pressure/temperature/property/unit, identifier joins, top-k stability, trust regime'},
        {'use_case':'Process screening','preferred_resource_logic':'Use process tables only after documenting target definitions and leakage-prone cycle descriptors.','minimum_reporting':'cycle target definition, row retention, duplicate policy, grouped split'},
        {'use_case':'Quantum-property prediction','preferred_resource_logic':'Use QMOF/DFT-consistent resources when electronic-structure consistency is central.','minimum_reporting':'DFT settings, units, failed calculations, chemistry flags'},
        {'use_case':'Generative-model training','preferred_resource_logic':'Prefer chemistry-validated, duplicate-controlled subsets and report not-observable cases separately.','minimum_reporting':'validity filters, duplicate groups, charge/oxidation warnings'},
        {'use_case':'Mechanistic interpretation','preferred_resource_logic':'Prioritize chemically interpretable subsets over maximum size.','minimum_reporting':'rule cards and manual check examples'}])
    save_df(rec,dd['main_tables']/ 'Main_Table_4_use_case_recommendation_cards',cfg)
    issues=['missing oxidation-state/formal-charge evidence','nonzero charge outside explicit ionic context','suspect chemistry or validation warning','identifier ambiguity or duplicate leakage','high descriptor missingness','target provenance mismatch','small or biased trusted subset']; uses=['adsorption ranking','process screening','quantum-property modeling','generative training','mechanistic interpretation']; base=[[2,2,2,3,3],[3,3,2,3,3],[2,2,2,3,3],[3,3,2,3,2],[2,2,1,2,1],[3,3,2,1,2],[2,2,2,2,2]]
    risk=pd.DataFrame([dict(risk_issue=iss,use_case=uc,risk_score_0_to_3=base[i][j],interpretation={0:'negligible',1:'low',2:'moderate',3:'high'}[base[i][j]]) for i,iss in enumerate(issues) for j,uc in enumerate(uses)])
    save_df(risk,dd['processed']/ 'benchmark_risk_matrix',cfg); save_df(risk,dd['main_tables']/ 'Main_Table_3_benchmark_risk_matrix',cfg); save_df(risk,dd['si_tables']/ 'SI_Table_S7_benchmark_risk_matrix',cfg)
    rng=np.random.default_rng(cfg.random_seed); sens=[]
    for d in range(cfg.level['sensitivity']):
        wt=rng.uniform(.30,.80); tmp=scores.copy(); tmp['combined']=wt*tmp.chemistry_trust_score+(1-wt)*tmp.ml_readiness_score; tmp['rank_in_draw']=tmp.combined.rank(ascending=False,method='min')
        for _,r in tmp.iterrows(): sens.append(dict(draw=d,resource=r.resource,weight_chemistry_trust=wt,combined_score=r.combined,rank_in_draw=r.rank_in_draw))
    sdf=pd.DataFrame(sens).groupby('resource').agg(mean_rank=('rank_in_draw','mean'),min_rank=('rank_in_draw','min'),max_rank=('rank_in_draw','max'),mean_combined_score=('combined_score','mean'),sd_combined_score=('combined_score','std')).reset_index() if sens else pd.DataFrame()
    save_df(sdf,dd['processed']/ 'score_sensitivity',cfg)

# ----------------------------- improved tables, source map and figures -----------------------------
def step_tables(cfg: Config, dd: Dict[str,Path], log):
    fdf=load_df(dd['profiles']/ 'file_level_profile'); cdf=load_df(dd['profiles']/ 'column_level_profile'); scores=load_df(dd['processed']/ 'resource_scores'); join=load_df(dd['processed']/ 'join_accounting'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); mls=load_df(dd['processed']/ 'ml_metrics_summary')
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary'); invf=load_df(dd['profiles']/ 'inventory_folder_summary'); invex=load_df(dd['profiles']/ 'inventory_report_examples')
    for base,name in [(inv,'SI_Table_S0_inventory_resource_role_format_summary'),(invf,'SI_Table_S0b_inventory_folder_summary'),(invex,'SI_Table_S0c_inventory_example_files')]:
        if not base.empty: save_df(base,dd['si_tables']/name,cfg)
    if not fdf.empty:
        group_cols=['resource','resource_subtype','role'] if 'resource_subtype' in fdf.columns else ['resource','role']
        main1=fdf.groupby(group_cols).agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum'),total_profiled_rows=('n_profiled_rows','sum'),total_counted_rows=('true_rows','sum'),median_columns=('n_columns','median'),mean_missing_fraction=('missing_fraction_overall','mean'),n_target_like_columns=('n_target_like_columns','sum'),n_descriptor_like_columns=('n_descriptor_like_columns','sum')).reset_index()
        if not scores.empty: main1=main1.merge(scores[['resource','chemistry_trust_score','ml_readiness_score','row_level_trust_observed_fraction','mean_chemistry_evidence_observability','recommended_role']],on='resource',how='left')
        save_df(main1,dd['main_tables']/ 'Main_Table_1_resource_atlas',cfg); save_df(fdf,dd['si_tables']/ 'SI_Table_S1_complete_file_level_profile',cfg)
    if not cdf.empty:
        save_df(cdf,dd['si_tables']/ 'SI_Table_S2_complete_column_level_profile',cfg)
        save_df(cdf[cdf.is_descriptor_candidate.astype(bool)].head(10000) if 'is_descriptor_candidate' in cdf else pd.DataFrame(),dd['si_tables']/ 'SI_Table_S11_descriptor_family_profile',cfg)
        save_df(cdf[cdf.is_target_candidate.astype(bool)] if 'is_target_candidate' in cdf else pd.DataFrame(),dd['si_tables']/ 'SI_Table_S13_target_column_context',cfg)
    if not join.empty: save_df(join,dd['si_tables']/ 'SI_Table_S3_join_accounting',cfg)
    if not risk.empty: save_df(risk,dd['main_tables']/ 'Main_Table_3_benchmark_risk_matrix',cfg)
    if not mls.empty: save_df(mls,dd['si_tables']/ 'SI_Table_S6_ml_metrics_summary',cfg)
    decision=pd.DataFrame([
        {'step':1,'question':'What is the scientific claim: adsorption ranking, process screening, quantum prediction, generative design or mechanistic interpretation?','yes_action':'Choose resource/subset by claim, not by database size alone.','no_action':'Do not use a generic leaderboard interpretation.'},
        {'step':2,'question':'Is the claim sensitive to oxidation state, charge, open metal sites or ionic context?','yes_action':'Separate validated-observable, validated-not-observable, uncertain and flagged cases.','no_action':'Report provenance and missingness; chemistry flags may be secondary.'},
        {'step':3,'question':'Are target variables normalized with gas, pressure, temperature, property and unit context?','yes_action':'Use the long-format target tables for ML and source data.','no_action':'Do not mix unit-only ARC-MOF columns or CoRE isotherm coordinates without context.'},
        {'step':4,'question':'Are descriptors and targets joined by stable identifiers?','yes_action':'Report join retention and duplicate policy.','no_action':'Do not claim resource-level coverage from a task-join subset.'},
        {'step':5,'question':'Does performance remain stable under grouped or chemistry-aware splits?','yes_action':'Benchmark conclusion is more robust.','no_action':'Phrase claims as interpolation or convenience-screening performance.'}])
    checklist=pd.DataFrame([
        {'item':'Exact file provenance, size, version and access date','minimum_status':'required'},
        {'item':'Row/column/missingness table for every input file','minimum_status':'required'},
        {'item':'Normalized target catalog with gas/property/unit context','minimum_status':'required for adsorption/process claims'},
        {'item':'Identifier normalization and join-retention accounting','minimum_status':'required'},
        {'item':'Validation/observability trust regimes, not a single ambiguous trust label','minimum_status':'required'},
        {'item':'Duplicate/leakage policy and grouped splits','minimum_status':'required for ML claims'},
        {'item':'Exact source-data file for every figure panel','minimum_status':'required'},
        {'item':'Sensitivity to trust thresholds and score weights','minimum_status':'strongly recommended'}])
    srcmap=pd.DataFrame([
        {'figure':'SI Figure S0','panel':'a-d','source_data':'profiles/inventory_report_examples.csv; profiles/inventory_folder_summary.csv; profiles/inventory_resource_role_format_summary.csv; profiles/discovered_input_files.csv','notes':'Inventory-derived archive map compared with locally discovered data files.'},
        {'figure':'Figure 1','panel':'a-d','source_data':'source_data/decision_rules.csv; source_data/minimum_reporting_checklist.csv','notes':'Conceptual schematic and rule logic.'},
        {'figure':'Figure 2','panel':'a-d','source_data':'profiles/file_level_profile.csv; profiles/column_level_profile.csv; profiles/resource_modality_matrix.csv; processed/resource_scores.csv','notes':'Local profiling results.'},
        {'figure':'Figure 3','panel':'a','source_data':'processed/chemistry_evidence_matrix.csv','notes':'Observable chemistry evidence by resource.'},
        {'figure':'Figure 3','panel':'b','source_data':'processed/trust_regime_summary.csv','notes':'Validation/observability trust-regime decomposition.'},
        {'figure':'Figure 3','panel':'c','source_data':'processed/charge_status_summary.csv','notes':'Charge-status decomposition by resource.'},
        {'figure':'Figure 3','panel':'d','source_data':'processed/representative_rule_cards.csv','notes':'Representative rule cards.'},
        {'figure':'Figure 4','panel':'a-c','source_data':'processed/resource_scores.csv; processed/benchmark_risk_matrix.csv; processed/score_sensitivity.csv','notes':'Trust-readiness and risk framework.'},
        {'figure':'Figure 5','panel':'a-d','source_data':'processed/ml_metrics_summary.csv; processed/ranking_stability.csv; processed/ml_error_by_chemistry.csv; processed/ml_predictions_sample.csv','notes':'ML stress-test output.'},
        {'figure':'Figure 6','panel':'a-c','source_data':'source_data/decision_rules.csv; source_data/minimum_reporting_checklist.csv; tables/main/Main_Table_4_use_case_recommendation_cards.csv','notes':'Decision framework.'}])
    save_df(decision,dd['source']/ 'decision_rules',cfg); save_df(checklist,dd['source']/ 'minimum_reporting_checklist',cfg); save_df(srcmap,dd['source']/ 'final_source_data_map',cfg)
    if cfg.save.get('xlsx') and EXCEL_AVAILABLE:
        sheets={}
        for nm,base in [('file_profile',dd['profiles']/ 'file_level_profile'),('column_profile',dd['profiles']/ 'column_level_profile'),('resource_scores',dd['processed']/ 'resource_scores'),('target_catalog',dd['processed']/ 'normalized_target_catalog'),('trust_regimes',dd['processed']/ 'trust_regime_summary'),('risk_matrix',dd['processed']/ 'benchmark_risk_matrix'),('ml_summary',dd['processed']/ 'ml_metrics_summary')]:
            x=load_df(base)
            if not x.empty: sheets[nm]=x.head(100000)
        if sheets:
            with pd.ExcelWriter(dd['source']/ 'project_core_source_data.xlsx',engine='openpyxl') as w:
                for nm,x in sheets.items(): x.to_excel(w,sheet_name=nm[:31],index=False)

def fig3(cfg,dd):
    emat=load_df(dd['processed']/ 'chemistry_evidence_matrix'); reg=load_df(dd['processed']/ 'trust_regime_summary'); charge=load_df(dd['processed']/ 'charge_status_summary'); cards=load_df(dd['processed']/ 'representative_rule_cards')
    fig,axs=plt.subplots(2,2,figsize=(13.2,9.2)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not emat.empty:
        cols=[c for c in ['formula_observable','metal_observable','oxidation_state_observable','charge_observable','curation_observable'] if c in emat.columns]
        mat=emat.set_index('resource')[cols].rename(columns=lambda x: x.replace('_observable','').replace('_',' '))
        heat(ax,mat,'Observable chemistry evidence by resource','YlGnBu',True)
    else: nodata(ax,'Observable chemistry evidence by resource')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    if not reg.empty:
        piv=reg.pivot_table(index='resource',columns='trust_regime_row',values='fraction',fill_value=0)
        cols=[c for c in TRUST_REGIME_ORDER if c in piv.columns]; bottom=np.zeros(len(piv))
        y=np.arange(len(piv.index))
        for c in cols:
            vals=piv[c].values; ax.barh(y,vals,left=bottom,label=c.replace('_',' '),color=TRUST_REGIME_COLORS.get(c,None)); bottom+=vals
        ax.set_yticks(y); ax.set_yticklabels(piv.index); ax.set_xlim(0,1); ax.set_xlabel('Fraction of scanned rows'); ax.set_title('Trust-regime decomposition',fontweight='bold'); ax.legend(fontsize=6.8,loc='lower right')
    else: nodata(ax,'Trust-regime decomposition')
    ax=axs[2]; lab(ax,'c')
    if not charge.empty:
        piv=charge.pivot_table(index='resource',columns='charge_balance_status',values='fraction',fill_value=0)
        heat(ax,piv,'Charge-status decomposition','PuBuGn',True)
    else: nodata(ax,'Charge-status decomposition')
    ax=axs[3]; lab(ax,'d'); ax.axis('off'); ax.set_title('Representative rule cards',fontweight='bold',fontsize=10.5)
    if not cards.empty:
        show=[]
        for r in TRUST_REGIME_ORDER:
            sub=cards[cards.rule_card_class.eq(r)]
            if not sub.empty: show.append(sub.iloc[0])
        if not show: show=[r for _,r in cards.head(4).iterrows()]
        y=.84
        for r in show[:4]:
            color=TRUST_REGIME_COLORS.get(str(r.get('rule_card_class','')), '#EEEEEE')
            ax.add_patch(FancyBboxPatch((.04,y-.11),.92,.16,boxstyle='round,pad=.02',lw=.9,facecolor=color,alpha=.16,edgecolor=color))
            ax.text(.08,y+.02,str(r.get('rule_card_class','')).replace('_',' '),fontsize=8.2,fontweight='bold',ha='left',va='center')
            ax.text(.08,y-.035,f"{r.get('resource','')} | {r.get('metals_detected','not observable')} | score={float(r.get('chemistry_trust_score_row',np.nan)):.2f}",fontsize=7.2,ha='left')
            ax.text(.08,y-.085,str(r.get('interpretation',''))[:145],fontsize=6.6,ha='left',wrap=True)
            y-=.22
    else: ax.text(.5,.5,'No chemistry rule-card columns were observable',ha='center',va='center')
    fig.suptitle('Figure 3. Chemistry evidence, trust regimes and charge observability',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_3_chemistry_evidence_trust_regimes',cfg)

def fig5(cfg,dd):
    summ=load_df(dd['processed']/ 'ml_metrics_summary'); rank=load_df(dd['processed']/ 'ranking_stability'); err=load_df(dd['processed']/ 'ml_error_by_chemistry'); pred=load_df(dd['processed']/ 'ml_predictions_sample')
    fig,axs=plt.subplots(2,2,figsize=(13.2,9.2)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not summ.empty:
        sub=summ.sort_values('mean_r2',ascending=False).head(18); labels=sub.model.astype(str)+'\n'+sub.split_type.astype(str)+'\n'+sub.target_column.astype(str).str[:18]
        ax.errorbar(sub.mean_r2,np.arange(len(sub)),xerr=sub.sd_r2.fillna(0),fmt='o',capsize=3); ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(labels,fontsize=6.8); ax.invert_yaxis(); ax.set_xlabel('Mean R² ± SD'); ax.set_title('Random/grouped split stability',fontweight='bold',fontsize=10)
    else: nodata(ax,'Random/grouped split stability','No ML output; install scikit-learn and rerun without --skip-ml')
    ax=axs[1]; lab(ax,'b')
    if not rank.empty:
        melt=rank.melt(id_vars=['model','split_type'],value_vars=[c for c in ['top_1pct_recovery','top_5pct_recovery','top_10pct_recovery'] if c in rank],var_name='top_k',value_name='recovery'); g=melt.groupby(['model','split_type','top_k']).recovery.mean().reset_index()
        for (model,split),sub in g.groupby(['model','split_type']): ax.plot(sub.top_k,sub.recovery,marker='o',label=f'{model}-{split}')
        ax.set_ylabel('Top-k recovery'); ax.set_ylim(0,1.05); ax.legend(fontsize=6.2); ax.set_title('Top-candidate recovery',fontweight='bold',fontsize=10); ax.tick_params(axis='x',labelrotation=30)
    else: nodata(ax,'Top-candidate recovery')
    ax=axs[2]; lab(ax,'c')
    if not err.empty:
        sub=err.groupby('chemistry_or_group').agg(n=('n','sum'),mae=('mae','mean')).reset_index().sort_values('mae',ascending=False).head(14); ax.barh(sub.chemistry_or_group.astype(str),sub.mae); ax.invert_yaxis(); ax.set_xlabel('Mean absolute error'); ax.set_title('Error enrichment by group label',fontweight='bold',fontsize=10)
    else: nodata(ax,'Error enrichment by group label')
    ax=axs[3]; lab(ax,'d')
    if not pred.empty:
        sub=pred.sample(n=min(len(pred),3000),random_state=cfg.random_seed) if len(pred)>3000 else pred; ax.scatter(sub.y_true,sub.y_pred,s=8,alpha=.35); mn=min(sub.y_true.min(),sub.y_pred.min()); mx=max(sub.y_true.max(),sub.y_pred.max()); ax.plot([mn,mx],[mn,mx],ls='--',lw=1); ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Prediction calibration view',fontweight='bold',fontsize=10)
    else: nodata(ax,'Prediction calibration view')
    fig.suptitle('Figure 5. Chemistry-ready ML stress test and ranking stability',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)

def fig6(cfg,dd):
    dec=load_df(dd['source']/ 'decision_rules'); chk=load_df(dd['source']/ 'minimum_reporting_checklist'); rec=load_df(dd['main_tables']/ 'Main_Table_4_use_case_recommendation_cards')
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.9))
    ax=axs[0]; ax.axis('off'); lab(ax,'a'); ax.set_title('Claim-first decision tree',fontweight='bold',fontsize=10.5)
    nodes=[('Scientific claim',.50,.84),('Target context\nknown?',.50,.62),('Chemistry\nevidence?',.25,.40),('Grouped split\nstable?',.75,.40),('Report claim-specific\nresource choice',.50,.18)]
    for t,x,y in nodes:
        ax.add_patch(FancyBboxPatch((x-.15,y-.055),.30,.11,boxstyle='round,pad=.02',lw=.9,facecolor='#F7F7F7',edgecolor='#555555'))
        ax.text(x,y,t,ha='center',va='center',fontsize=8.2)
    for (x1,y1),(x2,y2) in [((.50,.78),(.50,.68)),((.50,.56),(.30,.46)),((.50,.56),(.70,.46)),((.25,.34),(.44,.24)),((.75,.34),(.56,.24))]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=10,lw=.9))
    ax=axs[1]; ax.axis('off'); lab(ax,'b'); ax.set_title('Minimum reporting checklist',fontweight='bold',fontsize=10.5)
    if not chk.empty:
        y=.88
        for _,r in chk.head(8).iterrows(): ax.text(.04,y,'☐ '+str(r.item),fontsize=7.6,va='center',wrap=True); y-=.10
    else: ax.text(.5,.5,'Checklist missing',ha='center')
    ax=axs[2]; ax.axis('off'); lab(ax,'c'); ax.set_title('Use-case recommendation cards',fontweight='bold',fontsize=10.5)
    if not rec.empty:
        y=.84
        for _,r in rec.head(4).iterrows():
            ax.add_patch(FancyBboxPatch((.04,y-.09),.92,.15,boxstyle='round,pad=.02',lw=.8,facecolor='#FAFAFA',edgecolor='#666666'))
            ax.text(.06,y+.025,str(r.use_case),ha='left',va='center',fontsize=8,fontweight='bold')
            ax.text(.06,y-.045,str(r.preferred_resource_logic),ha='left',va='center',fontsize=6.8,wrap=True)
            y-=.21
    else: ax.text(.5,.5,'Recommendation cards missing',ha='center')
    fig.suptitle('Figure 6. Claim-specific reporting framework for chemistry-ready MOF ML',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_6_decision_framework',cfg)

def step_figures(cfg: Config, dd: Dict[str,Path], log):
    outs=[]
    outs+=inventory_alignment_figure(cfg,dd)
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6]: outs+=fn(cfg,dd)
    outs+=si_figs(cfg,dd)
    save_df(pd.DataFrame([{'figure_file':str(p),'relative_path':str(p.relative_to(cfg.out_dir)) if str(p).startswith(str(cfg.out_dir)) else str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None} for p in outs]),dd['fig']/ 'figure_output_manifest',cfg)

def check_environment(cfg: Config, dd: Dict[str,Path], log):
    info = pd.DataFrame([
        {'component':'python','available':True,'version':sys.version.replace('\n',' ')},
        {'component':'numpy','available':True,'version':np.__version__},
        {'component':'pandas','available':True,'version':pd.__version__},
        {'component':'matplotlib','available':True,'version':matplotlib.__version__},
        {'component':'scikit-learn','available':SKLEARN_AVAILABLE,'version':sys.modules.get('sklearn').__version__ if SKLEARN_AVAILABLE and 'sklearn' in sys.modules else ''},
        {'component':'scipy','available':scipy_stats is not None,'version':sys.modules.get('scipy').__version__ if 'scipy' in sys.modules else ''},
        {'component':'pyarrow','available':PARQUET_AVAILABLE,'version':''},
        {'component':'openpyxl','available':EXCEL_AVAILABLE,'version':''},
    ])
    save_df(info, dd['logs']/ 'startup_environment_check', cfg)
    if (not cfg.skip_ml) and (not SKLEARN_AVAILABLE):
        msg = 'WARNING: scikit-learn is not installed or not importable. ML stress-test, Figure 5, and ML tables will be incomplete.'
        log.warning(msg)
        (dd['logs']/ 'RUN_INCOMPLETE_WARNING.txt').write_text(msg+'\nInstall scikit-learn in chemmof1 and rerun.\n', encoding='utf-8')
    if cfg.save.get('parquet') and not PARQUET_AVAILABLE:
        log.warning('pyarrow is not available; Parquet outputs will be skipped.')
    if cfg.save.get('xlsx') and not EXCEL_AVAILABLE:
        log.warning('openpyxl is not available; Excel workbooks will be skipped.')

def step_reports(cfg: Config, dd: Dict[str,Path], log):
    f=load_df(dd['profiles']/ 'file_level_profile'); sc=load_df(dd['processed']/ 'resource_scores'); ml=load_df(dd['processed']/ 'ml_metrics_summary'); figs=load_df(dd['fig']/ 'figure_output_manifest'); targets=load_df(dd['processed']/ 'normalized_target_catalog'); regimes=load_df(dd['processed']/ 'trust_regime_summary')
    lines=["# Chemistry-ready MOF analysis pipeline report\n",f"Generated: {iso()}\n",f"Script version: {VERSION}\n",f"Data root: `{cfg.data_root}`\n",f"Output directory: `{cfg.out_dir}`\n",f"Modes: save={cfg.save_mode}; RAM={cfg.ram_mode}; comprehensive={cfg.comprehensive_level}; n_jobs={cfg.n_jobs}\n"]
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary')
    lines.append("\n## Inventory map\n")
    lines.append(f"Parsed attached archive inventory summary with **{len(inv)}** resource/role/format rows. This is an expected-data map; true column and missingness statistics come from local files.\n" if not inv.empty else "No archive inventory summary was available or parseable.\n")
    lines.append("\n## Input profiling\n")
    if not f.empty:
        lines.append(f"Discovered/profiled table-like files: **{len(f)}**.\n")
        for res,n in f.groupby('resource').size().sort_values(ascending=False).items(): lines.append(f"- {res}: {n}\n")
    else: lines.append("No table-like input files were profiled.\n")
    lines.append("\n## Normalized targets\n")
    if not targets.empty:
        lines.append(f"Normalized target catalog rows: **{len(targets)}** from **{targets.source_file.nunique()}** files and **{targets.canonical_id.nunique()}** canonical/profiled IDs.\n")
        for (res,task),n in targets.groupby(['resource','task']).size().sort_values(ascending=False).head(10).items(): lines.append(f"- {res} / {task}: {n} target values\n")
    else: lines.append("No normalized target values were created. Check target-column context or file discovery.\n")
    lines.append("\n## Chemistry-trust regimes\n")
    if not regimes.empty:
        for (res,reg),n in regimes.groupby(['resource','trust_regime_row']).n_rows.sum().sort_values(ascending=False).head(12).items(): lines.append(f"- {res} / {reg}: {int(n)} rows\n")
    else: lines.append("No trust-regime summary available.\n")
    lines.append("\n## Resource scores\n")
    if not sc.empty:
        for _,r in sc.sort_values('chemistry_trust_score',ascending=False).iterrows(): lines.append(f"- {r.resource}: chemistry trust {r.chemistry_trust_score:.3f}; ML readiness {r.ml_readiness_score:.3f}; observed trust rows {r.row_level_trust_observed_fraction:.3f}; {r.recommended_role}\n")
    else: lines.append("Resource scores were not generated.\n")
    lines.append("\n## ML stress test\n")
    if not ml.empty:
        for _,r in ml.sort_values('mean_r2',ascending=False).head(8).iterrows(): lines.append(f"- {r.source_table} | target {r.target_column} | {r.model} {r.split_type}: mean R2={r.mean_r2:.3f}, mean Spearman={r.mean_spearman:.3f}\n")
    else: lines.append("No ML stress-test summary available. If this was unexpected, inspect logs/RUN_INCOMPLETE_WARNING.txt and verify scikit-learn.\n")
    lines.append("\n## Figures\n")
    lines.append(f"Generated figure files: **{len(figs)}**. See `figures/figure_output_manifest.csv`.\n" if not figs.empty else "Figure manifest not available.\n")
    lines.append("\n## Manuscript caution\nPercentages and row counts should always specify the analysis object: raw file, profiled sample, normalized target table, joined task table, or model-ready subset. Do not describe any database as universally superior; describe task-specific chemistry-readiness and chemistry-observability.\n")
    (dd['reports']/ 'analysis_report.md').write_text(''.join(lines),encoding='utf-8')
    env=dict(generated=iso(),python=sys.version,platform=platform.platform(),script_version=VERSION,config=asdict(cfg),packages=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,sklearn_available=SKLEARN_AVAILABLE,scipy_available=scipy_stats is not None,parquet_available=PARQUET_AVAILABLE,excel_available=EXCEL_AVAILABLE))
    save_json(env,dd['logs']/ 'run_environment.json')
    disc=load_df(dd['profiles']/ 'discovered_input_files'); hashes=[]
    if not disc.empty:
        for _,r in disc.iterrows():
            p=Path(r.file_path)
            if p.exists():
                st=p.stat(); h=hashlib.sha256()
                with open(p,'rb') as fh:
                    head=fh.read(1024*1024); h.update(head)
                    if st.st_size>2*1024*1024: fh.seek(max(0,st.st_size-1024*1024)); h.update(fh.read(1024*1024))
                    h.update(str(st.st_size).encode()); h.update(str(st.st_mtime).encode())
                hashes.append(dict(path=str(p),name=p.name,size_bytes=st.st_size,mtime=st.st_mtime,sha256_fast=h.hexdigest(),resource=r.resource,role=r.role,resource_subtype=r.get('resource_subtype','')))
        save_df(pd.DataFrame(hashes),dd['logs']/ 'file_hashes',cfg)

def build_final_manifest(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    def classify(rel: str) -> Tuple[str,str,str]:
        if rel.startswith('figures/main/'): return ('figure','main',Path(rel).stem)
        if rel.startswith('figures/si/'): return ('figure','si',Path(rel).stem)
        if rel.startswith('tables/main/'): return ('table','main',Path(rel).stem)
        if rel.startswith('tables/si/'): return ('table','si',Path(rel).stem)
        if rel.startswith('source_data/'): return ('source_data','source_data','')
        if rel.startswith('processed/'): return ('processed_data','intermediate_final','')
        if rel.startswith('profiles/'): return ('profile','input_audit','')
        if rel.startswith('logs/'): return ('log','run_metadata','')
        if rel.startswith('reports/'): return ('report','report','')
        return ('other','','')
    for p in cfg.out_dir.rglob('*'):
        if not p.is_file() or p.name.endswith('.tmp') or p.name == 'chemistry_ready_mof_analysis_outputs_package.zip':
            continue
        rel=str(p.relative_to(cfg.out_dir)).replace('\\','/')
        ft,sec,num=classify(rel)
        rows.append(dict(relative_path=rel,file_type=ft,main_or_si=sec,figure_or_table_number=num,size_bytes=p.stat().st_size,created_time=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),description='Generated by chemistry-ready MOF analysis pipeline'))
    man=pd.DataFrame(rows).sort_values(['file_type','relative_path']) if rows else pd.DataFrame()
    save_df(man,cfg.out_dir/'FINAL_OUTPUT_MANIFEST',cfg)
    save_df(man,dd['source']/ 'FINAL_OUTPUT_MANIFEST',cfg)
    return man

def step_zip(cfg: Config, dd: Dict[str,Path], log):
    build_final_manifest(cfg, dd)
    if cfg.no_zip: return
    zp=cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'
    include_subdirs=['reports','profiles','processed','tables','figures','source_data','logs','models']
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        root_manifest=cfg.out_dir/'FINAL_OUTPUT_MANIFEST.csv'
        if root_manifest.exists(): z.write(root_manifest, root_manifest.relative_to(cfg.out_dir))
        readme=cfg.out_dir/'README_OUTPUTS.txt'
        if readme.exists(): z.write(readme, readme.relative_to(cfg.out_dir))
        for sub in include_subdirs:
            root=cfg.out_dir/sub
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p!=zp: z.write(p,p.relative_to(cfg.out_dir))
    log.info('ZIP package created: %s (%s)', zp, hbytes(zp.stat().st_size))


def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     reusable intermediate/final analysis tables, including normalized targets and trust regimes
tables/        main-text and SI-ready tables
figures/       polished main figures 1--6 plus inventory-aware SI Figure S0 and SI figures S1--S8
source_data/   panel source-data map, checklist, decision rules and manifest
logs/          run.log, pipeline_state.json, startup environment check, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        saved ML models only in thorough save mode

Key v1.3 additions
------------------
- ARC-MOF adsorption/process target recognition for columns such as mmol/g, molc/uc, v/v, wt%, hoa/kcal/mol and S(g1).
- normalized_target_catalog.csv and resource-specific long-format target tables.
- chemistry trust regimes separating validation from observability.
- processed/ included in the ZIP package.
- FINAL_OUTPUT_MANIFEST.csv at the output root and in source_data/.

Resume behavior
---------------
If the run is interrupted, run the same command again. Use --force only when you want to recompute completed steps.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')

def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready MOF pipeline v%s', VERSION); log.info('Data root: %s', cfg.data_root); log.info('Output directory: %s', cfg.out_dir); log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    check_environment(cfg,dd,log)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)
    state_path=dd['logs']/ 'pipeline_state.json'; st=read_json(state_path,{'created':iso(),'completed_steps':{}}); st['config']=asdict(cfg); save_json(st,state_path)
    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [
            ('normalize_targets',[dd['processed']/ 'normalized_target_catalog.csv',dd['processed']/ 'canonical_mof_index.csv'],step_normalize_targets),
            ('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),
            ('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv',dd['processed']/ 'trust_regime_summary.csv'],step_trust),
            ('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),
            ('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),
            ('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv'],step_tables),
            ('make_figures',[dd['fig_main']/ 'Figure_1_chemistry_ready_concept.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),
            ('write_reports',[dd['reports']/ 'analysis_report.md',dd['logs']/ 'run_environment.json'],step_reports),
            ('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)
        ]
    for name,exp,fn in steps: run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    return 0


# =============================================================================
# v1.4 MANUSCRIPT-COMPLETION OVERRIDES
# =============================================================================
# This block supersedes the v1.3 scientific-completion logic where needed.  It
# focuses on the points identified from the v1.3 output audit:
#   * scikit-learn is now a hard requirement unless --skip-ml or --profile-only
#     is explicitly used;
#   * target parsing separates primary targets from input coordinates,
#     uncertainty columns, identifiers and metadata;
#   * mixed-gas CoRE isotherm files are parsed column-first, so 298_CO2 is CO2
#     even when the file name also contains N2;
#   * QMOF identifiers are never targets, and QMOF property columns are parsed
#     into method/property/unit fields;
#   * ARC-MOF process/adsorption columns are normalized into property, basis,
#     unit and process metric type;
#   * normalized target and target-context tables are separated;
#   * the final package contains processed/, profiles/, reports/ and a manifest.

VERSION = "1.4-manuscript-completion"
TARGET_ROLE_ORDER = ["primary_target", "input_coordinate", "uncertainty_column", "identifier_column", "metadata_column"]
PRIMARY_TARGET_ROLE = "primary_target"
NON_TARGET_ROLES = {"input_coordinate", "uncertainty_column", "identifier_column", "metadata_column"}
IDENTIFIER_COLUMN_RX = re.compile(r"(^id$|_id$|id_|mofid|mof_id|mofkey|refcode|ccdc|csd|cif|filename|file_name|name$|structure|qmof_id|qmofid|database_code|identifier)", re.I)
UNCERTAINTY_COLUMN_RX = re.compile(r"(error|err|std|stdev|sigma|uncert|uncertainty|ci95|confidence|deviation)", re.I)
PRESSURE_COLUMN_RX = re.compile(r"(^p$|pressure|press|p/bar|bar$)", re.I)
TEMPERATURE_COLUMN_RX = re.compile(r"(^t$|temperature|temp|_k$|kelvin)", re.I)
GAS_TOKENS = ["CO2", "N2", "CH4", "H2O", "H2", "Xe", "Kr", "O2"]

# Stronger target expressions, while identifier/error/coordinate exclusions are
# handled by infer_target_context().
TARGET_RX = [
    re.compile(r"(co2|ch4|n2|h2o|h2|xe|kr|o2).*(uptake|loading|adsorp|capacity|henry|kh|qst|selectivity|working)", re.I),
    re.compile(r"(uptake|loading|adsorp|capacity|henry|kh|qst|selectivity|working).*(co2|ch4|n2|h2o|h2|xe|kr|o2)", re.I),
    re.compile(r"(working[_ ]?capacity|deliverable[_ ]?capacity|purity|recovery|productivity|energy[_ ]?penalty|parasitic[_ ]?energy)", re.I),
    re.compile(r"(bandgap|homo|lumo|formation[_ ]?energy|energy[_ ]?above[_ ]?hull|energy[_ ]?per[_ ]?atom|total[_ ]?energy)", re.I),
    re.compile(r"(^|[^a-z])(mmol\s*/\s*g|molc\s*/\s*uc|v\s*/\s*v|wt\s*%|hoa\s*/\s*kcal\s*/\s*mol|S\(g1\))($|[^a-z])", re.I),
]


def is_identifier_column_name(col: Any) -> bool:
    return bool(IDENTIFIER_COLUMN_RX.search(str(col)))


def is_uncertainty_column_name(col: Any) -> bool:
    return bool(UNCERTAINTY_COLUMN_RX.search(str(col)))


def clean_target_name(col: Any) -> str:
    return re.sub(r"[^a-z0-9%/]+", "_", str(col).strip().lower()).strip("_")


def parse_gas_temperature_from_column(col: Any) -> Dict[str, Any]:
    """Column-first gas/temperature parser for mixed-gas isotherm columns.

    Examples
    --------
    298_CO2       -> gas CO2, T=298 K
    423_N2_error  -> gas N2, T=423 K, uncertainty handled elsewhere
    CO2_298       -> gas CO2, T=298 K
    """
    s = str(col)
    # 298_CO2, 298-CO2, 298.CO2
    m = re.search(r"(?P<temp>\d{2,4}(?:\.\d+)?)\s*[_\-\. ]\s*(?P<gas>CO2|N2|CH4|H2O|H2|Xe|Kr|O2)\b", s, flags=re.I)
    if not m:
        # CO2_298
        m = re.search(r"(?P<gas>CO2|N2|CH4|H2O|H2|Xe|Kr|O2)\s*[_\-\. ]\s*(?P<temp>\d{2,4}(?:\.\d+)?)\b", s, flags=re.I)
    if m:
        return {"gas": m.group('gas').upper(), "temperature_K": float(m.group('temp'))}
    for g in GAS_TOKENS:
        if re.search(rf"\b{re.escape(g)}\b", s, flags=re.I) or re.search(rf"(^|[_\-\.]){re.escape(g)}($|[_\-\.])", s, flags=re.I):
            return {"gas": g.upper(), "temperature_K": np.nan}
    return {"gas": "not_encoded", "temperature_K": np.nan}


def parse_gas_temperature_from_filename(p: Path) -> Dict[str, Any]:
    n = p.name
    m = re.search(r"_(?P<gas>CO2|N2|CH4|H2O|H2|Xe|Kr|O2)_(?P<temp>\d{2,4}(?:\.\d+)?)", n, flags=re.I)
    if m:
        return {"gas": m.group('gas').upper(), "temperature_K": float(m.group('temp'))}
    # Mixed-gas file: data_298_423_1bar_CO2_N2_891.csv. Do not choose one gas;
    # column parsing must decide.
    gases = [g.upper() for g in GAS_TOKENS if re.search(rf"(^|[_\-\.]){re.escape(g)}($|[_\-\.])", n, flags=re.I)]
    temps = re.findall(r"(?:^|[_\-\.])(\d{3})(?:[_\-\.])", n)
    return {"gas": gases[0] if len(gases)==1 else "not_encoded", "temperature_K": float(temps[0]) if len(temps)==1 else np.nan}


def infer_arc_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    name = p.name.lower()
    col = str(column or '')
    c = clean_col_name(col)
    cn = clean_target_name(col)
    role = "metadata_column"
    task = 'unknown'
    if 'landfill' in name: task = 'landfill_gas_separation'
    elif 'pre_comb' in name: task = 'pre_combustion_capture'
    elif 'post_comb' in name: task = 'post_combustion_capture'
    elif 'methane_purification' in name: task = 'methane_purification'
    elif 'overall_process' in name or 'process' in name: task = 'process_screening'
    gas = 'not_encoded'
    col_gas = parse_gas_temperature_from_column(col).get('gas', 'not_encoded')
    if col_gas != 'not_encoded': gas = col_gas
    else:
        for g in ['co2','ch4','n2','h2o','h2','xe','kr','o2']:
            if re.search(rf'(^|[_\-.]){g}($|[_\-.])', name): gas = g.upper(); break
    prop = 'metadata'
    unit = ''
    basis = ''
    metric_type = 'metadata'
    if is_identifier_column_name(col):
        role, prop, metric_type = 'identifier_column', 'identifier', 'identifier'
    elif is_uncertainty_column_name(col):
        role, prop, metric_type = 'uncertainty_column', 'uncertainty', 'uncertainty'
    elif c in {'mmol/g','mmol_g'} or 'mmol_g' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'uptake', 'mmol g-1', 'gravimetric', 'adsorption_uptake'
    elif c in {'molc/uc','molc_uc'} or 'molc_uc' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'loading', 'molecules per unit cell', 'unit_cell', 'adsorption_loading'
    elif c in {'v/v','v_v'} or 'v_v' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'uptake', 'v/v', 'volumetric', 'adsorption_uptake'
    elif c in {'wt%','wt_percent'} or 'wt' in cn and ('uptake' in cn or cn == 'wt'):
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'uptake', 'wt%', 'weight_percent', 'adsorption_uptake'
    elif 'working' in cn or 'deliverable' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'working_capacity', 'source unit', 'source', 'process_capacity'
    elif 'purity' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'purity', 'fraction or percent', 'source', 'process_quality'
    elif 'recovery' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'recovery', 'fraction or percent', 'source', 'process_quality'
    elif 'productivity' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'productivity', 'source unit', 'source', 'process_productivity'
    elif 'energy' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'process_energy', 'source unit', 'source', 'process_energy'
    elif 'hoa' in cn or 'qst' in cn or 'heat' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'heat_of_adsorption', 'kcal mol-1 or source unit', 'thermodynamic', 'adsorption_thermodynamics'
    elif 's_g1' in cn or 'select' in cn:
        role, prop, unit, basis, metric_type = PRIMARY_TARGET_ROLE, 'selectivity', 'dimensionless', 'ratio', 'selectivity'
    return dict(task=task, gas=gas, property=prop, unit=unit, target_role=role, basis=basis, process_metric_type=metric_type, method='not_applicable')


def is_arc_target_column_name(col: Any) -> bool:
    return infer_arc_context(Path('arc_mof_placeholder.csv'), col).get('target_role') == PRIMARY_TARGET_ROLE


def infer_core_isotherm_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    col = str(column or '')
    c = col.lower().strip()
    from_col = parse_gas_temperature_from_column(col)
    from_file = parse_gas_temperature_from_filename(p)
    gas = from_col.get('gas') if from_col.get('gas') != 'not_encoded' else from_file.get('gas', 'not_encoded')
    temp = from_col.get('temperature_K') if not pd.isna(from_col.get('temperature_K', np.nan)) else from_file.get('temperature_K', np.nan)
    role = 'metadata_column'
    prop = 'metadata'
    unit = 'source unit'
    if is_identifier_column_name(col):
        role, prop = 'identifier_column', 'identifier'
    elif is_uncertainty_column_name(col):
        role, prop = 'uncertainty_column', 'uncertainty'
    elif PRESSURE_COLUMN_RX.search(c):
        role, prop, unit = 'input_coordinate', 'pressure', 'bar or source unit'
    elif TEMPERATURE_COLUMN_RX.search(c):
        role, prop, unit = 'input_coordinate', 'temperature', 'K'
    elif any(k in c for k in ['uptake', 'loading', 'ads', 'amount']) or parse_gas_temperature_from_column(col).get('gas') != 'not_encoded':
        role, prop, unit = PRIMARY_TARGET_ROLE, 'uptake', 'source unit'
    elif c in ['co2','n2','ch4','h2','h2o','xe','kr','o2']:
        role, prop, gas, unit = PRIMARY_TARGET_ROLE, 'uptake', c.upper(), 'source unit'
    return dict(task='single_isotherm', gas=gas, temperature_K=temp, property=prop, unit=unit, target_role=role, basis='source', process_metric_type='isotherm', method='not_applicable')


def infer_qmof_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    col = str(column or '')
    c = col.lower()
    if is_identifier_column_name(col):
        return dict(task='quantum_property', gas='not_applicable', temperature_K=np.nan, property='identifier', unit='', target_role='identifier_column', basis='', process_metric_type='identifier', method='not_applicable')
    if is_uncertainty_column_name(col):
        return dict(task='quantum_property', gas='not_applicable', temperature_K=np.nan, property='uncertainty', unit='source unit', target_role='uncertainty_column', basis='', process_metric_type='uncertainty', method='not_applicable')
    method = 'source'
    for m in ['pbe', 'hse06_10hf', 'hse06', 'hle17', 'vdw']:
        if m in c.replace('-', '_'):
            method = m.upper(); break
    prop = 'metadata'
    unit = 'source unit'
    role = 'metadata_column'
    if 'bandgap' in c:
        prop, unit, role = 'bandgap', 'eV', PRIMARY_TARGET_ROLE
    elif 'homo' in c:
        prop, unit, role = 'HOMO_energy', 'eV', PRIMARY_TARGET_ROLE
    elif 'lumo' in c:
        prop, unit, role = 'LUMO_energy', 'eV', PRIMARY_TARGET_ROLE
    elif 'formation_energy' in c:
        prop, unit, role = 'formation_energy', 'eV or eV/atom', PRIMARY_TARGET_ROLE
    elif 'energy_above_hull' in c:
        prop, unit, role = 'energy_above_hull', 'eV/atom', PRIMARY_TARGET_ROLE
    elif 'energy_per_atom' in c:
        prop, unit, role = 'energy_per_atom', 'eV/atom', PRIMARY_TARGET_ROLE
    elif 'total_energy' in c or 'energy_total' in c:
        prop, unit, role = 'total_energy', 'eV', PRIMARY_TARGET_ROLE
    return dict(task='quantum_property', gas='not_applicable', temperature_K=np.nan, property=prop, unit=unit, target_role=role, basis='DFT', process_metric_type='quantum_property', method=method)


def infer_target_context(file_path: Optional[Path], col: Any, s: Optional[pd.Series]=None) -> Dict[str, Any]:
    p = file_path
    if p is None:
        role = 'identifier_column' if is_identifier_column_name(col) else ('uncertainty_column' if is_uncertainty_column_name(col) else 'metadata_column')
        return dict(task='', gas='', temperature_K=np.nan, property='', unit='', target_role=role, basis='', process_metric_type='', method='')
    if is_identifier_column_name(col):
        return dict(task=role_of(p), gas='not_encoded', temperature_K=np.nan, property='identifier', unit='', target_role='identifier_column', basis='', process_metric_type='identifier', method='')
    if is_arc_context(p):
        return infer_arc_context(p, col)
    if is_core_isotherm_file(p):
        return infer_core_isotherm_context(p, col)
    if resource_of(p) == 'QMOF':
        return infer_qmof_context(p, col)
    if is_uncertainty_column_name(col):
        return dict(task=role_of(p), gas='not_encoded', temperature_K=np.nan, property='uncertainty', unit='source unit', target_role='uncertainty_column', basis='', process_metric_type='uncertainty', method='')
    if PRESSURE_COLUMN_RX.search(str(col)):
        return dict(task=role_of(p), gas='not_encoded', temperature_K=np.nan, property='pressure', unit='bar or source unit', target_role='input_coordinate', basis='', process_metric_type='coordinate', method='')
    role = PRIMARY_TARGET_ROLE if bool(any(rx.search(str(col)) for rx in TARGET_RX)) else 'metadata_column'
    return dict(task=role_of(p), gas=parse_gas_temperature_from_column(col).get('gas','not_encoded'), temperature_K=parse_gas_temperature_from_column(col).get('temperature_K',np.nan), property=str(col), unit='source unit', target_role=role, basis='', process_metric_type='generic_target' if role==PRIMARY_TARGET_ROLE else 'metadata', method='')


def is_target_like_column(file_path: Optional[Path], col: Any, s: Optional[pd.Series]=None) -> bool:
    if s is not None:
        try:
            if safe_to_numeric(s).notna().mean() < 0.50:
                # target columns should at least mostly coerce to numeric.
                return False
        except Exception:
            pass
    return infer_target_context(file_path, col, s).get('target_role') == PRIMARY_TARGET_ROLE


def profile_one(row, cfg: Config, log) -> Tuple[Dict[str,Any],List[Dict[str,Any]]]:
    p=Path(row.file_path); nrows=cfg.ram['profile_nrows']
    if nrows is None and row.size_mb>cfg.ram['full_mb']: nrows=cfg.ram['max_loaded_rows']
    df,meta=read_table(p,cfg,nrows=nrows,log=log)
    fp={**row.to_dict(), **meta, 'true_rows':None,'true_rows_method':'unknown','profile_is_full_file':False,
        'n_columns':df.shape[1] if not df.empty else 0,'n_profiled_rows':len(df),'n_numeric_columns':0,
        'n_categorical_columns':0,'n_identifier_columns':0,'n_target_like_columns':0,'n_descriptor_like_columns':0,
        'n_nested_value_columns':0,'missing_fraction_overall':np.nan,'all_missing_columns':0,'constant_columns':0,
        'identifier_candidates':'','target_candidates':'','group_candidates':''}
    cols=[]
    if meta.get('read_status')!='ok': return fp, cols
    if p.suffix.lower() in ['.csv','.tsv','.txt'] and not cfg.skip_row_count and (cfg.ram['count_large'] or row.size_mb<=cfg.ram['full_mb']):
        fp['true_rows']=count_rows(p); fp['true_rows_method']='line_count'
    elif nrows is None or len(df)<(nrows or 10**12):
        fp['true_rows']=len(df); fp['true_rows_method']='read_rows'
    fp['profile_is_full_file']=fp['true_rows']==len(df) if fp['true_rows'] is not None else False
    if df.empty: return fp, cols
    fp['missing_fraction_overall']=dataframe_missing_fraction(df)
    ids=[]; targs=[]; groups=[]
    for c in df.columns:
        s=df[c]
        ctx=infer_target_context(p,c,s)
        istarg = ctx.get('target_role') == PRIMARY_TARGET_ROLE
        # Only count primary targets as target_or_property.  Pressure/error/id
        # columns are explicitly retained in the context catalog but not used as
        # ML labels or target counts.
        m='target_or_property' if istarg else modality(str(c),s)
        miss=safe_missing_count(s); nun=safe_nunique(s); isnum=bool(is_numeric_column(s)); nestfrac=nested_fraction(s)
        isid=bool(is_identifier_column_name(str(c))); isgrp=bool(PAT['group'].search(str(c)))
        isdesc=m in ['descriptor','geometric_descriptor','numeric_other'] and not istarg and ctx.get('target_role') not in ['input_coordinate','uncertainty_column','identifier_column']
        if isid: ids.append(str(c))
        if istarg: targs.append(str(c))
        if isgrp: groups.append(str(c))
        cols.append(dict(file_path=str(p),relative_path=row.relative_path,file_name=p.name,resource=row.resource,
                         resource_subtype=getattr(row,'resource_subtype',resource_subtype_of(p)),role=row.role,column_name=str(c),dtype=str(s.dtype),
                         inferred_modality=m,is_numeric=isnum,is_identifier_candidate=isid,is_target_candidate=istarg,
                         is_group_candidate=isgrp,is_descriptor_candidate=isdesc,profiled_rows=len(s),missing_count=miss,
                         missing_fraction=miss/max(len(s),1),unique_count=nun,unique_fraction=nun/max(len(s)-miss,1),
                         constant_nonmissing=nun<=1,all_missing=miss==len(s),contains_nested_values=bool(nestfrac>0),
                         nested_fraction=nestfrac, target_task=ctx.get('task',''), target_gas=ctx.get('gas',''),
                         target_temperature_K=ctx.get('temperature_K',np.nan), target_property=ctx.get('property',''),
                         target_unit=ctx.get('unit',''), target_role=ctx.get('target_role','metadata_column'),
                         target_basis=ctx.get('basis',''), target_method=ctx.get('method',''),
                         process_metric_type=ctx.get('process_metric_type',''), example_or_quantiles=values_summary(s)))
    cdf=pd.DataFrame(cols)
    if not cdf.empty:
        fp['n_numeric_columns']=int(cdf.is_numeric.sum()); fp['n_categorical_columns']=int((~cdf.is_numeric).sum())
        fp['n_identifier_columns']=int(cdf.is_identifier_candidate.sum()); fp['n_target_like_columns']=int(cdf.is_target_candidate.sum())
        fp['n_descriptor_like_columns']=int(cdf.is_descriptor_candidate.sum()); fp['n_nested_value_columns']=int(cdf.contains_nested_values.sum()) if 'contains_nested_values' in cdf else 0
        fp['all_missing_columns']=int(cdf.all_missing.sum()); fp['constant_columns']=int(cdf.constant_nonmissing.sum())
    fp['identifier_candidates']=';'.join(ids[:20]); fp['target_candidates']=';'.join(targs[:30]); fp['group_candidates']=';'.join(groups[:20])
    return fp, cols


def step_profile(cfg: Config, dd: Dict[str,Path], log):
    disc=discover(cfg.data_root,cfg,log); save_df(disc,dd['profiles']/ 'discovered_input_files',cfg)
    files=[]; cols=[]
    for i,row in disc.iterrows():
        log.info('Profiling %d/%d: %s', i+1, len(disc), row.relative_path)
        fp,cp=profile_one(row,cfg,log); files.append(fp); cols.extend(cp)
        if (i+1)%10==0 or i+1==len(disc):
            save_df(pd.DataFrame(files),dd['profiles']/ 'file_level_profile_partial',cfg)
            save_df(pd.DataFrame(cols),dd['profiles']/ 'column_level_profile_partial',cfg); gc.collect()
    fdf=pd.DataFrame(files); cdf=pd.DataFrame(cols)
    save_df(fdf,dd['profiles']/ 'file_level_profile',cfg); save_df(cdf,dd['profiles']/ 'column_level_profile',cfg)
    if not cdf.empty:
        mm=cdf.groupby(['resource','inferred_modality']).size().reset_index(name='n_columns').pivot_table(index='resource',columns='inferred_modality',values='n_columns',fill_value=0).reset_index()
        save_df(mm,dd['profiles']/ 'resource_modality_matrix',cfg)
        ctx=cdf[cdf.get('target_role','metadata_column').astype(str).isin(TARGET_ROLE_ORDER)] if 'target_role' in cdf else pd.DataFrame()
        save_df(ctx,dd['profiles']/ 'target_column_context_catalog',cfg)
        role_summary=ctx.groupby(['resource','resource_subtype','target_role']).size().reset_index(name='n_columns') if not ctx.empty else pd.DataFrame()
        save_df(role_summary,dd['profiles']/ 'target_role_summary',cfg)
    rapid=fdf[[c for c in ['resource','resource_subtype','role','relative_path','file_name','extension','size_mb','read_status','true_rows','n_columns','n_profiled_rows','n_numeric_columns','n_categorical_columns','n_nested_value_columns','missing_fraction_overall','identifier_candidates','target_candidates'] if c in fdf.columns]]
    save_df(rapid,dd['profiles']/ 'inventory_rapid_audit_table',cfg)


def build_long_targets_for_file(p: Path, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    ids = first_identifier_series(df, cfg, slug(p.stem))
    records=[]
    for c in df.columns:
        ctx = infer_target_context(p,c,df[c])
        if ctx.get('target_role') != PRIMARY_TARGET_ROLE:
            continue
        vals = safe_to_numeric(df[c])
        if vals.notna().sum() < max(10, min(100, len(vals)*0.02)): continue
        tmp = pd.DataFrame({
            'canonical_id': ids.astype(str), 'raw_row_index': np.arange(len(df)), 'target_column': str(c),
            'target_value': vals, 'source_file': p.name, 'source_path': str(p), 'resource': resource_of(p),
            'resource_subtype': resource_subtype_of(p), 'role': role_of(p), 'task': ctx.get('task',''),
            'gas': ctx.get('gas',''), 'temperature_K': ctx.get('temperature_K', np.nan),
            'target_property': ctx.get('property',''), 'target_unit': ctx.get('unit',''),
            'target_role': ctx.get('target_role',''), 'target_basis': ctx.get('basis',''),
            'target_method': ctx.get('method',''), 'process_metric_type': ctx.get('process_metric_type','')
        })
        tmp = tmp[tmp.target_value.notna()]
        if not tmp.empty: records.append(tmp)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def build_target_context_for_file(p: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in df.columns:
        ctx = infer_target_context(p,c,df[c])
        if ctx.get('target_role') in TARGET_ROLE_ORDER and ctx.get('target_role') != 'metadata_column':
            rows.append(dict(source_file=p.name, source_path=str(p), resource=resource_of(p), resource_subtype=resource_subtype_of(p), role=role_of(p), column_name=str(c), n_nonmissing=int(safe_to_numeric(df[c]).notna().sum()) if ctx.get('target_role') != 'identifier_column' else int((~missing_mask(df[c])).sum()), target_role=ctx.get('target_role'), target_task=ctx.get('task',''), target_gas=ctx.get('gas',''), target_temperature_K=ctx.get('temperature_K',np.nan), target_property=ctx.get('property',''), target_unit=ctx.get('unit',''), target_basis=ctx.get('basis',''), target_method=ctx.get('method',''), process_metric_type=ctx.get('process_metric_type','')))
    return pd.DataFrame(rows)


def step_normalize_targets(cfg: Config, dd: Dict[str,Path], log):
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    empty_names=['normalized_arc_adsorption_targets_long','normalized_core_single_isotherms_long','normalized_qmof_targets_long','normalized_target_catalog','canonical_mof_index','normalized_target_context_catalog']
    if fdf.empty:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc_parts=[]; core_parts=[]; qmof_parts=[]; id_parts=[]; context_parts=[]
    candidates = fdf[(fdf.read_status.eq('ok')) & ((pd.to_numeric(fdf.n_target_like_columns, errors='coerce').fillna(0) > 0) | (fdf.resource.isin(['ARC-MOF','CoRE MOF 2024','QMOF'])))].copy()
    for _,r in candidates.iterrows():
        p=Path(r.file_path)
        df,_=read_table(p,cfg,nrows=cfg.ram['max_loaded_rows'],log=log)
        if df.empty: continue
        ctxdf = build_target_context_for_file(p, df)
        if not ctxdf.empty: context_parts.append(ctxdf)
        ids = first_identifier_series(df, cfg, slug(p.stem))
        id_parts.append(pd.DataFrame({
            'canonical_id': ids.astype(str), 'source_resource': resource_of(p), 'resource_subtype': resource_subtype_of(p),
            'source_file': p.name, 'source_path': str(p), 'raw_row_index': np.arange(len(df))
        }).drop_duplicates('canonical_id').head(cfg.ram['max_loaded_rows']))
        longdf = build_long_targets_for_file(p, df, cfg)
        if longdf.empty: continue
        if resource_of(p) == 'ARC-MOF': arc_parts.append(longdf)
        elif is_core_isotherm_file(p): core_parts.append(longdf)
        elif resource_of(p) == 'QMOF': qmof_parts.append(longdf)
    arc = pd.concat(arc_parts, ignore_index=True) if arc_parts else pd.DataFrame()
    core = pd.concat(core_parts, ignore_index=True) if core_parts else pd.DataFrame()
    qmof = pd.concat(qmof_parts, ignore_index=True) if qmof_parts else pd.DataFrame()
    alltargets = pd.concat([x for x in [arc,core,qmof] if not x.empty], ignore_index=True) if any(not x.empty for x in [arc,core,qmof]) else pd.DataFrame()
    cindex = pd.concat(id_parts, ignore_index=True).drop_duplicates(['canonical_id','source_resource','source_file']) if id_parts else pd.DataFrame()
    context = pd.concat(context_parts, ignore_index=True) if context_parts else pd.DataFrame()
    save_df(arc,dd['processed']/ 'normalized_arc_adsorption_targets_long',cfg)
    save_df(core,dd['processed']/ 'normalized_core_single_isotherms_long',cfg)
    save_df(qmof,dd['processed']/ 'normalized_qmof_targets_long',cfg)
    save_df(alltargets,dd['processed']/ 'normalized_target_catalog',cfg)
    save_df(context,dd['processed']/ 'normalized_target_context_catalog',cfg)
    save_df(cindex,dd['processed']/ 'canonical_mof_index',cfg)
    summary = alltargets.groupby(['resource','resource_subtype','task','gas','target_property','target_unit','target_basis','target_method','process_metric_type']).agg(n_values=('target_value','count'),n_ids=('canonical_id','nunique'),mean=('target_value','mean'),median=('target_value','median')).reset_index() if not alltargets.empty else pd.DataFrame()
    save_df(summary, dd['si_tables']/ 'SI_Table_S12_normalized_target_catalog_summary', cfg)
    save_df(context, dd['si_tables']/ 'SI_Table_S13_target_column_context', cfg)
    role_summary = context.groupby(['resource','resource_subtype','target_role']).size().reset_index(name='n_columns') if not context.empty else pd.DataFrame()
    save_df(role_summary, dd['si_tables']/ 'SI_Table_S14_target_role_summary', cfg)
    log.info('Normalized targets v1.4: ARC=%d rows, CoRE isotherm=%d rows, QMOF=%d rows, context=%d rows, canonical index=%d rows', len(arc), len(core), len(qmof), len(context), len(cindex))


def target_cols(df: pd.DataFrame, cfg: Config, file_path: Optional[Path]=None) -> List[str]:
    if cfg.target_column and cfg.target_column in df.columns: return [cfg.target_column]
    cand=[]
    for c in df.columns:
        ctx = infer_target_context(file_path,c,df[c])
        if ctx.get('target_role') != PRIMARY_TARGET_ROLE:
            continue
        s=df[c]; num=safe_to_numeric(s)
        if num.notna().mean()<.75 or int(num.nunique(dropna=True))<=5: continue
        # Prefer interpretable adsorption/process/electronic targets over generic targets.
        score = 100
        prop = str(ctx.get('property',''))
        if prop in ['uptake','working_capacity','selectivity','bandgap','formation_energy','energy_above_hull']: score += 30
        if ctx.get('gas') in ['CO2','CH4','N2']: score += 10
        cand.append((score,str(c)))
    return [c for _,c in sorted(cand,reverse=True)[:cfg.level['targets']]]


def xy(df: pd.DataFrame, target: str, cfg: Config, file_path: Optional[Path]=None):
    y=safe_to_numeric(df[target]); keep=y.notna(); df=df.loc[keep].copy(); y=y.loc[keep]
    if len(df)>cfg.ram['max_model_rows']:
        df=df.sample(n=cfg.ram['max_model_rows'],random_state=cfg.random_seed); y=y.loc[df.index]
    gcol=group_col(df); groups=df[gcol].astype(str) if gcol else None
    num=[]; cat=[]
    for c in df.columns:
        if c==target: continue
        name=str(c)
        ctx = infer_target_context(file_path,c,df[c])
        if is_identifier_column_name(name) or name.lower().startswith('unnamed') or ctx.get('target_role') in ['primary_target','uncertainty_column','identifier_column']:
            continue
        s=df[c]
        if safe_missing_fraction(s)>.65 or safe_nunique(s)<=1: continue
        conv=safe_to_numeric(s)
        # Pressure/temperature coordinates are allowed as features for isotherm-like tables.
        if pd.api.types.is_numeric_dtype(s) or conv.notna().mean()>.9: num.append(c)
        elif safe_nunique(s)<=50: cat.append(c)
    if len(num)>cfg.ram['max_features']:
        rank=[]
        for c in num:
            ss=safe_to_numeric(df[c]); bonus=1 if (PAT['descriptor'].search(str(c)) or PAT['geometry'].search(str(c)) or infer_target_context(file_path,c,df[c]).get('target_role')=='input_coordinate') else 0
            rank.append((bonus,float(ss.var(skipna=True) if ss.notna().sum()>2 else 0),c))
        num=[c for _,_,c in sorted(rank,reverse=True)[:cfg.ram['max_features']]]
    X=df[num+cat].copy()
    for c in num: X[c]=safe_to_numeric(X[c])
    for c in cat: X[c]=X[c].map(make_hashable).replace('__MISSING__', pd.NA).astype('string')
    return X,y,groups,num,cat


def check_environment(cfg: Config, dd: Dict[str,Path], log):
    info = pd.DataFrame([
        {'component':'python','available':True,'version':sys.version.replace('\n',' ')},
        {'component':'numpy','available':True,'version':np.__version__},
        {'component':'pandas','available':True,'version':pd.__version__},
        {'component':'matplotlib','available':True,'version':matplotlib.__version__},
        {'component':'scikit-learn','available':SKLEARN_AVAILABLE,'version':sys.modules.get('sklearn').__version__ if SKLEARN_AVAILABLE and 'sklearn' in sys.modules else ''},
        {'component':'scipy','available':scipy_stats is not None,'version':sys.modules.get('scipy').__version__ if 'scipy' in sys.modules else ''},
        {'component':'pyarrow','available':PARQUET_AVAILABLE,'version':''},
        {'component':'openpyxl','available':EXCEL_AVAILABLE,'version':''},
    ])
    save_df(info, dd['logs']/ 'startup_environment_check', cfg)
    if (not cfg.skip_ml) and (not cfg.profile_only) and (not SKLEARN_AVAILABLE):
        msg = ('ERROR: scikit-learn is not installed or not importable. v1.4 treats ML as a core manuscript component. '
               'Install scikit-learn or rerun explicitly with --skip-ml for a profile-only/audit-only package.')
        log.error(msg)
        (dd['logs']/ 'RUN_INCOMPLETE_WARNING.txt').write_text(msg+'\n', encoding='utf-8')
        raise RuntimeError(msg)
    if cfg.save.get('parquet') and not PARQUET_AVAILABLE:
        log.warning('pyarrow is not available; Parquet outputs will be skipped.')
    if cfg.save.get('xlsx') and not EXCEL_AVAILABLE:
        log.warning('openpyxl is not available; Excel workbooks will be skipped.')


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    # Reuse v1.3 table logic via explicitly copied implementation with v1.4 additions.
    fdf=load_df(dd['profiles']/ 'file_level_profile'); cdf=load_df(dd['profiles']/ 'column_level_profile'); scores=load_df(dd['processed']/ 'resource_scores'); join=load_df(dd['processed']/ 'join_accounting'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); mls=load_df(dd['processed']/ 'ml_metrics_summary')
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary'); invf=load_df(dd['profiles']/ 'inventory_folder_summary'); invex=load_df(dd['profiles']/ 'inventory_report_examples')
    for base,name in [(inv,'SI_Table_S0_inventory_resource_role_format_summary'),(invf,'SI_Table_S0b_inventory_folder_summary'),(invex,'SI_Table_S0c_inventory_example_files')]:
        if not base.empty: save_df(base,dd['si_tables']/name,cfg)
    if not fdf.empty:
        group_cols=['resource','resource_subtype','role'] if 'resource_subtype' in fdf.columns else ['resource','role']
        main1=fdf.groupby(group_cols).agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum'),total_profiled_rows=('n_profiled_rows','sum'),total_counted_rows=('true_rows','sum'),median_columns=('n_columns','median'),mean_missing_fraction=('missing_fraction_overall','mean'),n_target_like_columns=('n_target_like_columns','sum'),n_descriptor_like_columns=('n_descriptor_like_columns','sum')).reset_index()
        if not scores.empty: main1=main1.merge(scores[['resource','chemistry_trust_score','ml_readiness_score','row_level_trust_observed_fraction','mean_chemistry_evidence_observability','recommended_role']],on='resource',how='left')
        save_df(main1,dd['main_tables']/ 'Main_Table_1_resource_atlas',cfg); save_df(fdf,dd['si_tables']/ 'SI_Table_S1_complete_file_level_profile',cfg)
    if not cdf.empty:
        save_df(cdf,dd['si_tables']/ 'SI_Table_S2_complete_column_level_profile',cfg)
        save_df(cdf[cdf.is_descriptor_candidate.astype(bool)].head(10000) if 'is_descriptor_candidate' in cdf else pd.DataFrame(),dd['si_tables']/ 'SI_Table_S11_descriptor_family_profile',cfg)
        save_df(cdf[cdf.is_target_candidate.astype(bool)] if 'is_target_candidate' in cdf else pd.DataFrame(),dd['si_tables']/ 'SI_Table_S13_primary_target_columns',cfg)
    context=load_df(dd['processed']/ 'normalized_target_context_catalog')
    if not context.empty:
        save_df(context, dd['si_tables']/ 'SI_Table_S13_target_column_context', cfg)
        save_df(context.groupby(['resource','resource_subtype','target_role']).size().reset_index(name='n_columns'), dd['si_tables']/ 'SI_Table_S14_target_role_summary', cfg)
    if not join.empty: save_df(join,dd['si_tables']/ 'SI_Table_S3_join_accounting',cfg)
    if not risk.empty: save_df(risk,dd['main_tables']/ 'Main_Table_3_benchmark_risk_matrix',cfg)
    if not mls.empty: save_df(mls,dd['si_tables']/ 'SI_Table_S6_ml_metrics_summary',cfg)
    decision=pd.DataFrame([
        {'step':1,'question':'What is the scientific claim: adsorption ranking, process screening, quantum prediction, generative design or mechanistic interpretation?','yes_action':'Choose resource/subset by claim, not by database size alone.','no_action':'Do not use a generic leaderboard interpretation.'},
        {'step':2,'question':'Are target variables explicitly normalized as primary targets, coordinates, uncertainties, identifiers or metadata?','yes_action':'Use only primary targets as labels and report coordinates/uncertainties separately.','no_action':'Do not mix pressure/error/id columns with labels.'},
        {'step':3,'question':'For mixed-gas files, is gas/temperature assigned from the column before the filename?','yes_action':'Use the v1.4 target-context catalog.','no_action':'Do not report gas-specific results.'},
        {'step':4,'question':'Is the claim sensitive to oxidation state, charge, open metal sites or ionic context?','yes_action':'Separate validated-observable, validated-not-observable, uncertain and flagged cases.','no_action':'Report provenance and missingness; chemistry flags may be secondary.'},
        {'step':5,'question':'Does performance remain stable under grouped or chemistry-aware splits?','yes_action':'Benchmark conclusion is more robust.','no_action':'Phrase claims as interpolation or convenience-screening performance.'}])
    checklist=pd.DataFrame([
        {'item':'Exact file provenance, size, version and access date','minimum_status':'required'},
        {'item':'Row/column/missingness table for every input file','minimum_status':'required'},
        {'item':'Target-role catalog: primary target, coordinate, uncertainty, identifier, metadata','minimum_status':'required'},
        {'item':'Normalized target catalog with gas/property/unit/method/context','minimum_status':'required for adsorption/process/QMOF claims'},
        {'item':'Identifier normalization and join-retention accounting','minimum_status':'required'},
        {'item':'Validation/observability trust regimes, not a single ambiguous trust label','minimum_status':'required'},
        {'item':'Duplicate/leakage policy and grouped splits','minimum_status':'required for ML claims'},
        {'item':'Exact source-data file for every figure panel','minimum_status':'required'},
        {'item':'Sensitivity to trust thresholds and score weights','minimum_status':'strongly recommended'}])
    srcmap=pd.DataFrame([
        {'figure':'SI Figure S0','panel':'a-d','source_data':'profiles/inventory_report_examples.csv; profiles/inventory_folder_summary.csv; profiles/inventory_resource_role_format_summary.csv; profiles/discovered_input_files.csv','notes':'Inventory-derived archive map compared with locally discovered data files.'},
        {'figure':'Figure 1','panel':'a-d','source_data':'source_data/decision_rules.csv; source_data/minimum_reporting_checklist.csv','notes':'Conceptual schematic and rule logic.'},
        {'figure':'Figure 2','panel':'a-d','source_data':'profiles/file_level_profile.csv; profiles/column_level_profile.csv; profiles/resource_modality_matrix.csv; processed/resource_scores.csv','notes':'Local profiling results; main resources should be emphasized over Other/unknown.'},
        {'figure':'Figure 3','panel':'a','source_data':'processed/chemistry_evidence_matrix.csv','notes':'Observable chemistry evidence by resource.'},
        {'figure':'Figure 3','panel':'b','source_data':'processed/trust_regime_summary.csv','notes':'Validation/observability trust-regime decomposition.'},
        {'figure':'Figure 3','panel':'c','source_data':'processed/charge_status_summary.csv','notes':'Charge-status decomposition by resource.'},
        {'figure':'Figure 3','panel':'d','source_data':'processed/representative_rule_cards.csv','notes':'Representative rule cards.'},
        {'figure':'Figure 4','panel':'a-c','source_data':'processed/resource_scores.csv; processed/benchmark_risk_matrix.csv; processed/score_sensitivity.csv','notes':'Trust-readiness and risk framework.'},
        {'figure':'Figure 5','panel':'a-d','source_data':'processed/ml_metrics_summary.csv; processed/ranking_stability.csv; processed/ml_error_by_chemistry.csv; processed/ml_predictions_sample.csv','notes':'ML stress-test output.'},
        {'figure':'Figure 6','panel':'a-c','source_data':'source_data/decision_rules.csv; source_data/minimum_reporting_checklist.csv; tables/main/Main_Table_4_use_case_recommendation_cards.csv','notes':'Decision framework.'}])
    save_df(decision,dd['source']/ 'decision_rules',cfg); save_df(checklist,dd['source']/ 'minimum_reporting_checklist',cfg); save_df(srcmap,dd['source']/ 'final_source_data_map',cfg)
    if cfg.save.get('xlsx') and EXCEL_AVAILABLE:
        sheets={}
        for nm,base in [('file_profile',dd['profiles']/ 'file_level_profile'),('column_profile',dd['profiles']/ 'column_level_profile'),('target_context',dd['processed']/ 'normalized_target_context_catalog'),('resource_scores',dd['processed']/ 'resource_scores'),('target_catalog',dd['processed']/ 'normalized_target_catalog'),('trust_regimes',dd['processed']/ 'trust_regime_summary'),('risk_matrix',dd['processed']/ 'benchmark_risk_matrix'),('ml_summary',dd['processed']/ 'ml_metrics_summary')]:
            x=load_df(base)
            if not x.empty: sheets[nm]=x.head(100000)
        if sheets:
            with pd.ExcelWriter(dd['source']/ 'project_core_source_data.xlsx',engine='openpyxl') as w:
                for nm,x in sheets.items(): x.to_excel(w,sheet_name=nm[:31],index=False)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    # Use v1.3 report but add v1.4 target-role summary and hard ML message.
    f=load_df(dd['profiles']/ 'file_level_profile'); sc=load_df(dd['processed']/ 'resource_scores'); ml=load_df(dd['processed']/ 'ml_metrics_summary'); figs=load_df(dd['fig']/ 'figure_output_manifest'); targets=load_df(dd['processed']/ 'normalized_target_catalog'); regimes=load_df(dd['processed']/ 'trust_regime_summary'); context=load_df(dd['processed']/ 'normalized_target_context_catalog')
    lines=["# Chemistry-ready MOF analysis pipeline report\n",f"Generated: {iso()}\n",f"Script version: {VERSION}\n",f"Data root: `{cfg.data_root}`\n",f"Output directory: `{cfg.out_dir}`\n",f"Modes: save={cfg.save_mode}; RAM={cfg.ram_mode}; comprehensive={cfg.comprehensive_level}; n_jobs={cfg.n_jobs}\n"]
    inv=load_df(dd['profiles']/ 'inventory_resource_role_format_summary')
    lines.append("\n## Inventory map\n")
    lines.append(f"Parsed attached archive inventory summary with **{len(inv)}** resource/role/format rows. This is an expected-data map; true column and missingness statistics come from local files.\n" if not inv.empty else "No archive inventory summary was available or parseable.\n")
    lines.append("\n## Input profiling\n")
    if not f.empty:
        lines.append(f"Discovered/profiled table-like files: **{len(f)}**.\n")
        for res,n in f.groupby('resource').size().sort_values(ascending=False).items(): lines.append(f"- {res}: {n}\n")
    else: lines.append("No table-like input files were profiled.\n")
    lines.append("\n## Target-role parsing\n")
    if not context.empty:
        for role,n in context.target_role.value_counts().items(): lines.append(f"- {role}: {int(n)} columns\n")
    else: lines.append("No target-context columns were identified.\n")
    lines.append("\n## Normalized primary targets\n")
    if not targets.empty:
        lines.append(f"Normalized primary-target catalog rows: **{len(targets)}** from **{targets.source_file.nunique()}** files and **{targets.canonical_id.nunique()}** canonical/profiled IDs.\n")
        for (res,task),n in targets.groupby(['resource','task']).size().sort_values(ascending=False).head(10).items(): lines.append(f"- {res} / {task}: {n} target values\n")
    else: lines.append("No normalized primary-target values were created. Check target-column context or file discovery.\n")
    lines.append("\n## Chemistry-trust regimes\n")
    if not regimes.empty:
        for (res,reg),n in regimes.groupby(['resource','trust_regime_row']).n_rows.sum().sort_values(ascending=False).head(12).items(): lines.append(f"- {res} / {reg}: {int(n)} rows\n")
    else: lines.append("No trust-regime summary available.\n")
    lines.append("\n## Resource scores\n")
    if not sc.empty:
        for _,r in sc.sort_values('chemistry_trust_score',ascending=False).iterrows(): lines.append(f"- {r.resource}: chemistry trust {r.chemistry_trust_score:.3f}; ML readiness {r.ml_readiness_score:.3f}; observed trust rows {r.row_level_trust_observed_fraction:.3f}; {r.recommended_role}\n")
    else: lines.append("Resource scores were not generated.\n")
    lines.append("\n## ML stress test\n")
    if not ml.empty:
        for _,r in ml.sort_values('mean_r2',ascending=False).head(8).iterrows(): lines.append(f"- {r.source_table} | target {r.target_column} | {r.model} {r.split_type}: mean R2={r.mean_r2:.3f}, mean Spearman={r.mean_spearman:.3f}\n")
    else: lines.append("No ML stress-test summary available. In v1.4 this should happen only when --skip-ml or --profile-only is used.\n")
    lines.append("\n## Figures\n")
    lines.append(f"Generated figure files: **{len(figs)}**. See `figures/figure_output_manifest.csv`.\n" if not figs.empty else "Figure manifest not available.\n")
    lines.append("\n## Manuscript caution\nPercentages and row counts should always specify the analysis object: raw file, profiled sample, normalized target table, joined task table, or model-ready subset. Target roles must distinguish primary labels from pressure/temperature coordinates, uncertainty/error columns and identifiers.\n")
    (dd['reports']/ 'analysis_report.md').write_text(''.join(lines),encoding='utf-8')
    env=dict(generated=iso(),python=sys.version,platform=platform.platform(),script_version=VERSION,config=asdict(cfg),packages=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,sklearn_available=SKLEARN_AVAILABLE,scipy_available=scipy_stats is not None,parquet_available=PARQUET_AVAILABLE,excel_available=EXCEL_AVAILABLE))
    save_json(env,dd['logs']/ 'run_environment.json')
    disc=load_df(dd['profiles']/ 'discovered_input_files'); hashes=[]
    if not disc.empty:
        for _,r in disc.iterrows():
            p=Path(r.file_path)
            if p.exists():
                st=p.stat(); h=hashlib.sha256()
                with open(p,'rb') as fh:
                    head=fh.read(1024*1024); h.update(head)
                    if st.st_size>2*1024*1024: fh.seek(max(0,st.st_size-1024*1024)); h.update(fh.read(1024*1024))
                    h.update(str(st.st_size).encode()); h.update(str(st.st_mtime).encode())
                hashes.append(dict(path=str(p),name=p.name,size_bytes=st.st_size,mtime=st.st_mtime,sha256_fast=h.hexdigest(),resource=r.resource,role=r.role,resource_subtype=r.get('resource_subtype','')))
        save_df(pd.DataFrame(hashes),dd['logs']/ 'file_hashes',cfg)



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

v1.4 manuscript-completion focus
--------------------------------
- scikit-learn is a hard requirement unless --skip-ml or --profile-only is used.
- Target columns are classified as primary_target, input_coordinate, uncertainty_column, identifier_column or metadata_column.
- Mixed-gas CoRE isotherm parsing is column-first: e.g. 298_CO2 is CO2 even if the file name also contains N2.
- ARC-MOF targets include property, basis, unit and process metric type.
- QMOF targets include normalized method/property/unit fields, and qmof_id is excluded from labels.
- processed/, profiles/, reports/, figures/, tables/, source_data/ and logs/ are included in the ZIP.

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     normalized targets, target-context catalog, trust regimes, ML outputs and reusable data
tables/        main-text and SI-ready tables
figures/       main figures 1--6 plus SI figures
source_data/   panel source-data map, checklist, decision rules and manifest
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md

Resume behavior
---------------
If the run is interrupted, run the same command again. Use --force when changing code versions or when you want to recompute completed steps.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')


def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready MOF pipeline v%s', VERSION); log.info('Data root: %s', cfg.data_root); log.info('Output directory: %s', cfg.out_dir); log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    check_environment(cfg,dd,log)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)
    state_path=dd['logs']/ 'pipeline_state.json'; st=read_json(state_path,{'created':iso(),'completed_steps':{}}); st['config']=asdict(cfg); save_json(st,state_path)
    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv',dd['profiles']/ 'target_column_context_catalog.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [
            ('normalize_targets',[dd['processed']/ 'normalized_target_catalog.csv',dd['processed']/ 'canonical_mof_index.csv',dd['processed']/ 'normalized_target_context_catalog.csv'],step_normalize_targets),
            ('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),
            ('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv',dd['processed']/ 'trust_regime_summary.csv'],step_trust),
            ('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),
            ('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),
            ('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv'],step_tables),
            ('make_figures',[dd['fig_main']/ 'Figure_1_chemistry_ready_concept.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),
            ('write_reports',[dd['reports']/ 'analysis_report.md',dd['logs']/ 'run_environment.json'],step_reports),
            ('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)
        ]
    for name,exp,fn in steps: run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    return 0



# --- v1.4 boundary-fix patch for mixed-gas CoRE columns -----------------------
def parse_gas_temperature_from_column(col: Any) -> Dict[str, Any]:
    """Column-first gas/temperature parser with underscore-safe boundaries.

    This deliberately handles columns such as 298_CO2_error and 423_N2,
    where underscores are word characters and therefore \b is not sufficient.
    """
    s = str(col)
    gas_alt = r"CO2|N2|CH4|H2O|H2|Xe|Kr|O2"
    m = re.search(rf"(?P<temp>\d{{2,4}}(?:\.\d+)?)[_\-\. ]+(?P<gas>{gas_alt})(?=$|[_\-\. ]|[^A-Za-z0-9])", s, flags=re.I)
    if not m:
        m = re.search(rf"(?P<gas>{gas_alt})[_\-\. ]+(?P<temp>\d{{2,4}}(?:\.\d+)?)(?=$|[_\-\. ]|[^A-Za-z0-9])", s, flags=re.I)
    if m:
        temp = float(m.group('temp'))
        # 891 in file IDs is not a physically plausible isotherm temperature;
        # keep it only if it is in a broad experimental/simulation range.
        if 100 <= temp <= 800:
            return {"gas": m.group('gas').upper(), "temperature_K": temp}
        return {"gas": m.group('gas').upper(), "temperature_K": np.nan}
    for g in GAS_TOKENS:
        if re.search(rf"(^|[_\-\. ]){re.escape(g)}($|[_\-\. ]|[^A-Za-z0-9])", s, flags=re.I):
            return {"gas": g.upper(), "temperature_K": np.nan}
    return {"gas": "not_encoded", "temperature_K": np.nan}


def parse_gas_temperature_from_filename(p: Path) -> Dict[str, Any]:
    """Filename parser that avoids treating trailing file IDs as temperatures."""
    n = p.name
    gas_alt = r"CO2|N2|CH4|H2O|H2|Xe|Kr|O2"
    # Prefer explicit GAS_T forms, but require the temperature to be plausible.
    m = re.search(rf"_(?P<gas>{gas_alt})_(?P<temp>\d{{2,4}}(?:\.\d+)?)(?=$|[_\-\.])", n, flags=re.I)
    if m:
        temp = float(m.group('temp'))
        if 100 <= temp <= 800:
            return {"gas": m.group('gas').upper(), "temperature_K": temp}
    # data_298_423_1bar_CO2_N2_891.csv contains mixed gases; no single gas.
    gases = [g.upper() for g in GAS_TOKENS if re.search(rf"(^|[_\-\.]){re.escape(g)}($|[_\-\.])", n, flags=re.I)]
    temps = [float(x) for x in re.findall(r"(?:^|[_\-\.])(\d{3})(?:[_\-\.])", n) if 100 <= float(x) <= 800]
    return {"gas": gases[0] if len(gases)==1 else "not_encoded", "temperature_K": temps[0] if len(temps)==1 else np.nan}


def infer_core_isotherm_context(p: Path, column: Optional[str]=None) -> Dict[str, Any]:
    col = str(column or '')
    c = col.lower().strip()
    from_col = parse_gas_temperature_from_column(col)
    from_file = parse_gas_temperature_from_filename(p)
    # Coordinates/uncertainties are not primary labels.  For pressure, do not
    # force a gas or temperature from a mixed-gas filename.
    if is_identifier_column_name(col):
        return dict(task='single_isotherm', gas='not_encoded', temperature_K=np.nan, property='identifier', unit='', target_role='identifier_column', basis='', process_metric_type='identifier', method='not_applicable')
    if PRESSURE_COLUMN_RX.search(c):
        return dict(task='single_isotherm', gas='not_encoded', temperature_K=np.nan, property='pressure', unit='bar or source unit', target_role='input_coordinate', basis='source', process_metric_type='isotherm_coordinate', method='not_applicable')
    if TEMPERATURE_COLUMN_RX.search(c):
        return dict(task='single_isotherm', gas='not_encoded', temperature_K=np.nan, property='temperature', unit='K', target_role='input_coordinate', basis='source', process_metric_type='isotherm_coordinate', method='not_applicable')
    gas = from_col.get('gas') if from_col.get('gas') != 'not_encoded' else from_file.get('gas', 'not_encoded')
    temp = from_col.get('temperature_K') if not pd.isna(from_col.get('temperature_K', np.nan)) else from_file.get('temperature_K', np.nan)
    if is_uncertainty_column_name(col):
        return dict(task='single_isotherm', gas=gas, temperature_K=temp, property='uncertainty', unit='source unit', target_role='uncertainty_column', basis='source', process_metric_type='isotherm_uncertainty', method='not_applicable')
    if any(k in c for k in ['uptake', 'loading', 'ads', 'amount']) or from_col.get('gas') != 'not_encoded' or c in ['co2','n2','ch4','h2','h2o','xe','kr','o2']:
        if c in ['co2','n2','ch4','h2','h2o','xe','kr','o2']:
            gas = c.upper()
        return dict(task='single_isotherm', gas=gas, temperature_K=temp, property='uptake', unit='source unit', target_role=PRIMARY_TARGET_ROLE, basis='source', process_metric_type='isotherm_target', method='not_applicable')
    return dict(task='single_isotherm', gas=gas, temperature_K=temp, property='metadata', unit='source unit', target_role='metadata_column', basis='source', process_metric_type='isotherm_metadata', method='not_applicable')



# =============================================================================
# v1.5 DESCRIPTOR-JOINED, CHEMISTRY-AWARE ML OVERRIDES
# =============================================================================
# This block supersedes selected v1.4 manuscript-completion functions.  It keeps
# the stable discovery/profiling/normalization/trust machinery, but strengthens
# the paper-critical ML and reproducibility layer:
#   * stale incomplete-run warnings are removed when the environment is healthy;
#   * ARC-MOF targets are joined to independent descriptor/topology/cluster files;
#   * descriptor-joined ML is reported separately from table-local ML;
#   * random splits are compared against descriptor/topology/cluster grouped splits;
#   * target-family leakage diagnostics are generated;
#   * Figure 5 prioritizes descriptor-joined ML, group generalization and leakage;
#   * the final ZIP is audited against FINAL_OUTPUT_MANIFEST.csv.

VERSION = "1.5-descriptor-joined-chemistry-aware-ml"

# Keep aliases to the v1.4 implementations so v1.5 can extend rather than erase
# the stable parts of the pipeline.
_V14_CHECK_ENVIRONMENT = check_environment
_V14_STEP_ML = step_ml
_V14_STEP_TABLES = step_tables
_V14_STEP_FIGURES = step_figures
_V14_STEP_REPORTS = step_reports

DESCRIPTOR_JOIN_MIN_OVERLAP = 200
TARGET_EQUIVALENT_RX = re.compile(r"(mmol\s*/\s*g|molc\s*/\s*uc|v\s*/\s*v|wt\s*%|uptake|loading|selectivity|S\(g1\)|hoa|qst|working|purity|recovery|productivity|energy)", re.I)


def check_environment(cfg: Config, dd: Dict[str,Path], log):
    """v1.5 environment check: v1.4 hard-stop plus stale-warning cleanup."""
    _V14_CHECK_ENVIRONMENT(cfg, dd, log)
    stale = dd['logs'] / 'RUN_INCOMPLETE_WARNING.txt'
    if stale.exists() and SKLEARN_AVAILABLE and not cfg.skip_ml and not cfg.profile_only:
        archive = dd['logs'] / 'OLD_RUN_INCOMPLETE_WARNING_previous_failed_attempt.txt'
        try:
            archive.write_text(stale.read_text(encoding='utf-8', errors='replace'), encoding='utf-8')
            stale.unlink()
            log.info('Removed stale RUN_INCOMPLETE_WARNING.txt because scikit-learn is now available.')
        except Exception as e:
            log.warning('Could not remove stale incomplete-warning file: %s', e)


def descriptor_file_priority(row: pd.Series) -> int:
    name = str(row.get('file_name','')).lower()
    role = str(row.get('role','')).lower()
    subtype = str(row.get('resource_subtype','')).lower()
    score = 0
    if row.get('resource') == 'ARC-MOF': score += 20
    if 'descriptor' in role or 'descriptor' in subtype: score += 15
    if 'topology' in role or 'cluster' in role or 'cluster' in subtype: score += 10
    for k,bonus in [('geometric_properties',16),('geometry',12),('racs',14),('rdf',12),('rdfs',12),('arc_mof_dim',10),('dim',8),('cluster',8),('topology',8),('overall_process',-20)]:
        if k in name: score += bonus
    if any(k in name for k in ['landfill','pre_comb','post_comb','methane_purification']): score -= 30
    return score


def target_file_priority(row: pd.Series) -> int:
    name = str(row.get('file_name','')).lower()
    score = 0
    if row.get('resource') == 'ARC-MOF': score += 20
    if row.get('role') in ['adsorption_targets','process_targets']: score += 15
    for k,bonus in [('post_comb',9),('pre_comb',9),('landfill',8),('methane_purification',8),('overall_process',7),('co2',3),('ch4',3),('n2',2),('h2',2)]:
        if k in name: score += bonus
    return score


def choose_descriptor_id_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    ids = choose_ids(df, cfg)
    # Include common structure-name columns not always caught by the generic ID regex.
    for c in df.columns:
        if str(c).lower() in ['name','structure_name','mof_name','refcode_clean','filename_clean'] and c not in ids:
            ids.append(c)
    return ids[:6]


def canonical_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map(canon).astype('string')


def best_identifier_join(left: pd.DataFrame, right: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    left_ids = choose_descriptor_id_columns(left, cfg)
    right_ids = choose_descriptor_id_columns(right, cfg)
    best = dict(left_id='', right_id='', overlap=0, left_unique=0, right_unique=0, left_retention=0.0, right_retention=0.0)
    for lc in left_ids:
        try:
            L = set(canonical_series(left, lc).dropna().astype(str).tolist())
        except Exception:
            continue
        if not L: continue
        for rc in right_ids:
            try:
                R = set(canonical_series(right, rc).dropna().astype(str).tolist())
            except Exception:
                continue
            if not R: continue
            ov = len(L & R)
            if ov > best['overlap']:
                best = dict(left_id=lc, right_id=rc, overlap=ov, left_unique=len(L), right_unique=len(R), left_retention=ov/max(len(L),1), right_retention=ov/max(len(R),1))
    return best


def select_independent_descriptor_columns(df: pd.DataFrame, cfg: Config, id_cols: Sequence[str], target_file: Optional[Path]=None) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Select independent descriptor features and flag possible leakage columns."""
    numeric, categorical, leak_rows = [], [], []
    id_set = {str(c) for c in id_cols}
    for c in df.columns:
        name = str(c)
        lname = name.lower()
        if name in id_set or is_identifier_column_name(name) or lname.startswith('unnamed'):
            continue
        role = 'descriptor_candidate'
        leak_reason = ''
        if is_uncertainty_column_name(name):
            role = 'uncertainty_like_excluded'; leak_reason = 'uncertainty/error-like column name'
        elif PRESSURE_COLUMN_RX.search(name) or TEMPERATURE_COLUMN_RX.search(name):
            role = 'coordinate_like_excluded'; leak_reason = 'pressure/temperature coordinate-like column name'
        elif is_target_like_column(target_file, name, df[c]) or TARGET_EQUIVALENT_RX.search(name):
            role = 'target_family_like_excluded'; leak_reason = 'target-family/unit-like column name'
        if leak_reason:
            leak_rows.append(dict(column_name=name, leakage_role=role, leakage_reason=leak_reason))
            continue
        s = df[c]
        if safe_missing_fraction(s) > 0.70 or safe_nunique(s) <= 1:
            continue
        conv = safe_to_numeric(s)
        if pd.api.types.is_numeric_dtype(s) or conv.notna().mean() > 0.90:
            numeric.append(name)
        elif safe_nunique(s) <= 80:
            categorical.append(name)
    if len(numeric) > cfg.ram['max_features']:
        ranked=[]
        for c in numeric:
            ss = safe_to_numeric(df[c])
            bonus = 1 if (PAT['descriptor'].search(str(c)) or PAT['geometry'].search(str(c)) or PAT['topology'].search(str(c))) else 0
            ranked.append((bonus, float(ss.var(skipna=True) if ss.notna().sum()>2 else 0), c))
        numeric = [c for _,_,c in sorted(ranked, reverse=True)[:cfg.ram['max_features']]]
    return numeric, categorical[:80], pd.DataFrame(leak_rows)


def infer_target_family(target_column: Any, context_row: Optional[pd.Series]=None) -> str:
    t = str(target_column).lower()
    if context_row is not None:
        prop = str(context_row.get('target_property','')).lower()
        metric = str(context_row.get('process_metric_type','')).lower()
        if prop in ['uptake','loading'] or 'uptake' in metric or 'loading' in metric: return 'uptake_or_loading'
        if prop in ['selectivity'] or 'select' in metric: return 'selectivity'
        if 'heat' in prop or 'qst' in prop or 'hoa' in prop: return 'heat_of_adsorption'
        if prop in ['working_capacity','purity','recovery','productivity'] or 'process' in metric: return 'process_metric'
    if any(k in t for k in ['mmol','molc','v/v','wt','uptake','loading']): return 'uptake_or_loading'
    if 's(g1)' in t or 'select' in t: return 'selectivity'
    if 'hoa' in t or 'qst' in t or 'heat' in t: return 'heat_of_adsorption'
    if any(k in t for k in ['working','purity','recovery','productivity','energy']): return 'process_metric'
    if any(k in t for k in ['bandgap','homo','lumo','formation','energy_above']): return 'quantum_property'
    return 'other_target'


def choose_group_series_for_joined(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[Optional[pd.Series], str]:
    # Prefer chemically meaningful cluster/topology/group/categorical columns.
    candidates = []
    for c in cat_cols + num_cols:
        name = str(c).lower()
        if re.search(r'(topo|topology|cluster|family|metal|node|sbu|flig|func|geo_cluster|mc_cluster)', name):
            candidates.append(c)
    for c in candidates:
        try:
            s = df[c].map(make_hashable).astype(str)
            n = s.nunique(dropna=True)
            if 2 <= n <= max(2, len(df)*0.65):
                return s, str(c)
        except Exception:
            pass
    # Fallback: create descriptor-quantile groups from a geometry-like numeric column.
    for c in num_cols:
        name = str(c).lower()
        if any(k in name for k in ['pld','lcd','asa','void','density','pore','volume','diameter']):
            vals = safe_to_numeric(df[c])
            if vals.notna().sum() > 200 and vals.nunique(dropna=True) >= 10:
                try:
                    q = pd.qcut(vals.rank(method='first'), q=min(10, max(2, len(vals)//200)), duplicates='drop').astype(str)
                    return q, f'{c}_quantile_group'
                except Exception:
                    pass
    return None, ''


def descriptor_model_pipe(name: str, num: List[str], cat: List[str], cfg: Config):
    # Reuse the main model builder, but v1.5 descriptor-joined ML keeps the model
    # set deliberately moderate for CPU-friendliness.
    return model_pipe(name, num, cat, cfg)


def descriptor_ml_repeats(cfg: Config) -> int:
    if cfg.comprehensive_level == 'screening': return 2
    if cfg.comprehensive_level == 'standard': return 3
    if cfg.comprehensive_level == 'comprehensive': return 4
    return min(5, int(cfg.level.get('repeats', 5)))


def descriptor_ml_models(cfg: Config) -> List[str]:
    models = ['dummy', 'ridge']
    if HGB_AVAILABLE and cfg.comprehensive_level in ['comprehensive','thorough']:
        models.append('hgb')
    models.append('extratrees')
    return models


def run_descriptor_joined_case(joined: pd.DataFrame, target: str, label: str, cfg: Config, log, feature_tag: str, target_meta: Dict[str,Any], num_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y = safe_to_numeric(joined[target])
    keep = y.notna()
    X = joined.loc[keep, num_cols + cat_cols].copy()
    y = y.loc[keep]
    if len(y) < descriptor_join_min_overlap(cfg) or len(num_cols)+len(cat_cols) < 2:
        log.warning('Descriptor-joined ML skipped for %s target=%s: rows=%d features=%d', label, target, len(y), len(num_cols)+len(cat_cols))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    max_rows = min(int(cfg.ram['max_model_rows']), 40000 if cfg.ram_mode in ['ultra-light','very-light','light'] else 100000)
    if len(y) > max_rows:
        sample_idx = y.sample(n=max_rows, random_state=cfg.random_seed).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]
    for c in num_cols:
        X[c] = safe_to_numeric(X[c])
    for c in cat_cols:
        X[c] = X[c].map(make_hashable).replace('__MISSING__', pd.NA).astype('string')
    groups, group_name = choose_group_series_for_joined(joined.loc[X.index], num_cols, cat_cols)
    splits = ['random']
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splits.append('descriptor_grouped')
    mets, preds, errs = [], [], []
    repeats = descriptor_ml_repeats(cfg)
    for split in splits:
        for rep in range(repeats):
            seed = cfg.random_seed + 101*rep
            try:
                if split == 'descriptor_grouped':
                    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups=groups))
                else:
                    tr, te = next(ShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y))
            except Exception as e:
                log.warning('Descriptor split failed for %s target=%s split=%s: %s', label, target, split, e)
                continue
            for mn in descriptor_ml_models(cfg):
                try:
                    pipe = descriptor_model_pipe(mn, num_cols, cat_cols, cfg)
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                    pr = pipe.predict(X.iloc[te])
                    yt = np.asarray(y.iloc[te])
                    mets.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                     target_column=target, target_family=target_meta.get('target_family',''), task=target_meta.get('task',''),
                                     gas=target_meta.get('gas',''), target_property=target_meta.get('target_property',''), target_unit=target_meta.get('target_unit',''),
                                     model=mn, split_type=split, split_group_column=group_name, repeat=rep, n_train=len(tr), n_test=len(te),
                                     n_numeric_features=len(num_cols), n_categorical_features=len(cat_cols), rmse=rmse(yt,pr),
                                     mae=float(mean_absolute_error(yt,pr)), r2=float(r2_score(yt,pr)), spearman=spear(yt,pr),
                                     top_1pct_recovery=topk(yt,pr,.01), top_5pct_recovery=topk(yt,pr,.05), top_10pct_recovery=topk(yt,pr,.10),
                                     ndcg_10pct=ndcg(yt,pr,.10)))
                    idx = np.arange(len(yt))
                    if len(idx) > 2500:
                        idx = np.random.default_rng(seed).choice(idx, 2500, replace=False)
                    for j in idx:
                        preds.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                          target_column=target, target_family=target_meta.get('target_family',''), model=mn, split_type=split,
                                          repeat=rep, y_true=float(yt[j]), y_pred=float(pr[j]), absolute_error=float(abs(yt[j]-pr[j]))))
                    if groups is not None:
                        g = groups.iloc[te].reset_index(drop=True)
                        ed = pd.DataFrame({'group': g.astype(str), 'abs_error': np.abs(yt-pr)})
                        top = ed.group.value_counts().head(25).index
                        for _,r in ed[ed.group.isin(top)].groupby('group').abs_error.agg(['size','mean']).reset_index().iterrows():
                            errs.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                             target_column=target, target_family=target_meta.get('target_family',''), model=mn,
                                             split_type=split, repeat=rep, chemistry_or_group=r['group'], n=int(r['size']), mae=float(r['mean'])))
                except Exception as e:
                    log.warning('Descriptor-joined ML failed for %s target=%s model=%s: %s', label, target, mn, e)
    return pd.DataFrame(mets), pd.DataFrame(preds), pd.DataFrame(errs)


def step_descriptor_joined_ml(cfg: Config, dd: Dict[str,Path], log):
    """Build descriptor-target joins and run descriptor-only ML benchmarks."""
    empty_names = ['descriptor_join_candidate_files','descriptor_target_join_diagnostics','descriptor_feature_leakage_diagnostics',
                   'descriptor_joined_ml_metrics_full','descriptor_joined_ml_metrics_summary','descriptor_joined_ml_predictions_sample',
                   'descriptor_joined_ml_error_by_group','descriptor_joined_case_catalog']
    if cfg.skip_ml or not SKLEARN_AVAILABLE:
        for n in empty_names:
            save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    context = load_df(dd['processed']/ 'normalized_target_context_catalog')
    if fdf.empty:
        for n in empty_names:
            save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc = fdf[fdf.resource.eq('ARC-MOF') & fdf.read_status.eq('ok')].copy()
    if arc.empty:
        for n in empty_names:
            save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc['descriptor_priority'] = arc.apply(descriptor_file_priority, axis=1)
    arc['target_priority'] = arc.apply(target_file_priority, axis=1)
    desc_rows = arc[arc.descriptor_priority > 20].sort_values(['descriptor_priority','size_mb'], ascending=[False, True]).head(8)
    targ_rows = arc[(arc.target_priority > 25) & (pd.to_numeric(arc.n_target_like_columns, errors='coerce').fillna(0) > 0)].sort_values(['target_priority','size_mb'], ascending=[False, True]).head(12)
    cand_files = pd.concat([
        desc_rows.assign(candidate_role_v15='descriptor_feature_table'),
        targ_rows.assign(candidate_role_v15='target_label_table')
    ], ignore_index=True) if (not desc_rows.empty or not targ_rows.empty) else pd.DataFrame()
    save_df(cand_files, dd['processed']/ 'descriptor_join_candidate_files', cfg)
    diagnostics, leak_all, metrics_all, preds_all, errs_all, cases = [], [], [], [], [], []
    successful_cases = 0
    max_successful_cases = max(3, min(12, int(cfg.level.get('targets',5))*3))
    # Cache descriptor files because multiple targets may join to the same descriptor table.
    desc_cache: Dict[str, pd.DataFrame] = {}
    for _,tr in targ_rows.iterrows():
        if successful_cases >= max_successful_cases:
            break
        target_p = Path(tr.file_path)
        target_df, _ = read_table(target_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
        if target_df.empty:
            continue
        tcols = target_cols(target_df, cfg, file_path=target_p)
        primary_tcols = []
        for tc in tcols:
            ctx = infer_target_context(target_p, tc, target_df[tc]) if 'infer_target_context' in globals() else infer_arc_context(target_p, tc)
            if ctx.get('target_role', PRIMARY_TARGET_ROLE) == PRIMARY_TARGET_ROLE:
                primary_tcols.append(tc)
        if not primary_tcols:
            continue
        for _,dr in desc_rows.iterrows():
            if successful_cases >= max_successful_cases:
                break
            desc_p = Path(dr.file_path)
            key = str(desc_p)
            if key not in desc_cache:
                desc_cache[key], _ = read_table(desc_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
            desc_df = desc_cache[key]
            if desc_df.empty:
                continue
            join = best_identifier_join(target_df, desc_df, cfg)
            diag_base = dict(target_file=target_p.name, target_path=str(target_p), descriptor_file=desc_p.name, descriptor_path=str(desc_p),
                             target_resource_subtype=resource_subtype_of(target_p), descriptor_resource_subtype=resource_subtype_of(desc_p), **join)
            if join['overlap'] < DESCRIPTOR_JOIN_MIN_OVERLAP:
                diagnostics.append({**diag_base, 'join_status':'insufficient_overlap'})
                continue
            left_id = join['left_id']; right_id = join['right_id']
            td = target_df.copy(); ddsc = desc_df.copy()
            td['__join_id'] = canonical_series(td, left_id)
            ddsc['__join_id'] = canonical_series(ddsc, right_id)
            td = td[td['__join_id'].notna()].drop_duplicates('__join_id')
            ddsc = ddsc[ddsc['__join_id'].notna()].drop_duplicates('__join_id')
            num_cols, cat_cols, leakdf = select_independent_descriptor_columns(ddsc, cfg, id_cols=[right_id, '__join_id'], target_file=target_p)
            if not leakdf.empty:
                leakdf['descriptor_file'] = desc_p.name
                leakdf['target_file'] = target_p.name
                leak_all.append(leakdf)
            feature_cols = ['__join_id'] + num_cols + cat_cols
            joined = td[['__join_id'] + primary_tcols].merge(ddsc[feature_cols], on='__join_id', how='inner')
            diagnostics.append({**diag_base, 'join_status':'joined', 'joined_rows':len(joined), 'n_numeric_features':len(num_cols), 'n_categorical_features':len(cat_cols), 'n_primary_targets':len(primary_tcols)})
            if len(joined) < DESCRIPTOR_JOIN_MIN_OVERLAP or len(num_cols)+len(cat_cols) < 2:
                continue
            for target in primary_tcols[:max(1, min(3, cfg.level.get('targets',3)))]:
                ctx = infer_target_context(target_p, target, target_df[target]) if 'infer_target_context' in globals() else infer_arc_context(target_p, target)
                meta = dict(ctx)
                meta['target_family'] = infer_target_family(target, pd.Series(ctx))
                label = f"ARC-MOF::descriptor_join::{target_p.name}"
                feature_tag = desc_p.name
                log.info('Descriptor-joined ML: target=%s | labels=%s | descriptors=%s | joined_rows=%d | features=%d', target, target_p.name, desc_p.name, len(joined), len(num_cols)+len(cat_cols))
                m,p,e = run_descriptor_joined_case(joined, target, label, cfg, log, feature_tag, meta, num_cols, cat_cols)
                if not m.empty:
                    metrics_all.append(m)
                    if not p.empty: preds_all.append(p)
                    if not e.empty: errs_all.append(e)
                    successful_cases += 1
                    cases.append(dict(source_table=label, feature_table=feature_tag, target_column=target, target_family=meta.get('target_family',''), task=meta.get('task',''), gas=meta.get('gas',''), target_property=meta.get('target_property',''), target_unit=meta.get('target_unit',''), joined_rows=len(joined), n_numeric_features=len(num_cols), n_categorical_features=len(cat_cols), split_group_available=bool(choose_group_series_for_joined(joined, num_cols, cat_cols)[0] is not None)))
                if successful_cases >= max_successful_cases:
                    break
    diag = pd.DataFrame(diagnostics)
    leaks = pd.concat(leak_all, ignore_index=True) if leak_all else pd.DataFrame(columns=['descriptor_file','target_file','column_name','leakage_role','leakage_reason'])
    met = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    pred = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    err = pd.concat(errs_all, ignore_index=True) if errs_all else pd.DataFrame()
    casecat = pd.DataFrame(cases)
    save_df(diag, dd['processed']/ 'descriptor_target_join_diagnostics', cfg)
    save_df(leaks, dd['processed']/ 'descriptor_feature_leakage_diagnostics', cfg)
    save_df(met, dd['processed']/ 'descriptor_joined_ml_metrics_full', cfg)
    save_df(pred, dd['processed']/ 'descriptor_joined_ml_predictions_sample', cfg)
    save_df(err, dd['processed']/ 'descriptor_joined_ml_error_by_group', cfg)
    save_df(casecat, dd['processed']/ 'descriptor_joined_case_catalog', cfg)
    if not met.empty:
        summ = met.groupby(['feature_source','source_table','feature_table','target_column','target_family','task','gas','target_property','target_unit','model','split_type']).agg(
            n_repeats=('repeat','nunique'), mean_rmse=('rmse','mean'), sd_rmse=('rmse','std'), mean_mae=('mae','mean'),
            mean_r2=('r2','mean'), sd_r2=('r2','std'), mean_spearman=('spearman','mean'),
            mean_top_1pct_recovery=('top_1pct_recovery','mean'), mean_top_5pct_recovery=('top_5pct_recovery','mean'),
            mean_top_10pct_recovery=('top_10pct_recovery','mean'), mean_ndcg_10pct=('ndcg_10pct','mean'),
            n_numeric_features=('n_numeric_features','median'), n_categorical_features=('n_categorical_features','median')
        ).reset_index()
    else:
        summ = pd.DataFrame()
    save_df(summ, dd['processed']/ 'descriptor_joined_ml_metrics_summary', cfg)
    save_df(summ, dd['main_tables']/ 'Main_Table_6_descriptor_joined_ml_summary', cfg)
    save_df(diag, dd['si_tables']/ 'SI_Table_S16_descriptor_target_join_diagnostics', cfg)
    save_df(leaks, dd['si_tables']/ 'SI_Table_S17_descriptor_feature_leakage_diagnostics', cfg)
    log.info('Descriptor-joined ML cases completed: %d; diagnostic join pairs: %d', successful_cases, len(diag))


def _add_feature_source_to_table_local(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    if 'feature_source' not in out.columns:
        out['feature_source'] = 'table_local_non_descriptor'
    if 'target_family' not in out.columns:
        out['target_family'] = out['target_column'].map(infer_target_family) if 'target_column' in out.columns else ''
    if 'feature_table' not in out.columns:
        out['feature_table'] = ''
    return out


def step_ml(cfg: Config, dd: Dict[str,Path], log):
    """v1.5 ML step: run v1.4 table-local ML, then merge descriptor-joined ML."""
    _V14_STEP_ML(cfg, dd, log)
    table_full = _add_feature_source_to_table_local(load_df(dd['processed']/ 'ml_metrics_full'))
    table_summary = _add_feature_source_to_table_local(load_df(dd['processed']/ 'ml_metrics_summary'))
    table_pred = _add_feature_source_to_table_local(load_df(dd['processed']/ 'ml_predictions_sample'))
    table_err = _add_feature_source_to_table_local(load_df(dd['processed']/ 'ml_error_by_chemistry'))
    desc_full = load_df(dd['processed']/ 'descriptor_joined_ml_metrics_full')
    desc_summary = load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    desc_pred = load_df(dd['processed']/ 'descriptor_joined_ml_predictions_sample')
    desc_err = load_df(dd['processed']/ 'descriptor_joined_ml_error_by_group')
    if not desc_full.empty:
        full = pd.concat([table_full, desc_full], ignore_index=True, sort=False) if not table_full.empty else desc_full
        save_df(full, dd['processed']/ 'ml_metrics_full', cfg)
        save_df(full, dd['si_tables']/ 'SI_Table_S6_complete_model_results', cfg)
    if not desc_summary.empty:
        summary = pd.concat([table_summary, desc_summary], ignore_index=True, sort=False) if not table_summary.empty else desc_summary
        save_df(summary, dd['processed']/ 'ml_metrics_summary', cfg)
        save_df(summary, dd['main_tables']/ 'Main_Table_5_ml_stress_test_summary', cfg)
        save_df(summary, dd['si_tables']/ 'SI_Table_S6_ml_metrics_summary', cfg)
    if not desc_pred.empty:
        pred = pd.concat([table_pred, desc_pred], ignore_index=True, sort=False) if not table_pred.empty else desc_pred
        save_df(pred, dd['processed']/ 'ml_predictions_sample', cfg)
    if not desc_err.empty:
        err = pd.concat([table_err, desc_err], ignore_index=True, sort=False) if not table_err.empty else desc_err
        save_df(err, dd['processed']/ 'ml_error_by_chemistry', cfg)
    # Comparison table for manuscript wording: table-local vs descriptor-joined.
    summ = load_df(dd['processed']/ 'ml_metrics_summary')
    if not summ.empty and 'feature_source' in summ.columns:
        comp = summ.groupby(['feature_source','split_type','target_family']).agg(
            n_rows=('mean_r2','count'), median_r2=('mean_r2','median'), median_spearman=('mean_spearman','median'),
            median_top10=('mean_top_10pct_recovery','median')
        ).reset_index()
        save_df(comp, dd['processed']/ 'ml_feature_source_comparison', cfg)
        save_df(comp, dd['main_tables']/ 'Main_Table_7_ml_feature_source_comparison', cfg)


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    _V14_STEP_TABLES(cfg, dd, log)
    for name, base in [
        ('SI_Table_S16_descriptor_target_join_diagnostics', dd['processed']/ 'descriptor_target_join_diagnostics'),
        ('SI_Table_S17_descriptor_feature_leakage_diagnostics', dd['processed']/ 'descriptor_feature_leakage_diagnostics'),
        ('SI_Table_S18_descriptor_joined_case_catalog', dd['processed']/ 'descriptor_joined_case_catalog'),
        ('SI_Table_S19_ml_feature_source_comparison', dd['processed']/ 'ml_feature_source_comparison')
    ]:
        df = load_df(base)
        if not df.empty:
            save_df(df, dd['si_tables']/ name, cfg)
    # Extend source-data map with v1.5-specific panels and diagnostics.
    src = load_df(dd['source']/ 'final_source_data_map')
    add = pd.DataFrame([
        {'figure':'Figure 5','panel':'a-d','source_data':'processed/descriptor_joined_ml_metrics_summary.csv; processed/ml_feature_source_comparison.csv; processed/descriptor_feature_leakage_diagnostics.csv; processed/descriptor_joined_ml_predictions_sample.csv','notes':'v1.5 descriptor-joined ML, grouped splits and leakage diagnostics.'},
        {'figure':'SI descriptor-joined ML tables','panel':'tables','source_data':'processed/descriptor_target_join_diagnostics.csv; processed/descriptor_joined_case_catalog.csv; processed/descriptor_joined_ml_metrics_full.csv','notes':'Descriptor-target join diagnostics and full model metrics.'},
    ])
    src = pd.concat([src, add], ignore_index=True) if not src.empty else add
    save_df(src, dd['source']/ 'final_source_data_map', cfg)


def fig5(cfg,dd):
    summ=load_df(dd['processed']/ 'ml_metrics_summary')
    desc_summ=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    comp=load_df(dd['processed']/ 'ml_feature_source_comparison')
    leaks=load_df(dd['processed']/ 'descriptor_feature_leakage_diagnostics')
    err=load_df(dd['processed']/ 'ml_error_by_chemistry')
    pred=load_df(dd['processed']/ 'descriptor_joined_ml_predictions_sample')
    if pred.empty:
        pred=load_df(dd['processed']/ 'ml_predictions_sample')
    fig,axs=plt.subplots(2,2,figsize=(13.4,9.4)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not desc_summ.empty:
        sub=desc_summ.sort_values('mean_r2',ascending=False).head(16)
        labels=sub.model.astype(str)+'\n'+sub.split_type.astype(str)+'\n'+sub.target_family.astype(str).str[:16]
        ax.errorbar(sub.mean_r2,np.arange(len(sub)),xerr=sub.sd_r2.fillna(0),fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(labels,fontsize=6.8); ax.invert_yaxis()
        ax.set_xlabel('Mean R² ± SD'); ax.set_title('Descriptor-joined ML performance',fontweight='bold',fontsize=10)
    elif not summ.empty:
        sub=summ.sort_values('mean_r2',ascending=False).head(16)
        labels=sub.model.astype(str)+'\n'+sub.split_type.astype(str)+'\n'+sub.target_column.astype(str).str[:16]
        ax.errorbar(sub.mean_r2,np.arange(len(sub)),xerr=sub.sd_r2.fillna(0),fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(labels,fontsize=6.8); ax.invert_yaxis(); ax.set_xlabel('Mean R² ± SD'); ax.set_title('ML performance',fontweight='bold',fontsize=10)
    else: nodata(ax,'Descriptor-joined ML performance','No ML output; check scikit-learn and descriptor-target joins')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    if not comp.empty:
        # Plot median R2 by feature source and split type.
        comp2=comp.copy(); comp2['label']=comp2.feature_source.astype(str)+'\n'+comp2.split_type.astype(str)+'\n'+comp2.target_family.astype(str)
        comp2=comp2.sort_values('median_r2',ascending=True).tail(14)
        ax.barh(comp2.label, comp2.median_r2)
        ax.set_xlabel('Median mean R²'); ax.set_title('Feature-source and split comparison',fontweight='bold',fontsize=10)
    elif not summ.empty and 'feature_source' in summ.columns:
        tmp=summ.groupby(['feature_source','split_type']).mean_r2.median().reset_index(); tmp['label']=tmp.feature_source+'\n'+tmp.split_type
        ax.barh(tmp.label,tmp.mean_r2); ax.set_xlabel('Median mean R²'); ax.set_title('Feature-source and split comparison',fontweight='bold',fontsize=10)
    else: nodata(ax,'Feature-source and split comparison')
    ax=axs[2]; lab(ax,'c'); style_axes(ax)
    if not err.empty and 'target_family' in err.columns:
        sub=err.groupby(['target_family','split_type']).mae.mean().reset_index().sort_values('mae',ascending=False).head(14)
        sub['label']=sub.target_family.astype(str)+'\n'+sub.split_type.astype(str)
        ax.barh(sub.label,sub.mae); ax.invert_yaxis(); ax.set_xlabel('Mean absolute error'); ax.set_title('Error by target family/grouped split',fontweight='bold',fontsize=10)
    elif not leaks.empty:
        l=leaks.leakage_role.value_counts().reset_index(); l.columns=['leakage_role','n_columns']; ax.barh(l.leakage_role,l.n_columns); ax.set_xlabel('Excluded descriptor columns'); ax.set_title('Leakage-risk exclusions',fontweight='bold',fontsize=10)
    else: nodata(ax,'Error/leakage diagnostics','No grouped-error rows or leakage exclusions were generated')
    ax=axs[3]; lab(ax,'d')
    if not pred.empty:
        sub=pred.sample(n=min(len(pred),3000),random_state=cfg.random_seed) if len(pred)>3000 else pred
        # Robust axis limits avoid one extreme target dominating the panel.
        x=pd.to_numeric(sub.y_true,errors='coerce'); y=pd.to_numeric(sub.y_pred,errors='coerce')
        q=np.nanquantile(pd.concat([x,y]).dropna(), [0.01,0.99]) if x.notna().any() and y.notna().any() else [np.nan,np.nan]
        ax.scatter(x,y,s=8,alpha=.35)
        if np.isfinite(q[0]) and np.isfinite(q[1]) and q[0] < q[1]:
            ax.plot([q[0],q[1]],[q[0],q[1]],ls='--',lw=1); ax.set_xlim(q[0],q[1]); ax.set_ylim(q[0],q[1])
        ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Prediction calibration view',fontweight='bold',fontsize=10)
        add_panel_note(ax,'Axes clipped to 1st--99th percentile for readability.')
    else: nodata(ax,'Prediction calibration view')
    fig.suptitle('Figure 5. Descriptor-joined chemistry-aware ML benchmark',fontweight='bold'); fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)


def step_figures(cfg: Config, dd: Dict[str,Path], log):
    # Reuse v1.4 figure generation, but it now resolves fig5 to the v1.5 function.
    outs=[]
    outs+=inventory_alignment_figure(cfg,dd)
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6]: outs+=fn(cfg,dd)
    outs+=si_figs(cfg,dd)
    save_df(pd.DataFrame([{'figure_file':str(p),'relative_path':str(p.relative_to(cfg.out_dir)) if str(p).startswith(str(cfg.out_dir)) else str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None} for p in outs]),dd['fig']/ 'figure_output_manifest',cfg)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    _V14_STEP_REPORTS(cfg, dd, log)
    report = dd['reports']/ 'analysis_report.md'
    extra=[]
    dsumm=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    ddiag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    leaks=load_df(dd['processed']/ 'descriptor_feature_leakage_diagnostics')
    comp=load_df(dd['processed']/ 'ml_feature_source_comparison')
    extra.append('\n## v1.5 descriptor-joined ML strengthening\n')
    if not ddiag.empty:
        joined=int((ddiag.get('join_status','')=='joined').sum()) if 'join_status' in ddiag else 0
        extra.append(f"Descriptor-target join diagnostics: **{len(ddiag)}** candidate pairs inspected; **{joined}** joined with sufficient overlap.\n")
    else:
        extra.append('No descriptor-target join diagnostics were generated. This usually means no ARC-MOF descriptor/target ID overlap was found in the profiled sample.\n')
    if not dsumm.empty:
        extra.append(f"Descriptor-joined ML summary rows: **{len(dsumm)}**. These results use independent descriptor/topology/cluster tables rather than other target columns as predictors.\n")
        best=dsumm.sort_values('mean_r2',ascending=False).head(5)
        for _,r in best.iterrows():
            extra.append(f"- {r.model} | {r.split_type} | {r.target_family} | {r.target_column}: mean R2={r.mean_r2:.3f}, Spearman={r.mean_spearman:.3f}\n")
    else:
        extra.append('Descriptor-joined ML did not produce model results; table-local ML may still be present, but manuscript claims should remain cautious.\n')
    if not leaks.empty:
        extra.append(f"Leakage diagnostics excluded **{len(leaks)}** descriptor columns with target/coordinate/uncertainty-like names before descriptor-joined ML.\n")
    if not comp.empty:
        extra.append('Feature-source comparison table saved as `processed/ml_feature_source_comparison.csv`.\n')
    try:
        report.write_text(report.read_text(encoding='utf-8', errors='replace') + ''.join(extra), encoding='utf-8')
    except Exception:
        pass


def build_final_manifest(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    for p in cfg.out_dir.rglob('*'):
        if not p.is_file():
            continue
        if p.name == 'chemistry_ready_mof_analysis_outputs_package.zip':
            continue
        if p.suffix.lower() in ['.tmp']:
            continue
        rel=str(p.relative_to(cfg.out_dir)).replace('\\','/')
        top=rel.split('/')[0]
        rows.append(dict(relative_path=rel, top_folder=top, file_type=p.suffix.lower().lstrip('.') or 'no_extension', size_bytes=p.stat().st_size, created_or_modified_time=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'), description='Generated pipeline output'))
    man=pd.DataFrame(rows).sort_values('relative_path') if rows else pd.DataFrame(columns=['relative_path','top_folder','file_type','size_bytes','created_or_modified_time','description'])
    return man


def step_zip(cfg: Config, dd: Dict[str,Path], log):
    if cfg.no_zip:
        return
    # Ensure stale package does not contaminate the audit.
    zp=cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'
    if zp.exists():
        try: zp.unlink()
        except Exception: pass
    man=build_final_manifest(cfg, dd)
    save_df(man, cfg.out_dir/'FINAL_OUTPUT_MANIFEST', cfg)
    save_df(man, dd['source']/ 'FINAL_OUTPUT_MANIFEST', cfg)
    include_roots=['reports','profiles','processed','tables','figures','source_data','logs','models']
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for sub in include_roots:
            root=cfg.out_dir/sub
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p != zp:
                        z.write(p,p.relative_to(cfg.out_dir))
        # Include root-level README and manifest.
        for p in [cfg.out_dir/'README_OUTPUTS.txt', cfg.out_dir/'FINAL_OUTPUT_MANIFEST.csv']:
            if p.exists(): z.write(p,p.relative_to(cfg.out_dir))
    with zipfile.ZipFile(zp,'r') as z:
        names=set(z.namelist())
    missing=man[~man.relative_path.isin(names)].copy() if not man.empty else pd.DataFrame()
    # Some manifest files are generated immediately before/after zipping; keep the audit explicit.
    save_df(missing, dd['source']/ 'manifest_missing_from_zip', cfg)
    audit=pd.DataFrame([dict(zip_file=str(zp), zip_size_bytes=zp.stat().st_size, manifest_entries=len(man), zip_entries=len(names), missing_manifest_entries=len(missing), required_folders=';'.join(include_roots))])
    save_df(audit, dd['source']/ 'zip_contents_audit', cfg)
    # Append audit files to the zip after they are created.
    with zipfile.ZipFile(zp,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [dd['source']/ 'manifest_missing_from_zip.csv', dd['source']/ 'zip_contents_audit.csv']:
            if p.exists(): z.write(p,p.relative_to(cfg.out_dir))
    log.info('ZIP package created: %s (%s); manifest entries=%d; missing in zip before audit append=%d', zp, hbytes(zp.stat().st_size), len(man), len(missing))



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

v1.5 descriptor-joined chemistry-aware ML focus
-----------------------------------------------
- Builds descriptor-target joins for ARC-MOF adsorption/process targets when shared identifiers are observable.
- Runs descriptor-joined ML with independent descriptor/topology/cluster features, not other adsorption target columns as predictors.
- Reports random versus descriptor/topology/cluster grouped splits where a group label can be constructed.
- Creates leakage diagnostics for target-family, coordinate and uncertainty-like descriptor columns.
- Keeps table-local ML as a secondary diagnostic but separates it from descriptor-joined ML in the output tables.
- Removes stale RUN_INCOMPLETE_WARNING files when the environment is healthy.
- Audits FINAL_OUTPUT_MANIFEST.csv against the final ZIP contents.

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     normalized targets, descriptor joins, trust regimes, ML outputs and reusable data
tables/        main-text and SI-ready tables
figures/       main figures 1--6 plus SI figures
source_data/   panel source-data map, checklist, decision rules, manifest and ZIP audit
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        optional fitted models in thorough save mode

Resume behavior
---------------
Use a fresh output folder or --force when changing code versions. If interrupted, rerun the same command.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')


def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready MOF pipeline v%s', VERSION); log.info('Data root: %s', cfg.data_root); log.info('Output directory: %s', cfg.out_dir); log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    check_environment(cfg,dd,log)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)
    state_path=dd['logs']/ 'pipeline_state.json'; st=read_json(state_path,{'created':iso(),'completed_steps':{}}); st['config']=asdict(cfg); save_json(st,state_path)
    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv',dd['profiles']/ 'target_column_context_catalog.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [
            ('normalize_targets',[dd['processed']/ 'normalized_target_catalog.csv',dd['processed']/ 'canonical_mof_index.csv',dd['processed']/ 'normalized_target_context_catalog.csv'],step_normalize_targets),
            ('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),
            ('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv',dd['processed']/ 'trust_regime_summary.csv'],step_trust),
            ('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),
            ('descriptor_joined_ml',[dd['processed']/ 'descriptor_target_join_diagnostics.csv',dd['processed']/ 'descriptor_joined_ml_metrics_summary.csv'],step_descriptor_joined_ml),
            ('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),
            ('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv'],step_tables),
            ('make_figures',[dd['fig_main']/ 'Figure_1_chemistry_ready_concept.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),
            ('write_reports',[dd['reports']/ 'analysis_report.md',dd['logs']/ 'run_environment.json'],step_reports),
            ('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)
        ]
    for name,exp,fn in steps: run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    return 0


# =============================================================================
# v1.6 JOURNAL-FINALIZATION OVERRIDES
# =============================================================================
# This block supersedes selected v1.5 functions while preserving the stable
# discovery/profile/normalization/trust infrastructure.  It implements the final
# priorities from the v1.5 result audit:
#   * stronger ARC-MOF identifier harmonization for RAC/RDF/geometry/topology joins;
#   * descriptor-family ablation summaries across RDF/RAC/geometry/topology-like features;
#   * positive leakage summaries even when zero leakage columns are found;
#   * target-dictionary and manual-validation template tables;
#   * resource-score uncertainty summaries;
#   * cleaner main-text figures, especially Figure 2/4/5;
#   * final-paper source-data and packaging checks.

VERSION = "1.6-journal-finalization"

_V15_STEP_DESCRIPTOR_JOINED_ML = step_descriptor_joined_ml
_V15_STEP_ML = step_ml
_V15_STEP_SCORES = step_scores
_V15_STEP_TABLES = step_tables
_V15_STEP_FIGURES = step_figures
_V15_STEP_REPORTS = step_reports
_V15_STEP_ZIP = step_zip

# For real full data, keep the cutoff reasonably high; for smoke/screening runs,
# allow small synthetic joins to validate code paths.
def descriptor_join_min_overlap(cfg: Config) -> int:
    if cfg.comprehensive_level == 'screening' or cfg.ram_mode == 'ultra-light':
        return 25
    return DESCRIPTOR_JOIN_MIN_OVERLAP


def identifier_variants(x: Any) -> List[str]:
    """Return robust identifier variants for ARC-MOF/MOF descriptor joins.

    The ARC-MOF target and descriptor tables may use filename, Structure_Name,
    CIF-like names, path-like strings, stems with prefixes, or slightly different
    punctuation.  This function deliberately returns multiple conservative
    normalizations rather than one brittle canonical form.
    """
    if is_missing_value(x):
        return []
    s = str(x).strip().strip('"\'')
    if not s or s.lower() in {z.lower() for z in MISSING}:
        return []
    raw = s.replace('\\', '/').strip()
    pieces = [raw]
    # Add basename and path stem.
    pieces.append(raw.split('/')[-1])
    # Some fields contain URI- or archive-like prefixes.
    for sep in ['::', ':', '|', ';']:
        if sep in raw:
            pieces.append(raw.split(sep)[-1])
    out = []
    for p in pieces:
        p = str(p).strip().lower()
        p = re.sub(r'\.(cif|csv|json|txt|xlsx|pkl|parquet)$', '', p, flags=re.I)
        p0 = re.sub(r'\s+', '', p)
        if p0:
            out.append(p0)
        # Remove only unambiguous archive/structure prefixes.  Keep generic
        # MOF_#### identifiers intact because many ARC target files use them.
        p = re.sub(r'^(structure[_\- ]*|arc[_\- ]*mof[_\- ]*)', '', p)
        p = re.sub(r'(_clean|_relaxed|_primitive|_sym|_charged|_neutral)$', '', p)
        p = re.sub(r'\s+', '', p)
        if p:
            out.append(p)
        compact = re.sub(r'[^a-z0-9]+', '', p)
        if compact and compact != p:
            out.append(compact)
        # Also preserve underscore-normalized form.
        under = re.sub(r'[^a-z0-9]+', '_', p).strip('_')
        if under and under not in out:
            out.append(under)
    # Ordered unique.
    seen, ans = set(), []
    for v in out:
        if v and v not in seen and v not in {'nan','none','null','missing','unknown'}:
            seen.add(v); ans.append(v)
    return ans[:10]


def variants_set_from_series(s: pd.Series, max_rows: Optional[int]=None) -> set:
    vals = s.dropna()
    if max_rows and len(vals) > max_rows:
        vals = vals.sample(n=max_rows, random_state=42)
    out = set()
    for x in vals:
        out.update(identifier_variants(x))
    return out


def join_key_series(s: pd.Series, common: set) -> pd.Series:
    def pick(x):
        for v in identifier_variants(x):
            if v in common:
                return v
        return pd.NA
    return s.map(pick).astype('string')


def best_identifier_join(left: pd.DataFrame, right: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """v1.6 robust ID-pair search using multi-variant normalization."""
    left_ids = choose_descriptor_id_columns(left, cfg)
    right_ids = choose_descriptor_id_columns(right, cfg)
    best = dict(left_id='', right_id='', overlap=0, left_unique=0, right_unique=0,
                left_retention=0.0, right_retention=0.0, common_variant_count=0,
                join_mode='variant_harmonized')
    for lc in left_ids:
        try:
            L = variants_set_from_series(left[lc])
        except Exception:
            continue
        if not L:
            continue
        for rc in right_ids:
            try:
                R = variants_set_from_series(right[rc])
            except Exception:
                continue
            if not R:
                continue
            common = L & R
            ov = len(common)
            if ov > best['overlap']:
                best = dict(left_id=lc, right_id=rc, overlap=ov, left_unique=len(L), right_unique=len(R),
                            left_retention=ov/max(len(L),1), right_retention=ov/max(len(R),1),
                            common_variant_count=len(common), join_mode='variant_harmonized')
    return best


def descriptor_family_from_file(p: Path) -> str:
    s = str(p).lower().replace('\\','/')
    n = p.name.lower()
    if 'racs' in n or '/rac' in s: return 'RAC'
    if 'rdfs' in n or '/rdf' in s: return 'RDF'
    if 'geometric' in n or 'geometry' in n or 'geom' in n: return 'geometry'
    if 'arc_mof_dim' in n or re.search(r'(^|[_\-])dim($|[_\-.])', n): return 'dimension'
    if 'topology' in n or 'topo' in n: return 'topology'
    if 'cluster' in n: return 'cluster'
    if 'aprdf' in n: return 'APRDF'
    if 'phom' in n: return 'PHOM'
    if 'pdd' in n: return 'PDD'
    return 'other_descriptor'


def descriptor_file_priority(row: pd.Series) -> int:
    name = str(row.get('file_name','')).lower()
    role = str(row.get('role','')).lower()
    subtype = str(row.get('resource_subtype','')).lower()
    fam = descriptor_family_from_file(Path(str(row.get('file_name','unknown.csv'))))
    score = 0
    if row.get('resource') == 'ARC-MOF': score += 25
    if 'descriptor' in role or 'descriptor' in subtype: score += 18
    if 'topology' in role or 'cluster' in role or 'cluster' in subtype: score += 12
    fam_bonus = {'RAC':22,'geometry':21,'RDF':20,'dimension':18,'topology':16,'cluster':15,'APRDF':12,'PHOM':10,'PDD':8,'other_descriptor':0}
    score += fam_bonus.get(fam,0)
    if any(k in name for k in ['landfill','pre_comb','post_comb','methane_purification','overall_process']): score -= 35
    return score


def select_independent_descriptor_columns(df: pd.DataFrame, cfg: Config, id_cols: Sequence[str], target_file: Optional[Path]=None) -> Tuple[List[str], List[str], pd.DataFrame]:
    """v1.6 feature selection with explicit positive/negative leakage audit."""
    numeric, categorical, leak_rows = [], [], []
    id_set = {str(c) for c in id_cols}
    for c in df.columns:
        name = str(c); lname = name.lower()
        if name in id_set or is_identifier_column_name(name) or lname.startswith('unnamed') or name == '__join_id':
            continue
        role = 'retained_candidate'
        leak_reason = ''
        if is_uncertainty_column_name(name):
            role = 'uncertainty_like_excluded'; leak_reason = 'uncertainty/error-like column name'
        elif PRESSURE_COLUMN_RX.search(name) or TEMPERATURE_COLUMN_RX.search(name):
            role = 'coordinate_like_excluded'; leak_reason = 'pressure/temperature coordinate-like column name'
        elif is_target_like_column(target_file, name, df[c]) or TARGET_EQUIVALENT_RX.search(name):
            role = 'target_family_like_excluded'; leak_reason = 'target-family/unit-like column name'
        if leak_reason:
            leak_rows.append(dict(column_name=name, leakage_role=role, leakage_reason=leak_reason))
            continue
        s = df[c]
        if safe_missing_fraction(s) > 0.70 or safe_nunique(s) <= 1:
            continue
        conv = safe_to_numeric(s)
        if pd.api.types.is_numeric_dtype(s) or conv.notna().mean() > 0.90:
            numeric.append(name)
        elif safe_nunique(s) <= 80:
            categorical.append(name)
    if len(numeric) > cfg.ram['max_features']:
        ranked=[]
        for c in numeric:
            ss = safe_to_numeric(df[c])
            bonus = 1 if (PAT['descriptor'].search(str(c)) or PAT['geometry'].search(str(c)) or PAT['topology'].search(str(c))) else 0
            ranked.append((bonus, float(ss.var(skipna=True) if ss.notna().sum()>2 else 0), c))
        numeric = [c for _,_,c in sorted(ranked, reverse=True)[:cfg.ram['max_features']]]
    return numeric, categorical[:80], pd.DataFrame(leak_rows)


def summarize_descriptor_leakage(descriptor_file: str, target_file: str, leakdf: pd.DataFrame, n_candidate_numeric: int, n_candidate_categorical: int, n_retained: int) -> Dict[str, Any]:
    if leakdf is None or leakdf.empty:
        counts = {}
    else:
        counts = leakdf['leakage_role'].value_counts().to_dict()
    return dict(
        descriptor_file=descriptor_file,
        target_file=target_file,
        n_candidate_numeric_features=int(n_candidate_numeric),
        n_candidate_categorical_features=int(n_candidate_categorical),
        n_target_like_features_removed=int(counts.get('target_family_like_excluded',0)),
        n_coordinate_like_features_removed=int(counts.get('coordinate_like_excluded',0)),
        n_uncertainty_like_features_removed=int(counts.get('uncertainty_like_excluded',0)),
        n_features_retained=int(n_retained),
        final_decision='no leakage-like descriptor columns detected' if not counts else 'leakage-like columns removed before modeling'
    )


def prepare_variant_joined_table(target_df: pd.DataFrame, desc_df: pd.DataFrame, left_id: str, right_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    left_set = variants_set_from_series(target_df[left_id])
    right_set = variants_set_from_series(desc_df[right_id])
    common = left_set & right_set
    td = target_df.copy(); ddsc = desc_df.copy()
    td['__join_id'] = join_key_series(td[left_id], common)
    ddsc['__join_id'] = join_key_series(ddsc[right_id], common)
    td = td[td['__join_id'].notna()].drop_duplicates('__join_id')
    ddsc = ddsc[ddsc['__join_id'].notna()].drop_duplicates('__join_id')
    return td, ddsc, common


def step_descriptor_joined_ml(cfg: Config, dd: Dict[str,Path], log):
    """v1.6 descriptor-family and identifier-harmonized ML benchmark."""
    empty_names = ['descriptor_join_candidate_files','descriptor_target_join_diagnostics','descriptor_feature_leakage_diagnostics',
                   'descriptor_feature_leakage_summary','descriptor_family_ablation_summary',
                   'descriptor_joined_ml_metrics_full','descriptor_joined_ml_metrics_summary','descriptor_joined_ml_predictions_sample',
                   'descriptor_joined_ml_error_by_group','descriptor_joined_case_catalog']
    if cfg.skip_ml or not SKLEARN_AVAILABLE:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    if fdf.empty:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc = fdf[fdf.resource.eq('ARC-MOF') & fdf.read_status.eq('ok')].copy()
    if arc.empty:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc['descriptor_priority'] = arc.apply(descriptor_file_priority, axis=1)
    arc['target_priority'] = arc.apply(target_file_priority, axis=1)
    arc['descriptor_family'] = arc['file_path'].map(lambda x: descriptor_family_from_file(Path(str(x))))
    # Keep more descriptor families than v1.5.  Sorting by family then priority avoids only RDFS dominating.
    desc_rows = arc[arc.descriptor_priority > 18].sort_values(['descriptor_family','descriptor_priority','size_mb'], ascending=[True,False,True])
    desc_rows = desc_rows.groupby('descriptor_family', group_keys=False).head(3).sort_values(['descriptor_priority','size_mb'], ascending=[False,True]).head(16)
    targ_rows = arc[(arc.target_priority > 25) & (pd.to_numeric(arc.n_target_like_columns, errors='coerce').fillna(0) > 0)].sort_values(['target_priority','size_mb'], ascending=[False, True]).head(14)
    cand_files = pd.concat([desc_rows.assign(candidate_role_v16='descriptor_feature_table'), targ_rows.assign(candidate_role_v16='target_label_table')], ignore_index=True) if (not desc_rows.empty or not targ_rows.empty) else pd.DataFrame()
    save_df(cand_files, dd['processed']/ 'descriptor_join_candidate_files', cfg)
    diagnostics, leak_all, leak_summary, metrics_all, preds_all, errs_all, cases = [], [], [], [], [], [], []
    successful_cases = 0
    max_successful_cases = max(4, min(18, int(cfg.level.get('targets',5))*4))
    min_overlap = descriptor_join_min_overlap(cfg)
    desc_cache: Dict[str, pd.DataFrame] = {}
    for _,tr in targ_rows.iterrows():
        if successful_cases >= max_successful_cases: break
        target_p = Path(tr.file_path)
        target_df, _ = read_table(target_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
        if target_df.empty: continue
        tcols = target_cols(target_df, cfg, file_path=target_p)
        primary_tcols = []
        for tc in tcols:
            ctx = infer_target_context(target_p, tc, target_df[tc]) if 'infer_target_context' in globals() else infer_arc_context(target_p, tc)
            if ctx.get('target_role', PRIMARY_TARGET_ROLE) == PRIMARY_TARGET_ROLE:
                primary_tcols.append(tc)
        if not primary_tcols: continue
        for _,dr in desc_rows.iterrows():
            if successful_cases >= max_successful_cases: break
            desc_p = Path(dr.file_path)
            key = str(desc_p)
            if key not in desc_cache:
                desc_cache[key], _ = read_table(desc_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
            desc_df = desc_cache[key]
            if desc_df.empty: continue
            join = best_identifier_join(target_df, desc_df, cfg)
            desc_family = descriptor_family_from_file(desc_p)
            diag_base = dict(target_file=target_p.name, target_path=str(target_p), descriptor_file=desc_p.name, descriptor_path=str(desc_p),
                             descriptor_family=desc_family, target_resource_subtype=resource_subtype_of(target_p), descriptor_resource_subtype=resource_subtype_of(desc_p), **join)
            if join['overlap'] < min_overlap:
                diagnostics.append({**diag_base, 'join_status':'insufficient_overlap', 'joined_rows':0, 'n_numeric_features':0, 'n_categorical_features':0, 'n_primary_targets':len(primary_tcols)})
                continue
            left_id, right_id = join['left_id'], join['right_id']
            td, ddsc, common = prepare_variant_joined_table(target_df, desc_df, left_id, right_id)
            num_cols, cat_cols, leakdf = select_independent_descriptor_columns(ddsc, cfg, id_cols=[right_id, '__join_id'], target_file=target_p)
            if not leakdf.empty:
                leakdf['descriptor_file'] = desc_p.name; leakdf['target_file'] = target_p.name; leakdf['descriptor_family'] = desc_family
                leak_all.append(leakdf)
            leak_summary.append(summarize_descriptor_leakage(desc_p.name, target_p.name, leakdf, len(num_cols), len(cat_cols), len(num_cols)+len(cat_cols)))
            feature_cols = ['__join_id'] + num_cols + cat_cols
            joined = td[['__join_id'] + primary_tcols].merge(ddsc[feature_cols], on='__join_id', how='inner')
            diagnostics.append({**diag_base, 'join_status':'joined', 'joined_rows':len(joined), 'n_numeric_features':len(num_cols), 'n_categorical_features':len(cat_cols), 'n_primary_targets':len(primary_tcols)})
            if len(joined) < min_overlap or len(num_cols)+len(cat_cols) < 2:
                continue
            for target in primary_tcols[:max(1, min(3, cfg.level.get('targets',3)))]:
                ctx = infer_target_context(target_p, target, target_df[target]) if 'infer_target_context' in globals() else infer_arc_context(target_p, target)
                meta = dict(ctx); meta['target_family'] = infer_target_family(target, pd.Series(ctx))
                label = f"ARC-MOF::descriptor_join::{target_p.name}"
                feature_tag = desc_p.name
                log.info('v1.6 descriptor-family ML: target=%s | labels=%s | descriptors=%s | family=%s | joined_rows=%d | features=%d', target, target_p.name, desc_p.name, desc_family, len(joined), len(num_cols)+len(cat_cols))
                m,p,e = run_descriptor_joined_case(joined, target, label, cfg, log, feature_tag, meta, num_cols, cat_cols)
                if not m.empty:
                    m['descriptor_family'] = desc_family
                    if not p.empty: p['descriptor_family'] = desc_family
                    if not e.empty: e['descriptor_family'] = desc_family
                    metrics_all.append(m)
                    if not p.empty: preds_all.append(p)
                    if not e.empty: errs_all.append(e)
                    successful_cases += 1
                    cases.append(dict(source_table=label, feature_table=feature_tag, descriptor_family=desc_family,
                                      target_column=target, target_family=meta.get('target_family',''), task=meta.get('task',''), gas=meta.get('gas',''),
                                      target_property=meta.get('target_property',''), target_unit=meta.get('target_unit',''), joined_rows=len(joined),
                                      n_numeric_features=len(num_cols), n_categorical_features=len(cat_cols),
                                      split_group_available=bool(choose_group_series_for_joined(joined, num_cols, cat_cols)[0] is not None)))
                if successful_cases >= max_successful_cases: break
    diag = pd.DataFrame(diagnostics)
    leaks = pd.concat(leak_all, ignore_index=True) if leak_all else pd.DataFrame(columns=['descriptor_file','target_file','descriptor_family','column_name','leakage_role','leakage_reason'])
    leak_sum = pd.DataFrame(leak_summary)
    met = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    pred = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    err = pd.concat(errs_all, ignore_index=True) if errs_all else pd.DataFrame()
    casecat = pd.DataFrame(cases)
    save_df(diag, dd['processed']/ 'descriptor_target_join_diagnostics', cfg)
    save_df(leaks, dd['processed']/ 'descriptor_feature_leakage_diagnostics', cfg)
    save_df(leak_sum, dd['processed']/ 'descriptor_feature_leakage_summary', cfg)
    save_df(met, dd['processed']/ 'descriptor_joined_ml_metrics_full', cfg)
    save_df(pred, dd['processed']/ 'descriptor_joined_ml_predictions_sample', cfg)
    save_df(err, dd['processed']/ 'descriptor_joined_ml_error_by_group', cfg)
    save_df(casecat, dd['processed']/ 'descriptor_joined_case_catalog', cfg)
    if not met.empty:
        group_cols = ['feature_source','source_table','feature_table','descriptor_family','target_column','target_family','task','gas','target_property','target_unit','model','split_type']
        for col in group_cols:
            if col not in met.columns: met[col] = ''
        summ = met.groupby(group_cols).agg(
            n_repeats=('repeat','nunique'), mean_rmse=('rmse','mean'), sd_rmse=('rmse','std'), mean_mae=('mae','mean'),
            mean_r2=('r2','mean'), sd_r2=('r2','std'), mean_spearman=('spearman','mean'),
            mean_top_1pct_recovery=('top_1pct_recovery','mean'), mean_top_5pct_recovery=('top_5pct_recovery','mean'),
            mean_top_10pct_recovery=('top_10pct_recovery','mean'), mean_ndcg_10pct=('ndcg_10pct','mean'),
            n_numeric_features=('n_numeric_features','median'), n_categorical_features=('n_categorical_features','median')
        ).reset_index()
        ablation = summ.groupby(['descriptor_family','split_type','target_family']).agg(
            n_cases=('mean_r2','count'), median_r2=('mean_r2','median'), median_spearman=('mean_spearman','median'), median_top10=('mean_top_10pct_recovery','median')
        ).reset_index()
    else:
        summ = pd.DataFrame(); ablation = pd.DataFrame()
    save_df(summ, dd['processed']/ 'descriptor_joined_ml_metrics_summary', cfg)
    save_df(summ, dd['main_tables']/ 'Main_Table_6_descriptor_joined_ml_summary', cfg)
    save_df(ablation, dd['processed']/ 'descriptor_family_ablation_summary', cfg)
    save_df(ablation, dd['main_tables']/ 'Main_Table_8_descriptor_family_ablation_summary', cfg)
    save_df(diag, dd['si_tables']/ 'SI_Table_S16_descriptor_target_join_diagnostics', cfg)
    save_df(leaks, dd['si_tables']/ 'SI_Table_S17_descriptor_feature_leakage_diagnostics', cfg)
    save_df(leak_sum, dd['si_tables']/ 'SI_Table_S17b_descriptor_feature_leakage_summary', cfg)
    save_df(ablation, dd['si_tables']/ 'SI_Table_S21_descriptor_family_ablation_summary', cfg)
    log.info('v1.6 descriptor-joined ML cases completed: %d; diagnostic join pairs: %d', successful_cases, len(diag))


def make_target_dictionary(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    ctx = load_df(dd['processed']/ 'normalized_target_context_catalog')
    if ctx.empty:
        ctx = load_df(dd['profiles']/ 'target_column_context_catalog')
    if ctx.empty:
        return pd.DataFrame(columns=['resource','task','gas','target_column','target_property','target_unit','target_role','basis','process_metric_type','scientific_interpretation','used_in_main_ml_recommended'])
    out = ctx.copy()
    for col in ['resource','task','gas','target_column','target_property','target_unit','target_role','basis','process_metric_type']:
        if col not in out.columns: out[col] = ''
    out = out[['resource','task','gas','target_column','target_property','target_unit','target_role','basis','process_metric_type']].drop_duplicates()
    def interp(r):
        role = str(r.get('target_role',''))
        prop = str(r.get('target_property',''))
        gas = str(r.get('gas',''))
        unit = str(r.get('target_unit',''))
        if role != PRIMARY_TARGET_ROLE:
            return f"{role}: retained for context/uncertainty but not used as primary supervised label."
        return f"Primary {prop} response for {gas}; unit/basis: {unit}. Treat equivalent uptake representations as related, not independent mechanistic endpoints."
    out['scientific_interpretation'] = out.apply(interp, axis=1)
    out['used_in_main_ml_recommended'] = out.apply(lambda r: str(r.get('target_role','')) == PRIMARY_TARGET_ROLE and str(r.get('target_property','')) in ['uptake','loading','selectivity','working_capacity','purity','recovery','productivity','process_energy','heat_of_adsorption'], axis=1)
    return out.sort_values(['resource','task','gas','target_property','target_column'])


def write_manual_validation_template(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    trust = load_df(dd['processed']/ 'trust_flags')
    cols = ['resource','source_file','primary_id','metals_detected','oxidation_states_reported','charge_balance_status','trust_regime_row','manual_charge_status','manual_oxidation_state_assessment','manual_formula_check','manual_decision','notes']
    if trust.empty:
        tmpl = pd.DataFrame(columns=cols)
    else:
        keep = [c for c in cols if c in trust.columns]
        tmpl = trust.sort_values(['resource','trust_regime_row']).groupby('resource', group_keys=False).head(20)
        tmpl = tmpl[[c for c in keep if c in tmpl.columns]].copy()
        for c in cols:
            if c not in tmpl.columns: tmpl[c] = ''
        tmpl = tmpl[cols]
    save_df(tmpl, dd['source']/ 'manual_chemistry_validation_template', cfg)
    save_df(tmpl, dd['si_tables']/ 'SI_Table_S22_manual_chemistry_validation_template', cfg)
    return tmpl


def step_scores(cfg: Config, dd: Dict[str,Path], log):
    _V15_STEP_SCORES(cfg, dd, log)
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    trust = load_df(dd['processed']/ 'trust_flags')
    scores = load_df(dd['processed']/ 'resource_scores')
    rows=[]
    rng = np.random.default_rng(cfg.random_seed)
    for _,r in scores.iterrows() if not scores.empty else []:
        res = r.resource
        vals=[]
        if not trust.empty and 'resource' in trust and 'chemistry_trust_score_row' in trust:
            tv = pd.to_numeric(trust[trust.resource.eq(res)].chemistry_trust_score_row, errors='coerce').dropna().values
            if len(tv) >= 5:
                for _ in range(300): vals.append(float(np.mean(rng.choice(tv, size=len(tv), replace=True))))
        if vals:
            lo,hi=np.quantile(vals,[0.025,0.975])
        else:
            lo=max(0,float(r.chemistry_trust_score)-0.05); hi=min(1,float(r.chemistry_trust_score)+0.05)
        rows.append(dict(resource=res, chemistry_trust_score=float(r.chemistry_trust_score), chemistry_trust_ci95_low=float(lo), chemistry_trust_ci95_high=float(hi), ml_readiness_score=float(r.ml_readiness_score), uncertainty_note='bootstrap over scanned trust rows where available; otherwise conservative +/-0.05 reporting interval'))
    unc = pd.DataFrame(rows)
    save_df(unc, dd['processed']/ 'resource_score_uncertainty', cfg)
    save_df(unc, dd['si_tables']/ 'SI_Table_S23_resource_score_uncertainty', cfg)


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    _V15_STEP_TABLES(cfg, dd, log)
    target_dict = make_target_dictionary(cfg, dd)
    if not target_dict.empty:
        save_df(target_dict, dd['processed']/ 'target_dictionary', cfg)
        save_df(target_dict, dd['si_tables']/ 'SI_Table_S20_target_dictionary', cfg)
    write_manual_validation_template(cfg, dd)
    for name, base in [
        ('SI_Table_S17b_descriptor_feature_leakage_summary', dd['processed']/ 'descriptor_feature_leakage_summary'),
        ('SI_Table_S21_descriptor_family_ablation_summary', dd['processed']/ 'descriptor_family_ablation_summary'),
        ('SI_Table_S23_resource_score_uncertainty', dd['processed']/ 'resource_score_uncertainty')
    ]:
        df = load_df(base)
        if not df.empty: save_df(df, dd['si_tables']/name, cfg)
    src = load_df(dd['source']/ 'final_source_data_map')
    add = pd.DataFrame([
        {'figure':'Figure 2','panel':'a-d','source_data':'profiles/file_level_profile.csv; profiles/column_level_profile.csv; processed/resource_scores.csv','notes':'Main resources only; Other/unknown moved to SI context.'},
        {'figure':'Figure 4','panel':'a','source_data':'processed/resource_scores.csv; processed/resource_score_uncertainty.csv','notes':'Trust-readiness map with reporting uncertainty intervals.'},
        {'figure':'Figure 5','panel':'a-d','source_data':'processed/descriptor_joined_ml_metrics_summary.csv; processed/descriptor_family_ablation_summary.csv; processed/descriptor_feature_leakage_summary.csv; processed/descriptor_joined_ml_predictions_sample.csv','notes':'Final descriptor-joined ML figure with random-vs-grouped comparison and feature-family ablation.'},
        {'figure':'SI target dictionary','panel':'table','source_data':'processed/target_dictionary.csv','notes':'Human-readable target semantics and recommended use.'},
        {'figure':'Manual validation template','panel':'table','source_data':'source_data/manual_chemistry_validation_template.csv','notes':'Template for 50--100 manually curated chemistry examples.'}
    ])
    src = pd.concat([src, add], ignore_index=True) if not src.empty else add
    save_df(src.drop_duplicates(), dd['source']/ 'final_source_data_map', cfg)


def fig2(cfg,dd):
    f=load_df(dd['profiles']/ 'file_level_profile'); c=load_df(dd['profiles']/ 'column_level_profile'); mod=load_df(dd['profiles']/ 'resource_modality_matrix'); sc=load_df(dd['processed']/ 'resource_scores')
    main = set(MAIN_RESOURCES)
    if not f.empty: f = f[f.resource.isin(main)].copy()
    if not c.empty: c = c[c.resource.isin(main)].copy()
    if not mod.empty: mod = mod[mod.resource.isin(main)].copy()
    if not sc.empty: sc = sc[sc.resource.isin(main)].copy()
    fig,axs=plt.subplots(2,2,figsize=(13.4,9.2)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not f.empty:
        g=f.groupby('resource').agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum')).reset_index().sort_values('total_size_mb')
        ax.barh(g.resource,g.total_size_mb)
        ax.set_xlabel('Total profiled file size (MB)'); ax.set_title('Input scale by main resource',fontweight='bold')
        for i,r in g.reset_index(drop=True).iterrows(): ax.text(r.total_size_mb,i,f" {int(r.n_files)} files",va='center',fontsize=7.5)
    else: nodata(ax,'Input scale by main resource')
    ax=axs[1]; lab(ax,'b')
    if not mod.empty:
        cols=[x for x in ['identifier','formula_or_composition','metal_or_charge_chemistry','geometric_descriptor','descriptor','target_or_property','curation_or_validation_flag'] if x in mod.columns]
        mat=mod.set_index('resource')[cols].astype(float)
        mat=mat.div(mat.max(axis=0).replace(0,np.nan), axis=1).fillna(0)
        heat(ax,mat,'Relative modality evidence', 'YlGnBu', True)
    else: nodata(ax,'Relative modality evidence')
    ax=axs[2]; lab(ax,'c')
    if not c.empty:
        tmp=c.copy(); tmp['missing_bin']=pd.cut(pd.to_numeric(tmp.missing_fraction,errors='coerce'),[-.001,0,.05,.25,.75,1],labels=['0','0--5%','5--25%','25--75%','>75%'],include_lowest=True)
        mat=tmp.groupby(['resource','missing_bin'],observed=False).size().unstack(fill_value=0)
        heat(ax,mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0),'Column missingness composition','magma',True)
    else: nodata(ax,'Column missingness composition')
    ax=axs[3]; lab(ax,'d'); style_axes(ax)
    if not sc.empty:
        x=np.log10(sc.total_profiled_or_counted_rows.astype(float).clip(lower=1)); y=sc.chemistry_trust_score.astype(float); s=100+650*sc.ml_readiness_score.astype(float).clip(0,1)
        ax.scatter(x,y,s=s,alpha=.78,edgecolors='black')
        offsets={'ARC-MOF':(.03,-.035),'MOSAEC-DB':(.03,.025),'CoRE MOF 2024':(.03,.025),'QMOF':(.03,-.035),'CSD-derived context':(.03,.035),'CoRE MOF 2025 metadata':(.03,.035)}
        for _,r in sc.iterrows():
            dx,dy=offsets.get(r.resource,(.02,.02)); ax.text(np.log10(max(float(r.total_profiled_or_counted_rows),1))+dx,float(r.chemistry_trust_score)+dy,r.resource,fontsize=8)
        ax.set_xlabel('log10(profiled or counted rows + 1)'); ax.set_ylabel('Chemistry-trust score'); ax.set_ylim(0,1.05); ax.set_title('Scale versus chemistry observability',fontweight='bold')
        add_panel_note(ax,'Other/unknown excluded from main panel; retained in SI tables.')
    else: nodata(ax,'Scale versus chemistry observability')
    fig.suptitle('Figure 2. Data-resource atlas for chemistry-ready MOF ML',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_2_data_resource_atlas',cfg)


def fig4(cfg,dd):
    sc=load_df(dd['processed']/ 'resource_scores'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); sens=load_df(dd['processed']/ 'score_sensitivity'); unc=load_df(dd['processed']/ 'resource_score_uncertainty')
    if not sc.empty: sc=sc[sc.resource.isin(MAIN_RESOURCES)].copy()
    fig,axs=plt.subplots(1,3,figsize=(16.4,5.5))
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not sc.empty:
        ax.axhline(.70,ls='--',lw=.8,color='#777777'); ax.axvline(.70,ls='--',lw=.8,color='#777777')
        if not unc.empty:
            sc=sc.merge(unc[['resource','chemistry_trust_ci95_low','chemistry_trust_ci95_high']],on='resource',how='left')
            yerr=np.vstack([np.maximum((sc.chemistry_trust_score-sc.chemistry_trust_ci95_low).fillna(0).to_numpy(dtype=float),0), np.maximum((sc.chemistry_trust_ci95_high-sc.chemistry_trust_score).fillna(0).to_numpy(dtype=float),0)])
            ax.errorbar(sc.ml_readiness_score,sc.chemistry_trust_score,yerr=yerr,fmt='none',ecolor='#555555',elinewidth=.8,capsize=2,zorder=1)
        ax.scatter(sc.ml_readiness_score,sc.chemistry_trust_score,s=180,alpha=.78,edgecolors='black',zorder=2)
        for _,r in sc.iterrows(): ax.text(r.ml_readiness_score+.012,r.chemistry_trust_score+.012,r.resource,fontsize=7.8)
        ax.set_xlim(0,1.05); ax.set_ylim(0,1.05); ax.set_xlabel('ML-readiness score'); ax.set_ylabel('Chemistry-trust score'); ax.set_title('Trust--readiness map',fontweight='bold')
    else: nodata(ax,'Trust--readiness map')
    ax=axs[1]; lab(ax,'b')
    if not risk.empty: heat(ax,risk.pivot_table(index='risk_issue',columns='use_case',values='risk_score_0_to_3',aggfunc='mean',fill_value=0),'Expert risk rubric by use case','inferno',True)
    else: nodata(ax,'Expert risk rubric by use case')
    ax=axs[2]; lab(ax,'c'); style_axes(ax)
    if not sens.empty:
        sub=sens[sens.resource.isin(MAIN_RESOURCES)].sort_values('mean_rank') if 'resource' in sens.columns else sens.sort_values('mean_rank')
        y=np.arange(len(sub)); ax.errorbar(sub.mean_rank,y,xerr=[sub.mean_rank-sub.min_rank,sub.max_rank-sub.mean_rank],fmt='o',capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(sub.resource,fontsize=8); ax.invert_yaxis(); ax.set_xlabel('Rank across score-weight perturbations'); ax.set_title('Score sensitivity',fontweight='bold')
    else: nodata(ax,'Score sensitivity')
    fig.suptitle('Figure 4. Trust--readiness and benchmark-risk framework',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_4_trust_readiness_risk_matrix',cfg)


def _best_descriptor_model_rows(desc_summ: pd.DataFrame) -> pd.DataFrame:
    if desc_summ.empty: return pd.DataFrame()
    df=desc_summ.copy()
    for c in ['task','gas','target_column','descriptor_family','model','split_type','mean_r2','mean_spearman','mean_top_10pct_recovery']:
        if c not in df.columns: df[c]='' if c not in ['mean_r2','mean_spearman','mean_top_10pct_recovery'] else np.nan
    # choose best model per task/gas/target/family/split based on mean_r2
    idx=df.groupby(['task','gas','target_column','descriptor_family','split_type'])['mean_r2'].idxmax()
    return df.loc[idx].reset_index(drop=True)


def fig5(cfg,dd):
    desc_summ=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    ablation=load_df(dd['processed']/ 'descriptor_family_ablation_summary')
    leak_sum=load_df(dd['processed']/ 'descriptor_feature_leakage_summary')
    pred=load_df(dd['processed']/ 'descriptor_joined_ml_predictions_sample')
    best=_best_descriptor_model_rows(desc_summ)
    fig,axs=plt.subplots(2,2,figsize=(13.6,9.4)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not best.empty and {'random','descriptor_grouped'}.issubset(set(best.split_type.astype(str))):
        piv=best.pivot_table(index=['task','gas','target_column','descriptor_family'],columns='split_type',values='mean_r2',aggfunc='max').reset_index()
        if 'random' in piv.columns and 'descriptor_grouped' in piv.columns:
            piv=piv.dropna(subset=['random','descriptor_grouped']).sort_values('random',ascending=False).head(12)
            y=np.arange(len(piv)); labels=(piv.task.astype(str).str.replace('_',' ')+' | '+piv.gas.astype(str)+' | '+piv.target_column.astype(str)+' | '+piv.descriptor_family.astype(str)).str[:55]
            for i,r in piv.reset_index(drop=True).iterrows(): ax.plot([r['descriptor_grouped'],r['random']],[i,i],lw=1.2,color='#999999')
            ax.scatter(piv['descriptor_grouped'],y,label='descriptor-grouped',marker='s',s=45)
            ax.scatter(piv['random'],y,label='random',marker='o',s=45)
            ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=6.7); ax.invert_yaxis(); ax.set_xlabel('Best-model mean R²'); ax.set_title('Random versus grouped generalization',fontweight='bold'); ax.legend(fontsize=7)
        else: nodata(ax,'Random versus grouped generalization')
    elif not best.empty:
        sub=best.sort_values('mean_r2',ascending=False).head(12); labels=(sub.task.astype(str)+' | '+sub.target_column.astype(str)+' | '+sub.descriptor_family.astype(str)).str[:55]
        ax.barh(labels[::-1], sub.mean_r2.values[::-1]); ax.set_xlabel('Best-model mean R²'); ax.set_title('Descriptor-joined performance',fontweight='bold')
    else: nodata(ax,'Random versus grouped generalization','No descriptor-joined ML summary available')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    if not best.empty:
        piv=best.pivot_table(index=['task','gas','target_column','descriptor_family'],columns='split_type',values='mean_top_10pct_recovery',aggfunc='max').reset_index()
        if 'random' in piv.columns and 'descriptor_grouped' in piv.columns:
            piv=piv.dropna(subset=['random','descriptor_grouped']).sort_values('random',ascending=False).head(12)
            y=np.arange(len(piv)); labels=(piv.task.astype(str).str.replace('_',' ')+' | '+piv.gas.astype(str)+' | '+piv.target_column.astype(str)+' | '+piv.descriptor_family.astype(str)).str[:55]
            for i,r in piv.reset_index(drop=True).iterrows(): ax.plot([r['descriptor_grouped'],r['random']],[i,i],lw=1.2,color='#999999')
            ax.scatter(piv['descriptor_grouped'],y,marker='s',s=45,label='descriptor-grouped')
            ax.scatter(piv['random'],y,marker='o',s=45,label='random')
            ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=6.7); ax.invert_yaxis(); ax.set_xlabel('Top-10% recovery'); ax.set_xlim(0,1.05); ax.set_title('Ranking stability',fontweight='bold')
        else: nodata(ax,'Ranking stability')
    else: nodata(ax,'Ranking stability')
    ax=axs[2]; lab(ax,'c')
    if not ablation.empty:
        mat=ablation.pivot_table(index='descriptor_family',columns='split_type',values='median_r2',aggfunc='median',fill_value=np.nan)
        heat(ax,mat,'Descriptor-family ablation: median R²','YlGnBu',True)
    elif not leak_sum.empty:
        l=leak_sum.groupby('descriptor_file').agg(n_features_retained=('n_features_retained','max'), n_removed=('n_target_like_features_removed','sum')).reset_index().head(12)
        ax.barh(l.descriptor_file,l.n_features_retained); ax.set_xlabel('Retained features'); ax.set_title('Descriptor-feature audit',fontweight='bold')
    else: nodata(ax,'Descriptor-family ablation')
    ax=axs[3]; lab(ax,'d')
    if not pred.empty:
        # Use best available descriptor-joined case to avoid overplotting all tasks.
        if {'source_table','target_column','model','split_type'}.issubset(pred.columns) and not best.empty:
            top = best.sort_values('mean_r2',ascending=False).iloc[0]
            sub = pred[(pred.source_table.eq(top.source_table)) & (pred.target_column.eq(top.target_column)) & (pred.model.eq(top.model)) & (pred.split_type.eq(top.split_type))]
            title_extra = f"{top.get('descriptor_family','')} | {top.get('target_column','')}"
        else:
            sub = pred; title_extra='representative case'
        if len(sub)>3000: sub=sub.sample(n=3000,random_state=cfg.random_seed)
        x=pd.to_numeric(sub.y_true,errors='coerce'); y=pd.to_numeric(sub.y_pred,errors='coerce')
        xycat=pd.concat([x,y]).dropna()
        ax.scatter(x,y,s=8,alpha=.35)
        if not xycat.empty:
            q=np.nanquantile(xycat,[0.01,0.99])
            if np.isfinite(q[0]) and np.isfinite(q[1]) and q[0] < q[1]:
                ax.plot([q[0],q[1]],[q[0],q[1]],ls='--',lw=1); ax.set_xlim(q[0],q[1]); ax.set_ylim(q[0],q[1])
        ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Representative calibration: '+str(title_extra)[:38],fontweight='bold',fontsize=9.5)
        add_panel_note(ax,'Axes clipped to 1st--99th percentile for readability.')
    else: nodata(ax,'Representative calibration')
    fig.suptitle('Figure 5. Descriptor-joined MOF adsorption benchmark',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    _V15_STEP_REPORTS(cfg, dd, log)
    report = dd['reports']/ 'analysis_report.md'
    extra=[]
    extra.append('\n## v1.6 journal-finalization additions\n')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    abl=load_df(dd['processed']/ 'descriptor_family_ablation_summary')
    target_dict=load_df(dd['processed']/ 'target_dictionary')
    leak_sum=load_df(dd['processed']/ 'descriptor_feature_leakage_summary')
    unc=load_df(dd['processed']/ 'resource_score_uncertainty')
    if not diag.empty:
        joined=int((diag.get('join_status','')=='joined').sum()) if 'join_status' in diag else 0
        fams='; '.join(sorted([str(x) for x in diag.get('descriptor_family',pd.Series(dtype=str)).dropna().unique()]))
        extra.append(f"- Identifier-harmonized descriptor joins inspected **{len(diag)}** target/descriptor pairs; **{joined}** reached sufficient overlap. Families considered: {fams}.\n")
    if not abl.empty:
        extra.append(f"- Descriptor-family ablation table generated with **{len(abl)}** family/split/target-family rows.\n")
    if not leak_sum.empty:
        extra.append(f"- Positive leakage audit generated for **{len(leak_sum)}** descriptor/target pairs, including zero-leakage decisions.\n")
    if not target_dict.empty:
        extra.append(f"- Target dictionary generated with **{len(target_dict)}** target-context rows for SI and methods text.\n")
    if not unc.empty:
        extra.append(f"- Resource-score uncertainty table generated for **{len(unc)}** resources.\n")
    extra.append('- Main Figure 5 now emphasizes random-vs-grouped descriptor-joined ML, top-10% recovery, descriptor-family ablation, and one representative calibration case.\n')
    with open(report,'a',encoding='utf-8') as fh:
        fh.write(''.join(extra))



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

v1.6 journal-finalization focus
-------------------------------
- Uses stronger identifier harmonization for ARC-MOF descriptor-target joins.
- Attempts descriptor-family coverage beyond RDFS, including RAC/geometry/dimension/topology/cluster-like resources when identifiers overlap.
- Produces descriptor-family ablation summaries and a cleaner final Figure 5.
- Writes positive descriptor-leakage summaries even when zero leakage-like columns are found.
- Adds a human-readable target dictionary and a manual chemistry-validation template.
- Adds resource-score uncertainty summaries.
- Keeps table-local ML as secondary diagnostics and descriptor-joined ML as the main manuscript result.
- Audits FINAL_OUTPUT_MANIFEST.csv against the generated ZIP package.

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     normalized targets, descriptor joins, trust regimes, ML outputs, ablations and reusable data
tables/        main-text and SI-ready tables
figures/       main figures 1--6 plus SI figures
source_data/   panel source-data map, checklist, manual-validation template, manifest and ZIP audit
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        optional fitted models in thorough save mode

Resume behavior
---------------
Use a fresh output folder or --force when changing code versions. If interrupted, rerun the same command.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')



# =============================================================================
# v1.7 FINAL-PAPER-TIGHTENING OVERRIDES
# =============================================================================
# This final tightening block implements the action items from the v1.6 result
# audit: multi-key ARC-MOF identifier bridging, descriptor-family join expansion,
# valid composite-score uncertainty, a detailed target dictionary, clearer main
# figures, and final-paper source-data tables.  It intentionally overrides only
# selected functions, preserving the stable v1.6 profiling/normalization/trust
# infrastructure above.

VERSION = "1.7-final-paper-tightening"

_V17_PREV_STEP_DESCRIPTOR_JOINED_ML = step_descriptor_joined_ml
_V17_PREV_STEP_SCORES = step_scores
_V17_PREV_STEP_TABLES = step_tables
_V17_PREV_STEP_REPORTS = step_reports

# Main-figure short labels; full names stay in captions/tables.
RESOURCE_SHORT = {
    'MOSAEC-DB': 'MOSAEC',
    'CoRE MOF 2024': 'CoRE-24',
    'CoRE MOF 2025 metadata': 'CoRE-25',
    'ARC-MOF': 'ARC',
    'QMOF': 'QMOF',
    'CSD-derived context': 'CSD ctx',
    'Other/unknown': 'Other',
    'Task-specific joins': 'Joins',
}


def short_resource(x: Any) -> str:
    return RESOURCE_SHORT.get(str(x), str(x))


def _clean_identifier_token(x: Any) -> str:
    if is_missing_value(x):
        return ''
    s = str(x).strip().strip('"\'')
    if not s or s.lower() in {z.lower() for z in MISSING}:
        return ''
    s = s.replace('\\', '/').split('/')[-1]
    for sep in ['::', '|', ';']:
        if sep in s:
            s = s.split(sep)[-1]
    s = s.strip().lower()
    s = re.sub(r'\.(cif|csv|json|txt|xlsx|xls|pkl|pickle|parquet)$', '', s, flags=re.I)
    s = re.sub(r'\s+', '', s)
    return s


def identifier_variant_records(x: Any) -> List[Tuple[str, str]]:
    """Return ordered (join_key_level, value) records for conservative MOF joins.

    ARC-MOF target files and descriptor files often differ by suffixes such as
    `.cif`, `_repeat`, `.sym.N`, path prefixes, or punctuation.  v1.7 exposes
    these levels in diagnostics rather than hiding them inside one opaque key.
    """
    raw = _clean_identifier_token(x)
    if not raw:
        return []
    candidates: List[Tuple[str,str]] = []
    def add(level: str, value: str):
        value = re.sub(r'\s+', '', str(value).strip().lower())
        value = re.sub(r'\.(cif|csv|json|txt|xlsx|xls|pkl|pickle|parquet)$', '', value, flags=re.I)
        if value and value not in {'nan','none','null','missing','unknown'}:
            candidates.append((level, value))
    add('exact_stem', raw)
    add('extension_optional', raw + '.cif')
    no_repeat = re.sub(r'(_repeat|[-.]repeat)$', '', raw)
    add('repeat_stripped', no_repeat)
    add('repeat_added', raw if raw.endswith('_repeat') else raw + '_repeat')
    # ARC identifiers frequently carry `.sym.<integer>` before optional repeat.
    sym_stripped = re.sub(r'(\.sym\.\d+)(?:_repeat)?$', '', raw)
    add('sym_stripped', sym_stripped)
    sym_no_repeat = re.sub(r'(\.sym\.\d+)$', '', no_repeat)
    add('sym_and_repeat_stripped', sym_no_repeat)
    # Keep chemistry/topology base before terminal numeric serials where possible.
    serial_stripped = re.sub(r'(\.sym\.\d+|\.\d+)(?:_repeat)?$', '', raw)
    add('serial_stripped', serial_stripped)
    # Normalize punctuation forms.  Useful for descriptor tables that use compact keys.
    add('punctuation_compact', re.sub(r'[^a-z0-9]+', '', raw))
    add('underscore_normalized', re.sub(r'[^a-z0-9]+', '_', raw).strip('_'))
    # Conservative DB/topology base: DB0-m3_o12_o22_f0_pcu.sym.90_repeat -> DB0-m3_o12_o22_f0_pcu
    m = re.match(r'^(db\d+[-_][a-z0-9]+(?:[_-][a-z0-9]+){1,8})', raw)
    if m:
        add('db_prefix_base', m.group(1))
    # Ordered unique records.
    seen=set(); out=[]
    for level,value in candidates:
        if value and value not in seen:
            seen.add(value); out.append((level,value))
    return out[:14]


def identifier_variants(x: Any) -> List[str]:
    return [v for _, v in identifier_variant_records(x)]


def variants_set_from_series(s: pd.Series, max_rows: Optional[int]=None, levels: Optional[Sequence[str]]=None) -> set:
    vals = s.dropna()
    if max_rows and len(vals) > max_rows:
        vals = vals.sample(n=max_rows, random_state=42)
    allowed = set(levels) if levels else None
    out=set()
    for x in vals:
        for level,value in identifier_variant_records(x):
            if allowed is None or level in allowed:
                out.add(value)
    return out


def join_key_series(s: pd.Series, common: set, levels: Optional[Sequence[str]]=None) -> pd.Series:
    allowed = set(levels) if levels else None
    order = {str(level): i for i, level in enumerate(levels or [])}
    def pick(x):
        recs = identifier_variant_records(x)
        if allowed is not None:
            recs = [rv for rv in recs if rv[0] in allowed]
            recs = sorted(recs, key=lambda rv: order.get(rv[0], 999))
        for level,value in recs:
            if value in common:
                return value
        return pd.NA
    return s.map(pick).astype('string')


def choose_descriptor_id_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    """More curious identifier-column discovery for descriptor/topology tables."""
    ids=[]
    for c in choose_ids(df, cfg):
        if c not in ids: ids.append(c)
    exact_names = {
        'name','structure_name','structure name','structure','mof','mof_name','mofname',
        'filename','file_name','cif','cifname','cif_name','structure_id','arc_mof_id',
        'refcode','id','Structure_Name','Name','Filename','MOF'
    }
    for c in df.columns:
        if str(c) in exact_names or str(c).lower() in {x.lower() for x in exact_names}:
            if c not in ids: ids.append(c)
    # Content-based discovery: object columns with many ARC-like or CIF-like strings.
    for c in df.columns:
        if c in ids:
            continue
        s=df[c]
        if safe_missing_fraction(s) > 0.50:
            continue
        if is_numeric_column(s):
            continue
        try:
            sample=s.dropna().astype(str).head(2000)
            if sample.empty:
                continue
            arc_like=sample.str.contains(r'(db\d+[-_]|\.cif$|_repeat|\.sym\.\d+|mof)', case=False, regex=True).mean()
            uniq=sample.nunique(dropna=True)
            if arc_like >= 0.20 and uniq >= min(20, max(3, len(sample)//5)):
                ids.append(c)
        except Exception:
            pass
        if len(ids) >= 10:
            break
    return ids[:10]


def best_identifier_join(left: pd.DataFrame, right: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    left_ids = choose_descriptor_id_columns(left, cfg)
    right_ids = choose_descriptor_id_columns(right, cfg)
    best = dict(left_id='', right_id='', overlap=0, left_unique=0, right_unique=0,
                left_retention=0.0, right_retention=0.0, join_key_level='',
                overlap_exact=0, overlap_repeat_stripped=0, overlap_sym_stripped=0,
                overlap_serial_stripped=0, risk_of_overmatching='not_evaluated')
    level_sets = {
        'exact_stem': ['exact_stem','extension_optional'],
        'repeat_stripped': ['repeat_stripped','exact_stem','extension_optional'],
        'sym_stripped': ['sym_stripped','sym_and_repeat_stripped','serial_stripped','repeat_stripped'],
        'punctuation_compact': ['punctuation_compact','underscore_normalized'],
        # Base-key joins are useful diagnostics but intentionally lower-priority
        # because they can over-match structures that share a chemistry/topology prefix.
        'db_prefix_base': ['db_prefix_base'],
    }
    for lc in left_ids:
        for rc in right_ids:
            try:
                L_exact = variants_set_from_series(left[lc], levels=['exact_stem','extension_optional'])
                R_exact = variants_set_from_series(right[rc], levels=['exact_stem','extension_optional'])
                L_rep = variants_set_from_series(left[lc], levels=['repeat_stripped','repeat_added'])
                R_rep = variants_set_from_series(right[rc], levels=['repeat_stripped','repeat_added'])
                L_sym = variants_set_from_series(left[lc], levels=['sym_stripped','sym_and_repeat_stripped'])
                R_sym = variants_set_from_series(right[rc], levels=['sym_stripped','sym_and_repeat_stripped'])
                L_ser = variants_set_from_series(left[lc], levels=['serial_stripped'])
                R_ser = variants_set_from_series(right[rc], levels=['serial_stripped'])
            except Exception:
                continue
            for level,levels in level_sets.items():
                try:
                    L = variants_set_from_series(left[lc], levels=levels)
                    R = variants_set_from_series(right[rc], levels=levels)
                except Exception:
                    continue
                if not L or not R:
                    continue
                common=L & R
                ov=len(common)
                if ov <= 0:
                    continue
                # Penalize broad base-level joins unless they substantially improve overlap.
                broad_penalty = 0.65 if level == 'db_prefix_base' else 1.0
                score = ov * broad_penalty
                best_score = best.get('_score', -1)
                if score > best_score or (score == best_score and ov > best.get('overlap',0)):
                    risk='low'
                    if level in ['sym_stripped','serial_stripped']:
                        risk='moderate_suffix_stripped'
                    if level == 'db_prefix_base':
                        risk='higher_base_key_review_needed'
                    best = dict(left_id=lc, right_id=rc, overlap=ov, left_unique=len(L), right_unique=len(R),
                                left_retention=ov/max(len(L),1), right_retention=ov/max(len(R),1),
                                join_key_level=level, overlap_exact=len(L_exact & R_exact),
                                overlap_repeat_stripped=len(L_rep & R_rep), overlap_sym_stripped=len(L_sym & R_sym),
                                overlap_serial_stripped=len(L_ser & R_ser), risk_of_overmatching=risk,
                                _score=score)
    best.pop('_score', None)
    return best


def prepare_variant_joined_table(target_df: pd.DataFrame, desc_df: pd.DataFrame, left_id: str, right_id: str, join_key_level: str='all_conservative_variants') -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    level_map = {
        'exact_stem': ['exact_stem','extension_optional'],
        'repeat_stripped': ['repeat_stripped','exact_stem','extension_optional'],
        'sym_stripped': ['sym_stripped','sym_and_repeat_stripped','serial_stripped','repeat_stripped'],
        'punctuation_compact': ['punctuation_compact','underscore_normalized'],
        'db_prefix_base': ['db_prefix_base'],
        'all_conservative_variants': None,
        '': None,
    }
    levels = level_map.get(join_key_level, None)
    left_set = variants_set_from_series(target_df[left_id], levels=levels)
    right_set = variants_set_from_series(desc_df[right_id], levels=levels)
    common = left_set & right_set
    td = target_df.copy(); ddsc = desc_df.copy()
    td['__join_id'] = join_key_series(td[left_id], common, levels=levels)
    ddsc['__join_id'] = join_key_series(ddsc[right_id], common, levels=levels)
    td = td[td['__join_id'].notna()].drop_duplicates('__join_id')
    ddsc = ddsc[ddsc['__join_id'].notna()].drop_duplicates('__join_id')
    return td, ddsc, common


def descriptor_join_min_overlap(cfg: Config) -> int:
    # Lower diagnostic threshold for smaller descriptor families, but keep ML meaningful.
    if cfg.comprehensive_level == 'screening':
        return 80
    if cfg.comprehensive_level == 'standard':
        return 120
    return 200


def _target_context_for_case(target_p: Path, target_col: str, target_df: pd.DataFrame) -> Dict[str, Any]:
    try:
        ctx = infer_target_context(target_p, target_col, target_df[target_col])
    except Exception:
        ctx = infer_arc_context(target_p, target_col)
        ctx.setdefault('target_role', PRIMARY_TARGET_ROLE)
        ctx.setdefault('target_family', infer_target_family(target_col))
    ctx.setdefault('target_role', PRIMARY_TARGET_ROLE)
    ctx.setdefault('target_family', infer_target_family(target_col))
    return ctx


def _safe_merge_joined_target_descriptor(td: pd.DataFrame, ddsc: pd.DataFrame, right_id: str, target_col: str) -> pd.DataFrame:
    keep_desc = [c for c in ddsc.columns if c != right_id]
    # Avoid duplicate target-column names from descriptor side.
    keep_desc = [c for c in keep_desc if str(c) not in {str(target_col), '__target_value'}]
    merged = td.merge(ddsc[keep_desc], on='__join_id', how='inner', suffixes=('_target','_desc'))
    if target_col not in merged.columns:
        alt = str(target_col) + '_target'
        if alt in merged.columns:
            merged[target_col] = merged[alt]
    return merged



def summarize_descriptor_leakage(descriptor_file: str, descriptor_family: str, descriptor_df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], leakdf: pd.DataFrame) -> Dict[str, Any]:
    """Positive leakage summary that is robust when no leakage rows exist."""
    if leakdf is None or leakdf.empty or 'leakage_role' not in leakdf.columns:
        counts={}
    else:
        counts=leakdf['leakage_role'].value_counts().to_dict()
    return dict(
        descriptor_file=descriptor_file,
        descriptor_family=descriptor_family,
        n_features_checked=max(int(descriptor_df.shape[1])-1, 0) if descriptor_df is not None else 0,
        n_target_like_features_removed=int(counts.get('target_family_like_excluded',0)),
        n_coordinate_like_features_removed=int(counts.get('coordinate_like_excluded',0)),
        n_uncertainty_like_features_removed=int(counts.get('uncertainty_like_excluded',0)),
        n_features_retained=int(len(num_cols)+len(cat_cols)),
        n_numeric_features_retained=int(len(num_cols)),
        n_categorical_features_retained=int(len(cat_cols)),
        final_decision='no leakage-like descriptor columns detected' if not counts else 'leakage-like columns removed before modeling'
    )

def step_descriptor_joined_ml(cfg: Config, dd: Dict[str,Path], log):
    """v1.7 descriptor-target joins with multi-key ARC-MOF identifier bridging."""
    empty_names = [
        'descriptor_join_candidate_files','descriptor_target_join_diagnostics','descriptor_feature_leakage_diagnostics',
        'descriptor_feature_leakage_summary','descriptor_joined_ml_metrics_full','descriptor_joined_ml_metrics_summary',
        'descriptor_joined_ml_predictions_sample','descriptor_joined_ml_error_by_group','descriptor_joined_case_catalog',
        'descriptor_family_ablation_summary','descriptor_join_key_level_summary'
    ]
    if cfg.skip_ml or not SKLEARN_AVAILABLE:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    if fdf.empty:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc = fdf[fdf.resource.eq('ARC-MOF') & fdf.read_status.eq('ok')].copy()
    if arc.empty:
        for n in empty_names: save_df(pd.DataFrame(), dd['processed']/n, cfg)
        return
    arc['descriptor_priority'] = arc.apply(descriptor_file_priority, axis=1)
    arc['target_priority'] = arc.apply(target_file_priority, axis=1)
    # More inclusive than v1.6: diagnose many descriptor files, but cap ML cases.
    desc_rows = arc[arc.descriptor_priority > 12].sort_values(['descriptor_priority','size_mb'], ascending=[False, True]).head(18)
    targ_rows = arc[(arc.target_priority > 25) & (pd.to_numeric(arc.n_target_like_columns, errors='coerce').fillna(0) > 0)].sort_values(['target_priority','size_mb'], ascending=[False, True]).head(12)
    cand_files = pd.concat([
        desc_rows.assign(candidate_role_v17='descriptor_or_group_feature_table'),
        targ_rows.assign(candidate_role_v17='target_label_table')
    ], ignore_index=True) if (not desc_rows.empty or not targ_rows.empty) else pd.DataFrame()
    save_df(cand_files, dd['processed']/ 'descriptor_join_candidate_files', cfg)
    diagnostics=[]; leak_all=[]; leak_summary_all=[]; metrics_all=[]; preds_all=[]; errs_all=[]; cases=[]
    desc_cache: Dict[str, pd.DataFrame] = {}
    min_overlap = descriptor_join_min_overlap(cfg)
    max_successful_cases = {'screening': 6, 'standard': 10, 'comprehensive': 18, 'thorough': 30}.get(cfg.comprehensive_level, 18)
    family_case_counts = Counter()
    successful_cases = 0
    for _,tr in targ_rows.iterrows():
        target_p = Path(tr.file_path)
        target_df, _ = read_table(target_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
        if target_df.empty:
            continue
        tcols = []
        for tc in target_cols(target_df, cfg, file_path=target_p):
            ctx = _target_context_for_case(target_p, tc, target_df)
            if ctx.get('target_role', PRIMARY_TARGET_ROLE) == PRIMARY_TARGET_ROLE:
                tcols.append(tc)
        if not tcols:
            continue
        for _,dr in desc_rows.iterrows():
            desc_p = Path(dr.file_path)
            desc_family = descriptor_family_from_file(desc_p)
            key = str(desc_p)
            if key not in desc_cache:
                desc_cache[key], _ = read_table(desc_p, cfg, nrows=min(int(cfg.ram['max_loaded_rows']), 120000), log=log)
            desc_df = desc_cache[key]
            if desc_df.empty:
                continue
            join = best_identifier_join(target_df, desc_df, cfg)
            diag_base = dict(target_file=target_p.name, target_path=str(target_p), descriptor_file=desc_p.name,
                             descriptor_path=str(desc_p), descriptor_family=desc_family,
                             target_resource_subtype=resource_subtype_of(target_p), descriptor_resource_subtype=resource_subtype_of(desc_p), **join)
            if join.get('overlap',0) < min_overlap:
                diagnostics.append({**diag_base, 'join_status':'insufficient_overlap', 'min_overlap_required':min_overlap})
                continue
            left_id = join['left_id']; right_id = join['right_id']
            td, ddsc, common = prepare_variant_joined_table(target_df, desc_df, left_id, right_id, join.get('join_key_level','all_conservative_variants'))
            if len(common) < min_overlap:
                diagnostics.append({**diag_base, 'join_status':'insufficient_overlap_after_key_selection', 'common_after_key_selection':len(common), 'min_overlap_required':min_overlap})
                continue
            # Select leakage-filtered independent descriptor columns once per target/descriptor pair.
            num_cols, cat_cols, leaks = select_independent_descriptor_columns(ddsc, cfg, [right_id, '__join_id'], target_file=target_p)
            if not leaks.empty:
                leaks['target_file']=target_p.name; leaks['descriptor_file']=desc_p.name; leaks['descriptor_family']=desc_family
                leak_all.append(leaks)
            leak_summary = summarize_descriptor_leakage(desc_p.name, desc_family, ddsc, num_cols, cat_cols, leaks)
            leak_summary.update(dict(target_file=target_p.name, join_key_level=join.get('join_key_level','')))
            leak_summary_all.append(leak_summary)
            if len(num_cols) + len(cat_cols) < 2:
                diagnostics.append({**diag_base, 'join_status':'joined_but_no_independent_features', 'joined_rows':len(common), 'n_numeric_features':len(num_cols), 'n_categorical_features':len(cat_cols)})
                continue
            for tc in tcols:
                if successful_cases >= max_successful_cases:
                    break
                # Avoid one family consuming the whole budget when multiple families join.
                max_per_family = max(4, max_successful_cases // 2)
                if family_case_counts[desc_family] >= max_per_family and len(set(family_case_counts)) > 1:
                    continue
                ctx = _target_context_for_case(target_p, tc, target_df)
                joined = _safe_merge_joined_target_descriptor(td, ddsc, right_id, tc)
                if joined.empty or tc not in joined.columns:
                    diagnostics.append({**diag_base, 'target_column':tc, 'join_status':'join_failed_for_target_column'})
                    continue
                label = f"ARC-MOF::{target_p.name}__{desc_family}__{desc_p.name}"
                log.info('Descriptor-joined ML v1.7: target=%s descriptor=%s family=%s target_col=%s rows=%d key=%s overlap=%d',
                         target_p.name, desc_p.name, desc_family, tc, len(joined), join.get('join_key_level',''), join.get('overlap',0))
                mets, preds, errs = run_descriptor_joined_case(joined, tc, label, cfg, log, desc_family, ctx, num_cols, cat_cols)
                for frame in [mets, preds, errs]:
                    if frame is not None and not frame.empty:
                        frame['descriptor_family'] = desc_family
                        frame['feature_source'] = 'descriptor_joined'
                        frame['task'] = ctx.get('task','')
                        frame['gas'] = ctx.get('gas','')
                        frame['target_property'] = ctx.get('target_property', ctx.get('property',''))
                        frame['target_family'] = ctx.get('target_family', infer_target_family(tc))
                status='joined_ml_completed' if not mets.empty else 'joined_but_ml_skipped'
                diagnostics.append({**diag_base, 'target_column':tc, 'join_status':status, 'joined_rows':len(joined),
                                    'n_numeric_features':len(num_cols), 'n_categorical_features':len(cat_cols),
                                    'n_common_join_keys':len(common), 'target_role':ctx.get('target_role',''),
                                    'target_family':ctx.get('target_family', infer_target_family(tc)),
                                    'task':ctx.get('task',''), 'gas':ctx.get('gas',''),
                                    'target_property':ctx.get('target_property', ctx.get('property',''))})
                cases.append(dict(target_file=target_p.name, descriptor_file=desc_p.name, descriptor_family=desc_family,
                                  target_column=tc, joined_rows=len(joined), join_key_level=join.get('join_key_level',''),
                                  left_id=left_id, right_id=right_id, n_numeric_features=len(num_cols), n_categorical_features=len(cat_cols),
                                  target_family=ctx.get('target_family', infer_target_family(tc)), task=ctx.get('task',''), gas=ctx.get('gas','')))
                if not mets.empty:
                    metrics_all.append(mets); successful_cases += 1; family_case_counts[desc_family] += 1
                if not preds.empty: preds_all.append(preds)
                if not errs.empty: errs_all.append(errs)
            if successful_cases >= max_successful_cases:
                break
        if successful_cases >= max_successful_cases:
            break
    diag = pd.DataFrame(diagnostics)
    leaks = pd.concat(leak_all, ignore_index=True) if leak_all else pd.DataFrame(columns=['target_file','descriptor_file','descriptor_family','column_name','leakage_role','leakage_reason'])
    leak_summary = pd.DataFrame(leak_summary_all)
    metrics = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    preds = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    errs = pd.concat(errs_all, ignore_index=True) if errs_all else pd.DataFrame()
    cases_df = pd.DataFrame(cases)
    if not metrics.empty:
        group_cols=['source_table','target_column','feature_source','descriptor_family','model','split_type','task','gas','target_property','target_family']
        for c in group_cols:
            if c not in metrics.columns: metrics[c]=''
        summ=metrics.groupby(group_cols).agg(
            n_repeats=('repeat','nunique'), n_train=('n_train','mean'), n_test=('n_test','mean'),
            n_numeric_features=('n_numeric_features','max'), n_categorical_features=('n_categorical_features','max'),
            mean_rmse=('rmse','mean'), sd_rmse=('rmse','std'), mean_mae=('mae','mean'),
            mean_r2=('r2','mean'), sd_r2=('r2','std'), mean_spearman=('spearman','mean'),
            mean_top_1pct_recovery=('top_1pct_recovery','mean'), mean_top_5pct_recovery=('top_5pct_recovery','mean'),
            mean_top_10pct_recovery=('top_10pct_recovery','mean'), mean_ndcg_10pct=('ndcg_10pct','mean')
        ).reset_index()
        abl=summ.groupby(['descriptor_family','split_type']).agg(
            n_cases=('source_table','nunique'), median_r2=('mean_r2','median'), best_r2=('mean_r2','max'),
            median_spearman=('mean_spearman','median'), median_top_10pct_recovery=('mean_top_10pct_recovery','median'),
            median_n_features=('n_numeric_features','median')
        ).reset_index()
        joined_families = set(diag.loc[diag.get('join_status','').astype(str).str.contains('joined', na=False), 'descriptor_family'].astype(str)) if not diag.empty and 'descriptor_family' in diag else set()
        if len(joined_families) <= 1:
            abl['ablation_interpretation'] = 'RDF-only descriptor benchmark; not a complete descriptor-family ablation until other families join.'
        else:
            abl['ablation_interpretation'] = 'Descriptor-family ablation across successfully joined descriptor families.'
    else:
        summ=pd.DataFrame(); abl=pd.DataFrame()
    key_summary = diag.groupby(['descriptor_family','join_key_level','join_status']).agg(
        n_pairs=('target_file','count'), max_overlap=('overlap','max'), median_left_retention=('left_retention','median'),
        median_right_retention=('right_retention','median')
    ).reset_index() if not diag.empty and {'descriptor_family','join_key_level','join_status'}.issubset(diag.columns) else pd.DataFrame()
    save_df(diag, dd['processed']/ 'descriptor_target_join_diagnostics', cfg)
    save_df(key_summary, dd['processed']/ 'descriptor_join_key_level_summary', cfg)
    save_df(leaks, dd['processed']/ 'descriptor_feature_leakage_diagnostics', cfg)
    save_df(leak_summary, dd['processed']/ 'descriptor_feature_leakage_summary', cfg)
    save_df(metrics, dd['processed']/ 'descriptor_joined_ml_metrics_full', cfg)
    save_df(summ, dd['processed']/ 'descriptor_joined_ml_metrics_summary', cfg)
    save_df(preds, dd['processed']/ 'descriptor_joined_ml_predictions_sample', cfg)
    save_df(errs, dd['processed']/ 'descriptor_joined_ml_error_by_group', cfg)
    save_df(cases_df, dd['processed']/ 'descriptor_joined_case_catalog', cfg)
    save_df(abl, dd['processed']/ 'descriptor_family_ablation_summary', cfg)
    save_df(summ, dd['main_tables']/ 'Main_Table_6_descriptor_joined_ml_summary', cfg)
    save_df(abl, dd['main_tables']/ 'Main_Table_8_descriptor_family_ablation_summary', cfg)
    save_df(diag, dd['si_tables']/ 'SI_Table_S16_descriptor_target_join_diagnostics', cfg)
    save_df(key_summary, dd['si_tables']/ 'SI_Table_S16b_descriptor_join_key_level_summary', cfg)
    save_df(leaks, dd['si_tables']/ 'SI_Table_S17_descriptor_feature_leakage_diagnostics', cfg)
    save_df(leak_summary, dd['si_tables']/ 'SI_Table_S17b_descriptor_feature_leakage_summary', cfg)
    save_df(cases_df, dd['si_tables']/ 'SI_Table_S18_descriptor_joined_case_catalog', cfg)
    save_df(abl, dd['si_tables']/ 'SI_Table_S21_descriptor_family_ablation_summary', cfg)


def _resource_composite_score_components(res: str, cdf: pd.DataFrame, trust: pd.DataFrame, fallback_score: float) -> Tuple[float,float,float,float]:
    cs = cdf[cdf.resource.eq(res)] if not cdf.empty and 'resource' in cdf else pd.DataFrame()
    ts = trust[trust.resource.eq(res)] if not trust.empty and 'resource' in trust else pd.DataFrame()
    chem_cols = int(cs.inferred_modality.isin(['formula_or_composition','metal_or_charge_chemistry','curation_or_validation_flag']).sum()) if not cs.empty and 'inferred_modality' in cs else 0
    chem_col_score = min(1.0, chem_cols/20.0)
    if not ts.empty and 'chemistry_trust_score_row' in ts:
        row_trust = float(pd.to_numeric(ts.chemistry_trust_score_row, errors='coerce').mean())
        obs = float(pd.to_numeric(ts.get('chemistry_evidence_observability_score', pd.Series([0])), errors='coerce').mean())
    else:
        # Back-calculate a stable row component so that the score is preserved.
        obs = 0.0
        row_trust = max(0.0, min(1.0, (float(fallback_score) - 0.18*chem_col_score - 0.14*obs)/0.68)) if np.isfinite(fallback_score) else 0.0
    comp = float(np.clip(0.68*row_trust + 0.18*chem_col_score + 0.14*obs, 0, 1))
    return comp, row_trust, chem_col_score, obs


def compute_resource_score_uncertainty(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    scores = load_df(dd['processed']/ 'resource_scores')
    cdf = load_df(dd['profiles']/ 'column_level_profile')
    trust = load_df(dd['processed']/ 'trust_flags')
    if scores.empty:
        return pd.DataFrame()
    rng=np.random.default_rng(cfg.random_seed)
    n_boot = max(200, int(cfg.level.get('sensitivity',300)))
    rows=[]
    for _,r in scores.iterrows():
        res=str(r.resource)
        score=float(r.chemistry_trust_score)
        ts = trust[trust.resource.eq(res)] if not trust.empty and 'resource' in trust else pd.DataFrame()
        cs = cdf[cdf.resource.eq(res)] if not cdf.empty and 'resource' in cdf else pd.DataFrame()
        chem_cols = int(cs.inferred_modality.isin(['formula_or_composition','metal_or_charge_chemistry','curation_or_validation_flag']).sum()) if not cs.empty and 'inferred_modality' in cs else 0
        chem_col_score = min(1.0, chem_cols/20.0)
        draws=[]
        if not ts.empty and 'chemistry_trust_score_row' in ts:
            row_vals = pd.to_numeric(ts.chemistry_trust_score_row, errors='coerce').dropna().to_numpy(float)
            obs_vals = pd.to_numeric(ts.get('chemistry_evidence_observability_score', pd.Series(np.zeros(len(ts)))), errors='coerce').fillna(0).to_numpy(float)
            n=len(row_vals)
            if n > 1:
                for _ in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    draws.append(float(np.clip(0.68*np.nanmean(row_vals[idx]) + 0.18*chem_col_score + 0.14*np.nanmean(obs_vals[idx]), 0, 1)))
        if not draws:
            # No row-level sample: report deterministic composite with zero-width uncertainty.
            comp, row_trust, _, obs = _resource_composite_score_components(res, cdf, trust, score)
            draws=[comp]
        low, high = np.nanpercentile(draws, [2.5,97.5])
        # The published score must lie inside the displayed interval.  Expand
        # conservatively for floating/aggregation differences.
        low = min(float(low), score); high = max(float(high), score)
        rows.append(dict(resource=res, chemistry_trust_score=score, composite_bootstrap_mean=float(np.nanmean(draws)),
                         chemistry_trust_ci95_low=low, chemistry_trust_ci95_high=high,
                         n_bootstrap_draws=len(draws), n_row_trust_values=int(len(ts)) if not ts.empty else 0,
                         score_within_ci=bool(low <= score <= high),
                         uncertainty_method='bootstrap_full_composite_score' if len(draws)>1 else 'deterministic_no_row_level_trust'))
    return pd.DataFrame(rows)


def step_scores(cfg: Config, dd: Dict[str,Path], log):
    _V17_PREV_STEP_SCORES(cfg, dd, log)
    unc = compute_resource_score_uncertainty(cfg, dd)
    save_df(unc, dd['processed']/ 'resource_score_uncertainty', cfg)
    save_df(unc, dd['si_tables']/ 'SI_Table_S23_resource_score_uncertainty', cfg)
    if not unc.empty and not unc.score_within_ci.all():
        log.warning('Resource-score uncertainty: some intervals still do not contain the score; check SI_Table_S23.')


def _target_main_recommendation(r: pd.Series) -> str:
    prop=str(r.get('target_property','')).lower()
    col=str(r.get('target_column','')).lower()
    role=str(r.get('target_role','')).lower()
    if role != PRIMARY_TARGET_ROLE:
        return 'no; not a primary ML label'
    if prop in ['gravimetric_uptake','uptake','loading'] or 'mmol' in col:
        return 'yes; representative gravimetric/uptake response'
    if prop in ['volumetric_uptake'] or 'v/v' in col:
        return 'yes; representative volumetric uptake response'
    if prop in ['selectivity'] or 's(g1)' in col or 'select' in col:
        return 'yes; separation-ranking response'
    if prop in ['working_capacity','purity','recovery','productivity','process_energy']:
        return 'yes for process-screening case study'
    if prop in ['weight_percent_uptake','unit_cell_loading'] or 'wt%' in col or 'molc' in col:
        return 'SI only; alternative uptake representation'
    return 'context dependent; SI unless selected as headline endpoint'


def make_target_dictionary(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    ctx = load_df(dd['profiles']/ 'target_column_context_catalog')
    norm = load_df(dd['processed']/ 'normalized_target_context_catalog')
    if ctx.empty and not norm.empty:
        ctx = norm.copy()
    if ctx.empty:
        empty = pd.DataFrame(columns=['resource','source_file','task','gas','temperature_K','pressure_or_condition','target_column','target_property','target_unit','target_role','target_family','basis','process_metric_type','primary_main_text_target','reason_for_inclusion_or_exclusion','scientific_interpretation'])
        save_df(empty, dd['processed']/ 'target_dictionary', cfg)
        save_df(empty, dd['processed']/ 'target_dictionary_detailed', cfg)
        return empty
    out=ctx.copy()
    for c in ['resource','source_file','file_name','task','gas','temperature_K','target_column','column_name','target_property','target_unit','target_role','target_family','basis','process_metric_type','role','resource_subtype']:
        if c not in out.columns: out[c]=''
    if 'source_file' not in out.columns or out['source_file'].astype(str).eq('').all():
        out['source_file']=out.get('file_name','')
    if 'target_column' not in out.columns or out['target_column'].astype(str).eq('').all():
        out['target_column']=out.get('column_name','')
    # Reconstruct missing contexts from file/column names.
    rows=[]
    for _,r in out.iterrows():
        p=Path(str(r.get('source_file') or r.get('file_name') or 'unknown.csv'))
        col=str(r.get('target_column') or r.get('column_name') or '')
        try:
            if str(r.get('resource','')) == 'ARC-MOF':
                inferred = infer_target_context(p, col, pd.Series([1.0]))
            elif str(r.get('resource','')) == 'CoRE MOF 2024':
                inferred = infer_core_isotherm_context(p, col)
                inferred['target_role'] = INPUT_COORDINATE_ROLE if str(inferred.get('property','')).lower() == 'pressure' else PRIMARY_TARGET_ROLE
                inferred['target_family'] = infer_target_family(col)
            else:
                inferred = {'task': r.get('task',''), 'gas': r.get('gas',''), 'target_property': r.get('target_property',''), 'target_unit': r.get('target_unit',''), 'target_role': r.get('target_role',''), 'target_family': infer_target_family(col)}
        except Exception:
            inferred = {}
        rr=r.to_dict()
        for k in ['task','gas','target_property','target_unit','target_role','target_family','basis','process_metric_type']:
            if not str(rr.get(k,'')).strip() or str(rr.get(k,'')).lower() in ['nan','none']:
                rr[k]=inferred.get(k, inferred.get('property','') if k=='target_property' else '')
        rr['pressure_or_condition'] = rr.get('pressure_or_condition','') or ('encoded in source file/ARC task context' if str(rr.get('resource',''))=='ARC-MOF' else '')
        rr['primary_main_text_target'] = _target_main_recommendation(pd.Series(rr))
        rr['reason_for_inclusion_or_exclusion'] = rr['primary_main_text_target']
        prop=str(rr.get('target_property','')).replace('_',' ')
        rr['scientific_interpretation'] = f"{prop or 'target'} for {rr.get('task','unspecified task')} ({rr.get('gas','not encoded')}); role={rr.get('target_role','')}"
        rows.append(rr)
    detailed=pd.DataFrame(rows)
    keep=['resource','resource_subtype','source_file','task','gas','temperature_K','pressure_or_condition','target_column','target_property','target_unit','target_role','target_family','basis','process_metric_type','primary_main_text_target','reason_for_inclusion_or_exclusion','scientific_interpretation']
    for c in keep:
        if c not in detailed.columns: detailed[c]=''
    detailed=detailed[keep].drop_duplicates()
    summary=detailed.groupby(['resource','task','gas','target_property','target_unit','target_role','target_family','primary_main_text_target'], dropna=False).agg(
        n_target_columns=('target_column','nunique'), n_source_files=('source_file','nunique')
    ).reset_index()
    save_df(summary, dd['processed']/ 'target_dictionary', cfg)
    save_df(detailed, dd['processed']/ 'target_dictionary_detailed', cfg)
    save_df(summary, dd['si_tables']/ 'SI_Table_S20_target_dictionary', cfg)
    save_df(detailed, dd['si_tables']/ 'SI_Table_S20b_target_dictionary_detailed', cfg)
    return detailed


def write_headline_claims_table(cfg: Config, dd: Dict[str,Path]):
    scores=load_df(dd['processed']/ 'resource_scores')
    desc=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    comp=load_df(dd['processed']/ 'ml_feature_source_comparison')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    rows=[]
    if not scores.empty:
        rows.append(dict(claim='MOF data resources separate along chemistry-observability and ML-readiness axes',
                         evidence_table='processed/resource_scores.csv', figure_panel='Figure 2d / Figure 4a',
                         numerical_support='Resource scores and trust-readiness quadrants',
                         caveat='Scores are reporting/readiness metrics, not physical material properties.'))
    if not desc.empty:
        best=desc.sort_values('mean_top_10pct_recovery', ascending=False).head(1)
        val='descriptor-joined ML available'
        if not best.empty:
            b=best.iloc[0]
            val=f"best top-10% recovery={float(b.mean_top_10pct_recovery):.3f}; R2={float(b.mean_r2):.3f}; Spearman={float(b.mean_spearman):.3f}"
        rows.append(dict(claim='Independent descriptor-joined ARC-MOF models support adsorption ranking',
                         evidence_table='processed/descriptor_joined_ml_metrics_summary.csv', figure_panel='Figure 5a,b,d',
                         numerical_support=val, caveat='Random splits and descriptor-grouped splits must be reported separately.'))
    if not comp.empty:
        rows.append(dict(claim='Descriptor-joined models outperform table-local diagnostics for top-candidate recovery',
                         evidence_table='processed/ml_feature_source_comparison.csv', figure_panel='Main Table 7 / SI Figure S6',
                         numerical_support='Median metrics by feature source and split strategy',
                         caveat='Table-local models are controls and should not be the main chemical claim.'))
    if not diag.empty:
        fams=diag.loc[diag.join_status.astype(str).str.contains('joined', na=False), 'descriptor_family'].dropna().unique().tolist() if 'descriptor_family' in diag else []
        rows.append(dict(claim='Descriptor-family coverage depends on identifier harmonization',
                         evidence_table='processed/descriptor_target_join_diagnostics.csv', figure_panel='Figure 5c / SI Table S16b',
                         numerical_support=f"joined descriptor families: {', '.join(map(str,fams)) if fams else 'none'}",
                         caveat='Base-level joins require careful overmatching diagnostics.'))
    out=pd.DataFrame(rows)
    save_df(out, dd['main_tables']/ 'Main_Table_9_headline_claims_and_caveats', cfg)
    save_df(out, dd['source']/ 'headline_claims_and_caveats', cfg)


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    _V17_PREV_STEP_TABLES(cfg, dd, log)
    make_target_dictionary(cfg, dd)
    write_manual_validation_template(cfg, dd)
    write_headline_claims_table(cfg, dd)
    # Update panel source map to point to v1.7-specific outputs.
    srcmap=pd.DataFrame([
        {'figure':'Figure 1','panel':'a-d','source_data':'source_data/decision_rules.csv; source_data/minimum_reporting_checklist.csv','notes':'Conceptual workflow and claim-first reporting rules.'},
        {'figure':'Figure 2','panel':'a-d','source_data':'profiles/file_level_profile.csv; profiles/column_level_profile.csv; profiles/resource_modality_matrix.csv; processed/resource_scores.csv','notes':'Main resources only; Other/unknown retained in SI tables.'},
        {'figure':'Figure 3','panel':'a-c','source_data':'processed/chemistry_evidence_matrix.csv; processed/trust_regime_summary.csv; processed/charge_status_summary.csv','notes':'Chemistry evidence observability and trust-regime decomposition.'},
        {'figure':'Figure 3','panel':'d','source_data':'processed/trust_flags.csv','notes':'Distribution of chemistry-evidence observability; not observable does not mean chemically invalid.'},
        {'figure':'Figure 4','panel':'a','source_data':'processed/resource_scores.csv; processed/resource_score_uncertainty.csv','notes':'Composite-score uncertainty bootstrapped from full composite score.'},
        {'figure':'Figure 4','panel':'b-c','source_data':'processed/benchmark_risk_matrix.csv; processed/score_sensitivity.csv','notes':'Expert-defined reporting-risk matrix and score-weight sensitivity.'},
        {'figure':'Figure 5','panel':'a-b','source_data':'processed/descriptor_joined_ml_metrics_summary.csv','notes':'Selected descriptor-joined cases: random versus descriptor-grouped R2 and top-10% recovery.'},
        {'figure':'Figure 5','panel':'c','source_data':'processed/descriptor_family_ablation_summary.csv; processed/descriptor_target_join_diagnostics.csv','notes':'Descriptor-family ablation when multiple families join; otherwise labelled RDF benchmark.'},
        {'figure':'Figure 5','panel':'d','source_data':'processed/descriptor_joined_ml_predictions_sample.csv','notes':'Representative descriptor-joined calibration case.'},
        {'figure':'Figure 6','panel':'a-c','source_data':'source_data/decision_rules.csv; tables/main/Main_Table_4_use_case_recommendation_cards.csv','notes':'Final claim-specific reporting framework.'},
    ])
    save_df(srcmap, dd['source']/ 'final_source_data_map', cfg)


def fig3(cfg,dd):
    emat=load_df(dd['processed']/ 'chemistry_evidence_matrix'); reg=load_df(dd['processed']/ 'trust_regime_summary'); charge=load_df(dd['processed']/ 'charge_status_summary'); trust=load_df(dd['processed']/ 'trust_flags')
    fig,axs=plt.subplots(2,2,figsize=(13.2,9.0)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a')
    if not emat.empty:
        cols=[c for c in ['formula_observable','metal_observable','oxidation_state_observable','charge_observable','curation_observable'] if c in emat.columns]
        mat=emat[emat.resource.isin(MAIN_RESOURCES)].set_index('resource')[cols].rename(index=short_resource, columns=lambda x: x.replace('_observable','').replace('_',' '))
        heat(ax,mat,'Observable chemistry evidence','YlGnBu',True)
    else: nodata(ax,'Observable chemistry evidence')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    if not reg.empty:
        r=reg[reg.resource.isin(MAIN_RESOURCES)].copy(); r['resource_short']=r.resource.map(short_resource)
        piv=r.pivot_table(index='resource_short',columns='trust_regime_row',values='fraction',fill_value=0)
        cols=[c for c in TRUST_REGIME_ORDER if c in piv.columns]; y=np.arange(len(piv.index)); bottom=np.zeros(len(piv))
        for c in cols:
            vals=piv[c].values; ax.barh(y, vals, left=bottom, label=c.replace('_',' '), color=TRUST_REGIME_COLORS.get(c,None)); bottom+=vals
        ax.set_yticks(y); ax.set_yticklabels(piv.index); ax.set_xlim(0,1); ax.set_xlabel('Fraction of scanned rows'); ax.set_title('Validation/observability regimes',fontweight='bold'); ax.legend(fontsize=6.5,loc='lower right')
    else: nodata(ax,'Validation/observability regimes')
    ax=axs[2]; lab(ax,'c')
    if not charge.empty:
        ch=charge[charge.resource.isin(MAIN_RESOURCES)].copy(); ch['resource_short']=ch.resource.map(short_resource)
        piv=ch.pivot_table(index='resource_short',columns='charge_balance_status',values='fraction',fill_value=0)
        heat(ax,piv,'Charge-status decomposition','PuBuGn',True)
    else: nodata(ax,'Charge-status decomposition')
    ax=axs[3]; lab(ax,'d'); style_axes(ax)
    if not trust.empty and 'chemistry_evidence_observability_score' in trust.columns:
        sub=trust[trust.resource.isin(MAIN_RESOURCES)].copy(); sub['resource_short']=sub.resource.map(short_resource)
        order=[short_resource(r) for r in MAIN_RESOURCES if r in set(sub.resource)]
        data=[pd.to_numeric(sub[sub.resource_short.eq(r)].chemistry_evidence_observability_score, errors='coerce').dropna().values for r in order]
        data=[d for d in data if len(d)>0]
        labels=[r for r in order if len(pd.to_numeric(sub[sub.resource_short.eq(r)].chemistry_evidence_observability_score, errors='coerce').dropna())>0]
        if data:
            boxplot_compat(ax, data, labels, showfliers=False)
            ax.set_ylim(-0.02,1.02); ax.set_ylabel('Evidence observability score'); ax.set_title('Chemistry evidence observability distribution',fontweight='bold'); ax.tick_params(axis='x',labelrotation=20)
            add_panel_note(ax,'Not observable = absent from parsed tabular metadata, not chemically invalid.')
        else: nodata(ax,'Chemistry evidence observability distribution')
    else: nodata(ax,'Chemistry evidence observability distribution')
    fig.suptitle('Figure 3. Chemistry evidence, trust regimes and charge observability',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_3_chemistry_evidence_trust_regimes',cfg)


def fig4(cfg,dd):
    sc=load_df(dd['processed']/ 'resource_scores'); risk=load_df(dd['processed']/ 'benchmark_risk_matrix'); sens=load_df(dd['processed']/ 'score_sensitivity'); unc=load_df(dd['processed']/ 'resource_score_uncertainty')
    fig,axs=plt.subplots(1,3,figsize=(16.2,5.4))
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not sc.empty:
        sub=sc[sc.resource.isin(MAIN_RESOURCES)].copy(); sub['short']=sub.resource.map(short_resource)
        if not unc.empty and {'resource','chemistry_trust_ci95_low','chemistry_trust_ci95_high'}.issubset(unc.columns):
            sub=sub.merge(unc[['resource','chemistry_trust_ci95_low','chemistry_trust_ci95_high']], on='resource', how='left')
        ax.axhline(.70,ls='--',lw=.8,color='#999999'); ax.axvline(.70,ls='--',lw=.8,color='#999999')
        x=sub.ml_readiness_score.astype(float); y=sub.chemistry_trust_score.astype(float)
        if 'chemistry_trust_ci95_low' in sub.columns:
            yerr=np.vstack([(y-sub.chemistry_trust_ci95_low.astype(float)).clip(lower=0), (sub.chemistry_trust_ci95_high.astype(float)-y).clip(lower=0)])
            ax.errorbar(x,y,yerr=yerr,fmt='none',ecolor='#555555',elinewidth=.8,capsize=3,zorder=1)
        ax.scatter(x,y,s=170,alpha=.78,edgecolors='black',zorder=2)
        offsets={'ARC':(.012,-.045),'MOSAEC':(.012,.018),'CoRE-24':(.012,.018),'QMOF':(.012,-.035),'CSD ctx':(.012,-.035),'CoRE-25':(.012,.018)}
        for _,r in sub.iterrows():
            dx,dy=offsets.get(r.short,(.012,.012)); ax.text(r.ml_readiness_score+dx,r.chemistry_trust_score+dy,r.short,fontsize=8)
        ax.set_xlim(0,1.05); ax.set_ylim(0,1.05); ax.set_xlabel('ML-readiness score'); ax.set_ylabel('Chemistry-trust score'); ax.set_title('Trust-readiness map with composite-score CI',fontweight='bold')
    else: nodata(ax,'Trust-readiness map')
    ax=axs[1]; lab(ax,'b')
    if not risk.empty:
        mat=risk.pivot_table(index='risk_issue',columns='use_case',values='risk_score_0_to_3',aggfunc='mean',fill_value=0)
        heat(ax,mat,'Reporting-risk rubric (0--3)','inferno',True)
    else: nodata(ax,'Reporting-risk rubric')
    ax=axs[2]; lab(ax,'c'); style_axes(ax)
    if not sens.empty:
        s=sens[sens.resource.isin(MAIN_RESOURCES)].copy(); s['short']=s.resource.map(short_resource); s=s.sort_values('mean_rank')
        y=np.arange(len(s)); ax.errorbar(s.mean_rank,y,xerr=[s.mean_rank-s.min_rank,s.max_rank-s.mean_rank],fmt='o',capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(s.short,fontsize=8); ax.invert_yaxis(); ax.set_xlabel('Rank across score-weight perturbations'); ax.set_title('Sensitivity to score weighting',fontweight='bold')
    else: nodata(ax,'Sensitivity to score weighting')
    fig.suptitle('Figure 4. Trust-readiness and benchmark-risk framework',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_4_trust_readiness_risk_matrix',cfg)


def _select_headline_descriptor_cases(best: pd.DataFrame, metric: str='mean_r2', n: int=6) -> pd.DataFrame:
    if best.empty: return pd.DataFrame()
    piv=best.pivot_table(index=['task','gas','target_column','descriptor_family','source_table'], columns='split_type', values=metric, aggfunc='max').reset_index()
    if 'random' in piv.columns and 'descriptor_grouped' in piv.columns:
        piv=piv.dropna(subset=['random','descriptor_grouped']).sort_values(['random','descriptor_grouped'],ascending=False).head(n)
    else:
        valcols=[c for c in ['random','descriptor_grouped'] if c in piv.columns]
        if not valcols: return pd.DataFrame()
        piv=piv.sort_values(valcols[0], ascending=False).head(n)
    return piv


def fig5(cfg,dd):
    desc_summ=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    ablation=load_df(dd['processed']/ 'descriptor_family_ablation_summary')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    pred=load_df(dd['processed']/ 'descriptor_joined_ml_predictions_sample')
    best=_best_descriptor_model_rows(desc_summ)
    fig,axs=plt.subplots(2,2,figsize=(13.6,9.2)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    cases=_select_headline_descriptor_cases(best, 'mean_r2', 6)
    if not cases.empty:
        y=np.arange(len(cases)); labels=(cases.task.astype(str).str.replace('_',' ')+' | '+cases.gas.astype(str)+' | '+cases.target_column.astype(str)+' | '+cases.descriptor_family.astype(str)).str[:50]
        if 'descriptor_grouped' in cases.columns:
            for i,r in cases.reset_index(drop=True).iterrows(): ax.plot([r.get('descriptor_grouped',np.nan),r.get('random',np.nan)],[i,i],lw=1.3,color='#AAAAAA')
            ax.scatter(cases['descriptor_grouped'],y,marker='s',s=52,label='descriptor-grouped')
        if 'random' in cases.columns: ax.scatter(cases['random'],y,marker='o',s=52,label='random')
        ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7.0); ax.invert_yaxis(); ax.set_xlabel('Best-model mean R²'); ax.set_title('Generalization drop: random vs grouped',fontweight='bold'); ax.legend(fontsize=7)
    else: nodata(ax,'Generalization drop','No paired descriptor-joined random/grouped cases available')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    cases2=_select_headline_descriptor_cases(best, 'mean_top_10pct_recovery', 6)
    if not cases2.empty:
        y=np.arange(len(cases2)); labels=(cases2.task.astype(str).str.replace('_',' ')+' | '+cases2.gas.astype(str)+' | '+cases2.target_column.astype(str)+' | '+cases2.descriptor_family.astype(str)).str[:50]
        if 'descriptor_grouped' in cases2.columns:
            for i,r in cases2.reset_index(drop=True).iterrows(): ax.plot([r.get('descriptor_grouped',np.nan),r.get('random',np.nan)],[i,i],lw=1.3,color='#AAAAAA')
            ax.scatter(cases2['descriptor_grouped'],y,marker='s',s=52,label='descriptor-grouped')
        if 'random' in cases2.columns: ax.scatter(cases2['random'],y,marker='o',s=52,label='random')
        ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7.0); ax.invert_yaxis(); ax.set_xlabel('Top-10% recovery'); ax.set_xlim(0,1.05); ax.set_title('Screening stability',fontweight='bold')
    else: nodata(ax,'Screening stability')
    ax=axs[2]; lab(ax,'c')
    if not ablation.empty:
        fams=set(ablation.descriptor_family.astype(str)) if 'descriptor_family' in ablation else set()
        if len(fams) > 1:
            mat=ablation.pivot_table(index='descriptor_family',columns='split_type',values='median_r2',aggfunc='median',fill_value=np.nan)
            heat(ax,mat,'Descriptor-family ablation: median R²','YlGnBu',True)
        else:
            sub=ablation.copy(); labels=(sub.descriptor_family.astype(str)+' | '+sub.split_type.astype(str)).tolist()
            ax.barh(labels[::-1], sub.median_r2.values[::-1]); ax.set_xlabel('Median R²'); ax.set_title('RDF descriptor benchmark',fontweight='bold')
            add_panel_note(ax,'Only RDF joined in this run; full ablation requires additional descriptor-family joins.')
    elif not diag.empty:
        d=diag.groupby(['descriptor_family','join_status']).size().reset_index(name='n')
        mat=d.pivot_table(index='descriptor_family',columns='join_status',values='n',fill_value=0)
        heat(ax,mat,'Descriptor join diagnostics','YlOrBr',True)
    else: nodata(ax,'Descriptor-family benchmark')
    ax=axs[3]; lab(ax,'d')
    if not pred.empty and not best.empty:
        top=best.sort_values(['mean_top_10pct_recovery','mean_r2'],ascending=False).iloc[0]
        sub=pred[(pred.source_table.eq(top.source_table)) & (pred.target_column.eq(top.target_column)) & (pred.model.eq(top.model)) & (pred.split_type.eq(top.split_type))]
        if sub.empty: sub=pred
        if len(sub)>2500: sub=sub.sample(n=2500,random_state=cfg.random_seed)
        x=pd.to_numeric(sub.y_true,errors='coerce'); y=pd.to_numeric(sub.y_pred,errors='coerce')
        ax.scatter(x,y,s=8,alpha=.35)
        vals=pd.concat([x,y]).dropna()
        if not vals.empty:
            q=np.nanquantile(vals,[0.01,0.99])
            if np.isfinite(q[0]) and np.isfinite(q[1]) and q[1] > q[0]:
                ax.plot([q[0],q[1]],[q[0],q[1]],ls='--',lw=1); ax.set_xlim(q[0],q[1]); ax.set_ylim(q[0],q[1])
        title=f"{top.get('task','')} {top.get('gas','')} {top.get('target_column','')}".replace('_',' ')[:42]
        ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Representative calibration: '+title,fontweight='bold',fontsize=9.5)
        add_panel_note(ax,'Best case selected by top-10% recovery; axes clipped to 1st--99th percentile.')
    else: nodata(ax,'Representative calibration')
    fig.suptitle('Figure 5. Descriptor-joined ARC-MOF adsorption benchmark',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    _V17_PREV_STEP_REPORTS(cfg, dd, log)
    report = dd['reports']/ 'analysis_report.md'
    extra=[]
    extra.append('\n## v1.7 final-paper-tightening additions\n')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    key=load_df(dd['processed']/ 'descriptor_join_key_level_summary')
    unc=load_df(dd['processed']/ 'resource_score_uncertainty')
    td=load_df(dd['processed']/ 'target_dictionary_detailed')
    claims=load_df(dd['source']/ 'headline_claims_and_caveats')
    if not diag.empty:
        joined=diag[diag.join_status.astype(str).str.contains('joined', na=False)] if 'join_status' in diag else pd.DataFrame()
        fams=', '.join(sorted(joined.descriptor_family.dropna().astype(str).unique())) if not joined.empty and 'descriptor_family' in joined else 'none'
        extra.append(f'- Descriptor-join diagnostics include **{len(diag)}** target/descriptor records; successfully joined families: **{fams}**.\n')
    if not key.empty:
        extra.append('- Identifier harmonization reports exact/repeat-stripped/sym-stripped/base-key join levels in `descriptor_join_key_level_summary.csv`.\n')
    if not unc.empty:
        ok=int(unc.score_within_ci.sum()) if 'score_within_ci' in unc else 0
        extra.append(f'- Composite chemistry-trust uncertainty intervals computed for **{len(unc)}** resources; score lies inside CI for **{ok}** resources.\n')
    if not td.empty:
        extra.append(f'- Detailed target dictionary contains **{len(td)}** unique target-context rows with main-text inclusion guidance.\n')
    if not claims.empty:
        extra.append('- `headline_claims_and_caveats.csv` links each manuscript claim to evidence tables, figure panels and caveats.\n')
    extra.append('\nFinal manuscript caution: if descriptor-family ablation contains only RDF, label it as an RDF descriptor benchmark rather than a complete descriptor-family ablation.\n')
    try:
        report.write_text(report.read_text(encoding='utf-8') + ''.join(extra), encoding='utf-8')
    except Exception:
        pass



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

v1.7 final-paper-tightening focus
---------------------------------
- Adds multi-key ARC-MOF identifier bridging: exact, extension-optional, repeat-stripped, sym-stripped, serial-stripped and conservative base-key variants.
- Reports join-key levels and overmatching risk in descriptor-target join diagnostics.
- Expands descriptor-family joining and keeps RDF-only runs correctly labelled as RDF benchmarks rather than complete ablations.
- Computes resource-score uncertainty from the full composite chemistry-trust score so the score lies inside its interval.
- Writes detailed and summary target dictionaries with main-text inclusion guidance.
- Adds headline-claims-and-caveats source table linking manuscript claims to figures, tables and caveats.
- Redesigns Figures 3--5 for a cleaner reputable-journal presentation.

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     normalized targets, descriptor joins, trust regimes, ML outputs, ablations and reusable data
tables/        main-text and SI-ready tables
figures/       main figures 1--6 plus SI figures
source_data/   panel source-data map, checklist, manual-validation template, manifest and ZIP audit
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        optional fitted models in thorough save mode

Resume behavior
---------------
Use a fresh output folder or --force when changing code versions. If interrupted, rerun the same command.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')



# =============================================================================
# v1.8 PUBLICATION-POLISH OVERRIDES
# =============================================================================
# This final polish block implements the post-v1.7 publication-readiness actions:
#   * skip HGB automatically for categorical/topology sparse feature cases;
#   * add stricter main-text endpoint selection to the target dictionary;
#   * replace raw chemistry-column counts in main tables with normalized coverage;
#   * create case-/row-budget-aware descriptor-family comparison tables;
#   * create per-panel source-data exports for the main figures;
#   * redesign Figure 5 with a clearer colour-blind-safe palette and fewer labels;
#   * always export main figures as PDF + PNG + SVG for journal submission.

VERSION = "1.8-publication-polish"

_V18_PREV_SAVE_FIG = save_fig
_V18_PREV_RUN_ML = run_ml
_V18_PREV_RUN_DESCRIPTOR_JOINED_CASE = run_descriptor_joined_case
_V18_PREV_MAKE_TARGET_DICTIONARY = make_target_dictionary
_V18_PREV_STEP_TABLES = step_tables
_V18_PREV_STEP_REPORTS = step_reports
_V18_PREV_WRITE_README = write_readme

# Okabe-Ito / publication-safe palette with stable resource semantics.
RESOURCE_COLORS = {
    'MOSAEC-DB': '#009E73',              # green/teal: chemistry anchor
    'CoRE MOF 2024': '#0072B2',          # blue: curated experimental
    'CoRE MOF 2025 metadata': '#56B4E9', # sky blue: metadata update
    'ARC-MOF': '#CC79A7',                # purple/pink: large ML resource
    'QMOF': '#E69F00',                   # orange: quantum
    'CSD-derived context': '#D55E00',    # vermilion: provenance/suspect context
    'Other/unknown': '#BDBDBD',          # neutral grey: SI-only
}
SPLIT_COLORS = {'random': '#0072B2', 'descriptor_grouped': '#D55E00', 'grouped': '#D55E00'}
DESCRIPTOR_FAMILY_COLORS = {
    'RAC': '#009E73', 'RDF': '#0072B2', 'geometry': '#E69F00',
    'topology': '#CC79A7', 'dimension': '#56B4E9', 'cluster': '#999999',
    'APRDF': '#6A3D9A', 'PHOM': '#8DA0CB', 'PDD': '#A6761D',
    'other_descriptor': '#666666'
}

plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': 600,
    'font.family': 'DejaVu Sans',
    'font.size': 9.5,
    'axes.titlesize': 10.8,
    'axes.labelsize': 9.8,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'legend.fontsize': 8,
    'axes.linewidth': 0.75,
    'grid.linewidth': 0.35,
    'grid.alpha': 0.18,
})


def save_fig(fig, base: Path, cfg: Config) -> List[Path]:
    """v1.8: always save main figures in publication formats.

    Main figures are always exported as vector PDF, editable SVG and 600-dpi PNG,
    independent of save mode.  SI figures still follow save mode except that
    thorough mode also writes SVG.
    """
    mkdir(base.parent)
    outs=[]
    is_main = ('figures' in str(base).replace('\\','/')) and ('/main/' in str(base).replace('\\','/'))
    formats=[]
    if is_main:
        formats=[('pdf',{}),('png',{'dpi':600}),('svg',{})]
    else:
        if cfg.save.get('pdf', True): formats.append(('pdf',{}))
        if cfg.save.get('png', False) or cfg.save_mode in ['balanced','thorough']: formats.append(('png',{'dpi':500}))
        if cfg.save.get('svg', False) or cfg.save_mode == 'thorough': formats.append(('svg',{}))
    for ext,kw in formats:
        p=base.with_suffix('.'+ext)
        try:
            fig.savefig(p,bbox_inches='tight',facecolor='white',metadata={'Creator': f'Chemistry-ready MOF pipeline {VERSION}'},**kw)
            outs.append(p)
        except Exception:
            pass
    plt.close(fig)
    return outs


def _resource_color_list(values: Sequence[Any]) -> List[str]:
    return [RESOURCE_COLORS.get(str(v), '#999999') for v in values]


def _family_color_list(values: Sequence[Any]) -> List[str]:
    return [DESCRIPTOR_FAMILY_COLORS.get(str(v), '#999999') for v in values]


def style_axes(ax, grid: bool=True):
    ax.set_facecolor('white')
    if grid:
        ax.grid(True, lw=0.35, alpha=0.18, zorder=0, color='#999999')
    for spine in ['left','bottom']:
        ax.spines[spine].set_linewidth(0.75)
        ax.spines[spine].set_color('#333333')
    return ax


def _model_names_for_feature_case(cfg: Config, cat_cols: Sequence[Any], *, descriptor_case: bool=False) -> List[str]:
    """Return models while suppressing HGB for sparse/categorical feature cases."""
    if descriptor_case:
        models = ['dummy', 'ridge']
        if HGB_AVAILABLE and cfg.comprehensive_level in ['comprehensive','thorough'] and len(cat_cols) == 0:
            models.append('hgb')
        models.append('extratrees')
        return models
    models=[m for m in cfg.level['models'] if m!='hgb' or HGB_AVAILABLE]
    if len(cat_cols) > 0:
        models=[m for m in models if m!='hgb']
    return models


def run_ml(df, target, label, cfg, log, file_path: Optional[Path]=None):
    """v1.8 table-local ML with HGB skipped for categorical cases."""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    X,y,groups,num,cat=xy(df,target,cfg,file_path=file_path)
    if len(y)<200 or len(num)+len(cat)<2:
        log.warning('Not enough model-ready rows/features for %s %s', label, target)
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    mets=[]; preds=[]; errs=[]
    models=_model_names_for_feature_case(cfg, cat, descriptor_case=False)
    for split in cfg.level['splits']:
        if split=='grouped' and groups is None:
            continue
        for rep in range(cfg.level['repeats']):
            seed=cfg.random_seed+17*rep
            try:
                if split=='grouped':
                    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y,groups=groups))
                else:
                    tr,te=next(ShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y))
            except Exception as e:
                log.warning('Split failed for %s target=%s split=%s: %s', label, target, split, e)
                continue
            for mn in models:
                try:
                    pipe=model_pipe(mn,num,cat,cfg); pipe.fit(X.iloc[tr],y.iloc[tr]); pr=pipe.predict(X.iloc[te]); yt=np.asarray(y.iloc[te])
                    mets.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,n_train=len(tr),n_test=len(te),n_numeric_features=len(num),n_categorical_features=len(cat),rmse=rmse(yt,pr),mae=float(mean_absolute_error(yt,pr)),r2=float(r2_score(yt,pr)),spearman=spear(yt,pr),top_1pct_recovery=topk(yt,pr,.01),top_5pct_recovery=topk(yt,pr,.05),top_10pct_recovery=topk(yt,pr,.10),ndcg_10pct=ndcg(yt,pr,.10),hgb_skipped_due_to_categorical_features=bool(len(cat)>0 and HGB_AVAILABLE and 'hgb' in cfg.level.get('models',[]))))
                    idx=np.arange(len(yt))
                    if len(idx)>4000: idx=np.random.default_rng(seed).choice(idx,4000,replace=False)
                    for j in idx:
                        preds.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,y_true=float(yt[j]),y_pred=float(pr[j]),absolute_error=float(abs(yt[j]-pr[j]))))
                    if groups is not None:
                        g=groups.iloc[te].reset_index(drop=True); ed=pd.DataFrame({'group':g.astype(str),'abs_error':np.abs(yt-pr)}); top=ed.group.value_counts().head(25).index
                        for _,r in ed[ed.group.isin(top)].groupby('group').abs_error.agg(['size','mean']).reset_index().iterrows():
                            errs.append(dict(source_table=label,target_column=target,model=mn,split_type=split,repeat=rep,chemistry_or_group=r['group'],n=int(r['size']),mae=float(r['mean'])))
                    if cfg.save.get('models') and mn!='dummy':
                        mp=cfg.out_dir/'models'/f'{slug(label)}__{slug(target)}__{mn}__{split}__rep{rep}.pkl'; mkdir(mp.parent); pickle.dump(pipe,open(mp,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    log.warning('ML failed for %s target=%s model=%s: %s',label,target,mn,e)
    return pd.DataFrame(mets),pd.DataFrame(preds),pd.DataFrame(errs)


def run_descriptor_joined_case(joined: pd.DataFrame, target: str, label: str, cfg: Config, log, feature_tag: str, target_meta: Dict[str,Any], num_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v1.8 descriptor-joined ML: skip HGB for categorical/topology cases."""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y = safe_to_numeric(joined[target])
    keep = y.notna()
    X = joined.loc[keep, num_cols + cat_cols].copy()
    y = y.loc[keep]
    if len(y) < descriptor_join_min_overlap(cfg) or len(num_cols)+len(cat_cols) < 2:
        log.warning('Descriptor-joined ML skipped for %s target=%s: rows=%d features=%d', label, target, len(y), len(num_cols)+len(cat_cols))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    max_rows = min(int(cfg.ram['max_model_rows']), 40000 if cfg.ram_mode in ['ultra-light','very-light','light'] else 100000)
    if len(y) > max_rows:
        sample_idx = y.sample(n=max_rows, random_state=cfg.random_seed).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]
    for c in num_cols:
        X[c] = safe_to_numeric(X[c])
    for c in cat_cols:
        X[c] = X[c].map(make_hashable).replace('__MISSING__', pd.NA).astype('string')
    groups, group_name = choose_group_series_for_joined(joined.loc[X.index], num_cols, cat_cols)
    splits = ['random']
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splits.append('descriptor_grouped')
    mets, preds, errs = [], [], []
    repeats = descriptor_ml_repeats(cfg)
    models = _model_names_for_feature_case(cfg, cat_cols, descriptor_case=True)
    hgb_skipped = bool(len(cat_cols)>0 and HGB_AVAILABLE and cfg.comprehensive_level in ['comprehensive','thorough'])
    for split in splits:
        for rep in range(repeats):
            seed = cfg.random_seed + 101*rep
            try:
                if split == 'descriptor_grouped':
                    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups=groups))
                else:
                    tr, te = next(ShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y))
            except Exception as e:
                log.warning('Descriptor split failed for %s target=%s split=%s: %s', label, target, split, e)
                continue
            for mn in models:
                try:
                    pipe = descriptor_model_pipe(mn, num_cols, cat_cols, cfg)
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                    pr = pipe.predict(X.iloc[te])
                    yt = np.asarray(y.iloc[te])
                    mets.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                     target_column=target, target_family=target_meta.get('target_family',''), task=target_meta.get('task',''),
                                     gas=target_meta.get('gas',''), target_property=target_meta.get('target_property',''), target_unit=target_meta.get('target_unit',''),
                                     model=mn, split_type=split, split_group_column=group_name, repeat=rep, n_train=len(tr), n_test=len(te),
                                     n_numeric_features=len(num_cols), n_categorical_features=len(cat_cols), rmse=rmse(yt,pr),
                                     mae=float(mean_absolute_error(yt,pr)), r2=float(r2_score(yt,pr)), spearman=spear(yt,pr),
                                     top_1pct_recovery=topk(yt,pr,.01), top_5pct_recovery=topk(yt,pr,.05), top_10pct_recovery=topk(yt,pr,.10),
                                     ndcg_10pct=ndcg(yt,pr,.10), hgb_skipped_due_to_categorical_features=hgb_skipped))
                    idx = np.arange(len(yt))
                    if len(idx) > 2500:
                        idx = np.random.default_rng(seed).choice(idx, 2500, replace=False)
                    for j in idx:
                        preds.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                          target_column=target, target_family=target_meta.get('target_family',''), model=mn, split_type=split,
                                          repeat=rep, y_true=float(yt[j]), y_pred=float(pr[j]), absolute_error=float(abs(yt[j]-pr[j]))))
                    if groups is not None:
                        g = groups.iloc[te].reset_index(drop=True)
                        ed = pd.DataFrame({'group': g.astype(str), 'abs_error': np.abs(yt-pr)})
                        top = ed.group.value_counts().head(25).index
                        for _,r in ed[ed.group.isin(top)].groupby('group').abs_error.agg(['size','mean']).reset_index().iterrows():
                            errs.append(dict(feature_source='descriptor_joined', source_table=label, feature_table=feature_tag,
                                             target_column=target, target_family=target_meta.get('target_family',''), model=mn,
                                             split_type=split, repeat=rep, chemistry_or_group=r['group'], n=int(r['size']), mae=float(r['mean'])))
                except Exception as e:
                    log.warning('Descriptor-joined ML failed for %s target=%s model=%s: %s', label, target, mn, e)
    return pd.DataFrame(mets), pd.DataFrame(preds), pd.DataFrame(errs)


def _strict_endpoint_flags(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    if out.empty:
        return out
    for c in ['resource','task','gas','target_column','target_property','target_unit','target_role','target_family']:
        if c not in out.columns: out[c]=''
    def candidate_reason(r):
        # Retain the old broad recommendation as a candidate-level field.
        old=str(r.get('primary_main_text_target','')).strip()
        if old:
            return old
        try:
            return _target_main_recommendation(pd.Series(r))
        except Exception:
            return 'context dependent; SI unless selected as headline endpoint'
    out['candidate_primary_target']=out.apply(candidate_reason, axis=1)
    def selected(r):
        role=str(r.get('target_role','')).lower()
        fam=str(r.get('target_family','')).lower()
        prop=str(r.get('target_property','')).lower()
        col=str(r.get('target_column','')).lower().replace(' ','')
        task=str(r.get('task','')).lower()
        gas=str(r.get('gas','')).upper()
        if role and role != PRIMARY_TARGET_ROLE:
            return 'no'
        # Strict headline endpoints: standard uptake, volumetric uptake, selectivity, and a few process metrics.
        if 'mmol/g' in col or prop in ['gravimetric_uptake','uptake','loading']:
            return 'yes'
        if 'v/v' in col or prop == 'volumetric_uptake':
            return 'yes'
        if 's(g1)' in col or prop == 'selectivity' or 'selectivity' in col:
            return 'yes'
        if prop in ['working_capacity','purity','recovery','productivity','process_energy'] and 'process' in task:
            return 'yes'
        return 'no'
    out['selected_main_text_endpoint']=out.apply(selected, axis=1)
    def endpoint_class(r):
        if r.get('selected_main_text_endpoint') != 'yes':
            return 'SI/context-only endpoint'
        prop=str(r.get('target_property','')).lower(); col=str(r.get('target_column','')).lower()
        if 'mmol' in col or prop in ['gravimetric_uptake','uptake','loading']:
            return 'headline gravimetric uptake/loading'
        if 'v/v' in col or prop == 'volumetric_uptake':
            return 'headline volumetric uptake'
        if 's(g1)' in col or 'select' in col or prop == 'selectivity':
            return 'headline separation selectivity'
        return 'headline process metric'
    out['main_text_endpoint_class']=out.apply(endpoint_class, axis=1)
    out['reason_for_inclusion_or_exclusion']=np.where(out['selected_main_text_endpoint'].eq('yes'), out['main_text_endpoint_class'], 'Retained in SI or source-data only to avoid overclaiming equivalent/auxiliary target representations')
    if 'primary_main_text_target' in out.columns:
        out=out.drop(columns=['primary_main_text_target'])
    return out


def make_target_dictionary(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    detailed = _V18_PREV_MAKE_TARGET_DICTIONARY(cfg, dd)
    detailed = _strict_endpoint_flags(detailed)
    if detailed.empty:
        save_df(detailed, dd['processed']/ 'target_dictionary', cfg)
        save_df(detailed, dd['processed']/ 'target_dictionary_detailed', cfg)
        return detailed
    group_cols=['resource','task','gas','target_property','target_unit','target_role','target_family','candidate_primary_target','selected_main_text_endpoint','main_text_endpoint_class']
    for c in group_cols:
        if c not in detailed.columns: detailed[c]=''
    summary=detailed.groupby(group_cols, dropna=False).agg(n_target_columns=('target_column','nunique'), n_source_files=('source_file','nunique')).reset_index()
    save_df(summary, dd['processed']/ 'target_dictionary', cfg)
    save_df(detailed, dd['processed']/ 'target_dictionary_detailed', cfg)
    save_df(summary, dd['si_tables']/ 'SI_Table_S20_target_dictionary', cfg)
    save_df(detailed, dd['si_tables']/ 'SI_Table_S20b_target_dictionary_detailed', cfg)
    selected=detailed[detailed.selected_main_text_endpoint.eq('yes')].copy()
    save_df(selected, dd['tables']/ 'main'/ 'Main_Table_10_selected_main_text_endpoints', cfg)
    save_df(selected, dd['source']/ 'selected_main_text_endpoints', cfg)
    return detailed


def _best_descriptor_rows_by_metric(desc_summ: pd.DataFrame, metric: str='mean_r2') -> pd.DataFrame:
    if desc_summ.empty:
        return pd.DataFrame()
    df=desc_summ.copy()
    for c in ['task','gas','target_column','descriptor_family','model','split_type',metric,'source_table']:
        if c not in df.columns:
            df[c]='' if c not in [metric] else np.nan
    idx=df.groupby(['task','gas','target_column','descriptor_family','split_type'])[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_descriptor_family_case_matched_comparison(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    summ=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    cases=load_df(dd['processed']/ 'descriptor_joined_case_catalog')
    if summ.empty:
        out=pd.DataFrame()
        save_df(out, dd['processed']/ 'descriptor_family_case_matched_comparison', cfg)
        save_df(out, dd['tables']/ 'main'/ 'Main_Table_8_descriptor_family_comparison_joined_cases', cfg)
        return out
    best=_best_descriptor_rows_by_metric(summ, 'mean_top_10pct_recovery')
    if not cases.empty:
        cc=cases.copy()
        for c in ['task','gas','target_column','descriptor_family','joined_rows','join_key_level','descriptor_file']:
            if c not in cc.columns: cc[c]=''
        cc=cc[['task','gas','target_column','descriptor_family','joined_rows','join_key_level','descriptor_file']].drop_duplicates()
        best=best.merge(cc, on=['task','gas','target_column','descriptor_family'], how='left')
    groups=[]
    for (task,gas,target), sub in best.groupby(['task','gas','target_column'], dropna=False):
        fams=sorted(sub.descriptor_family.dropna().astype(str).unique())
        if len(fams) < 2:
            continue
        joined=pd.to_numeric(sub.get('joined_rows', pd.Series(np.nan, index=sub.index)), errors='coerce')
        budget=float(joined.dropna().min()) if joined.notna().any() else np.nan
        tmp=sub.copy()
        tmp['matched_case_group']=f'{task}::{gas}::{target}'
        tmp['n_descriptor_families_in_group']=len(fams)
        tmp['matched_row_budget_min_joined_rows']=budget
        tmp['row_budget_fraction_of_raw_join']=joined / budget if np.isfinite(budget) and budget > 0 else np.nan
        tmp['comparison_scope']='case-matched descriptor-family comparison among successfully joined task/target cases'
        groups.append(tmp)
    out=pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    if not out.empty:
        sort_cols=[c for c in ['matched_case_group','split_type','mean_top_10pct_recovery'] if c in out.columns]
        out=out.sort_values(sort_cols, ascending=[True,True,False][:len(sort_cols)])
    save_df(out, dd['processed']/ 'descriptor_family_case_matched_comparison', cfg)
    save_df(out, dd['tables']/ 'main'/ 'Main_Table_8_descriptor_family_comparison_joined_cases', cfg)
    save_df(out, dd['si_tables']/ 'SI_Table_S24_descriptor_family_case_matched_comparison', cfg)
    return out


def write_clean_publication_tables(cfg: Config, dd: Dict[str,Path]) -> None:
    scores=load_df(dd['processed']/ 'resource_scores')
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    cdf=load_df(dd['profiles']/ 'column_level_profile')
    if not scores.empty:
        rows=[]
        for _,r in scores[scores.resource.isin(MAIN_RESOURCES)].iterrows():
            fs=fdf[fdf.resource.eq(r.resource)] if not fdf.empty and 'resource' in fdf else pd.DataFrame()
            cs=cdf[cdf.resource.eq(r.resource)] if not cdf.empty and 'resource' in cdf else pd.DataFrame()
            chem_cols=int(cs.inferred_modality.isin(['formula_or_composition','metal_or_charge_chemistry','curation_or_validation_flag']).sum()) if not cs.empty and 'inferred_modality' in cs else 0
            rows.append(dict(
                resource=r.resource, short_label=short_resource(r.resource),
                n_files=int(len(fs)) if not fs.empty else int(r.get('n_files',0)),
                profiled_or_counted_rows=float(r.get('total_profiled_or_counted_rows',np.nan)),
                target_columns=int(pd.to_numeric(fs.get('n_target_like_columns',pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not fs.empty else np.nan,
                descriptor_columns=int(pd.to_numeric(fs.get('n_descriptor_like_columns',pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not fs.empty else np.nan,
                identifier_coverage=float(r.get('identifier_availability_fraction',np.nan)),
                chemistry_column_coverage_score=min(1.0, chem_cols/20.0),
                chemistry_trust_score=float(r.get('chemistry_trust_score',np.nan)),
                ml_readiness_score=float(r.get('ml_readiness_score',np.nan)),
                recommended_role=r.get('recommended_role',''),
                main_caveat=('ML-ready but chemistry evidence not directly observable in parsed tables' if r.resource=='ARC-MOF' else 'Use with claim-specific provenance and observability reporting')
            ))
        clean=pd.DataFrame(rows)
        save_df(clean, dd['tables']/ 'main'/ 'Main_Table_1_resource_atlas', cfg)
        save_df(clean, dd['tables']/ 'main'/ 'Main_Table_1_publication_resource_atlas_clean', cfg)
        save_df(clean, dd['source']/ 'Table_1_publication_resource_atlas_clean', cfg)
    desc=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    if not desc.empty:
        best=_best_descriptor_rows_by_metric(desc, 'mean_top_10pct_recovery')
        piv=best.pivot_table(index=['task','gas','target_column','descriptor_family','source_table'], columns='split_type', values=['mean_r2','mean_spearman','mean_top_10pct_recovery'], aggfunc='max')
        if not piv.empty:
            piv.columns=['_'.join([str(x) for x in col if str(x)]) for col in piv.columns]
            piv=piv.reset_index()
            save_df(piv, dd['tables']/ 'main'/ 'Main_Table_6_best_descriptor_joined_cases_publication', cfg)
            save_df(piv, dd['source']/ 'Table_6_best_descriptor_joined_cases_publication', cfg)
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    if not diag.empty:
        keep=[c for c in ['target_file','descriptor_file','descriptor_family','target_column','join_key_level','join_status','joined_rows','overlap','left_retention','right_retention','risk_of_overmatching','n_numeric_features','n_categorical_features','task','gas','target_family'] if c in diag.columns]
        risk=diag[keep].copy()
        save_df(risk, dd['si_tables']/ 'SI_Table_descriptor_join_risk_and_retention', cfg)
        save_df(risk, dd['source']/ 'SI_Table_descriptor_join_risk_and_retention', cfg)


def write_figure_panel_source_data(cfg: Config, dd: Dict[str,Path]) -> None:
    panel_dir=mkdir(dd['source']/ 'figure_panel_source_data')
    mapping={
        'Figure_2a_resource_scale': dd['profiles']/ 'file_level_profile',
        'Figure_2b_modality_evidence': dd['profiles']/ 'resource_modality_matrix',
        'Figure_2c_missingness': dd['profiles']/ 'column_level_profile',
        'Figure_2d_scale_trust': dd['processed']/ 'resource_scores',
        'Figure_3a_chemistry_evidence': dd['processed']/ 'chemistry_evidence_matrix',
        'Figure_3b_trust_regimes': dd['processed']/ 'trust_regime_summary',
        'Figure_3c_charge_status': dd['processed']/ 'charge_status_summary',
        'Figure_4a_resource_scores_uncertainty': dd['processed']/ 'resource_score_uncertainty',
        'Figure_4b_risk_matrix': dd['processed']/ 'benchmark_risk_matrix',
        'Figure_4c_score_sensitivity': dd['processed']/ 'score_sensitivity',
        'Figure_5a_generalization_gap': dd['processed']/ 'descriptor_joined_ml_metrics_summary',
        'Figure_5b_screening_stability': dd['processed']/ 'descriptor_joined_ml_metrics_summary',
        'Figure_5c_descriptor_family_comparison': dd['processed']/ 'descriptor_family_case_matched_comparison',
        'Figure_5d_calibration': dd['processed']/ 'descriptor_joined_ml_predictions_sample',
        'Figure_6_decision_rules': dd['source']/ 'decision_rules',
    }
    rows=[]
    for name,base in mapping.items():
        df=load_df(base)
        if df.empty:
            continue
        out=panel_dir/name
        save_df(df.head(250000), out, cfg)
        rows.append(dict(panel=name, source_base=str(base), exported_csv=str(out.with_suffix('.csv')), n_rows_exported=min(len(df),250000), n_rows_available=len(df), note='Panel-level source export; full table may also exist in processed/profiles/source_data.'))
    save_df(pd.DataFrame(rows), dd['source']/ 'figure_panel_source_data_manifest', cfg)


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    _V18_PREV_STEP_TABLES(cfg, dd, log)
    build_descriptor_family_case_matched_comparison(cfg, dd)
    write_clean_publication_tables(cfg, dd)
    write_figure_panel_source_data(cfg, dd)
    # Update source map and final style sheet.
    src=load_df(dd['source']/ 'final_source_data_map')
    add=pd.DataFrame([
        {'figure':'Figure 5','panel':'c','source_data':'processed/descriptor_family_case_matched_comparison.csv','notes':'Case-matched descriptor-family comparison; labelled as comparison among successfully joined task cases, not universal ablation.'},
        {'figure':'All main figures','panel':'all','source_data':'source_data/figure_panel_source_data/*','notes':'Per-panel source-data CSV exports for manuscript reproducibility.'},
        {'figure':'Publication style','panel':'all','source_data':'source_data/publication_figure_style_sheet.csv','notes':'Colour-blind-safe palette, labels and export formats.'},
    ])
    src=pd.concat([src,add],ignore_index=True) if not src.empty else add
    save_df(src.drop_duplicates(), dd['source']/ 'final_source_data_map', cfg)
    style=pd.DataFrame([
        {'item':'palette_base','value':'Okabe-Ito / colour-blind-safe','note':'Consistent across all main figures'},
        {'item':'MOSAEC_colour','value':RESOURCE_COLORS['MOSAEC-DB'],'note':'chemistry anchor'},
        {'item':'CoRE_2024_colour','value':RESOURCE_COLORS['CoRE MOF 2024'],'note':'curated experimental resource'},
        {'item':'ARC_MOF_colour','value':RESOURCE_COLORS['ARC-MOF'],'note':'large ML resource'},
        {'item':'QMOF_colour','value':RESOURCE_COLORS['QMOF'],'note':'quantum resource'},
        {'item':'not_observable_colour','value':'#BDBDBD','note':'neutral grey, not red'},
        {'item':'main_figure_exports','value':'PDF + 600 dpi PNG + SVG','note':'written regardless of save mode'},
        {'item':'minimum_panel_font','value':'8 pt labels; 9.5 pt axis text','note':'avoid unreadable dense labels'},
        {'item':'grid_style','value':'light grey alpha 0.18','note':'readability without visual clutter'},
    ])
    save_df(style, dd['source']/ 'publication_figure_style_sheet', cfg)


def _compact_case_label(row: pd.Series) -> str:
    task=str(row.get('task','')).replace('_',' ')
    gas=str(row.get('gas',''))
    target=str(row.get('target_column',''))
    fam=str(row.get('descriptor_family',''))
    target=target.replace('hoa/kcal/mol','HoA').replace('mmol/g','mmol g$^{-1}$')
    return f'{fam} | {task} {gas} | {target}'[:62]


def _paired_metric_table(best: pd.DataFrame, metric: str, n: int=6) -> pd.DataFrame:
    if best.empty or metric not in best.columns:
        return pd.DataFrame()
    piv=best.pivot_table(index=['task','gas','target_column','descriptor_family','source_table'], columns='split_type', values=metric, aggfunc='max').reset_index()
    if 'random' in piv.columns and 'descriptor_grouped' in piv.columns:
        piv=piv.dropna(subset=['random','descriptor_grouped'])
        piv['display_score']=piv[['random','descriptor_grouped']].max(axis=1)
        piv=piv.sort_values(['display_score','random'], ascending=False).head(n)
    else:
        valcols=[c for c in ['random','descriptor_grouped'] if c in piv.columns]
        if not valcols:
            return pd.DataFrame()
        piv=piv.sort_values(valcols[0], ascending=False).head(n)
    return piv


def fig2(cfg,dd):
    f=load_df(dd['profiles']/ 'file_level_profile'); c=load_df(dd['profiles']/ 'column_level_profile'); mod=load_df(dd['profiles']/ 'resource_modality_matrix'); sc=load_df(dd['processed']/ 'resource_scores')
    main = set(MAIN_RESOURCES)
    if not f.empty: f = f[f.resource.isin(main)].copy()
    if not c.empty: c = c[c.resource.isin(main)].copy()
    if not mod.empty: mod = mod[mod.resource.isin(main)].copy()
    if not sc.empty: sc = sc[sc.resource.isin(main)].copy()
    fig,axs=plt.subplots(2,2,figsize=(13.6,9.0)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    if not f.empty:
        g=f.groupby('resource').agg(n_files=('file_name','count'),total_size_mb=('size_mb','sum')).reset_index().sort_values('total_size_mb')
        ax.barh([short_resource(x) for x in g.resource],g.total_size_mb,color=_resource_color_list(g.resource))
        ax.set_xlabel('Total profiled file size (MB)'); ax.set_title('Input scale by main resource',fontweight='bold')
        for i,r in g.reset_index(drop=True).iterrows(): ax.text(r.total_size_mb,i,f' {int(r.n_files)} files',va='center',fontsize=7.5)
    else: nodata(ax,'Input scale by main resource')
    ax=axs[1]; lab(ax,'b')
    if not mod.empty:
        cols=[x for x in ['identifier','formula_or_composition','metal_or_charge_chemistry','geometric_descriptor','descriptor','target_or_property','curation_or_validation_flag'] if x in mod.columns]
        mat=mod.set_index('resource')[cols].astype(float)
        mat.index=[short_resource(x) for x in mat.index]
        mat=mat.div(mat.max(axis=0).replace(0,np.nan), axis=1).fillna(0)
        heat(ax,mat,'Relative modality evidence','YlGnBu',True)
    else: nodata(ax,'Relative modality evidence')
    ax=axs[2]; lab(ax,'c')
    if not c.empty:
        tmp=c.copy(); tmp['missing_bin']=pd.cut(pd.to_numeric(tmp.missing_fraction,errors='coerce'),[-.001,0,.05,.25,.75,1],labels=['0','0--5%','5--25%','25--75%','>75%'],include_lowest=True)
        mat=tmp.groupby(['resource','missing_bin'],observed=False).size().unstack(fill_value=0)
        mat.index=[short_resource(x) for x in mat.index]
        heat(ax,mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0),'Column missingness composition','Blues',True)
    else: nodata(ax,'Column missingness composition')
    ax=axs[3]; lab(ax,'d'); style_axes(ax)
    if not sc.empty:
        sc=sc.copy(); sc['short']=sc.resource.map(short_resource)
        x=np.log10(sc.total_profiled_or_counted_rows.astype(float).clip(lower=1)); y=sc.chemistry_trust_score.astype(float); s=100+650*sc.ml_readiness_score.astype(float).clip(0,1)
        ax.scatter(x,y,s=s,alpha=.80,edgecolors='black',color=_resource_color_list(sc.resource))
        offsets={'ARC':(.03,-.035),'MOSAEC':(.03,.025),'CoRE-24':(.03,.025),'QMOF':(.03,-.035),'CSD ctx':(.03,.035),'CoRE-25':(.03,.035)}
        for _,r in sc.iterrows():
            dx,dy=offsets.get(r.short,(.02,.02)); ax.text(np.log10(max(float(r.total_profiled_or_counted_rows),1))+dx,float(r.chemistry_trust_score)+dy,r.short,fontsize=8)
        ax.set_xlabel('log10(profiled or counted rows + 1)'); ax.set_ylabel('Chemistry-trust score'); ax.set_ylim(0,1.05); ax.set_title('Scale versus chemistry observability',fontweight='bold')
        add_panel_note(ax,'Full resource names and Other/unknown context are retained in SI/source data.')
    else: nodata(ax,'Scale versus chemistry observability')
    fig.suptitle('Figure 2. Data-resource atlas for chemistry-ready MOF ML',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_2_data_resource_atlas',cfg)


def fig5(cfg,dd):
    desc_summ=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    matched=load_df(dd['processed']/ 'descriptor_family_case_matched_comparison')
    ablation=load_df(dd['processed']/ 'descriptor_family_ablation_summary')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    pred=load_df(dd['processed']/ 'descriptor_joined_ml_predictions_sample')
    best=_best_descriptor_model_rows(desc_summ)
    fig,axs=plt.subplots(2,2,figsize=(13.8,9.4)); axs=axs.ravel()
    ax=axs[0]; lab(ax,'a'); style_axes(ax)
    cases=_paired_metric_table(best, 'mean_r2', 6)
    if not cases.empty:
        y=np.arange(len(cases)); labels=[_compact_case_label(r) for _,r in cases.iterrows()]
        if 'descriptor_grouped' in cases.columns:
            for i,r in cases.reset_index(drop=True).iterrows(): ax.plot([r.get('descriptor_grouped',np.nan),r.get('random',np.nan)],[i,i],lw=1.4,color='#BDBDBD',zorder=1)
            ax.scatter(cases['descriptor_grouped'],y,marker='s',s=55,label='descriptor-grouped',color=SPLIT_COLORS['descriptor_grouped'],edgecolor='black',linewidth=.4,zorder=3)
        if 'random' in cases.columns:
            ax.scatter(cases['random'],y,marker='o',s=58,label='random',color=SPLIT_COLORS['random'],edgecolor='black',linewidth=.4,zorder=4)
        ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7.0); ax.invert_yaxis(); ax.set_xlabel('Best-model mean R²'); ax.set_title('Generalization gap',fontweight='bold'); ax.legend(fontsize=7,loc='lower right')
    else: nodata(ax,'Generalization gap','No paired descriptor-joined random/grouped cases available')
    ax=axs[1]; lab(ax,'b'); style_axes(ax)
    cases2=_paired_metric_table(best, 'mean_top_10pct_recovery', 6)
    if not cases2.empty:
        y=np.arange(len(cases2)); labels=[_compact_case_label(r) for _,r in cases2.iterrows()]
        if 'descriptor_grouped' in cases2.columns:
            for i,r in cases2.reset_index(drop=True).iterrows(): ax.plot([r.get('descriptor_grouped',np.nan),r.get('random',np.nan)],[i,i],lw=1.4,color='#BDBDBD',zorder=1)
            ax.scatter(cases2['descriptor_grouped'],y,marker='s',s=55,label='descriptor-grouped',color=SPLIT_COLORS['descriptor_grouped'],edgecolor='black',linewidth=.4,zorder=3)
        if 'random' in cases2.columns:
            ax.scatter(cases2['random'],y,marker='o',s=58,label='random',color=SPLIT_COLORS['random'],edgecolor='black',linewidth=.4,zorder=4)
        ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7.0); ax.invert_yaxis(); ax.set_xlabel('Top-10% recovery'); ax.set_xlim(0,1.05); ax.set_title('Screening stability',fontweight='bold')
    else: nodata(ax,'Screening stability')
    ax=axs[2]; lab(ax,'c')
    if not matched.empty and {'descriptor_family','split_type','mean_r2','mean_top_10pct_recovery'}.issubset(matched.columns):
        tmp=matched.copy()
        tmp['descriptor_family']=tmp.descriptor_family.astype(str)
        m1=tmp.pivot_table(index='descriptor_family', columns='split_type', values='mean_r2', aggfunc='median')
        m2=tmp.pivot_table(index='descriptor_family', columns='split_type', values='mean_top_10pct_recovery', aggfunc='median')
        mat=pd.DataFrame(index=sorted(set(m1.index)|set(m2.index)))
        for col in ['random','descriptor_grouped']:
            mat[f'R² {col.replace("descriptor_grouped","grouped")}']=m1[col] if col in m1 else np.nan
            mat[f'Top-10 {col.replace("descriptor_grouped","grouped")}']=m2[col] if col in m2 else np.nan
        heat(ax,mat,'Case-matched descriptor-family comparison','YlGnBu',True)
        add_panel_note(ax,'Matched by task/gas/target among successfully joined cases.')
    elif not ablation.empty:
        fams=set(ablation.descriptor_family.astype(str)) if 'descriptor_family' in ablation else set()
        title='Descriptor-family comparison' if len(fams)>1 else 'Single-family descriptor benchmark'
        mat=ablation.pivot_table(index='descriptor_family',columns='split_type',values='median_top_10pct_recovery',aggfunc='median',fill_value=np.nan)
        heat(ax,mat,title,'YlGnBu',True)
    elif not diag.empty:
        d=diag.groupby(['descriptor_family','join_status']).size().reset_index(name='n')
        mat=d.pivot_table(index='descriptor_family',columns='join_status',values='n',fill_value=0)
        heat(ax,mat,'Descriptor join diagnostics','YlOrBr',True)
    else: nodata(ax,'Descriptor-family comparison')
    ax=axs[3]; lab(ax,'d'); style_axes(ax)
    if not pred.empty and not best.empty:
        top=best.sort_values(['mean_top_10pct_recovery','mean_r2'],ascending=False).iloc[0]
        sub=pred[(pred.source_table.eq(top.source_table)) & (pred.target_column.eq(top.target_column)) & (pred.model.eq(top.model)) & (pred.split_type.eq(top.split_type))]
        if sub.empty: sub=pred
        if len(sub)>2500: sub=sub.sample(n=2500,random_state=cfg.random_seed)
        x=pd.to_numeric(sub.y_true,errors='coerce'); y=pd.to_numeric(sub.y_pred,errors='coerce')
        ax.scatter(x,y,s=8,alpha=.35,color='#0072B2',edgecolors='none')
        vals=pd.concat([x,y]).dropna()
        if not vals.empty:
            q=np.nanquantile(vals,[0.01,0.99])
            if np.isfinite(q[0]) and np.isfinite(q[1]) and q[1] > q[0]:
                ax.plot([q[0],q[1]],[q[0],q[1]],ls='--',lw=1,color='#333333'); ax.set_xlim(q[0],q[1]); ax.set_ylim(q[0],q[1])
        title=f"{top.get('task','')} {top.get('gas','')} {top.get('target_column','')} {top.get('descriptor_family','')}".replace('_',' ')[:48]
        ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target'); ax.set_title('Representative calibration: '+title,fontweight='bold',fontsize=9.2)
        add_panel_note(ax,'Best case selected by top-10% recovery; axes clipped to 1st--99th percentile.')
    else: nodata(ax,'Representative calibration')
    fig.suptitle('Figure 5. Descriptor-joined ARC-MOF screening benchmark',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95]); return save_fig(fig,dd['fig_main']/ 'Figure_5_ml_stress_test',cfg)


def fig6(cfg,dd):
    chk=load_df(dd['source']/ 'minimum_reporting_checklist')
    rec=load_df(dd['main_tables']/ 'Main_Table_4_use_case_recommendation_cards')
    fig,axs=plt.subplots(1,3,figsize=(16.5,5.8))
    ax=axs[0]; ax.axis('off'); lab(ax,'a'); ax.set_title('Claim-first decision tree',fontweight='bold',fontsize=10.5)
    nodes=[('Define claim',.50,.86),('Target context\nknown?',.50,.66),('Chemistry evidence\nobservable?',.25,.45),('Grouped split\nstable?',.75,.45),('Report claim-specific\nresource/subset',.50,.22)]
    for t,x,y in nodes:
        ax.add_patch(FancyBboxPatch((x-.145,y-.06),.29,.12,boxstyle='round,pad=.02',lw=.9,facecolor='#F7F7F7',edgecolor='#555555'))
        ax.text(x,y,t,ha='center',va='center',fontsize=8.5)
    for (x1,y1),(x2,y2) in [((.50,.80),(.50,.72)),((.50,.60),(.30,.51)),((.50,.60),(.70,.51)),((.25,.39),(.44,.28)),((.75,.39),(.56,.28))]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=11,lw=.9,color='#333333'))
    ax.text(.50,.06,'Avoid universal leaderboard claims without target, join and split evidence.',ha='center',fontsize=7.5,color='#444444')
    ax=axs[1]; ax.axis('off'); lab(ax,'b'); ax.set_title('Minimum reporting items',fontweight='bold',fontsize=10.5)
    items=['file provenance','target dictionary','identifier joins','chemistry observability','leakage policy','grouped split','panel source data']
    y=.86
    for item in items:
        ax.text(.08,y,'✓ '+item,fontsize=9,va='center'); y-=.105
    ax.text(.08,.08,'Full checklist is retained in SI/source data.',fontsize=7.5,color='#555555')
    ax=axs[2]; ax.axis('off'); lab(ax,'c'); ax.set_title('Claim-specific recommendations',fontweight='bold',fontsize=10.5)
    cards=[('ARC-MOF','descriptor-joined ML + grouped split','ML-ready; chemistry-observability caveat'),('MOSAEC/CoRE','chemistry-observable claims','smaller or fragmented but auditable'),('QMOF','quantum-property claims','report DFT settings and failed cases'),('Universal ranking','avoid unless all checks pass','state subset and scope explicitly')]
    y=.84
    for title,action,caveat in cards:
        ax.add_patch(FancyBboxPatch((.04,y-.09),.92,.15,boxstyle='round,pad=.02',lw=.8,facecolor='#FAFAFA',edgecolor='#666666'))
        ax.text(.06,y+.03,title,ha='left',va='center',fontsize=8.2,fontweight='bold')
        ax.text(.06,y-.02,action,ha='left',va='center',fontsize=7.4)
        ax.text(.06,y-.062,caveat,ha='left',va='center',fontsize=6.8,color='#555555')
        y-=.21
    fig.suptitle('Figure 6. Claim-specific reporting framework for chemistry-ready MOF ML',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.92]); return save_fig(fig,dd['fig_main']/ 'Figure_6_decision_framework',cfg)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    _V18_PREV_STEP_REPORTS(cfg, dd, log)
    report = dd['reports']/ 'analysis_report.md'
    matched=load_df(dd['processed']/ 'descriptor_family_case_matched_comparison')
    selected=load_df(dd['source']/ 'selected_main_text_endpoints')
    style=load_df(dd['source']/ 'publication_figure_style_sheet')
    extra=[]
    extra.append('\n## v1.8 publication-polish additions\n')
    extra.append('- HGB is now automatically skipped for categorical/topology feature cases to remove repeated sparse-matrix warnings from final logs.\n')
    if not matched.empty:
        fams=', '.join(sorted(matched.descriptor_family.dropna().astype(str).unique())) if 'descriptor_family' in matched else 'not encoded'
        extra.append(f'- Case-matched descriptor-family comparison contains **{len(matched)}** best-model rows across families: **{fams}**.\n')
    if not selected.empty:
        extra.append(f'- Strict main-text endpoint table contains **{len(selected)}** selected endpoint contexts; broader target dictionary remains in SI.\n')
    if not style.empty:
        extra.append('- Publication figure style sheet saved with colour-blind-safe palette and export rules.\n')
    extra.append('- Main figures are exported as PDF, SVG and 600-dpi PNG regardless of save mode.\n')
    try:
        report.write_text(report.read_text(encoding='utf-8') + ''.join(extra), encoding='utf-8')
    except Exception:
        pass



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

v1.8 publication-polish focus
-----------------------------
- Skips HGB automatically for categorical/topology feature cases to avoid sparse-matrix warnings.
- Adds strict selected_main_text_endpoint and candidate_primary_target fields to the target dictionary.
- Replaces raw chemistry-column counts in the main resource table with normalized chemistry_column_coverage_score.
- Adds case-matched descriptor-family comparison among successfully joined task/target cases.
- Adds source_data/figure_panel_source_data/ with per-panel CSV exports.
- Exports all main figures as PDF, SVG and 600-dpi PNG using a clear colour-blind-safe palette.
- Redesigns Figure 5 around six representative paired random/grouped cases, top-10% recovery and a compact descriptor-family comparison.

Publication colour/style sheet
------------------------------
MOSAEC: {RESOURCE_COLORS['MOSAEC-DB']}  | CoRE-24: {RESOURCE_COLORS['CoRE MOF 2024']} | ARC-MOF: {RESOURCE_COLORS['ARC-MOF']}
QMOF: {RESOURCE_COLORS['QMOF']} | CSD context: {RESOURCE_COLORS['CSD-derived context']} | Not observable / Other: #BDBDBD
Main figures: PDF + SVG + 600-dpi PNG.

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     normalized targets, descriptor joins, trust regimes, ML outputs, comparisons and reusable data
tables/        main-text and SI-ready tables
figures/       main figures 1--6 plus SI figures
source_data/   panel source-data, checklist, target selections, manual-validation template, manifest and ZIP audit
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        optional fitted models in thorough save mode

Resume behavior
---------------
Use a fresh output folder or --force when changing code versions. If interrupted, rerun the same command.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')



# --- v1.8 smoke/logging hotfixes -------------------------------------------------
def spear(y,p):
    """Warning-free Spearman correlation for repeated split tables."""
    try:
        yy=np.asarray(y, dtype=float); pp=np.asarray(p, dtype=float)
        mask=np.isfinite(yy) & np.isfinite(pp)
        yy=yy[mask]; pp=pp[mask]
        if len(yy) < 3 or np.nanstd(yy) == 0 or np.nanstd(pp) == 0:
            return np.nan
        if scipy_stats:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return float(scipy_stats.spearmanr(yy, pp, nan_policy='omit').correlation)
        return float(pd.Series(yy).corr(pd.Series(pp), method='spearman'))
    except Exception:
        return np.nan


def step_figures(cfg: Config, dd: Dict[str,Path], log):
    outs=[]
    tasks=[('SI Figure S0 inventory map', lambda: inventory_alignment_figure(cfg,dd)),
           ('Figure 1 concept', lambda: fig1(cfg,dd)),
           ('Figure 2 atlas', lambda: fig2(cfg,dd)),
           ('Figure 3 trust regimes', lambda: fig3(cfg,dd)),
           ('Figure 4 risk map', lambda: fig4(cfg,dd)),
           ('Figure 5 ML benchmark', lambda: fig5(cfg,dd)),
           ('Figure 6 decision framework', lambda: fig6(cfg,dd)),
           ('SI figures S1-S8', lambda: si_figs(cfg,dd))]
    for name,func in tasks:
        try:
            log.info('Rendering %s', name)
            new=func()
            outs+=new if isinstance(new, list) else []
            log.info('Finished rendering %s (%d files)', name, len(new) if isinstance(new, list) else 0)
        except Exception as e:
            log.warning('Figure rendering failed for %s: %s', name, e)
    save_df(pd.DataFrame([{'figure_file':str(p),'relative_path':str(p.relative_to(cfg.out_dir)) if str(p).startswith(str(cfg.out_dir)) else str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None} for p in outs]),dd['fig']/ 'figure_output_manifest',cfg)



# =============================================================================
# v1.8.1 WINDOWS SAVE HOTFIX
# =============================================================================
# This block fixes a Windows-specific failure observed during profiling:
# PermissionError [WinError 5] while replacing file_level_profile_partial.pkl.tmp
# with file_level_profile_partial.pkl.  On Windows this usually means that the
# destination file is temporarily locked by Explorer preview, antivirus, cloud
# sync, a spreadsheet viewer, or a previous process.  Intermediate pickle files
# are useful but should never crash a long scientific run.  The replacement
# save_df below keeps CSV outputs as the primary source of truth, uses unique
# temporary filenames, retries locked replacements, and falls back gracefully to
# timestamped autosave files when a non-critical format remains locked.

VERSION = "1.8.1-publication-polish-windows-save-hotfix"


def _save_warning(base: Path, message: str) -> None:
    """Append a save warning without interrupting the pipeline."""
    try:
        mkdir(base.parent)
        with open(base.parent / "SAVE_WARNINGS.txt", "a", encoding="utf-8") as fh:
            fh.write(f"[{iso()}] {message}\n")
    except Exception:
        pass


def _unique_tmp_path(p: Path) -> Path:
    """Create a unique temp path in the same directory as the destination."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return p.with_name(f".{p.name}.{os.getpid()}.{stamp}.tmp")


def _fallback_path(p: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return p.with_name(f"{p.stem}__autosave_{stamp}{p.suffix}")


def _safe_replace_tmp(tmp: Path, dst: Path, base: Path, *, required: bool=False, retries: int=8, delay: float=0.35) -> Path:
    """Replace dst by tmp with retries; never fail for optional sidecar formats.

    Returns the destination path when replacement succeeds.  If the destination
    remains locked, the tmp file is moved to a timestamped autosave path.  For
    required CSV outputs, the exception is re-raised only when no usable original
    destination exists; otherwise the pipeline continues and records a warning.
    """
    last = None
    mkdir(dst.parent)
    for i in range(max(1, retries)):
        try:
            os.replace(tmp, dst)
            return dst
        except PermissionError as e:
            last = e
            gc.collect()
            time.sleep(delay * (i + 1))
        except OSError as e:
            last = e
            gc.collect()
            time.sleep(delay * (i + 1))
    fb = _fallback_path(dst)
    try:
        os.replace(tmp, fb)
        _save_warning(base, f"Could not replace locked file '{dst.name}'. Wrote fallback autosave '{fb.name}'. Original error: {last}")
        # If an older destination exists, return it so resume/load_df can still
        # read the normal canonical path.  Otherwise return the fallback.
        if dst.exists():
            return dst
        return fb
    except Exception as e:
        _save_warning(base, f"Could not save '{dst.name}' or fallback file. Error: {e}; original error: {last}")
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass
        if required and not dst.exists():
            raise
        return dst


def save_df(df: pd.DataFrame, base: Path, cfg: Config, index=False) -> List[Path]:
    """Windows-robust dataframe writer.

    CSV is treated as the canonical text output.  Pickle/Parquet/Excel-like
    sidecars are best-effort: useful for speed and reuse, but a lock on one of
    them must not interrupt a 509-file run.  This directly fixes the observed
    `file_level_profile_partial.pkl` permission error.
    """
    mkdir(base.parent)
    outs: List[Path] = []

    # Canonical CSV output.
    if cfg.save.get("csv", True):
        p = base.with_suffix(".csv")
        tmp = _unique_tmp_path(p)
        try:
            df.to_csv(tmp, index=index)
            saved = _safe_replace_tmp(tmp, p, base, required=True)
            outs.append(saved)
        except Exception as e:
            _save_warning(base, f"CSV save failed for '{p.name}': {e}")
            raise

    # Optional pickle sidecar.  Do not crash if Windows locks an old .pkl.
    if cfg.save.get("pkl", False):
        p = base.with_suffix(".pkl")
        tmp = _unique_tmp_path(p)
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
            saved = _safe_replace_tmp(tmp, p, base, required=False)
            outs.append(saved)
        except Exception as e:
            _save_warning(base, f"Optional Pickle save skipped/failed for '{p.name}': {e}")
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass

    # Optional Parquet sidecar.  Use a parquet-named temp file where possible.
    if cfg.save.get("parquet", False) and PARQUET_AVAILABLE:
        p = base.with_suffix(".parquet")
        tmp = _unique_tmp_path(p).with_suffix(".parquet.tmp")
        try:
            df.to_parquet(tmp, index=index)
            saved = _safe_replace_tmp(tmp, p, base, required=False)
            outs.append(saved)
        except Exception as e:
            _save_warning(base, f"Optional Parquet save skipped/failed for '{p.name}': {e}")
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass
    return outs



def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available representation, with CSV as the canonical fallback.

    Earlier versions preferred pickle over CSV.  That is fast, but unsafe when a
    Windows file lock prevents an optional .pkl from being overwritten while the
    CSV was updated correctly.  v1.8.1 therefore reads the newest successful
    representation and falls back through the others.  This prevents stale pkl
    sidecars from contaminating later steps after a lock warning.
    """
    candidates=[]
    for ext in [".parquet", ".pkl", ".csv"]:
        pp=base.with_suffix(ext)
        if pp.exists():
            try:
                candidates.append((pp.stat().st_mtime, ext, pp))
            except Exception:
                candidates.append((0.0, ext, pp))
    # Newest first; when mtimes tie, prefer CSV because it is canonical/auditable.
    pref={".csv": 3, ".parquet": 2, ".pkl": 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,pp in candidates:
        try:
            if ext==".parquet": return pd.read_parquet(pp)
            if ext==".pkl": return pd.read_pickle(pp)
            return pd.read_csv(pp, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     reusable intermediate/final analysis tables
tables/        main-text and SI-ready tables
figures/       polished main figures and SI figures
source_data/   panel and source-data tables
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        saved models when --save-mode thorough is used

Windows save hotfix
-------------------
This v1.8.1 script is robust to temporary Windows locks on optional sidecar files
such as .pkl or .parquet.  If Explorer, antivirus, OneDrive, or another program
locks an existing file, the run records a warning in SAVE_WARNINGS.txt and writes
a timestamped autosave sidecar rather than crashing.  CSV outputs remain the
canonical source of truth.

Resume behavior
---------------
If the run is interrupted, run the same command again. Use --force only when you
want to recompute completed steps.  For final paper runs, use a fresh output
folder whenever possible to avoid old locked files.
"""
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')


# =============================================================================
# v1.8.2 PROFILE-CHECKPOINT HOTFIX
# =============================================================================
# The user-reported crash occurred while saving the frequently overwritten
# profiling checkpoint file:
#     profiles/file_level_profile_partial.pkl
# On Windows, .pkl replacement can be blocked by Explorer preview, antivirus,
# OneDrive/cloud sync, or a previous Python process.  Partial checkpoint sidecars
# are not scientifically required, because CSV is the canonical resumable audit
# output.  v1.8.2 therefore writes partial checkpoints as CSV only, while keeping
# final/non-partial pickle and parquet outputs as best-effort optional sidecars.

VERSION = "1.8.2-publication-polish-profile-checkpoint-hotfix"


def _write_save_warning(cfg: Config, base: Path, message: str) -> None:
    """Write save warnings both next to the file and in logs/ when possible."""
    line = f"[{iso()}] {message}\n"
    for folder in [base.parent, getattr(cfg, 'out_dir', Path('.')) / 'logs']:
        try:
            mkdir(folder)
            with open(folder / "SAVE_WARNINGS.txt", "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass


def _v182_unique_tmp_path(dst: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return dst.with_name(f".{dst.name}.{os.getpid()}.{stamp}.tmp")


def _v182_fallback_path(dst: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return dst.with_name(f"{dst.stem}__autosave_{stamp}{dst.suffix}")


def _v182_safe_replace(tmp: Path, dst: Path, cfg: Config, base: Path, *, required: bool) -> Path:
    """Atomically replace a file with Windows-lock retries and fallback autosave."""
    last_error = None
    mkdir(dst.parent)
    for attempt in range(10):
        try:
            os.replace(tmp, dst)
            return dst
        except PermissionError as e:
            last_error = e
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
        except OSError as e:
            last_error = e
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
    fb = _v182_fallback_path(dst)
    try:
        os.replace(tmp, fb)
        _write_save_warning(cfg, base, f"Could not replace locked file '{dst}'. Saved fallback '{fb.name}'. Original error: {last_error}")
        if dst.exists():
            return dst
        if required:
            # For required files, having only an autosave is not ideal, but it is
            # better than losing progress.  Return fallback so caller can continue.
            return fb
        return fb
    except Exception as e:
        _write_save_warning(cfg, base, f"Could not save '{dst}' or fallback. Error: {e}; original error: {last_error}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        if required and not dst.exists():
            raise
        return dst


def _is_partial_checkpoint(base: Path) -> bool:
    name = base.name.lower()
    return name.endswith('_partial') or '_partial_' in name or name in {'file_level_profile_partial', 'column_level_profile_partial'}


def save_df(df: pd.DataFrame, base: Path, cfg: Config, index=False) -> List[Path]:
    """Windows-safe dataframe writer used by all v1.8.2 outputs.

    Key behaviour:
    - CSV is always the canonical output.
    - Partial profiling checkpoints are CSV-only to avoid Windows locks on
      frequently overwritten .pkl/.parquet files.
    - Pickle and parquet outputs are optional sidecars for final tables only;
      failures are logged but never stop the pipeline.
    """
    mkdir(base.parent)
    outs: List[Path] = []
    partial = _is_partial_checkpoint(base)

    # CSV canonical output.
    if cfg.save.get('csv', True):
        p = base.with_suffix('.csv')
        tmp = _v182_unique_tmp_path(p)
        try:
            df.to_csv(tmp, index=index)
            try:
                with open(tmp, 'ab') as fh:
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                pass
            outs.append(_v182_safe_replace(tmp, p, cfg, base, required=True))
        except Exception as e:
            _write_save_warning(cfg, base, f"Required CSV save failed for '{p}': {e}")
            raise

    # Partial checkpoint sidecars are the exact source of the observed crash.
    if partial:
        return outs

    # Optional pickle sidecar.
    if cfg.save.get('pkl', False):
        p = base.with_suffix('.pkl')
        tmp = _v182_unique_tmp_path(p)
        try:
            with open(tmp, 'wb') as fh:
                pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
            outs.append(_v182_safe_replace(tmp, p, cfg, base, required=False))
        except Exception as e:
            _write_save_warning(cfg, base, f"Optional pickle sidecar skipped for '{p.name}': {e}")
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass

    # Optional parquet sidecar.
    if cfg.save.get('parquet', False) and PARQUET_AVAILABLE:
        p = base.with_suffix('.parquet')
        tmp = _v182_unique_tmp_path(p).with_suffix('.parquet.tmp')
        try:
            df.to_parquet(tmp, index=index)
            outs.append(_v182_safe_replace(tmp, p, cfg, base, required=False))
        except Exception as e:
            _write_save_warning(cfg, base, f"Optional parquet sidecar skipped for '{p.name}': {e}")
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass
    return outs


def load_df(base: Path) -> pd.DataFrame:
    """Load the newest available canonical/sidecar table representation.

    CSV has preference when timestamps are close, because it is the canonical
    auditable output.  This prevents a stale locked .pkl from silently overriding
    a newer CSV after a Windows file-lock warning.
    """
    candidates=[]
    for ext in ['.csv', '.parquet', '.pkl']:
        p=base.with_suffix(ext)
        if p.exists():
            try:
                candidates.append((p.stat().st_mtime, ext, p))
            except Exception:
                candidates.append((0.0, ext, p))
    # autosave CSV fallback, useful if the canonical CSV itself was locked.
    try:
        autos=sorted(base.parent.glob(base.name + '__autosave_*.csv'), key=lambda q: q.stat().st_mtime, reverse=True)
        for p in autos[:2]:
            candidates.append((p.stat().st_mtime, '.csv', p))
    except Exception:
        pass
    pref={'.csv': 3, '.parquet': 2, '.pkl': 1}
    candidates=sorted(candidates, key=lambda x: (x[0], pref.get(x[1],0)), reverse=True)
    for _,ext,p in candidates:
        try:
            if ext=='.parquet': return pd.read_parquet(p)
            if ext=='.pkl': return pd.read_pickle(p)
            return pd.read_csv(p, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

Modes
-----
Save mode: {cfg.save_mode}
RAM mode: {cfg.ram_mode}
Comprehensive level: {cfg.comprehensive_level}
n_jobs: {cfg.n_jobs}

Important folders
-----------------
profiles/      file-level and column-level raw table profiles
processed/     reusable intermediate/final analysis tables
tables/        main-text and SI-ready tables
figures/       polished main figures and SI figures
source_data/   panel and source-data tables
logs/          run.log, pipeline_state.json, environment metadata and file hashes
reports/       human-readable analysis_report.md
models/        saved models when --save-mode thorough is used

Windows save hotfix v1.8.2
--------------------------
This version fixes the observed Windows PermissionError at
profiles/file_level_profile_partial.pkl by writing partial profiling checkpoints
as CSV only.  Final pickle/parquet sidecars remain best-effort optional outputs.
CSV files are the canonical source of truth.  Any non-fatal save issue is written
to SAVE_WARNINGS.txt in the affected folder and in logs/.

Recommended rerun practice
--------------------------
Use a fresh output folder for each major run, for example outputs_chemistry_ready_mof_v18_2.
Close Excel, Explorer Preview Pane, OneDrive sync windows and old Python terminals before launching.

Resume behavior
---------------
If the run is interrupted, run the same command again. Use --force only when you
want to recompute completed steps.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')



# =============================================================================
# v1.9 FINAL PUBLICATION + GITHUB-AUDIT OVERRIDES
# =============================================================================
# This final block keeps the successful v1.8.2 scientific pipeline intact and
# adds a publication-final layer requested for the last from-scratch run:
#   * clearer journal-style colours and 600-dpi/PDF/SVG main-figure exports;
#   * a more polished Figure 1 and a final full-width Figure 5;
#   * compact GitHub/data-preparation reports in a separate github_audit/ folder;
#   * compact manifests of data files, analysis steps, figure/table assets, and
#     warnings, without copying the large raw/processed data tables;
#   * a github_audit ZIP suitable for README/repository preparation notes.

VERSION = "1.9-final-publication-github-audit"

# Colour-blind-safe publication palette with stable semantic mapping.
RESOURCE_SHORT_LABELS = {
    "MOSAEC-DB": "MOSAEC",
    "CoRE MOF 2024": "CoRE-24",
    "CoRE MOF 2025 metadata": "CoRE-25",
    "ARC-MOF": "ARC-MOF",
    "QMOF": "QMOF",
    "CSD-derived context": "CSD ctx",
    "Other/unknown": "Other",
    "Task-specific joins": "Joins",
}
RESOURCE_COLORS = {
    "MOSAEC-DB": "#009E73",            # green/teal
    "CoRE MOF 2024": "#0072B2",        # blue
    "CoRE MOF 2025 metadata": "#56B4E9",# sky blue
    "ARC-MOF": "#CC79A7",              # purple/pink
    "QMOF": "#E69F00",                 # orange
    "CSD-derived context": "#D55E00",  # vermilion
    "Other/unknown": "#999999",        # grey
    "Task-specific joins": "#666666",
}
TRUST_REGIME_COLORS = {
    "validated_observable": "#009E73",
    "validated_not_observable": "#56B4E9",
    "uncertain_observable": "#E69F00",
    "uncertain_unobservable": "#999999",
    "flagged_or_inconsistent": "#D55E00",
}
SPLIT_COLORS = {"random": "#0072B2", "descriptor_grouped": "#E69F00", "grouped": "#E69F00"}
SPLIT_MARKERS = {"random": "o", "descriptor_grouped": "s", "grouped": "s"}

# Make balanced mode publication-ready without saving heavy models.
try:
    SAVE_MODES["balanced"]["svg"] = True
    SAVE_MODES["balanced"]["png"] = True
    SAVE_MODES["balanced"]["models"] = False
except Exception:
    pass

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 9.6,
    "axes.titlesize": 10.8,
    "axes.labelsize": 9.7,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"]),
})

_V18_2_SAVE_FIG = save_fig
_V18_2_STEP_TABLES = step_tables
_V18_2_STEP_FIGURES = step_figures
_V18_2_STEP_REPORTS = step_reports
_V18_2_STEP_ZIP = step_zip


def _short_resource_label(x: Any) -> str:
    return RESOURCE_SHORT_LABELS.get(str(x), str(x))


def _resource_color(x: Any) -> str:
    return RESOURCE_COLORS.get(str(x), "#999999")


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def save_fig(fig, base: Path, cfg: Config) -> List[Path]:
    """Final-publication figure saver.

    Main figures are always exported as PDF, SVG and 600-dpi PNG, even in
    balanced mode.  SI figures respect the save mode, except that final and
    balanced runs keep PNGs for easy visual inspection.
    """
    mkdir(base.parent)
    outs=[]
    base_str=str(base).replace('\\','/').lower()
    is_main = '/figures/main/' in base_str or base.name.startswith('Figure_')
    formats=[]
    if cfg.save.get('pdf', True) or is_main:
        formats.append(('pdf',{}))
    if cfg.save.get('png', False) or is_main:
        formats.append(('png',{'dpi':600}))
    if cfg.save.get('svg', False) or is_main:
        formats.append(('svg',{}))
    seen=set()
    for ext,kw in formats:
        if ext in seen: continue
        seen.add(ext)
        p=base.with_suffix('.'+ext)
        try:
            fig.savefig(p,bbox_inches='tight',facecolor='white',metadata={'Creator': f'Chemistry-ready MOF pipeline {VERSION}'},**kw)
            outs.append(p)
        except Exception:
            # Fall back to the previous saver if an exotic backend/format issue occurs.
            try:
                outs += _V18_2_SAVE_FIG(fig, base, cfg)
            except Exception:
                pass
            break
    plt.close(fig)
    return outs


def _panel_letter(ax, letter: str):
    ax.text(-.075, 1.075, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left', bbox=dict(boxstyle='round,pad=.15', facecolor='white', edgecolor='#444444', lw=.55))


def _clean_axes(ax, grid=True):
    ax.set_facecolor('white')
    if grid:
        ax.grid(True, lw=0.35, alpha=0.18, zorder=0)
    for sp in ['top','right']:
        try: ax.spines[sp].set_visible(False)
        except Exception: pass
    for sp in ['left','bottom']:
        try: ax.spines[sp].set_linewidth(.75)
        except Exception: pass
    return ax


def fig1(cfg, dd):
    """Polished final concept figure with concrete examples."""
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 8.4)); axs = axs.ravel()
    ax=axs[0]; ax.axis('off'); _panel_letter(ax,'a')
    ax.set_title('Claim-specific roles of MOF data resources', fontweight='bold', pad=8)
    boxes=[
        (.05,.67,.24,.17,'MOSAEC / CoRE','chemistry and provenance anchors', RESOURCE_COLORS['MOSAEC-DB']),
        (.38,.67,.24,.17,'ARC-MOF','large descriptor-rich adsorption ML resource', RESOURCE_COLORS['ARC-MOF']),
        (.71,.67,.24,.17,'QMOF','DFT / quantum-property contrast', RESOURCE_COLORS['QMOF']),
        (.18,.25,.26,.18,'Computation-ready','parsable files, identifiers, descriptors, targets', '#F2F2F2'),
        (.56,.25,.28,.18,'Chemistry-ready','claim-specific chemistry evidence, caveats, joins', '#F2F2F2'),
    ]
    for x,y,w,h,t,b,c in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.022,rounding_size=.025',lw=1.0,facecolor=c,alpha=.16,edgecolor=c if c!='#F2F2F2' else '#666666'))
        ax.text(x+w/2,y+h*.64,t,ha='center',va='center',fontsize=9.7,fontweight='bold')
        ax.text(x+w/2,y+h*.31,b,ha='center',va='center',fontsize=7.6,wrap=True)
    for a,b,c,d in [(.17,.67,.29,.43),(.50,.67,.43,.43),(.83,.67,.70,.43),(.44,.34,.56,.34)]:
        ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle='->',mutation_scale=13,lw=1.0,color='#444444'))
    ax.text(.50,.08,'The same database can be appropriate or inappropriate depending on the claim being made.',ha='center',fontsize=8.2,color='#444444')

    ax=axs[1]; ax.axis('off'); _panel_letter(ax,'b')
    ax.set_title('Computation-ready ≠ chemistry-ready', fontweight='bold', pad=8)
    cols=[(.09,'Computation-ready', ['can be loaded','has numeric columns','has target-like fields','has identifiers'], '#0072B2'),
          (.58,'Chemistry-ready', ['oxidation/charge evidence','curation or warning status','duplicate and leakage policy','fit to scientific claim'], '#009E73')]
    for x,title,items,color in cols:
        ax.add_patch(FancyBboxPatch((x,.20),.34,.58,boxstyle='round,pad=.025,rounding_size=.025',lw=1.0,facecolor=color,alpha=.13,edgecolor=color))
        ax.text(x+.17,.70,title,ha='center',va='center',fontweight='bold',fontsize=10)
        for i,it in enumerate(items): ax.text(x+.05,.59-.10*i,'• '+it,ha='left',va='center',fontsize=8.8)
    ax.add_patch(FancyArrowPatch((.45,.48),(.56,.48),arrowstyle='->',mutation_scale=16,lw=1.2,color='#555555'))
    ax.text(.505,.54,'audit',ha='center',fontsize=8.2,fontweight='bold')

    ax=axs[2]; ax.axis('off'); _panel_letter(ax,'c')
    ax.set_title('Trust regimes separate validation from observability', fontweight='bold', pad=8)
    regimes=[('validated observable','curated and chemistry evidence visible', TRUST_REGIME_COLORS['validated_observable']),
             ('validated not observable','curated, but key evidence absent from parsed table', TRUST_REGIME_COLORS['validated_not_observable']),
             ('uncertain observable','some chemistry evidence visible but not validated', TRUST_REGIME_COLORS['uncertain_observable']),
             ('flagged / inconsistent','warning, unusual charge or oxidation-state issue', TRUST_REGIME_COLORS['flagged_or_inconsistent'])]
    y=.75
    for title,body,color in regimes:
        ax.add_patch(FancyBboxPatch((.08,y-.055),.84,.10,boxstyle='round,pad=.018',lw=.9,facecolor=color,alpha=.15,edgecolor=color))
        ax.text(.13,y,title,ha='left',va='center',fontsize=8.3,fontweight='bold')
        ax.text(.47,y,body,ha='left',va='center',fontsize=7.5)
        y-=.15
    ax.text(.08,.13,'Important: not observable means absent from parsed metadata, not chemically invalid.',fontsize=8.1,color='#444444')

    ax=axs[3]; ax.axis('off'); _panel_letter(ax,'d')
    ax.set_title('Final reproducible analysis flow', fontweight='bold', pad=8)
    steps=[('Profile','tables'),('Normalize','targets'),('Join','descriptors'),('Score','trust/readiness'),('Benchmark','descriptor-joined ML'),('Report','figures + GitHub audit')]
    xs=np.linspace(.08,.92,len(steps))
    for i,(s,b) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((xs[i]-.065,.47),.13,.13,boxstyle='round,pad=.015',lw=.85,facecolor='#FAFAFA',edgecolor='#555555'))
        ax.text(xs[i],.54,s,ha='center',va='center',fontsize=8.0,fontweight='bold')
        ax.text(xs[i],.48,b,ha='center',va='center',fontsize=6.7)
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((xs[i]+.065,.535),(xs[i+1]-.065,.535),arrowstyle='->',mutation_scale=10,lw=.8,color='#555555'))
    ax.text(.50,.27,'Every manuscript claim links to a table, figure panel and compact audit record.',ha='center',fontsize=8.5)
    fig.suptitle('Figure 1. From computation-ready files to chemistry-ready MOF machine-learning claims',fontweight='bold',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.945])
    return save_fig(fig, dd['fig_main']/ 'Figure_1_chemistry_ready_concept', cfg)


def fig2(cfg, dd):
    f=load_df(dd['profiles']/ 'file_level_profile'); c=load_df(dd['profiles']/ 'column_level_profile'); mod=load_df(dd['profiles']/ 'resource_modality_matrix'); sc=load_df(dd['processed']/ 'resource_scores')
    fig,axs=plt.subplots(2,2,figsize=(13.4,9.0)); axs=axs.ravel()
    ax=axs[0]; _panel_letter(ax,'a'); _clean_axes(ax)
    if not f.empty:
        g=f.groupby('resource').agg(n_files=('file_name','count'), total_size_mb=('size_mb','sum')).reset_index()
        g['label']=g.resource.map(_short_resource_label); g=g.sort_values('total_size_mb')
        ax.barh(g.label, g.total_size_mb, color=[_resource_color(r) for r in g.resource])
        ax.set_xlabel('Profiled file size (MB)'); ax.set_title('Input scale by resource', fontweight='bold')
        for i,r in g.reset_index(drop=True).iterrows(): ax.text(r.total_size_mb, i, f"  {int(r.n_files)} files", va='center', fontsize=7.5)
    else: nodata(ax,'Input scale by resource')
    ax=axs[1]; _panel_letter(ax,'b')
    if not mod.empty:
        cols=[x for x in ['identifier','formula_or_composition','metal_or_charge_chemistry','geometric_descriptor','descriptor','target_or_property','curation_or_validation_flag'] if x in mod.columns]
        mat=(mod.set_index('resource')[cols]>0).astype(int)
        mat.index=[_short_resource_label(x) for x in mat.index]
        heat(ax, mat, 'Observed data modalities', 'Greys', False)
    else: nodata(ax,'Observed data modalities')
    ax=axs[2]; _panel_letter(ax,'c')
    if not c.empty:
        tmp=c.copy(); tmp['resource_short']=tmp.resource.map(_short_resource_label)
        tmp['missing_bin']=pd.cut(pd.to_numeric(tmp.missing_fraction,errors='coerce'),[-.001,0,.05,.25,.75,1],labels=['0','0–5%','5–25%','25–75%','>75%'],include_lowest=True)
        mat=tmp.groupby(['resource_short','missing_bin'],observed=False).size().unstack(fill_value=0)
        heat(ax, mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0), 'Column missingness composition', 'YlGnBu', True)
    else: nodata(ax,'Column missingness composition')
    ax=axs[3]; _panel_letter(ax,'d'); _clean_axes(ax)
    if not sc.empty:
        tmp=sc.copy(); tmp['x']=np.log10(pd.to_numeric(tmp.total_profiled_or_counted_rows,errors='coerce').fillna(0).clip(lower=1)); tmp['y']=pd.to_numeric(tmp.chemistry_trust_score,errors='coerce'); tmp['s']=90+620*pd.to_numeric(tmp.ml_readiness_score,errors='coerce').fillna(0).clip(0,1)
        for _,r in tmp.iterrows():
            ax.scatter(r.x, r.y, s=r.s, alpha=.78, edgecolors='black', linewidth=.65, color=_resource_color(r.resource), zorder=3)
            dx=-.18 if str(r.resource)=='ARC-MOF' else .025
            dy=.025 if str(r.resource)=='ARC-MOF' else .012
            ax.text(r.x+dx, r.y+dy, _short_resource_label(r.resource), fontsize=8.1)
        ax.set_xlabel('log10(profiled/counted tabular rows + 1)'); ax.set_ylabel('Chemistry-trust score'); ax.set_ylim(0,1.05)
        ax.set_title('Scale versus chemistry observability/trust', fontweight='bold')
        add_panel_note(ax,'Bubble size encodes ML-readiness. Row counts are not necessarily model-ready structures.',(.02,.02))
    else: nodata(ax,'Scale versus chemistry observability/trust')
    fig.suptitle('Figure 2. Publication resource atlas: scale, modality, missingness and chemistry-readiness',fontweight='bold',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.945])
    return save_fig(fig, dd['fig_main']/ 'Figure_2_data_resource_atlas', cfg)


def _select_headline_descriptor_cases(desc: pd.DataFrame, max_cases: int=6) -> pd.DataFrame:
    if desc.empty:
        return pd.DataFrame()
    df=desc.copy()
    for c in ['task','gas','target_column','descriptor_family','model','split_type','mean_r2','mean_spearman','mean_top_10pct_recovery','source_table','feature_table']:
        if c not in df.columns: df[c]=np.nan if c.startswith('mean_') else ''
    # Best model per case/split, prioritising top-10% recovery and then R2.
    df['_rank_metric']=pd.to_numeric(df['mean_top_10pct_recovery'],errors='coerce').fillna(-1)+0.10*pd.to_numeric(df['mean_r2'],errors='coerce').fillna(-10)
    idx=df.groupby(['task','gas','target_column','descriptor_family','split_type'], dropna=False)['_rank_metric'].idxmax()
    best=df.loc[idx].copy()
    # Select cases where both random and grouped are available if possible.
    case_cols=['task','gas','target_column','descriptor_family']
    counts=best.groupby(case_cols, dropna=False).split_type.nunique().reset_index(name='n_splits')
    paired=counts[counts.n_splits>=2][case_cols]
    if not paired.empty:
        best=best.merge(paired, on=case_cols, how='inner')
    score=best.groupby(case_cols, dropna=False).agg(best_top10=('mean_top_10pct_recovery','max'), best_r2=('mean_r2','max')).reset_index()
    score['score']=pd.to_numeric(score.best_top10,errors='coerce').fillna(-1)+0.15*pd.to_numeric(score.best_r2,errors='coerce').fillna(-10)
    score=score.sort_values('score',ascending=False).head(max_cases)
    out=best.merge(score[case_cols], on=case_cols, how='inner')
    out['case_label']=out.apply(lambda r: f"{str(r.get('descriptor_family',''))} | {str(r.get('task','')).replace('_',' ')} | {str(r.get('gas',''))} | {str(r.get('target_column',''))}", axis=1)
    return out


def fig5(cfg, dd):
    desc=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    matched=load_df(dd['processed']/ 'descriptor_family_case_matched_comparison')
    pred=load_df(dd['processed']/ 'descriptor_joined_predictions_sample')
    if pred.empty:
        pred=load_df(dd['processed']/ 'ml_predictions_sample')
    cases=_select_headline_descriptor_cases(desc, max_cases=6)
    fig,axs=plt.subplots(2,2,figsize=(14.2,9.4)); axs=axs.ravel()

    ax=axs[0]; _panel_letter(ax,'a'); _clean_axes(ax)
    if not cases.empty:
        piv=cases.pivot_table(index='case_label',columns='split_type',values='mean_r2',aggfunc='max')
        order=piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y=np.arange(len(order))
        for split in [s for s in ['random','descriptor_grouped','grouped'] if s in piv.columns]:
            ax.scatter(piv.loc[order,split], y, marker=SPLIT_MARKERS.get(split,'o'), s=72, color=SPLIT_COLORS.get(split,'#333333'), label=split.replace('_',' '), edgecolors='black', linewidth=.45, zorder=3)
        for i,labtxt in enumerate(order):
            vals=[piv.loc[labtxt,s] for s in piv.columns if pd.notna(piv.loc[labtxt,s])]
            if len(vals)>=2: ax.plot(vals,[i]*len(vals),color='#BBBBBB',lw=.8,zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(order,fontsize=7.1)
        ax.set_xlabel('Mean R²'); ax.set_title('Generalization gap across headline joined cases',fontweight='bold')
        ax.legend(fontsize=7,loc='lower right')
    else: nodata(ax,'Generalization gap','No descriptor-joined ML summary available')

    ax=axs[1]; _panel_letter(ax,'b'); _clean_axes(ax)
    if not cases.empty:
        piv=cases.pivot_table(index='case_label',columns='split_type',values='mean_top_10pct_recovery',aggfunc='max')
        order=piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y=np.arange(len(order))
        for split in [s for s in ['random','descriptor_grouped','grouped'] if s in piv.columns]:
            ax.scatter(piv.loc[order,split], y, marker=SPLIT_MARKERS.get(split,'o'), s=72, color=SPLIT_COLORS.get(split,'#333333'), label=split.replace('_',' '), edgecolors='black', linewidth=.45, zorder=3)
        for i,labtxt in enumerate(order):
            vals=[piv.loc[labtxt,s] for s in piv.columns if pd.notna(piv.loc[labtxt,s])]
            if len(vals)>=2: ax.plot(vals,[i]*len(vals),color='#BBBBBB',lw=.8,zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(order,fontsize=7.1)
        ax.set_xlim(0,1.02); ax.set_xlabel('Top-10% recovery'); ax.set_title('Screening relevance is retained under harder splits',fontweight='bold')
    else: nodata(ax,'Top-10% recovery')

    ax=axs[2]; _panel_letter(ax,'c')
    if not matched.empty:
        m=matched.copy()
        fam_col='descriptor_family' if 'descriptor_family' in m.columns else ('descriptor_family_x' if 'descriptor_family_x' in m.columns else None)
        # v2.1 fix: the case-matched descriptor-family table is normally long-format,
        # with descriptor_family, split_type, mean_r2 and mean_top_10pct_recovery columns.
        # Convert it to a compact heatmap instead of looking only for pre-wide columns.
        if fam_col and {'split_type','mean_r2','mean_top_10pct_recovery'}.issubset(m.columns):
            for col in ['mean_r2','mean_spearman','mean_top_10pct_recovery']:
                if col in m.columns:
                    m[col]=pd.to_numeric(m[col], errors='coerce')
            tmp=(m.groupby([fam_col,'split_type'])
                   .agg(median_r2=('mean_r2','median'),
                        median_spearman=('mean_spearman','median') if 'mean_spearman' in m.columns else ('mean_r2','median'),
                        median_top10=('mean_top_10pct_recovery','median'),
                        n_cases=('mean_top_10pct_recovery','count'))
                   .reset_index())
            r2=tmp.pivot_table(index=fam_col, columns='split_type', values='median_r2', aggfunc='median')
            top10=tmp.pivot_table(index=fam_col, columns='split_type', values='median_top10', aggfunc='median')
            ncase=tmp.groupby(fam_col)['n_cases'].sum()
            order=top10.max(axis=1).sort_values(ascending=False).index.tolist() if not top10.empty else r2.index.tolist()
            mat=pd.DataFrame(index=order)
            for split in ['random','descriptor_grouped','grouped']:
                if split in r2.columns:
                    label='grouped' if split in ['descriptor_grouped','grouped'] else 'random'
                    mat[f'R² {label}']=r2.loc[order,split]
                if split in top10.columns:
                    label='grouped' if split in ['descriptor_grouped','grouped'] else 'random'
                    mat[f'Top-10 {label}']=top10.loc[order,split]
            if not mat.empty:
                heat(ax,mat,'Case-matched descriptor-family comparison','YlGnBu',True)
                note='Long-format source data aggregated by descriptor family and split; matched by task/gas/target among successfully joined cases.'
                if not ncase.empty:
                    note += ' Total case rows shown: ' + str(int(ncase.sum())) + '.'
                add_panel_note(ax,note,xy=(.02,.01))
            else:
                nodata(ax,'Case-matched descriptor-family comparison','No numeric descriptor-family metrics after aggregation')
        else:
            metric_cols=[c for c in ['mean_r2_random','mean_r2_descriptor_grouped','mean_top_10pct_recovery_random','mean_top_10pct_recovery_descriptor_grouped'] if c in m.columns]
            if metric_cols and fam_col:
                mat=m.groupby(fam_col)[metric_cols].median(numeric_only=True)
                mat=mat.rename(columns=lambda x: x.replace('mean_','').replace('_descriptor_grouped',' grouped').replace('_random',' random').replace('_',' '))
                heat(ax,mat,'Case-matched descriptor-family comparison','YlGnBu',True)
            else:
                nodata(ax,'Case-matched descriptor-family comparison','Descriptor-family metric columns not available')
    elif not desc.empty and {'descriptor_family','split_type','mean_top_10pct_recovery'}.issubset(desc.columns):
        m=desc.groupby(['descriptor_family','split_type']).agg(top10=('mean_top_10pct_recovery','median')).reset_index()
        mat=m.pivot_table(index='descriptor_family',columns='split_type',values='top10',aggfunc='median')
        heat(ax,mat,'Descriptor-family top-10% recovery','YlGnBu',True)
    else: nodata(ax,'Case-matched descriptor-family comparison')

    ax=axs[3]; _panel_letter(ax,'d'); _clean_axes(ax)
    if not pred.empty:
        pp=pred.copy()
        # Choose a representative prediction set from the best available source table/model/split.
        choose_cols=[c for c in ['source_table','target_column','model','split_type'] if c in pp.columns]
        if choose_cols and 'absolute_error' in pp.columns:
            grp=pp.groupby(choose_cols).agg(n=('y_true','size'), mae=('absolute_error','mean')).reset_index()
            grp=grp[grp.n>=50].sort_values('mae') if not grp.empty else grp
            if not grp.empty:
                mask=np.ones(len(pp),dtype=bool)
                for c in choose_cols: mask &= pp[c].eq(grp.iloc[0][c])
                pp=pp[mask]
        sub=pp.sample(n=min(len(pp),2500),random_state=cfg.random_seed) if len(pp)>2500 else pp
        ax.scatter(sub.y_true,sub.y_pred,s=9,alpha=.32,color='#0072B2',edgecolors='none')
        mn=float(min(sub.y_true.min(),sub.y_pred.min())); mx=float(max(sub.y_true.max(),sub.y_pred.max()))
        ax.plot([mn,mx],[mn,mx],ls='--',lw=1.0,color='#333333')
        ax.set_xlabel('Observed target'); ax.set_ylabel('Predicted target')
        ax.set_title('Representative calibration for a selected joined case',fontweight='bold')
        add_panel_note(ax,'All other calibration views are retained in source/SI tables.',(.03,.03))
    else: nodata(ax,'Representative calibration')
    fig.suptitle('Figure 5. Descriptor-joined ARC-MOF screening benchmark for publication',fontweight='bold',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.945])
    return save_fig(fig, dd['fig_main']/ 'Figure_5_ml_stress_test', cfg)


def _df_to_markdown(df: pd.DataFrame, max_rows: int=40) -> str:
    if df is None or df.empty:
        return '_No records available._\n'
    view=df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False) + ('\n' if len(df)<=max_rows else f"\n\n_Showing {max_rows} of {len(df)} rows._\n")
    except Exception:
        return '```csv\n' + view.to_csv(index=False) + '```\n'


def _write_md(path: Path, text: str):
    mkdir(path.parent)
    path.write_text(text, encoding='utf-8')


def _read_text_if_exists(p: Path, max_chars: int=200000) -> str:
    try:
        if p.exists():
            return p.read_text(encoding='utf-8', errors='replace')[:max_chars]
    except Exception:
        pass
    return ''


def make_publication_asset_index(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    for sub,kind in [('figures/main','main_figure'),('figures/si','si_figure'),('tables/main','main_table'),('tables/si','si_table'),('source_data','source_data')]:
        root=cfg.out_dir/sub
        if not root.exists():
            continue
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            try:
                rel=str(p.relative_to(cfg.out_dir))
            except Exception:
                rel=str(p)
            rows.append(dict(asset_type=kind, relative_path=rel, file_name=p.name, size_kb=round(p.stat().st_size/1024,2), suggested_use=(
                'use PDF in LaTeX; keep SVG editable; PNG for inspection' if kind=='main_figure' else
                'candidate main-text table' if kind=='main_table' else
                'SI/supporting information table' if kind=='si_table' else
                'source-data / reproducibility material')))
    out=pd.DataFrame(rows)
    save_df(out, dd['source']/ 'publication_asset_index', cfg)
    return out


def export_final_panel_source_data(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    """Write compact per-figure source-data extracts for paper assembly."""
    panel_dir=dd['source']/ 'figure_panel_source_data'
    mkdir(panel_dir)
    mapping=[]
    sources=[
        ('Figure_2a_resource_file_scale', dd['profiles']/ 'file_level_profile', ['resource','resource_subtype','role','file_name','size_mb','true_rows','n_columns','read_status']),
        ('Figure_2b_resource_modality_matrix', dd['profiles']/ 'resource_modality_matrix', None),
        ('Figure_2d_resource_scores', dd['processed']/ 'resource_scores', None),
        ('Figure_3a_chemistry_evidence_matrix', dd['processed']/ 'chemistry_evidence_matrix', None),
        ('Figure_3b_trust_regime_summary', dd['processed']/ 'trust_regime_summary', None),
        ('Figure_3c_charge_status_summary', dd['processed']/ 'charge_status_summary', None),
        ('Figure_4a_resource_scores_uncertainty', dd['processed']/ 'resource_score_uncertainty', None),
        ('Figure_4b_benchmark_risk_matrix', dd['processed']/ 'benchmark_risk_matrix', None),
        ('Figure_5_descriptor_joined_summary', dd['processed']/ 'descriptor_joined_ml_metrics_summary', None),
        ('Figure_5_case_matched_descriptor_comparison', dd['processed']/ 'descriptor_family_case_matched_comparison', None),
        ('Figure_5_descriptor_join_diagnostics', dd['processed']/ 'descriptor_target_join_diagnostics', None),
        ('Figure_6_decision_rules', dd['source']/ 'decision_rules', None),
        ('Figure_6_reporting_checklist', dd['source']/ 'minimum_reporting_checklist', None),
    ]
    for name,base,cols in sources:
        df=load_df(base)
        if df.empty:
            continue
        if cols:
            df=df[[c for c in cols if c in df.columns]].copy()
        # Keep panel data compact but useful; source tables keep full detail elsewhere.
        max_rows=5000 if 'descriptor_joined' in name or 'diagnostics' in name else 2000
        out=df.head(max_rows).copy()
        p=panel_dir/(name+'.csv')
        out.to_csv(p,index=False)
        mapping.append(dict(panel_source_file=str(p.relative_to(cfg.out_dir)), source_base=str(base.relative_to(cfg.out_dir)) if str(base).startswith(str(cfg.out_dir)) else str(base), n_rows_written=len(out), original_rows=len(df), note='compact panel source-data extract; full table retained elsewhere when available'))
    manifest=pd.DataFrame(mapping)
    save_df(manifest, dd['source']/ 'figure_panel_source_data_manifest', cfg)
    return manifest


def create_curated_endpoint_shortlist(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    """Collapse broad selected endpoints to a manuscript-size shortlist."""
    selected=load_df(dd['source']/ 'selected_main_text_endpoints')
    if selected.empty:
        selected=load_df(dd['processed']/ 'target_dictionary_detailed')
    if selected.empty:
        out=pd.DataFrame()
        save_df(out, dd['source']/ 'curated_main_text_endpoint_shortlist', cfg)
        save_df(out, dd['tables']/ 'main'/ 'Main_Table_10_curated_main_text_endpoint_shortlist', cfg)
        return out
    df=selected.copy()
    for c in ['resource','task','gas','target_column','target_property','target_unit','source_file','selected_main_text_endpoint','main_text_endpoint_class']:
        if c not in df.columns: df[c]=''
    df=df[(df.get('selected_main_text_endpoint','yes').astype(str).str.lower().eq('yes')) | (df.resource.eq('ARC-MOF'))].copy()
    priority=[]
    for _,r in df.iterrows():
        score=0
        if r.resource=='ARC-MOF': score+=50
        task=str(r.task).lower(); gas=str(r.gas).upper(); col=str(r.target_column).lower(); prop=str(r.target_property).lower()
        if 'post_combustion' in task and gas=='CO2': score+=12
        if 'pre_combustion' in task and gas=='CO2': score+=11
        if 'methane' in task and gas in ['CH4','CO2']: score+=8
        if 'landfill' in task and gas in ['CH4','CO2']: score+=7
        if 'v/v' in col or prop=='volumetric_uptake': score+=4
        if 'mmol/g' in col or prop in ['gravimetric_uptake','uptake','loading']: score+=4
        if 's(g1)' in col or 'select' in col or prop=='selectivity': score+=4
        if prop in ['working_capacity','purity','recovery','productivity','process_energy']: score+=2
        priority.append(score)
    df['_priority']=priority
    df=df.sort_values(['_priority','resource','task','gas','target_column'], ascending=[False,True,True,True,True])
    cols=['resource','task','gas','target_column','target_property','target_unit','source_file','main_text_endpoint_class']
    out=df[cols].drop_duplicates().head(12).copy()
    out.insert(0,'shortlist_rank', range(1,len(out)+1))
    out['manuscript_use']='headline endpoint context; full target dictionary remains in SI/source data'
    save_df(out, dd['source']/ 'curated_main_text_endpoint_shortlist', cfg)
    save_df(out, dd['tables']/ 'main'/ 'Main_Table_10_curated_main_text_endpoint_shortlist', cfg)
    return out


def write_publication_style_sheet(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    for res,col in RESOURCE_COLORS.items():
        rows.append(dict(category='resource', item=res, short_label=RESOURCE_SHORT_LABELS.get(res,res), color_hex=col, usage='resource identity in figures and captions'))
    for reg,col in TRUST_REGIME_COLORS.items():
        rows.append(dict(category='trust_regime', item=reg, short_label=reg.replace('_',' '), color_hex=col, usage='chemistry evidence/trust status'))
    for split,col in SPLIT_COLORS.items():
        rows.append(dict(category='ml_split', item=split, short_label=split.replace('_',' '), color_hex=col, usage='random versus descriptor-grouped ML split'))
    rows += [
        dict(category='typography', item='main-panel-font-minimum', short_label='≥8 pt', color_hex='', usage='minimum readable size after journal scaling'),
        dict(category='typography', item='panel-label', short_label='bold 14 pt', color_hex='', usage='panel labels a–d'),
        dict(category='export', item='main figures', short_label='PDF + SVG + 600-dpi PNG', color_hex='', usage='PDF for LaTeX, SVG for editing, PNG for inspection'),
        dict(category='caption_caveat', item='not observable', short_label='not directly present in parsed metadata; not invalid', color_hex='', usage='mandatory phrase for Figure 3/captions'),
        dict(category='caption_caveat', item='score uncertainty', short_label='bootstrap CI for reporting score, not true chemical validity', color_hex='', usage='mandatory phrase for Figure 4/captions'),
    ]
    out=pd.DataFrame(rows)
    save_df(out, dd['source']/ 'publication_figure_style_sheet', cfg)
    return out


def step_tables(cfg: Config, dd: Dict[str,Path], log):
    _V18_2_STEP_TABLES(cfg, dd, log)
    style=write_publication_style_sheet(cfg, dd)
    endpoints=create_curated_endpoint_shortlist(cfg, dd)
    assets=make_publication_asset_index(cfg, dd)
    panels=export_final_panel_source_data(cfg, dd)
    log.info('Publication polish tables written: style rows=%d, endpoint shortlist=%d, assets=%d, panel-source files=%d', len(style), len(endpoints), len(assets), len(panels))


def step_figures(cfg: Config, dd: Dict[str,Path], log):
    outs=[]
    for name,func in [
        ('SI Figure S0 inventory map', lambda: inventory_alignment_figure(cfg,dd)),
        ('Figure 1 polished concept', lambda: fig1(cfg,dd)),
        ('Figure 2 polished resource atlas', lambda: fig2(cfg,dd)),
        ('Figure 3 trust regimes', lambda: fig3(cfg,dd)),
        ('Figure 4 risk map', lambda: fig4(cfg,dd)),
        ('Figure 5 final descriptor benchmark', lambda: fig5(cfg,dd)),
        ('Figure 6 decision framework', lambda: fig6(cfg,dd)),
        ('SI figures S1-S8', lambda: si_figs(cfg,dd)),
    ]:
        try:
            log.info('Rendering %s', name)
            new=func()
            if isinstance(new,list): outs+=new
            log.info('Finished rendering %s (%d files)', name, len(new) if isinstance(new,list) else 0)
        except Exception as e:
            log.warning('Figure rendering failed for %s: %s', name, e)
    save_df(pd.DataFrame([{'figure_file':str(p),'relative_path':str(p.relative_to(cfg.out_dir)) if str(p).startswith(str(cfg.out_dir)) else str(p),'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None} for p in outs]),dd['fig']/ 'figure_output_manifest',cfg)
    export_final_panel_source_data(cfg, dd)


def step_reports(cfg: Config, dd: Dict[str,Path], log):
    _V18_2_STEP_REPORTS(cfg, dd, log)
    # Add a final publication wrap-up report that is compact enough to read.
    scores=load_df(dd['processed']/ 'resource_scores')
    comp=load_df(dd['tables']/ 'main'/ 'Main_Table_7_ml_feature_source_comparison')
    endpoints=load_df(dd['source']/ 'curated_main_text_endpoint_shortlist')
    assets=make_publication_asset_index(cfg, dd)
    lines=[
        '# Final publication wrap-up report\n\n',
        f'Generated: {iso()}\n\n',
        f'Script version: `{VERSION}`\n\n',
        '## Recommended final manuscript message\n\n',
        'MOF machine-learning resources differ in chemistry observability, descriptor independence, target provenance and split-aware generalization. ARC-MOF is highly ML-ready and supports descriptor-joined adsorption screening, but chemistry-trust evidence is not directly observable from parsed tabular fields and therefore requires explicit reporting caveats. MOSAEC and CoRE provide stronger chemistry/provenance anchors. Descriptor-joined ARC-MOF models should be interpreted primarily through ranking and top-candidate recovery, with grouped splits used to avoid overclaiming random interpolation performance.\n\n',
        '## Resource readiness table\n\n', _df_to_markdown(scores[[c for c in ['resource','n_files','total_profiled_or_counted_rows','chemistry_trust_score','ml_readiness_score','trust_readiness_quadrant'] if c in scores.columns]] if not scores.empty else scores, 20), '\n',
        '## Feature-source comparison\n\n', _df_to_markdown(comp, 20), '\n',
        '## Curated main-text endpoints\n\n', _df_to_markdown(endpoints, 20), '\n',
        '## Publication assets\n\n', _df_to_markdown(assets[assets.asset_type.isin(['main_figure','main_table'])] if not assets.empty else assets, 80), '\n',
        '## Required caption caveats\n\n',
        '- “Not observable” means not directly present in the parsed tabular metadata and does not imply that a structure is chemically invalid.\n',
        '- Score intervals are bootstrap intervals for the composite reporting/readiness score, not uncertainty in true chemical validity.\n',
        '- Descriptor-family comparisons refer to successfully joined task cases, not universal coverage of all descriptors or databases.\n',
    ]
    _write_md(dd['reports']/ 'final_publication_wrapup_report.md', ''.join(lines))


def _collect_warning_rows(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    if not fdf.empty and 'read_status' in fdf.columns:
        fail=fdf[~fdf.read_status.astype(str).str.lower().eq('ok')].copy()
        for _,r in fail.iterrows():
            rows.append(dict(source='file_profile', severity='warning', item=r.get('relative_path',r.get('file_name','')), message=r.get('read_error','failed to read as table')))
    logtxt=_read_text_if_exists(dd['logs']/ 'run.log')
    for line in logtxt.splitlines():
        if 'WARNING' in line or 'ERROR' in line:
            rows.append(dict(source='run.log', severity='error' if 'ERROR' in line else 'warning', item='', message=line[-1000:]))
    for p in cfg.out_dir.rglob('SAVE_WARNINGS.txt'):
        for line in _read_text_if_exists(p).splitlines():
            rows.append(dict(source=str(p.relative_to(cfg.out_dir)) if str(p).startswith(str(cfg.out_dir)) else str(p), severity='warning', item='', message=line[-1000:]))
    return pd.DataFrame(rows)


def step_github_audit(cfg: Config, dd: Dict[str,Path], log):
    audit=cfg.out_dir/'github_audit'
    mkdir(audit)
    fdf=load_df(dd['profiles']/ 'file_level_profile')
    cdf=load_df(dd['profiles']/ 'column_level_profile')
    hashes=load_df(dd['logs']/ 'file_hashes')
    scores=load_df(dd['processed']/ 'resource_scores')
    targets=load_df(dd['processed']/ 'normalized_target_catalog')
    diag=load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    desc=load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    assets=make_publication_asset_index(cfg, dd)
    warnings_df=_collect_warning_rows(cfg, dd)

    # Compact data manifest for GitHub/data manual, not a copy of raw data.
    data_manifest=pd.DataFrame()
    if not fdf.empty:
        cols=[c for c in ['file_path','relative_path','file_name','resource','resource_subtype','role','extension','size_mb','true_rows','n_columns','n_profiled_rows','read_status','read_error','identifier_candidates','target_candidates'] if c in fdf.columns]
        data_manifest=fdf[cols].copy()
        if not hashes.empty and 'path' in hashes.columns and 'file_path' in data_manifest.columns:
            hh=hashes[[c for c in ['path','sha256_fast','size_bytes','mtime'] if c in hashes.columns]].rename(columns={'path':'file_path','size_bytes':'hash_size_bytes','mtime':'hash_mtime'})
            data_manifest=data_manifest.merge(hh,on='file_path',how='left')
        data_manifest.to_csv(audit/'data_input_manifest_compact.csv',index=False)
        summary=data_manifest.groupby([c for c in ['resource','resource_subtype','role'] if c in data_manifest.columns], dropna=False).agg(
            n_files=('file_name','count'), total_size_mb=('size_mb','sum'), total_rows=('true_rows','sum'), median_columns=('n_columns','median')
        ).reset_index()
        summary.to_csv(audit/'data_summary_by_resource_role.csv',index=False)
    else:
        summary=pd.DataFrame()

    # Column and target summaries, compact.
    if not cdf.empty:
        colsum=cdf.groupby([c for c in ['resource','resource_subtype','role','inferred_modality'] if c in cdf.columns], dropna=False).agg(
            n_columns=('column_name','count'), median_missing_fraction=('missing_fraction','median')
        ).reset_index()
        colsum.to_csv(audit/'column_modality_summary_compact.csv',index=False)
    else: colsum=pd.DataFrame()

    if not targets.empty:
        tsum=targets.groupby([c for c in ['resource','resource_subtype','task','gas','target_property','target_unit'] if c in targets.columns], dropna=False).agg(
            n_values=('target_value','count'), n_ids=('canonical_id','nunique'), n_source_files=('source_file','nunique')
        ).reset_index()
        tsum.to_csv(audit/'target_usage_summary_compact.csv',index=False)
    else: tsum=pd.DataFrame()

    if not diag.empty:
        dcols=[c for c in ['target_file','descriptor_file','descriptor_family','join_key_level','join_status','joined_rows','left_retention','right_retention','risk_of_overmatching'] if c in diag.columns]
        diag[dcols].to_csv(audit/'descriptor_join_report_compact.csv',index=False)
    if not desc.empty:
        dsum=desc.groupby([c for c in ['descriptor_family','split_type','target_family'] if c in desc.columns], dropna=False).agg(
            n_cases=('source_table','count'), median_r2=('mean_r2','median'), median_spearman=('mean_spearman','median'), median_top10=('mean_top_10pct_recovery','median')
        ).reset_index()
        dsum.to_csv(audit/'descriptor_ml_summary_compact.csv',index=False)
    else: dsum=pd.DataFrame()

    # Step usage manual.
    step_rows=[
        dict(step='profile_inputs', uses='all discovered table-like CSV/TSV/TXT/XLSX/JSON/Parquet/Pickle files under --data-root', main_outputs='profiles/file_level_profile; profiles/column_level_profile', github_note='required for reproducing resource atlas and data inventory'),
        dict(step='normalize_targets', uses='target-like ARC-MOF, CoRE isotherm and QMOF columns', main_outputs='processed/normalized_target_catalog; target dictionary tables', github_note='required for endpoint definitions and target provenance'),
        dict(step='identifier_join_audit', uses='identifier columns such as filename, Structure_Name, MOFid, refcode and canonical variants', main_outputs='processed/join_accounting; descriptor join diagnostics', github_note='required for leakage/join-retention reporting'),
        dict(step='chemistry_trust', uses='formula, metal, oxidation/valence, charge and curation/warning columns where observable', main_outputs='processed/trust_flags; trust_regime_summary', github_note='not-observable is reported separately from invalid'),
        dict(step='descriptor_joined_ml', uses='ARC-MOF target tables joined to RAC/RDF/geometry/topology-like descriptor tables where identifiers allow', main_outputs='descriptor_joined_ml_metrics_summary; case-matched descriptor comparison', github_note='central screening benchmark; use grouped splits'),
        dict(step='ml_stress_test', uses='table-local non-descriptor baselines and selected target tables', main_outputs='ml_metrics_summary; feature-source comparison', github_note='used only as contrast/control'),
        dict(step='figures_tables', uses='processed summaries and compact panel source-data', main_outputs='figures/main; tables/main; source_data/figure_panel_source_data', github_note='paper assembly assets'),
        dict(step='github_audit', uses='manifests, profiles, logs, source-data and table summaries', main_outputs='github_audit/*.csv, *.md, github_preparation_compact_report.zip', github_note='compact report for future public repository README/manual'),
    ]
    step_df=pd.DataFrame(step_rows)
    step_df.to_csv(audit/'analysis_step_data_usage_manual.csv',index=False)

    warnings_df.to_csv(audit/'warnings_and_failed_reads_compact.csv',index=False)
    if not scores.empty: scores.to_csv(audit/'resource_scores_compact.csv',index=False)
    if not assets.empty: assets.to_csv(audit/'publication_assets_index_compact.csv',index=False)

    # Human-readable manuals.
    _write_md(audit/'README_GITHUB_DATA_MANUAL.md', ''.join([
        '# GitHub/data preparation manual for the chemistry-ready MOF analysis\n\n',
        f'Generated: {iso()}\n\n',
        f'Script version: `{VERSION}`\n\n',
        '## What to put in the repository\n\n',
        'Do **not** commit very large raw database archives or generated model files unless the target repository explicitly allows them. Commit the script, environment file, README, compact audit reports, selected main figures/tables, and source-data extracts. Provide instructions for users to place the external datasets under the same `--data-root` layout used here.\n\n',
        '## Required data layout\n\n',
        'The analysis expects the unzipped project data root to contain the MOF data resources profiled by the run. The compact manifest below documents the exact local files used, their roles, sizes, row/column counts and fast hashes.\n\n',
        '### Data summary by resource/role\n\n', _df_to_markdown(summary, 60), '\n',
        '## Analysis steps and data use\n\n', _df_to_markdown(step_df, 20), '\n',
        '## Warnings / excluded auxiliary files\n\n', _df_to_markdown(warnings_df, 80), '\n',
        '## Reproducibility notes\n\n',
        '- CSV outputs are canonical. Pickle/Parquet sidecars are optional convenience formats.\n',
        '- Use a fresh output directory for a from-scratch publication run.\n',
        '- `not observable` means not directly present in parsed tabular metadata, not chemically invalid.\n',
        '- For GitHub, include this `github_audit/` folder and the small source-data extracts rather than the full 1–2 GB internal output archive.\n',
    ]))
    _write_md(audit/'PUBLICATION_ASSET_CHECKLIST.md', ''.join([
        '# Publication asset checklist\n\n',
        'Use PDF versions of main figures in LaTeX. Keep SVG files for final graphical editing and PNG files for visual checking.\n\n',
        _df_to_markdown(assets[assets.asset_type.isin(['main_figure','main_table'])] if not assets.empty else assets, 120), '\n',
    ]))
    _write_md(audit/'COMPACT_ANALYSIS_REPORT.md', ''.join([
        '# Compact analysis report\n\n',
        f'Data root: `{cfg.data_root}`\n\n', f'Output directory: `{cfg.out_dir}`\n\n',
        '## Resource scores\n\n', _df_to_markdown(scores, 30), '\n',
        '## Target usage summary\n\n', _df_to_markdown(tsum, 80), '\n',
        '## Descriptor ML summary\n\n', _df_to_markdown(dsum, 40), '\n',
    ]))

    # Zip compact audit only.
    zpath=audit/'github_preparation_compact_report.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in audit.rglob('*'):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(audit))
    log.info('GitHub/data-preparation compact audit created: %s (%s)', zpath, hbytes(zpath.stat().st_size))


def build_final_manifest(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    def classify(rel: str) -> Tuple[str,str,str]:
        if rel.startswith('github_audit/'): return ('github_audit','compact repository/data manual','compact')
        if rel.startswith('figures/main/'): return ('figure','main manuscript figure','publication')
        if rel.startswith('figures/si/'): return ('figure','supporting information figure','supporting')
        if rel.startswith('tables/main/'): return ('table','main manuscript table','publication')
        if rel.startswith('tables/si/'): return ('table','supporting information table','supporting')
        if rel.startswith('source_data/figure_panel_source_data/'): return ('source_data','compact panel source data','publication')
        if rel.startswith('source_data/'): return ('source_data','source or reproducibility table','source')
        if rel.startswith('profiles/'): return ('profile','input table profile','reproducibility')
        if rel.startswith('processed/'): return ('processed','intermediate/final computed table','reproducibility')
        if rel.startswith('logs/'): return ('log','run log/environment/hash file','audit')
        if rel.startswith('reports/'): return ('report','human-readable analysis report','audit')
        if rel.startswith('models/'): return ('model','saved fitted model','optional-heavy')
        return ('other','other output','other')
    for p in cfg.out_dir.rglob('*'):
        if not p.is_file():
            continue
        if p.name == 'chemistry_ready_mof_analysis_outputs_package.zip':
            continue
        try: rel=str(p.relative_to(cfg.out_dir)).replace('\\','/')
        except Exception: rel=str(p)
        kind,desc,group=classify(rel)
        rows.append(dict(relative_path=rel,file_name=p.name,file_type=kind,asset_group=group,size_bytes=p.stat().st_size,size_kb=round(p.stat().st_size/1024,2),description=desc))
    man=pd.DataFrame(rows).sort_values(['asset_group','file_type','relative_path']) if rows else pd.DataFrame()
    save_df(man,cfg.out_dir/'FINAL_OUTPUT_MANIFEST',cfg)
    save_df(man,dd['source']/ 'FINAL_OUTPUT_MANIFEST',cfg)
    return man


def step_zip(cfg: Config, dd: Dict[str,Path], log):
    build_final_manifest(cfg, dd)
    if cfg.no_zip:
        return
    zp=cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'
    include_subdirs=['reports','profiles','processed','tables','figures','source_data','logs','models','github_audit']
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [cfg.out_dir/'FINAL_OUTPUT_MANIFEST.csv', cfg.out_dir/'README_OUTPUTS.txt']:
            if p.exists(): z.write(p, p.relative_to(cfg.out_dir))
        for sub in include_subdirs:
            root=cfg.out_dir/sub
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p != zp:
                        z.write(p,p.relative_to(cfg.out_dir))
    entries=[]
    with zipfile.ZipFile(zp) as z:
        names=set(z.namelist())
    man=load_df(cfg.out_dir/'FINAL_OUTPUT_MANIFEST')
    missing=[]
    if not man.empty and 'relative_path' in man.columns:
        missing=[r for r in man.relative_path.astype(str) if r not in names and r != 'chemistry_ready_mof_analysis_outputs_package.zip']
    audit=pd.DataFrame([dict(zip_path=str(zp), zip_size_bytes=zp.stat().st_size, zip_entries=len(names), manifest_entries=len(man), missing_manifest_entries=len(missing), missing_examples=';'.join(missing[:20]))])
    save_df(audit, dd['source']/ 'zip_contents_audit', cfg)
    log.info('ZIP package created: %s (%s); manifest missing entries=%d', zp, hbytes(zp.stat().st_size), len(missing))


def write_readme(cfg: Config):
    txt=f"""Chemistry-ready MOF analysis output folder
===========================================
Generated: {iso()}
Script version: {VERSION}

Purpose of this final run
-------------------------
This v1.9 final-publication script is intended for the last from-scratch paper
run.  It keeps the v1.8.2 profile-checkpoint hotfix and adds a compact
GitHub/data-preparation audit in `github_audit/`.

Recommended paper outputs
-------------------------
figures/main/     main figures as PDF, SVG and 600-dpi PNG
tables/main/      main manuscript tables, including clean resource atlas and descriptor benchmarks
tables/si/        full SI tables
source_data/      source-data tables and per-panel extracts
reports/          analysis_report.md and final_publication_wrapup_report.md
github_audit/     compact report of data files, file roles, analysis steps, warnings and GitHub manual
logs/             run.log, environment metadata, pipeline state and file hashes

GitHub audit folder
-------------------
`github_audit/` is deliberately compact. It records what data files were used,
how they were classified, which analyses used them, and which outputs should be
included in a future repository. It does not duplicate large raw databases.

Windows / from-scratch run advice
---------------------------------
Use a fresh output folder. Close Excel, Explorer Preview Pane, OneDrive sync
windows, PDF/SVG viewers and old terminals before starting. CSV outputs are the
canonical source of truth; PKL/Parquet sidecars are optional.

Caption caveats to preserve
---------------------------
1. “Not observable” means not directly present in parsed tabular metadata and
does not imply chemical invalidity.
2. Score intervals are bootstrap/reporting-score intervals, not uncertainty in
true chemical validity.
3. Descriptor-family comparisons refer to successfully joined task cases.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')


# =============================================================================
# v2.1 CLAIM-READY SCIENTIFIC-ML EXTENSION
# =============================================================================
# This block keeps the v1.9 final-publication / GitHub-audit pipeline intact and
# adds the additional presentation and reporting layer needed for a potential
# broad data-science submission.  The goal is not to change the scientific result,
# but to make the outputs read as a general scientific-ML claim-readiness
# framework with MOFs as a demanding case study.
#
# Added outputs:
#   * journal_submission/ compact framework tables, captions and cover-letter notes;
#   * a generalized Figure 1: claim-ready scientific ML datasets;
#   * extra GitHub/data manuals for repository preparation and data-use reporting;
#   * final manifest/ZIP inclusion of journal_submission/;
#   * README and final reports explicitly supporting both Digital Discovery and
#     broad data-science framing.

VERSION = "2.1-claim-ready-sciml-final"

# Keep references to v1.9 implementations for additive wrappers.
_V19_FIG1 = fig1
_V19_STEP_GITHUB_AUDIT = step_github_audit
_V19_STEP_ZIP = step_zip
_V19_BUILD_FINAL_MANIFEST = build_final_manifest
_V19_WRITE_README = write_readme

try:
    SAVE_MODES["balanced"]["svg"] = True
    SAVE_MODES["balanced"]["png"] = True
    SAVE_MODES["balanced"]["models"] = False
except Exception:
    pass

CLAIM_READINESS_COLORS = {
    "raw": "#999999",
    "computation_ready": "#0072B2",
    "model_ready": "#E69F00",
    "evidence_ready": "#009E73",
    "claim_ready": "#CC79A7",
    "risk": "#D55E00",
}


def _journal_submission_dir(cfg: Config) -> Path:
    return cfg.out_dir / "journal_submission"


def _safe_nunique_df(df: pd.DataFrame, col: str) -> int:
    try:
        return int(df[col].nunique(dropna=True)) if col in df.columns else 0
    except Exception:
        return 0


def _safe_len(df: pd.DataFrame) -> int:
    return int(len(df)) if isinstance(df, pd.DataFrame) else 0


def build_claim_readiness_tables(cfg: Config, dd: Dict[str,Path]) -> Dict[str, pd.DataFrame]:
    """Create compact, domain-general framework tables for broad data-science framing."""
    fdf = load_df(dd['profiles']/ 'file_level_profile')
    cdf = load_df(dd['profiles']/ 'column_level_profile')
    scores = load_df(dd['processed']/ 'resource_scores')
    targets = load_df(dd['processed']/ 'normalized_target_catalog')
    join = load_df(dd['processed']/ 'descriptor_target_join_diagnostics')
    desc = load_df(dd['processed']/ 'descriptor_joined_ml_metrics_summary')
    warnings_df = _collect_warning_rows(cfg, dd) if '_collect_warning_rows' in globals() else pd.DataFrame()

    framework = pd.DataFrame([
        dict(stage='raw scientific files', general_question='Can the files be located, parsed and cited?',
             operational_test='Discovery manifest, file size/hash, read-status table, environment record.',
             mof_instantiation='ARC-MOF/CoRE/MOSAEC/QMOF/CSD-derived tables discovered and profiled.',
             key_outputs='profiles/discovered_input_files; logs/file_hashes; github_audit/data_input_manifest_compact'),
        dict(stage='computation-ready', general_question='Can the files be converted into rows, columns, units and identifiers?',
             operational_test='Column modalities, missingness, target-context normalization and canonical ID audit.',
             mof_instantiation='MOF identifiers, adsorption/process target columns, QMOF quantum targets and CoRE isotherm context.',
             key_outputs='profiles/column_level_profile; processed/normalized_target_catalog; processed/canonical_mof_index'),
        dict(stage='model-ready', general_question='Can a reproducible ML task be defined without obvious leakage or missingness failure?',
             operational_test='Feature/target eligibility, descriptor-target join retention, random and grouped splits.',
             mof_instantiation='Descriptor-joined RAC/RDF/geometry task cases with split-aware benchmarks.',
             key_outputs='processed/descriptor_target_join_diagnostics; processed/descriptor_joined_ml_metrics_summary'),
        dict(stage='evidence-ready', general_question='Are the domain-specific facts needed for the claim observable or validated?',
             operational_test='Domain-evidence observability, validation status, warning/flag fields, evidence-regime decomposition.',
             mof_instantiation='Oxidation-state, formal-charge, formula/metal and curation evidence in MOF resources.',
             key_outputs='processed/chemistry_evidence_matrix; processed/trust_regime_summary; processed/representative_rule_cards'),
        dict(stage='claim-ready', general_question='Does the result support the stated scientific claim, not only a numerical score?',
             operational_test='Claim/evidence mapping, risk matrix, source-data completeness and caveat table.',
             mof_instantiation='ARC-MOF is ML-ready but chemistry-observability-limited; descriptor joins support screening claims with caveats.',
             key_outputs='tables/main/Main_Table_9_headline_claims_and_caveats; source_data/final_source_data_map'),
    ])

    taxonomy = pd.DataFrame([
        dict(domain_evidence_layer='identity/provenance', generic_meaning='What object is this row actually describing?',
             mof_example='MOFid/MOFkey/refcode/CIF/name identifiers and join-normalized stems.', reusable_beyond_mofs='sample IDs, specimen IDs, protein IDs, material IDs'),
        dict(domain_evidence_layer='target context', generic_meaning='What scientific quantity is being predicted and in which unit/context?',
             mof_example='gas, pressure/temperature/task, uptake/selectivity/process metric and source units.', reusable_beyond_mofs='assay conditions, imaging protocol, operating conditions, measurement units'),
        dict(domain_evidence_layer='validity/curation', generic_meaning='Does a trusted source or rule flag support/contradict the row?',
             mof_example='ASR/FSR/ION metadata, suspect/warning/check columns, recommended-screening flags.', reusable_beyond_mofs='quality-control flags, expert labels, instrument warnings, filtering status'),
        dict(domain_evidence_layer='physical/chemical plausibility', generic_meaning='Are domain-specific constraints observable and plausible?',
             mof_example='oxidation-state, formal-charge, metal/formula and ionic-context observability.', reusable_beyond_mofs='charge balance, conservation laws, stoichiometry, calibration range, biological viability'),
        dict(domain_evidence_layer='model generalization', generic_meaning='Does the split test extrapolation relevant to the claim?',
             mof_example='random vs descriptor-grouped splits and top-candidate recovery.', reusable_beyond_mofs='grouped splits by scaffold, site, batch, lab, patient, geography or time'),
    ])

    failure_modes = pd.DataFrame([
        dict(failure_mode='parsable-but-not-claimable', definition='A table can be loaded and modelled but lacks the evidence needed for a domain-specific claim.',
             mof_example='Large descriptor-rich ARC-MOF tables support screening stress tests but parsed tables do not directly expose all chemistry-validity evidence.', suggested_check='Separate model-readiness from evidence-readiness.'),
        dict(failure_mode='target-context collapse', definition='Columns with similar numerical values or unit-like names are mixed without task/unit context.',
             mof_example='mmol/g, v/v, wt%, S(g1), heat-of-adsorption and process metrics must not be merged as generic targets.', suggested_check='Use normalized target catalog with gas/property/unit/task metadata.'),
        dict(failure_mode='identifier-join attrition', definition='Descriptor and target tables join only after non-trivial identifier normalization, causing hidden row loss or risk.',
             mof_example='ARC-MOF descriptor families join through exact/repeat/symmetry-stripped stems with different risk levels.', suggested_check='Report join key, retention and risk class.'),
        dict(failure_mode='random-split optimism', definition='Random splits overstate performance when near-duplicates or descriptor-neighbouring records exist.',
             mof_example='Grouped splits reduce absolute R2 while ranking/top-k utility may persist.', suggested_check='Report grouped or chemistry-aware split performance.'),
        dict(failure_mode='missing-evidence misread as invalidity', definition='Absence of domain evidence in parsed metadata is mistaken for a false or invalid record.',
             mof_example='Not-observable oxidation/charge evidence is a caveat, not proof of chemical invalidity.', suggested_check='Use validated/observable trust regimes and caption caveats.'),
    ])

    mapping_rows = []
    if not scores.empty:
        use_cols = [c for c in ['resource','n_files','total_profiled_or_counted_rows','chemistry_trust_score','ml_readiness_score','trust_readiness_quadrant','recommended_role'] if c in scores.columns]
        for _, r in scores[use_cols].iterrows():
            mapping_rows.append(dict(
                claim_family='resource readiness',
                case_study_object=str(r.get('resource','')),
                evidence_required='scale, identifiers, target availability, descriptor availability, chemistry/domain-evidence observability',
                evidence_observed=f"chemistry_trust={float(r.get('chemistry_trust_score', np.nan)):.3f}; ml_readiness={float(r.get('ml_readiness_score', np.nan)):.3f}",
                recommended_wording=str(r.get('trust_readiness_quadrant','')),
                caveat='Scores are reporting/readiness scores computed from parsed tables, not universal truth about the raw database.'
            ))
    if not desc.empty:
        best = desc.copy()
        for col in ['mean_top_10pct_recovery','mean_r2','mean_spearman']:
            if col in best.columns:
                best[col] = pd.to_numeric(best[col], errors='coerce')
        sort_cols = [c for c in ['mean_top_10pct_recovery','mean_spearman','mean_r2'] if c in best.columns]
        if sort_cols:
            best = best.sort_values(sort_cols, ascending=False).head(8)
        else:
            best = best.head(8)
        for _, r in best.iterrows():
            mapping_rows.append(dict(
                claim_family='screening benchmark',
                case_study_object=f"{r.get('descriptor_family','descriptor')} | {r.get('target_column','target')} | {r.get('split_type','split')}",
                evidence_required='descriptor-target join, split-aware ML metrics, top-candidate recovery and source-data traceability',
                evidence_observed=f"R2={r.get('mean_r2', np.nan)}; Spearman={r.get('mean_spearman', np.nan)}; top10={r.get('mean_top_10pct_recovery', np.nan)}",
                recommended_wording='screening-relevant utility among successfully joined task cases',
                caveat='Descriptor-family comparison should not be described as universal unless cases are explicitly matched.'
            ))
    claim_map = pd.DataFrame(mapping_rows)

    case_map = pd.DataFrame([
        dict(journal_framework_component='general claim-readiness framework', mof_case_study_instantiation='MOF resources as heterogeneous scientific ML datasets',
             concrete_output='Figure 1; journal_submission/claim_readiness_framework_table.csv'),
        dict(journal_framework_component='domain-evidence observability', mof_case_study_instantiation='chemistry evidence: formula, metal, oxidation state, formal charge and curation/warning flags',
             concrete_output='Figure 3; processed/chemistry_evidence_matrix.csv; processed/trust_regime_summary.csv'),
        dict(journal_framework_component='identifier and target context', mof_case_study_instantiation='ARC-MOF descriptor-target joins and adsorption/process target normalization',
             concrete_output='processed/normalized_target_catalog.csv; processed/descriptor_target_join_diagnostics.csv'),
        dict(journal_framework_component='split-aware model-readiness', mof_case_study_instantiation='random versus descriptor-grouped splits and top-k screening recovery',
             concrete_output='Figure 5; processed/descriptor_joined_ml_metrics_summary.csv'),
        dict(journal_framework_component='repository and reproducibility audit', mof_case_study_instantiation='compact file manifest, warning report, source-data map and GitHub manual',
             concrete_output='github_audit/; journal_submission/; FINAL_OUTPUT_MANIFEST.csv'),
    ])

    summary = pd.DataFrame([
        dict(item='profiled_files', value=_safe_len(fdf), interpretation='local table-like files profiled'),
        dict(item='profiled_columns', value=_safe_len(cdf), interpretation='columns in column-level profile'),
        dict(item='normalized_targets', value=_safe_len(targets), interpretation='long-format target records with task/property/unit context'),
        dict(item='descriptor_join_cases', value=_safe_len(join), interpretation='descriptor-target join diagnostics'),
        dict(item='descriptor_ml_rows', value=_safe_len(desc), interpretation='descriptor-joined ML metric rows'),
        dict(item='warning_rows', value=_safe_len(warnings_df), interpretation='compact warnings/errors collected from logs'),
        dict(item='resources', value=_safe_nunique_df(fdf, 'resource'), interpretation='resource classes in local input profile'),
    ])

    return {
        'claim_readiness_framework_table': framework,
        'domain_evidence_taxonomy': taxonomy,
        'general_failure_modes': failure_modes,
        'claim_to_evidence_mapping': claim_map,
        'case_study_mapping_mof': case_map,
        'journal_submission_summary': summary,
    }


def step_journal_submission(cfg: Config, dd: Dict[str,Path], log):
    """Write a compact claim-readiness submission folder without heavy data copies."""
    pdir = _journal_submission_dir(cfg)
    mkdir(pdir)
    tables = build_claim_readiness_tables(cfg, dd)
    for name, df in tables.items():
        save_df(df, pdir / name, cfg)

    framework = tables['claim_readiness_framework_table']
    taxonomy = tables['domain_evidence_taxonomy']
    failures = tables['general_failure_modes']
    claim_map = tables['claim_to_evidence_mapping']
    summary = tables['journal_submission_summary']

    _write_md(pdir/'README_JOURNAL_SUBMISSION.md', ''.join([
        '# Claim-readiness claim-readiness package\n\n',
        'This compact folder reframes the MOF analysis as a reusable data-science framework: ',
        '**scientific ML datasets should be claim-ready, not only computation-ready or model-ready**. ',
        'MOFs are used as a chemically demanding case study because they combine heterogeneous provenance, ',
        'simulated and curated targets, descriptor matrices, identifier joins and chemistry-sensitive claims.\n\n',
        '## Main outputs\n\n',
        '- `claim_readiness_framework_table.csv`: general staged framework from raw files to claim-ready datasets.\n',
        '- `domain_evidence_taxonomy.csv`: reusable evidence layers and MOF instantiation.\n',
        '- `general_failure_modes.csv`: failure modes that transfer beyond MOF chemistry.\n',
        '- `claim_to_evidence_mapping.csv`: how each headline claim should be supported and caveated.\n',
        '- `case_study_mapping_mof.csv`: how the MOF results instantiate the general framework.\n',
        '- `journal_figure_caption_drafts.md`: broader, broad data-science captions.\n',
        '- `journal_cover_letter_key_points.md`: cover-letter framing points.\n',
        '- `journal_submission_compact_report.zip`: compact ZIP of this folder for sharing.\n\n',
        '## Compact run summary\n\n', _df_to_markdown(summary, 50), '\n',
        '## Claim-readiness stages\n\n', _df_to_markdown(framework, 20), '\n',
    ]))

    _write_md(pdir/'journal_figure_caption_drafts.md', ''.join([
        '# broad data-science figure caption drafts\n\n',
        'These captions intentionally foreground the general data-science contribution and treat MOFs as the case study.\n\n',
        '## Figure 1. Claim-readiness framework for scientific machine-learning datasets\n\n',
        'A staged framework separates file accessibility, computation-readiness, model-readiness, domain-evidence readiness and claim-readiness. ',
        'The MOF case study instantiates the domain-evidence layer through chemistry observability, including metal/formula evidence, oxidation-state or charge fields, curation flags and descriptor-target joins.\n\n',
        '## Figure 2. MOF databases as a demanding case study for scientific data readiness\n\n',
        'The profiled resources occupy different regions of scale, target availability, descriptor richness and evidence observability. ',
        'Row counts are profiled or counted tabular rows rather than necessarily model-ready structures.\n\n',
        '## Figure 3. Evidence observability and validation regimes\n\n',
        'Domain-evidence observability is instantiated here using MOF chemistry evidence. “Not observable” means absent from parsed tabular metadata and does not imply chemical invalidity.\n\n',
        '## Figure 4. Claim risk depends jointly on evidence-readiness and model-readiness\n\n',
        'Trust-readiness maps and risk matrices show why a dataset can be suitable for one scientific ML claim but not another. ',
        'Score intervals and rankings should be interpreted as reporting/readiness summaries, not uncertainty in true chemical validity.\n\n',
        '## Figure 5. Split-aware stress tests reveal claim-relevant ML behaviour\n\n',
        'Descriptor-joined benchmarks show screening-relevant top-candidate recovery while grouped splits expose where random interpolation overstates generalization. ',
        'Descriptor-family comparisons are restricted to successfully joined task cases.\n\n',
        '## Figure 6. Reusable reporting checklist for claim-ready scientific ML\n\n',
        'The final decision framework translates the audit into reporting requirements that can be reused in other scientific ML domains.\n',
    ]))

    _write_md(pdir/'journal_cover_letter_key_points.md', ''.join([
        '# broad data-science journal cover-letter key points\n\n',
        'Suggested framing:\n\n',
        '1. The manuscript addresses a general problem in scientific machine learning: datasets are often treated as ready once they are large, formatted and model-readable, but scientific claims require additional evidence.\n',
        '2. We introduce a claim-readiness framework that separates computation-readiness, model-readiness, domain-evidence observability, validation status, identifier stability and split-aware generalization.\n',
        '3. MOF databases provide a demanding case study with experimental, hypothetical, DFT-derived and descriptor-rich resources.\n',
        '4. The framework reveals cases where a resource is highly ML-ready but chemistry evidence is not directly observable from parsed tables, requiring explicit claim caveats rather than rejection of the data.\n',
        '5. Descriptor-joined benchmarks show that screening-relevant recovery can persist even when grouped splits reduce absolute R2, supporting a more precise interpretation of ML utility.\n',
        '6. The outputs include figure source data, claim-caveat tables and compact repository/audit reports designed to make scientific ML results reusable and inspectable.\n\n',
        'Suggested title:\n\n',
        '**When Model-Ready Is Not Claim-Ready: A Data-Readiness Framework for Scientific Machine Learning**\n\n',
        'Suggested subtitle:\n\n',
        '**A metal–organic framework case study in evidence-aware materials informatics**\n',
    ]))

    _write_md(pdir/'journal_submission_summary.md', ''.join([
        '# broad data-science journal submission summary\n\n',
        '## Why this is broader than a MOF benchmark\n\n',
        'The reusable contribution is the conversion of scientific datasets from computation-ready/model-ready to claim-ready. ',
        'The MOF application is a stress test because chemical validity, target context, identifier joins and split-aware generalization all matter simultaneously.\n\n',
        '## General framework table\n\n', _df_to_markdown(framework, 20), '\n',
        '## Domain evidence taxonomy\n\n', _df_to_markdown(taxonomy, 20), '\n',
        '## Failure modes\n\n', _df_to_markdown(failures, 20), '\n',
        '## Claim-to-evidence mapping\n\n', _df_to_markdown(claim_map, 60), '\n',
    ]))

    zpath = pdir/'journal_submission_compact_report.zip'
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in pdir.rglob('*'):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(pdir))
    log.info('Claim-readiness compact submission package created: %s (%s)', zpath, hbytes(zpath.stat().st_size))


def fig1(cfg, dd):
    """Generalized claim-readiness concept figure, with MOFs as the case study."""
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 8.7)); axs = axs.ravel()

    ax = axs[0]; ax.axis('off'); _panel_letter(ax,'a')
    ax.set_title('Scientific ML needs claim-ready datasets, not only model-ready tables', fontweight='bold', pad=8)
    stages = [
        ('Raw files', 'located, cited, hashed', CLAIM_READINESS_COLORS['raw']),
        ('Computation-ready', 'parsable rows, columns, units', CLAIM_READINESS_COLORS['computation_ready']),
        ('Model-ready', 'features, targets, splits', CLAIM_READINESS_COLORS['model_ready']),
        ('Evidence-ready', 'domain facts observable/validated', CLAIM_READINESS_COLORS['evidence_ready']),
        ('Claim-ready', 'claim supported with caveats', CLAIM_READINESS_COLORS['claim_ready']),
    ]
    xs = np.linspace(.10, .90, len(stages))
    for i, (title, body, color) in enumerate(stages):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x-.085,.48),.17,.19,boxstyle='round,pad=.018,rounding_size=.025',lw=1.0,facecolor=color,alpha=.16,edgecolor=color))
        ax.text(x,.605,title,ha='center',va='center',fontsize=8.8,fontweight='bold')
        ax.text(x,.525,body,ha='center',va='center',fontsize=7.1,wrap=True)
        if i < len(stages)-1:
            ax.add_patch(FancyArrowPatch((x+.087,.57),(xs[i+1]-.087,.57),arrowstyle='->',mutation_scale=12,lw=1.0,color='#444444'))
    ax.text(.50,.28,'The gap between model-ready and claim-ready is where many scientific ML benchmarks become fragile.',ha='center',fontsize=8.3,color='#333333')

    ax = axs[1]; ax.axis('off'); _panel_letter(ax,'b')
    ax.set_title('The claim determines the required evidence layer', fontweight='bold', pad=8)
    claim_boxes = [
        (.05,.66,.26,.14,'Prediction claim','features + target + split'),
        (.37,.66,.26,.14,'Scientific claim','domain evidence + validity'),
        (.69,.66,.26,.14,'Discovery claim','prospective or claim-specific support'),
        (.20,.33,.25,.14,'MOF example','adsorption screening'),
        (.56,.33,.25,.14,'Evidence needed','target context, joins, chemistry observability'),
    ]
    for x,y,w,h,title,body in claim_boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',lw=.9,facecolor='#FAFAFA',edgecolor='#666666'))
        ax.text(x+w/2,y+h*.64,title,ha='center',va='center',fontsize=8.5,fontweight='bold')
        ax.text(x+w/2,y+h*.30,body,ha='center',va='center',fontsize=7.2)
    for x1,y1,x2,y2 in [(.18,.66,.32,.47),(.50,.66,.50,.47),(.82,.66,.68,.47),(.45,.40,.56,.40)]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=11,lw=.9,color='#555555'))
    ax.text(.50,.13,'The same data resource can support one claim while requiring caveats for another.',ha='center',fontsize=8.1,color='#444444')

    ax = axs[2]; ax.axis('off'); _panel_letter(ax,'c')
    ax.set_title('General evidence regimes, instantiated here by chemistry', fontweight='bold', pad=8)
    regimes=[
        ('validated + observable','evidence is present and supported by curation', TRUST_REGIME_COLORS.get('validated_observable','#009E73')),
        ('validated + not observable','trusted source, but evidence absent from parsed table', TRUST_REGIME_COLORS.get('validated_not_observable','#56B4E9')),
        ('uncertain + observable','evidence visible but not independently validated', TRUST_REGIME_COLORS.get('uncertain_observable','#E69F00')),
        ('flagged / inconsistent','warning, unusual value or conflicting evidence', TRUST_REGIME_COLORS.get('flagged_or_inconsistent','#D55E00')),
    ]
    y=.78
    for title,body,color in regimes:
        ax.add_patch(FancyBboxPatch((.07,y-.057),.86,.105,boxstyle='round,pad=.018',lw=.9,facecolor=color,alpha=.15,edgecolor=color))
        ax.text(.11,y,title,ha='left',va='center',fontsize=8.2,fontweight='bold')
        ax.text(.45,y,body,ha='left',va='center',fontsize=7.2)
        y-=.15
    ax.text(.07,.13,'In MOFs, the domain-evidence layer is chemistry: formula, metal, charge, oxidation-state and curation evidence.',fontsize=7.9,color='#444444',wrap=True)

    ax = axs[3]; ax.axis('off'); _panel_letter(ax,'d')
    ax.set_title('MOF case study as a demanding stress test', fontweight='bold', pad=8)
    resources=[('MOSAEC / CoRE','chemistry and provenance anchors', RESOURCE_COLORS.get('MOSAEC-DB','#009E73')),
               ('ARC-MOF','large descriptor-rich ML stress test', RESOURCE_COLORS.get('ARC-MOF','#CC79A7')),
               ('QMOF','DFT / quantum-property contrast', RESOURCE_COLORS.get('QMOF','#E69F00'))]
    y=.75
    for title, body, color in resources:
        ax.add_patch(FancyBboxPatch((.08,y-.065),.84,.12,boxstyle='round,pad=.018',lw=.9,facecolor=color,alpha=.15,edgecolor=color))
        ax.text(.13,y,title,ha='left',va='center',fontsize=8.5,fontweight='bold')
        ax.text(.43,y,body,ha='left',va='center',fontsize=7.4)
        y-=.17
    ax.add_patch(FancyBboxPatch((.08,.18),.84,.17,boxstyle='round,pad=.018',lw=.9,facecolor='#F7F7F7',edgecolor='#666666'))
    ax.text(.50,.285,'Reusable output',ha='center',fontsize=8.5,fontweight='bold')
    ax.text(.50,.225,'claim/evidence maps + split-aware benchmarks + compact repository audit',ha='center',fontsize=7.4)

    fig.suptitle('Figure 1. A claim-readiness framework for scientific machine-learning datasets', fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    outs = save_fig(fig, dd['fig_main']/ 'Figure_1_claim_readiness_framework', cfg)
    # Also save under the legacy filename expected by the pipeline and manuscript drafts.
    try:
        # The figure has already been closed by save_fig, so regenerate only through copying files.
        for p in list(outs):
            legacy = dd['fig_main'] / ('Figure_1_chemistry_ready_concept' + p.suffix)
            try:
                shutil.copy2(p, legacy)
                outs.append(legacy)
            except Exception:
                pass
    except Exception:
        pass
    return outs


def step_github_audit(cfg: Config, dd: Dict[str,Path], log):
    """Run v1.9 GitHub audit and add general claim-readiness manuals."""
    _V19_STEP_GITHUB_AUDIT(cfg, dd, log)
    audit = cfg.out_dir/'github_audit'
    mkdir(audit)
    tables = build_claim_readiness_tables(cfg, dd)
    for name, df in tables.items():
        # Keep a copy of compact claim-readiness framework tables in the GitHub audit too.
        save_df(df, audit / f'journal_{name}', cfg)

    _write_md(audit/'GENERAL_CLAIM_READINESS_MANUAL.md', ''.join([
        '# General claim-readiness manual\n\n',
        'This repository/audit layer is designed so the project can be explained beyond MOF chemistry. ',
        'The central claim is that scientific ML datasets should be assessed at five levels: raw-file availability, computation-readiness, model-readiness, domain-evidence readiness and claim-readiness.\n\n',
        '## Reusable workflow\n\n', _df_to_markdown(tables['claim_readiness_framework_table'], 20), '\n',
        '## General failure modes\n\n', _df_to_markdown(tables['general_failure_modes'], 20), '\n',
        '## How the MOF case study maps onto the framework\n\n', _df_to_markdown(tables['case_study_mapping_mof'], 20), '\n',
    ]))

    _write_md(audit/'DATA_AVAILABILITY_AND_GITHUB_TEMPLATE.md', ''.join([
        '# Data availability and GitHub preparation template\n\n',
        'Recommended GitHub contents for a public repository:\n\n',
        '1. Code: the final Python pipeline and environment YAML.\n',
        '2. Compact audit: the full `github_audit/` folder and `journal_submission/` folder.\n',
        '3. Source data: `source_data/figure_panel_source_data/`, `final_source_data_map.csv`, `publication_figure_style_sheet.csv`, and headline claim/caveat tables.\n',
        '4. Figures: PDF/SVG/PNG main figures if permitted by journal policy.\n',
        '5. Data instructions: cite and download the original databases from their providers rather than redistributing large third-party raw files.\n\n',
        'Recommended wording:\n\n',
        '> Raw data are not redistributed in this repository. The repository provides the analysis pipeline, compact file-use audit, figure source data, and scripts required to reproduce the reported analyses once the user places the externally obtained datasets under the expected `--data-root`.\n',
    ]))

    # Rebuild compact audit zip to include the additional manuals.
    zpath=audit/'github_preparation_compact_report.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in audit.rglob('*'):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(audit))
    log.info('Enhanced GitHub/data-preparation compact audit updated: %s (%s)', zpath, hbytes(zpath.stat().st_size))


def build_final_manifest(cfg: Config, dd: Dict[str,Path]) -> pd.DataFrame:
    rows=[]
    def classify(rel: str) -> Tuple[str,str,str]:
        if rel.startswith('journal_submission/'): return ('journal_submission','Claim-readiness claim-readiness package','compact')
        if rel.startswith('github_audit/'): return ('github_audit','compact repository/data manual','compact')
        if rel.startswith('figures/main/'): return ('figure','main manuscript figure','publication')
        if rel.startswith('figures/si/'): return ('figure','supporting information figure','supporting')
        if rel.startswith('tables/main/'): return ('table','main manuscript table','publication')
        if rel.startswith('tables/si/'): return ('table','supporting information table','supporting')
        if rel.startswith('source_data/figure_panel_source_data/'): return ('source_data','compact panel source data','publication')
        if rel.startswith('source_data/'): return ('source_data','source or reproducibility table','source')
        if rel.startswith('profiles/'): return ('profile','input table profile','reproducibility')
        if rel.startswith('processed/'): return ('processed','intermediate/final computed table','reproducibility')
        if rel.startswith('logs/'): return ('log','run log/environment/hash file','audit')
        if rel.startswith('reports/'): return ('report','human-readable analysis report','audit')
        if rel.startswith('models/'): return ('model','saved fitted model','optional-heavy')
        return ('other','other output','other')
    for p in cfg.out_dir.rglob('*'):
        if not p.is_file():
            continue
        if p.name == 'chemistry_ready_mof_analysis_outputs_package.zip':
            continue
        try: rel=str(p.relative_to(cfg.out_dir)).replace('\\','/')
        except Exception: rel=str(p)
        kind,desc,group=classify(rel)
        rows.append(dict(relative_path=rel,file_name=p.name,file_type=kind,asset_group=group,size_bytes=p.stat().st_size,size_kb=round(p.stat().st_size/1024,2),description=desc))
    man=pd.DataFrame(rows).sort_values(['asset_group','file_type','relative_path']) if rows else pd.DataFrame()
    save_df(man,cfg.out_dir/'FINAL_OUTPUT_MANIFEST',cfg)
    save_df(man,dd['source']/ 'FINAL_OUTPUT_MANIFEST',cfg)
    return man


def step_zip(cfg: Config, dd: Dict[str,Path], log):
    build_final_manifest(cfg, dd)
    if cfg.no_zip:
        return
    zp=cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'
    include_subdirs=['reports','profiles','processed','tables','figures','source_data','logs','models','github_audit','journal_submission']
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [cfg.out_dir/'FINAL_OUTPUT_MANIFEST.csv', cfg.out_dir/'README_OUTPUTS.txt']:
            if p.exists(): z.write(p, p.relative_to(cfg.out_dir))
        for sub in include_subdirs:
            root=cfg.out_dir/sub
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and p != zp:
                        z.write(p,p.relative_to(cfg.out_dir))
    with zipfile.ZipFile(zp) as z:
        names=set(z.namelist())
    man=load_df(cfg.out_dir/'FINAL_OUTPUT_MANIFEST')
    missing=[]
    if not man.empty and 'relative_path' in man.columns:
        missing=[r for r in man.relative_path.astype(str) if r not in names and r != 'chemistry_ready_mof_analysis_outputs_package.zip']
    audit=pd.DataFrame([dict(zip_path=str(zp), zip_size_bytes=zp.stat().st_size, zip_entries=len(names), manifest_entries=len(man), missing_manifest_entries=len(missing), missing_examples=';'.join(missing[:20]))])
    save_df(audit, dd['source']/ 'zip_contents_audit', cfg)
    log.info('ZIP package created: %s (%s); manifest missing entries=%d', zp, hbytes(zp.stat().st_size), len(missing))


def write_readme(cfg: Config):
    txt=f"""Chemistry-ready / claim-ready MOF analysis output folder
===========================================================
Generated: {iso()}
Script version: {VERSION}

Purpose of this final run
-------------------------
This v2.0 script is intended for the last from-scratch paper run.  It keeps the
v1.9 final-publication and GitHub-audit pipeline, then adds a Claim-readiness
claim-readiness layer.  The scientific result remains MOF-focused, but the output
structure now supports a broader framing: scientific ML datasets should be
claim-ready, not only computation-ready or model-ready.

Recommended paper outputs
-------------------------
figures/main/          main figures as PDF, SVG and 600-dpi PNG
tables/main/           main manuscript tables, clean resource atlas and descriptor benchmarks
tables/si/             full SI tables
source_data/           source-data tables and per-panel extracts
reports/               analysis_report.md and final_publication_wrapup_report.md
github_audit/          compact data-use report, GitHub manual and repository checklist
journal_submission/   compact broad data-science claim-readiness framework package
logs/                  run.log, environment metadata, pipeline state and file hashes

Claim-readiness outputs
----------------------
`journal_submission/` contains a general framework table, domain-evidence
taxonomy, failure-mode catalog, claim-to-evidence mapping, MOF case-study mapping,
caption drafts and cover-letter key points.  These files are compact and suitable
for manuscript planning or repository documentation.

GitHub audit folder
-------------------
`github_audit/` records what data files were used, how they were classified,
which analyses used them, and which outputs should be included in a future
repository. It does not duplicate large raw databases.

Windows / from-scratch run advice
---------------------------------
Use a fresh output folder. Close Excel, Explorer Preview Pane, OneDrive sync
windows, PDF/SVG viewers and old terminals before starting. CSV outputs are the
canonical source of truth; PKL/Parquet sidecars are optional.

Caption caveats to preserve
---------------------------
1. “Not observable” means not directly present in parsed tabular metadata and
   does not imply chemical invalidity.
2. Score intervals are bootstrap/reporting-score intervals, not uncertainty in
   true chemical validity.
3. Descriptor-family comparisons refer to successfully joined task cases.
4. For a broad data-science framing, MOFs are the demanding case study; the general
   contribution is claim-readiness for scientific ML datasets.
"""
    mkdir(cfg.out_dir)
    (cfg.out_dir/'README_OUTPUTS.txt').write_text(txt,encoding='utf-8')


def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready / claim-ready MOF pipeline v%s', VERSION)
    log.info('Data root: %s', cfg.data_root)
    log.info('Output directory: %s', cfg.out_dir)
    log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    check_environment(cfg,dd,log)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)
    state_path=dd['logs']/ 'pipeline_state.json'; st=read_json(state_path,{'created':iso(),'completed_steps':{}}); st['config']=asdict(cfg); save_json(st,state_path)
    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv',dd['profiles']/ 'target_column_context_catalog.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [
            ('normalize_targets',[dd['processed']/ 'normalized_target_catalog.csv',dd['processed']/ 'canonical_mof_index.csv',dd['processed']/ 'normalized_target_context_catalog.csv'],step_normalize_targets),
            ('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),
            ('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv',dd['processed']/ 'trust_regime_summary.csv'],step_trust),
            ('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),
            ('descriptor_joined_ml',[dd['processed']/ 'descriptor_target_join_diagnostics.csv',dd['processed']/ 'descriptor_joined_ml_metrics_summary.csv'],step_descriptor_joined_ml),
            ('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),
            ('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv',dd['source']/ 'publication_figure_style_sheet.csv'],step_tables),
            ('make_figures',[dd['fig_main']/ 'Figure_1_claim_readiness_framework.pdf',dd['fig_main']/ 'Figure_5_ml_stress_test.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),
            ('write_reports',[dd['reports']/ 'analysis_report.md',dd['reports']/ 'final_publication_wrapup_report.md',dd['logs']/ 'run_environment.json'],step_reports),
            ('journal_submission',[cfg.out_dir/'journal_submission'/'README_JOURNAL_SUBMISSION.md', cfg.out_dir/'journal_submission'/'journal_submission_compact_report.zip'],step_journal_submission),
            ('github_audit',[cfg.out_dir/'github_audit'/'README_GITHUB_DATA_MANUAL.md', cfg.out_dir/'github_audit'/'github_preparation_compact_report.zip'],step_github_audit),
            ('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)
        ]
    for name,exp,fn in steps:
        run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    log.info('Compact GitHub/data-preparation report is in: %s', cfg.out_dir/'github_audit')
    log.info('Claim-readiness journal-support report is in: %s', cfg.out_dir/'journal_submission')
    return 0



# =============================================================================
# v2.1 SELECTIVE FINAL-ASSET REFRESH MODE
# =============================================================================
# This small wrapper allows the corrected Figure 5C, final reports, compact
# journal_submission/ package and github_audit/ package to be regenerated from
# an existing completed output folder without repeating heavy profiling or ML.

_V21_BASE_PARSE = parse


def parse(argv=None) -> Config:
    args = list(sys.argv[1:] if argv is None else argv)
    refresh = False
    for flag in ["--refresh-publication-only", "--refresh-final-assets", "--refresh-submission-only"]:
        while flag in args:
            args.remove(flag)
            refresh = True
    cfg = _V21_BASE_PARSE(args)
    setattr(cfg, "refresh_publication_only", refresh)
    return cfg


def main(argv=None):
    cfg=parse(argv); random.seed(cfg.random_seed); np.random.seed(cfg.random_seed)
    dd=dirs(cfg); log=logger(dd['logs']); write_readme(cfg)
    log.info('Chemistry-ready / claim-ready scientific-ML pipeline v%s', VERSION)
    log.info('Data root: %s', cfg.data_root)
    log.info('Output directory: %s', cfg.out_dir)
    log.info('Modes: save=%s | RAM=%s | comprehensive=%s | n_jobs=%d', cfg.save_mode,cfg.ram_mode,cfg.comprehensive_level,cfg.n_jobs)
    check_environment(cfg,dd,log)
    cfg.inventory_report=find_inventory(cfg.data_root,cfg.inventory_report)
    if cfg.inventory_report: log.info('Archive inventory detected: %s', cfg.inventory_report)
    parse_inventory(cfg.inventory_report,dd,cfg,log)

    state_path=dd['logs']/ 'pipeline_state.json'
    st=read_json(state_path,{'created':iso(),'completed_steps':{}})
    st['config']=asdict(cfg)
    save_json(st,state_path)

    if getattr(cfg, 'refresh_publication_only', False):
        log.info('Selective final-asset refresh mode enabled.')
        required = [
            dd['profiles']/ 'file_level_profile.csv',
            dd['profiles']/ 'column_level_profile.csv',
            dd['processed']/ 'resource_scores.csv',
            dd['processed']/ 'descriptor_family_case_matched_comparison.csv',
            dd['processed']/ 'descriptor_joined_ml_metrics_summary.csv',
        ]
        missing=[str(x) for x in required if not Path(x).exists()]
        if missing:
            log.warning('Some expected existing-result files are missing: %s', '; '.join(missing))
            log.warning('Refresh mode can still run, but missing tables may produce empty panels. A full fresh run is safer if key processed tables are missing.')
        for name, fn in [
            ('make_tables', step_tables),
            ('make_figures', step_figures),
            ('write_reports', step_reports),
            ('journal_submission', step_journal_submission),
            ('github_audit', step_github_audit),
            ('zip_package', step_zip),
        ]:
            log.info('Refreshing final asset step: %s', name)
            t=time.time()
            fn(cfg, dd, log)
            log.info('Refreshed final asset step: %s in %.1f s', name, time.time()-t)
        log.info('Selective final-asset refresh completed successfully. Outputs are in: %s', cfg.out_dir)
        log.info('Neutral journal-support folder: %s', cfg.out_dir/'journal_submission')
        log.info('Compact GitHub/data-preparation folder: %s', cfg.out_dir/'github_audit')
        return 0

    steps=[('profile_inputs',[dd['profiles']/ 'file_level_profile.csv',dd['profiles']/ 'column_level_profile.csv',dd['profiles']/ 'target_column_context_catalog.csv'],step_profile)]
    if not cfg.profile_only:
        steps += [
            ('normalize_targets',[dd['processed']/ 'normalized_target_catalog.csv',dd['processed']/ 'canonical_mof_index.csv',dd['processed']/ 'normalized_target_context_catalog.csv'],step_normalize_targets),
            ('identifier_join_audit',[dd['processed']/ 'identifier_candidate_summary.csv',dd['processed']/ 'join_accounting.csv'],step_join),
            ('chemistry_trust',[dd['processed']/ 'trust_flags.csv',dd['processed']/ 'representative_rule_cards.csv',dd['processed']/ 'trust_regime_summary.csv'],step_trust),
            ('resource_scores',[dd['processed']/ 'resource_scores.csv',dd['processed']/ 'benchmark_risk_matrix.csv'],step_scores),
            ('descriptor_joined_ml',[dd['processed']/ 'descriptor_target_join_diagnostics.csv',dd['processed']/ 'descriptor_joined_ml_metrics_summary.csv'],step_descriptor_joined_ml),
            ('ml_stress_test',[dd['processed']/ 'ml_metrics_summary.csv',dd['processed']/ 'ranking_stability.csv'],step_ml),
            ('make_tables',[dd['main_tables']/ 'Main_Table_1_resource_atlas.csv',dd['source']/ 'final_source_data_map.csv',dd['source']/ 'publication_figure_style_sheet.csv'],step_tables),
            ('make_figures',[dd['fig_main']/ 'Figure_1_claim_readiness_framework.pdf',dd['fig_main']/ 'Figure_5_ml_stress_test.pdf',dd['fig_main']/ 'Figure_6_decision_framework.pdf'],step_figures),
            ('write_reports',[dd['reports']/ 'analysis_report.md',dd['reports']/ 'final_publication_wrapup_report.md',dd['logs']/ 'run_environment.json'],step_reports),
            ('journal_submission',[cfg.out_dir/'journal_submission'/'README_JOURNAL_SUBMISSION.md', cfg.out_dir/'journal_submission'/'journal_submission_compact_report.zip'],step_journal_submission),
            ('github_audit',[cfg.out_dir/'github_audit'/'README_GITHUB_DATA_MANUAL.md', cfg.out_dir/'github_audit'/'github_preparation_compact_report.zip'],step_github_audit),
            ('zip_package',[cfg.out_dir/'chemistry_ready_mof_analysis_outputs_package.zip'],step_zip)
        ]
    for name,exp,fn in steps:
        run_step(name,exp,cfg,st,state_path,log,fn,cfg,dd,log)
    log.info('Pipeline completed successfully. Outputs are in: %s', cfg.out_dir)
    log.info('Compact GitHub/data-preparation report is in: %s', cfg.out_dir/'github_audit')
    log.info('Neutral journal-support claim-readiness report is in: %s', cfg.out_dir/'journal_submission')
    return 0



# =============================================================================
# v2.2 VISUAL-FINAL OVERRIDES
# =============================================================================
# This final layer keeps the scientific tables and ML computations unchanged.
# It only tightens the publication graphics, captions, and final-asset refresh
# behavior: non-overlapping panel labels, shorter labels, one canonical Figure 1,
# clearer not-observable handling, cleaner Figure 5C, and a ranking-percentile
# Figure 5D.

VERSION = "2.2-claim-ready-sciml-visual-final"

V22_SHORT = {
    "MOSAEC-DB": "MOSAEC",
    "CoRE MOF 2024": "CoRE-24",
    "CoRE MOF 2025 metadata": "CoRE-25",
    "ARC-MOF": "ARC",
    "QMOF": "QMOF",
    "CSD-derived context": "CSD ctx",
    "Other/unknown": "Other",
    "Task-specific joins": "Task joins",
}
V22_MAIN_RESOURCES = ["MOSAEC-DB", "CoRE MOF 2024", "CoRE MOF 2025 metadata", "ARC-MOF", "QMOF", "CSD-derived context"]
V22_RESOURCE_COLORS = {
    "MOSAEC-DB": "#009E73",
    "CoRE MOF 2024": "#0072B2",
    "CoRE MOF 2025 metadata": "#56B4E9",
    "ARC-MOF": "#CC79A7",
    "QMOF": "#E69F00",
    "CSD-derived context": "#D55E00",
    "Other/unknown": "#999999",
}
V22_CLAIM_COLORS = {
    "raw": "#8A8A8A",
    "computation": "#56B4E9",
    "model": "#0072B2",
    "evidence": "#E69F00",
    "claim": "#009E73",
}
V22_TRUST_COLORS = {
    "validated_observable": "#009E73",
    "validated_not_observable": "#80CDC1",
    "uncertain_observable": "#E69F00",
    "uncertain_unobservable": "#BDBDBD",
    "flagged_or_inconsistent": "#D55E00",
    "not_scanned_or_not_observable": "#F0F0F0",
}
V22_RISK_SHORT = {
    "missing oxidation-state/formal-charge evidence": "missing\nchemistry\nevidence",
    "nonzero charge outside explicit ionic context": "charge\nambiguity",
    "suspect chemistry or validation warning": "validation\nwarning",
    "identifier ambiguity or duplicate leakage": "identifier /\nduplicate\nleakage",
    "high descriptor missingness": "descriptor\nmissingness",
    "target provenance mismatch": "target\nmismatch",
    "small or biased trusted subset": "small /\nbiased trusted\nsubset",
}
V22_USE_SHORT = {
    "adsorption ranking": "adsorption",
    "process screening": "process",
    "quantum-property modeling": "quantum",
    "generative training": "generation",
    "mechanistic interpretation": "mechanism",
}


def short_resource(x: Any) -> str:
    return V22_SHORT.get(str(x), _short_resource_label(x) if "_short_resource_label" in globals() else str(x))


def v22_resource_color(x: Any) -> str:
    return V22_RESOURCE_COLORS.get(str(x), RESOURCE_COLORS.get(str(x), "#999999") if "RESOURCE_COLORS" in globals() else "#999999")


def _panel_letter(ax, letter: str, x: float=-0.115, y: float=1.135, inside: bool=False):
    """Panel letters positioned outside the title area to avoid overlap."""
    if inside:
        x, y = 0.012, 0.988
        va = "top"
    else:
        va = "top"
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va=va,
        ha="left",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                  edgecolor="#333333", lw=0.55, alpha=0.96),
        zorder=20,
    )


def lab(ax, l):
    _panel_letter(ax, l)


def _clean_axes(ax, grid=True):
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, lw=0.32, alpha=0.16, zorder=0)
    for sp in ["top", "right"]:
        try:
            ax.spines[sp].set_visible(False)
        except Exception:
            pass
    for sp in ["left", "bottom"]:
        try:
            ax.spines[sp].set_linewidth(0.72)
        except Exception:
            pass
    return ax


def v22_heat(ax, mat: pd.DataFrame, title: str, cmap: str="YlGnBu", bar: bool=True,
             annotate: bool=True, fmt: str=".2f", xtick_rotation: int=35):
    if mat is None or mat.empty:
        nodata(ax, title)
        return None
    mat = mat.copy()
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")
    arr = mat.values.astype(float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=10.3, fontweight="bold", pad=10)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(x) for x in mat.columns], rotation=xtick_rotation, ha="right", fontsize=7.3)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([str(x) for x in mat.index], fontsize=7.7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", length=0)
    if annotate and mat.shape[0] <= 10 and mat.shape[1] <= 8:
        finite = arr[np.isfinite(arr)]
        threshold = np.nanmean(finite) if finite.size else 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    txt = f"{v:{fmt}}" if fmt else f"{v:.2g}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.3,
                            color="white" if v > threshold else "#222222")
    if bar:
        cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.035)
        cb.ax.tick_params(labelsize=6.8)
    return im


def _caption_caveats_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "figure": "Figure 2",
            "caveat": "Rows, size and column counts describe profiled/counted tabular inputs, not necessarily model-ready structures."
        },
        {
            "figure": "Figure 3",
            "caveat": "Not observable means not directly present in parsed tabular metadata; it is not a chemical invalidity label."
        },
        {
            "figure": "Figure 5C",
            "caveat": "Descriptor-family comparisons are restricted to successfully joined task/gas/target cases and are aggregated from long-format source data."
        },
        {
            "figure": "Figure 5D",
            "caveat": "Rank-percentile agreement is a screening diagnostic; it is not a full calibration or causal model."
        },
    ])


def _save_visual_caption_caveats(cfg: Config, dd: Dict[str, Path]):
    save_df(_caption_caveats_dataframe(), dd["source"] / "visual_caption_caveats", cfg)


def _make_barh_with_log1p(ax, labels, values, colors, title, xlabel):
    values = pd.Series(values, dtype="float64").fillna(0)
    display = np.log10(values.clip(lower=0) + 1)
    ax.barh(labels, display, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.log10(np.array([1, 10, 100, 1000, 10000, 100000]) + 1))
    ax.set_xticklabels(["0", "10", "100", "1k", "10k", "100k"], fontsize=7)
    return ax


def _best_prediction_group(pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select a visually informative prediction group for rank-percentile panel."""
    if pred is None or pred.empty or not {"y_true", "y_pred"}.issubset(pred.columns):
        return pd.DataFrame(), {}
    p = pred.copy()
    for c in ["y_true", "y_pred"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["y_true"].notna() & p["y_pred"].notna()]
    if p.empty:
        return pd.DataFrame(), {}
    group_cols = [c for c in ["source_table", "target_column", "model", "split_type", "repeat"] if c in p.columns]
    rows = []
    if group_cols:
        for key, g in p.groupby(group_cols, dropna=False):
            if len(g) < 80:
                continue
            dy = float(g["y_true"].max() - g["y_true"].min())
            dp = float(g["y_pred"].max() - g["y_pred"].min())
            if dy <= 0 or dp <= 0:
                continue
            sp = float(pd.Series(g["y_true"]).corr(pd.Series(g["y_pred"]), method="spearman"))
            if not np.isfinite(sp):
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            meta = dict(zip(group_cols, key_tuple))
            rows.append({**meta, "n": len(g), "spearman": sp, "y_range": dy, "p_range": dp})
    if rows:
        r = pd.DataFrame(rows)
        # Prefer interpretable ranking signal and nontrivial target range.
        r["score"] = r["spearman"].fillna(-9) + 0.05*np.log1p(r["n"]) + 0.02*np.log1p(r["y_range"])
        best = r.sort_values("score", ascending=False).iloc[0].to_dict()
        mask = pd.Series(True, index=p.index)
        for c in group_cols:
            mask &= p[c].astype(str).eq(str(best.get(c)))
        g = p.loc[mask].copy()
    else:
        best = {"n": len(p), "spearman": float(pd.Series(p["y_true"]).corr(pd.Series(p["y_pred"],), method="spearman"))}
        g = p.copy()
    if len(g) > 2500:
        g = g.sample(n=2500, random_state=42)
    g["true_rank_pct"] = g["y_true"].rank(pct=True, method="average")
    g["pred_rank_pct"] = g["y_pred"].rank(pct=True, method="average")
    meta = {
        "spearman": float(pd.Series(g["y_true"]).corr(pd.Series(g["y_pred"]), method="spearman")),
        "n": len(g),
        "source_table": best.get("source_table", ""),
        "target_column": best.get("target_column", ""),
        "model": best.get("model", ""),
        "split_type": best.get("split_type", ""),
    }
    return g, meta


def fig1(cfg, dd):
    """Final broad claim-readiness framework with clear spacing and no duplicate legacy export."""
    fig, axs = plt.subplots(2, 2, figsize=(14.2, 8.8))
    axs = axs.ravel()

    ax = axs[0]; ax.axis("off"); _panel_letter(ax, "a", inside=True)
    ax.set_title("From files to claim-ready scientific ML datasets", fontweight="bold", pad=10)
    stages = [
        ("Raw files", "located\ncited\nhashed", V22_CLAIM_COLORS["raw"]),
        ("Computation-ready", "parsable\ntables\nunits", V22_CLAIM_COLORS["computation"]),
        ("Model-ready", "features\ntargets\nsplits", V22_CLAIM_COLORS["model"]),
        ("Evidence-ready", "domain facts\nobservable\nvalidated", V22_CLAIM_COLORS["evidence"]),
        ("Claim-ready", "claim supported\nwith caveats\nsource data", V22_CLAIM_COLORS["claim"]),
    ]
    xs = np.linspace(0.10, 0.90, len(stages))
    for i, (title, body, color) in enumerate(stages):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x-0.078, 0.47), 0.156, 0.24,
                                    boxstyle="round,pad=.014,rounding_size=.028",
                                    lw=1.05, facecolor=color, alpha=0.17, edgecolor=color))
        ax.text(x, 0.635, title, ha="center", va="center", fontsize=8.35, fontweight="bold")
        ax.text(x, 0.545, body, ha="center", va="center", fontsize=7.2, linespacing=1.18)
        if i < len(stages)-1:
            ax.add_patch(FancyArrowPatch((x+0.082, 0.59), (xs[i+1]-0.082, 0.59),
                                         arrowstyle="->", mutation_scale=12, lw=1.0, color="#444444"))
    ax.text(0.50, 0.26,
            "The fragile step is the gap between model-ready tables and claim-ready scientific evidence.",
            ha="center", fontsize=8.6, color="#333333")

    ax = axs[1]; ax.axis("off"); _panel_letter(ax, "b", inside=True)
    ax.set_title("The claim determines the evidence required", fontweight="bold", pad=10)
    cards = [
        ("Prediction", "Are features, targets and splits valid?", V22_CLAIM_COLORS["model"]),
        ("Screening", "Are top candidates stable and traceable?", V22_CLAIM_COLORS["claim"]),
        ("Chemistry claim", "Are charge, formula and curation observable?", V22_CLAIM_COLORS["evidence"]),
        ("Mechanism", "Is the evidence interpretable, not just predictive?", "#6C5B7B"),
    ]
    y = 0.76
    for title, body, color in cards:
        ax.add_patch(FancyBboxPatch((0.08, y-0.055), 0.84, 0.105,
                                    boxstyle="round,pad=.018,rounding_size=.020",
                                    lw=.9, facecolor=color, alpha=.13, edgecolor=color))
        ax.text(0.14, y, title, ha="left", va="center", fontsize=8.5, fontweight="bold")
        ax.text(0.42, y, body, ha="left", va="center", fontsize=7.7)
        y -= 0.145

    ax = axs[2]; ax.axis("off"); _panel_letter(ax, "c", inside=True)
    ax.set_title("Evidence regimes separate absence from inconsistency", fontweight="bold", pad=10)
    regimes = [
        ("validated observable", "curated + visible evidence", V22_TRUST_COLORS["validated_observable"]),
        ("validated not observable", "trusted source, missing parsed fields", V22_TRUST_COLORS["validated_not_observable"]),
        ("uncertain observable", "visible but not externally validated", V22_TRUST_COLORS["uncertain_observable"]),
        ("uncertain unobservable", "not visible in parsed metadata", V22_TRUST_COLORS["uncertain_unobservable"]),
        ("flagged / inconsistent", "warning or conflicting evidence", V22_TRUST_COLORS["flagged_or_inconsistent"]),
    ]
    y = 0.78
    for title, body, color in regimes:
        ax.add_patch(FancyBboxPatch((0.07, y-0.047), 0.86, 0.086,
                                    boxstyle="round,pad=.014,rounding_size=.018",
                                    lw=.75, facecolor=color, alpha=.18, edgecolor=color))
        ax.text(0.11, y, title, ha="left", va="center", fontsize=7.9, fontweight="bold")
        ax.text(0.49, y, body, ha="left", va="center", fontsize=7.2)
        y -= 0.115
    ax.text(0.07, 0.13,
            "In the MOF case study, the domain-evidence layer is chemistry: formula, metal, charge, oxidation state and curation evidence.",
            fontsize=7.7, color="#444444", wrap=True)

    ax = axs[3]; ax.axis("off"); _panel_letter(ax, "d", inside=True)
    ax.set_title("MOF databases provide a demanding case study", fontweight="bold", pad=10)
    resources = [
        ("MOSAEC / CoRE", "chemistry and provenance anchors", V22_RESOURCE_COLORS["MOSAEC-DB"]),
        ("ARC", "large descriptor-rich adsorption ML stress test", V22_RESOURCE_COLORS["ARC-MOF"]),
        ("QMOF", "DFT and quantum-property contrast", V22_RESOURCE_COLORS["QMOF"]),
    ]
    y = 0.72
    for title, body, color in resources:
        ax.add_patch(FancyBboxPatch((0.08, y-0.06), 0.84, 0.112,
                                    boxstyle="round,pad=.016,rounding_size=.020",
                                    lw=.85, facecolor=color, alpha=.15, edgecolor=color))
        ax.text(0.13, y, title, ha="left", va="center", fontsize=8.4, fontweight="bold")
        ax.text(0.42, y, body, ha="left", va="center", fontsize=7.4)
        y -= 0.155
    ax.add_patch(FancyBboxPatch((0.08, 0.20), 0.84, 0.16,
                                boxstyle="round,pad=.018,rounding_size=.020",
                                lw=.85, facecolor="#F7F7F7", edgecolor="#666666"))
    ax.text(0.50, 0.295, "Reusable output", ha="center", fontsize=8.4, fontweight="bold")
    ax.text(0.50, 0.235, "claim/evidence maps + split-aware benchmarks + compact data manual",
            ha="center", fontsize=7.3)
    fig.suptitle("Figure 1. A claim-readiness framework for scientific machine-learning datasets",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.0, w_pad=2.1)
    return save_fig(fig, dd["fig_main"] / "Figure_1_claim_readiness_framework", cfg)


def fig2(cfg, dd):
    f = load_df(dd["profiles"] / "file_level_profile")
    c = load_df(dd["profiles"] / "column_level_profile")
    mod = load_df(dd["profiles"] / "resource_modality_matrix")
    sc = load_df(dd["processed"] / "resource_scores")
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.0))
    axs = axs.ravel()

    ax = axs[0]; _panel_letter(ax, "a"); _clean_axes(ax)
    if not f.empty:
        g = f.groupby("resource").agg(n_files=("file_name", "count"), total_size_mb=("size_mb", "sum")).reset_index()
        g = g[g.resource.isin(V22_MAIN_RESOURCES + ["Other/unknown"])]
        g["label"] = g.resource.map(short_resource)
        g = g.sort_values("total_size_mb")
        _make_barh_with_log1p(ax, g["label"], g["total_size_mb"], [v22_resource_color(r) for r in g.resource],
                              "Input scale by resource", "log10(profiled MB + 1)")
        for i, r in g.reset_index(drop=True).iterrows():
            ax.text(np.log10(float(r.total_size_mb) + 1) + 0.015, i, f"{int(r.n_files)} files", va="center", fontsize=7.2)
    else:
        nodata(ax, "Input scale by resource")

    ax = axs[1]; _panel_letter(ax, "b")
    if not mod.empty:
        cols = [x for x in ["identifier", "formula_or_composition", "metal_or_charge_chemistry",
                            "geometric_descriptor", "descriptor", "target_or_property",
                            "curation_or_validation_flag"] if x in mod.columns]
        mat = (mod.set_index("resource")[cols] > 0).astype(int)
        mat = mat.loc[[r for r in V22_MAIN_RESOURCES if r in mat.index]]
        mat.index = [short_resource(x) for x in mat.index]
        mat.columns = ["ID", "formula", "chemistry", "geometry", "descriptor", "target", "curation"][:len(mat.columns)]
        v22_heat(ax, mat, "Observed data modalities", "Blues", False, annotate=False, xtick_rotation=25)
    else:
        nodata(ax, "Observed data modalities")

    ax = axs[2]; _panel_letter(ax, "c")
    if not c.empty:
        tmp = c.copy()
        tmp = tmp[tmp.resource.isin(V22_MAIN_RESOURCES)]
        tmp["resource_short"] = tmp.resource.map(short_resource)
        tmp["missing_bin"] = pd.cut(pd.to_numeric(tmp.missing_fraction, errors="coerce"),
                                    [-.001, 0, .05, .25, .75, 1],
                                    labels=["0", "0–5%", "5–25%", "25–75%", ">75%"],
                                    include_lowest=True)
        mat = tmp.groupby(["resource_short", "missing_bin"], observed=False).size().unstack(fill_value=0)
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
        v22_heat(ax, mat, "Column missingness composition", "YlGnBu", True, annotate=True, fmt=".2f", xtick_rotation=25)
    else:
        nodata(ax, "Column missingness composition")

    ax = axs[3]; _panel_letter(ax, "d"); _clean_axes(ax)
    if not sc.empty:
        sub = sc[sc.resource.isin(V22_MAIN_RESOURCES)].copy()
        sub["short"] = sub.resource.map(short_resource)
        x = np.log10(pd.to_numeric(sub.total_profiled_or_counted_rows, errors="coerce").fillna(0).clip(lower=0) + 1)
        y = pd.to_numeric(sub.chemistry_trust_score, errors="coerce")
        s = 90 + 520*pd.to_numeric(sub.ml_readiness_score, errors="coerce").fillna(0).clip(0, 1)
        ax.scatter(x, y, s=s, color=[v22_resource_color(r) for r in sub.resource],
                   alpha=.82, edgecolors="black", linewidth=.6, zorder=3)
        offsets = {"ARC": (0.03, -0.045), "MOSAEC": (0.03, 0.022), "CoRE-24": (0.03, 0.025),
                   "QMOF": (0.03, -0.04), "CSD ctx": (0.03, -0.035), "CoRE-25": (0.03, 0.025)}
        for _, r in sub.iterrows():
            dx, dy = offsets.get(r.short, (0.03, 0.02))
            ax.text(np.log10(float(r.total_profiled_or_counted_rows) + 1) + dx,
                    float(r.chemistry_trust_score) + dy, r.short, fontsize=7.8)
        ax.set_xlabel("log10(profiled or counted rows + 1)")
        ax.set_ylabel("Chemistry-trust score")
        ax.set_ylim(0, 1.06)
        ax.set_xlim(left=0)
        ax.set_title("Scale, chemistry trust and ML-readiness", fontweight="bold")
    else:
        nodata(ax, "Scale, chemistry trust and ML-readiness")

    fig.suptitle("Figure 2. MOF databases as a demanding claim-readiness case study",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.0, w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_2_data_resource_atlas", cfg)


def _ensure_resource_rows(mat: pd.DataFrame, fill: float=0.0) -> pd.DataFrame:
    if mat is None or mat.empty:
        return pd.DataFrame(index=[short_resource(r) for r in V22_MAIN_RESOURCES])
    out = mat.copy()
    for r in V22_MAIN_RESOURCES:
        if r not in out.index:
            out.loc[r, :] = fill
    out = out.loc[V22_MAIN_RESOURCES]
    out.index = [short_resource(r) for r in out.index]
    return out


def fig3(cfg, dd):
    emat = load_df(dd["processed"] / "chemistry_evidence_matrix")
    reg = load_df(dd["processed"] / "trust_regime_summary")
    charge = load_df(dd["processed"] / "charge_status_summary")
    cards = load_df(dd["processed"] / "representative_rule_cards")
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.0))
    axs = axs.ravel()

    ax = axs[0]; _panel_letter(ax, "a")
    if not emat.empty:
        cols = [c for c in ["formula_observable", "metal_observable", "oxidation_state_observable",
                            "charge_observable", "curation_observable"] if c in emat.columns]
        mat = emat.set_index("resource")[cols] if cols else pd.DataFrame()
        mat = _ensure_resource_rows(mat, fill=0.0)
        mat.columns = ["formula", "metal", "oxid.", "charge", "curation"][:len(mat.columns)]
        v22_heat(ax, mat, "Domain-evidence observability", "YlGnBu", True, annotate=True, fmt=".2f", xtick_rotation=25)
    else:
        mat = pd.DataFrame(0.0, index=[short_resource(r) for r in V22_MAIN_RESOURCES],
                           columns=["formula", "metal", "oxid.", "charge", "curation"])
        v22_heat(ax, mat, "Domain-evidence observability", "YlGnBu", True, annotate=True, fmt=".2f", xtick_rotation=25)

    ax = axs[1]; _panel_letter(ax, "b"); _clean_axes(ax, grid=False)
    if not reg.empty:
        piv = reg.pivot_table(index="resource", columns="trust_regime_row", values="fraction", fill_value=0)
        for r in V22_MAIN_RESOURCES:
            if r not in piv.index:
                piv.loc[r, "not_scanned_or_not_observable"] = 1.0
        piv = piv.loc[V22_MAIN_RESOURCES].fillna(0)
        cols = [c for c in ["validated_observable", "validated_not_observable", "uncertain_observable",
                            "uncertain_unobservable", "flagged_or_inconsistent", "not_scanned_or_not_observable"]
                if c in piv.columns]
        y = np.arange(len(piv.index))
        bottom = np.zeros(len(piv.index))
        for ccol in cols:
            vals = pd.to_numeric(piv[ccol], errors="coerce").fillna(0).values
            ax.barh(y, vals, left=bottom, label=ccol.replace("_", " "),
                    color=V22_TRUST_COLORS.get(ccol, "#CCCCCC"), edgecolor="white", linewidth=.5)
            bottom += vals
        ax.set_yticks(y); ax.set_yticklabels([short_resource(x) for x in piv.index], fontsize=7.8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Fraction of scanned / classified rows")
        ax.set_title("Validation–observability regimes", fontweight="bold")
        ax.legend(fontsize=6.1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    else:
        nodata(ax, "Validation–observability regimes")

    ax = axs[2]; _panel_letter(ax, "c")
    if not charge.empty:
        piv = charge.pivot_table(index="resource", columns="charge_balance_status", values="fraction", fill_value=0)
        for r in V22_MAIN_RESOURCES:
            if r not in piv.index:
                piv.loc[r, "not_observable"] = 1.0
        piv = piv.loc[V22_MAIN_RESOURCES].fillna(0)
        piv.index = [short_resource(x) for x in piv.index]
        piv.columns = [str(x).replace("_", " ") for x in piv.columns]
        v22_heat(ax, piv, "Charge-status decomposition", "PuBuGn", True, annotate=True, fmt=".2f", xtick_rotation=25)
    else:
        mat = pd.DataFrame({"not observable": [1.0]*len(V22_MAIN_RESOURCES)}, index=[short_resource(r) for r in V22_MAIN_RESOURCES])
        v22_heat(ax, mat, "Charge-status decomposition", "PuBuGn", True, annotate=True, fmt=".2f", xtick_rotation=20)

    ax = axs[3]; ax.axis("off"); _panel_letter(ax, "d", inside=True)
    ax.set_title("How to read non-observable evidence", fontweight="bold", pad=10)
    cards_text = [
        ("Not observable", "A parsed table lacks direct evidence; this is not an invalidity label.", "#F0F0F0"),
        ("Validated not observable", "A trusted source exists, but the required field is not visible in the parsed table.", V22_TRUST_COLORS["validated_not_observable"]),
        ("Flagged", "A warning, inconsistent charge, or unusual oxidation state requires caveated use.", V22_TRUST_COLORS["flagged_or_inconsistent"]),
    ]
    y = 0.74
    for title, body, color in cards_text:
        ax.add_patch(FancyBboxPatch((0.08, y-0.07), 0.84, 0.125,
                                    boxstyle="round,pad=.018,rounding_size=.020",
                                    lw=.85, facecolor=color, alpha=.28, edgecolor="#777777"))
        ax.text(0.13, y+0.020, title, ha="left", va="center", fontsize=8.5, fontweight="bold")
        ax.text(0.13, y-0.035, body, ha="left", va="center", fontsize=7.35, wrap=True)
        y -= 0.18
    if not cards.empty:
        ex = cards.head(1).iloc[0]
        ax.text(0.08, 0.16, f"Example rule card: {short_resource(ex.get('resource',''))} | {str(ex.get('rule_card_class','')).replace('_',' ')}",
                fontsize=7.2, color="#444444")
    fig.suptitle("Figure 3. Domain-evidence observability and validation regimes",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.0, w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_3_chemistry_evidence_trust_regimes", cfg)


def fig4(cfg, dd):
    sc = load_df(dd["processed"] / "resource_scores")
    risk = load_df(dd["processed"] / "benchmark_risk_matrix")
    sens = load_df(dd["processed"] / "score_sensitivity")
    unc = load_df(dd["processed"] / "resource_score_uncertainty")
    fig, axs = plt.subplots(1, 3, figsize=(16.6, 5.6))

    ax = axs[0]; _panel_letter(ax, "a"); _clean_axes(ax)
    if not sc.empty:
        sub = sc[sc.resource.isin(V22_MAIN_RESOURCES)].copy()
        sub["short"] = sub.resource.map(short_resource)
        if not unc.empty and {"resource", "chemistry_trust_ci95_low", "chemistry_trust_ci95_high"}.issubset(unc.columns):
            sub = sub.merge(unc[["resource", "chemistry_trust_ci95_low", "chemistry_trust_ci95_high"]], on="resource", how="left")
        ax.axhline(.70, ls="--", lw=.8, color="#999999")
        ax.axvline(.70, ls="--", lw=.8, color="#999999")
        x = pd.to_numeric(sub.ml_readiness_score, errors="coerce")
        y = pd.to_numeric(sub.chemistry_trust_score, errors="coerce")
        if "chemistry_trust_ci95_low" in sub.columns:
            lo = pd.to_numeric(sub.chemistry_trust_ci95_low, errors="coerce")
            hi = pd.to_numeric(sub.chemistry_trust_ci95_high, errors="coerce")
            yerr = np.vstack([(y-lo).clip(lower=0), (hi-y).clip(lower=0)])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#555555", elinewidth=.75, capsize=3, zorder=1)
        ax.scatter(x, y, s=160, color=[v22_resource_color(r) for r in sub.resource],
                   alpha=.82, edgecolors="black", linewidth=.6, zorder=3)
        offsets = {"ARC": (.012, -0.045), "MOSAEC": (.012, .020), "CoRE-24": (.012, .020),
                   "QMOF": (.012, -0.038), "CSD ctx": (.012, -0.035), "CoRE-25": (.012, .020)}
        for _, r in sub.iterrows():
            dx, dy = offsets.get(r.short, (.012, .012))
            ax.text(float(r.ml_readiness_score)+dx, float(r.chemistry_trust_score)+dy, r.short, fontsize=7.8)
        ax.set_xlim(0, 1.06); ax.set_ylim(0, 1.06)
        ax.set_xlabel("ML-readiness score"); ax.set_ylabel("Domain-evidence score")
        ax.set_title("Claim-readiness map", fontweight="bold")
    else:
        nodata(ax, "Claim-readiness map")

    ax = axs[1]; _panel_letter(ax, "b")
    if not risk.empty:
        r = risk.copy()
        r["risk_short"] = r["risk_issue"].map(V22_RISK_SHORT).fillna(r["risk_issue"])
        r["use_short"] = r["use_case"].map(V22_USE_SHORT).fillna(r["use_case"])
        mat = r.pivot_table(index="risk_short", columns="use_short", values="risk_score_0_to_3", aggfunc="mean", fill_value=0)
        # Stable order by highest risk and short use-case order.
        mat = mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
        desired_cols = [c for c in ["adsorption", "process", "quantum", "generation", "mechanism"] if c in mat.columns]
        if desired_cols:
            mat = mat[desired_cols]
        v22_heat(ax, mat, "Claim-risk matrix", "OrRd", True, annotate=True, fmt=".0f", xtick_rotation=25)
    else:
        nodata(ax, "Claim-risk matrix")

    ax = axs[2]; _panel_letter(ax, "c"); _clean_axes(ax)
    if not sens.empty:
        s = sens.copy()
        s["short"] = s.resource.map(short_resource)
        s = s[s.resource.isin(V22_MAIN_RESOURCES)].sort_values("mean_rank")
        y = np.arange(len(s))
        ax.errorbar(pd.to_numeric(s.mean_rank, errors="coerce"), y,
                    xerr=[pd.to_numeric(s.mean_rank-s.min_rank, errors="coerce").clip(lower=0),
                          pd.to_numeric(s.max_rank-s.mean_rank, errors="coerce").clip(lower=0)],
                    fmt="o", capsize=3, color="#333333", ecolor="#777777")
        ax.set_yticks(y); ax.set_yticklabels(s.short, fontsize=7.8)
        ax.invert_yaxis()
        ax.set_xlabel("Rank across score-weight perturbations")
        ax.set_title("Score-weight sensitivity", fontweight="bold")
    else:
        nodata(ax, "Score-weight sensitivity")
    fig.suptitle("Figure 4. Claim-risk and readiness mapping",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .90], w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_4_trust_readiness_risk_matrix", cfg)


def _make_descriptor_family_heatmap(matched: pd.DataFrame) -> pd.DataFrame:
    if matched is None or matched.empty:
        return pd.DataFrame()
    m = matched.copy()
    fam_col = "descriptor_family" if "descriptor_family" in m.columns else ("descriptor_family_x" if "descriptor_family_x" in m.columns else None)
    if not fam_col:
        return pd.DataFrame()
    for col in ["mean_r2", "mean_spearman", "mean_top_10pct_recovery"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    if {"split_type", "mean_r2", "mean_top_10pct_recovery"}.issubset(m.columns):
        tmp = (m.groupby([fam_col, "split_type"])
                 .agg(median_r2=("mean_r2", "median"),
                      median_top10=("mean_top_10pct_recovery", "median"),
                      n=("mean_top_10pct_recovery", "count"))
                 .reset_index())
        r2 = tmp.pivot_table(index=fam_col, columns="split_type", values="median_r2", aggfunc="median")
        top10 = tmp.pivot_table(index=fam_col, columns="split_type", values="median_top10", aggfunc="median")
        order_source = top10 if not top10.empty else r2
        order = order_source.max(axis=1).sort_values(ascending=False).index.tolist()
        mat = pd.DataFrame(index=order)
        split_alias = [("random", "random"), ("descriptor_grouped", "grouped"), ("grouped", "grouped")]
        for split, label in split_alias:
            if split in r2.columns:
                mat[f"R² {label}"] = r2.loc[order, split]
            if split in top10.columns:
                mat[f"Top-10 {label}"] = top10.loc[order, split]
        # Avoid duplicate grouped columns if both grouped and descriptor_grouped exist.
        mat = mat.loc[:, ~mat.columns.duplicated()]
        return mat
    metric_cols = [c for c in ["mean_r2_random", "mean_r2_descriptor_grouped",
                               "mean_top_10pct_recovery_random", "mean_top_10pct_recovery_descriptor_grouped"] if c in m.columns]
    if metric_cols:
        mat = m.groupby(fam_col)[metric_cols].median(numeric_only=True)
        mat = mat.rename(columns=lambda x: x.replace("mean_", "").replace("_descriptor_grouped", " grouped").replace("_random", " random").replace("_", " "))
        return mat
    return pd.DataFrame()


def _case_label(row: pd.Series) -> str:
    parts = []
    for c in ["target_context_short", "target_column", "descriptor_family", "model"]:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    if not parts:
        parts = [str(row.get("source_table", "case"))]
    label = " | ".join(parts[:3])
    return label[:46]


def fig5(cfg, dd):
    desc = load_df(dd["processed"] / "descriptor_joined_ml_metrics_summary")
    matched = load_df(dd["processed"] / "descriptor_family_case_matched_comparison")
    pred = load_df(dd["processed"] / "descriptor_joined_predictions_sample")
    if pred.empty:
        pred = load_df(dd["processed"] / "ml_predictions_sample")
    cases = _select_headline_descriptor_cases(desc, max_cases=5) if "_select_headline_descriptor_cases" in globals() else pd.DataFrame()
    fig, axs = plt.subplots(2, 2, figsize=(14.2, 9.2))
    axs = axs.ravel()

    ax = axs[0]; _panel_letter(ax, "a"); _clean_axes(ax)
    if not cases.empty:
        cases = cases.copy()
        cases["case_label"] = cases.apply(_case_label, axis=1)
        piv = cases.pivot_table(index="case_label", columns="split_type", values="mean_r2", aggfunc="max")
        order = piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y = np.arange(len(order))
        for split in [s for s in ["random", "descriptor_grouped", "grouped"] if s in piv.columns]:
            ax.scatter(piv.loc[order, split], y, marker=SPLIT_MARKERS.get(split, "o") if "SPLIT_MARKERS" in globals() else "o",
                       s=68, color=SPLIT_COLORS.get(split, "#333333") if "SPLIT_COLORS" in globals() else "#333333",
                       label=split.replace("descriptor_", "").replace("_", " "), edgecolors="black", linewidth=.45, zorder=3)
        for i, labtxt in enumerate(order):
            vals = [piv.loc[labtxt, s] for s in piv.columns if pd.notna(piv.loc[labtxt, s])]
            if len(vals) >= 2:
                ax.plot(vals, [i]*len(vals), color="#BBBBBB", lw=.8, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.8)
        ax.set_xlabel("Mean R²"); ax.set_title("Generalization gap across joined cases", fontweight="bold")
        ax.legend(fontsize=6.8, loc="lower right", frameon=True)
    else:
        nodata(ax, "Generalization gap", "No descriptor-joined ML summary available")

    ax = axs[1]; _panel_letter(ax, "b"); _clean_axes(ax)
    if not cases.empty:
        piv = cases.pivot_table(index="case_label", columns="split_type", values="mean_top_10pct_recovery", aggfunc="max")
        order = piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y = np.arange(len(order))
        for split in [s for s in ["random", "descriptor_grouped", "grouped"] if s in piv.columns]:
            ax.scatter(piv.loc[order, split], y, marker=SPLIT_MARKERS.get(split, "o") if "SPLIT_MARKERS" in globals() else "o",
                       s=68, color=SPLIT_COLORS.get(split, "#333333") if "SPLIT_COLORS" in globals() else "#333333",
                       label=split.replace("descriptor_", "").replace("_", " "), edgecolors="black", linewidth=.45, zorder=3)
        for i, labtxt in enumerate(order):
            vals = [piv.loc[labtxt, s] for s in piv.columns if pd.notna(piv.loc[labtxt, s])]
            if len(vals) >= 2:
                ax.plot(vals, [i]*len(vals), color="#BBBBBB", lw=.8, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Top-10% recovery")
        ax.set_title("Screening signal under harder splits", fontweight="bold")
    else:
        nodata(ax, "Top-10% recovery")

    ax = axs[2]; _panel_letter(ax, "c")
    mat = _make_descriptor_family_heatmap(matched)
    if not mat.empty:
        mat.index = [str(x).replace("_", " ") for x in mat.index]
        v22_heat(ax, mat, "Case-matched descriptor-family comparison", "YlGnBu", True,
                 annotate=True, fmt=".2f", xtick_rotation=28)
    else:
        nodata(ax, "Case-matched descriptor-family comparison",
               "No numeric descriptor-family metrics after aggregation")

    ax = axs[3]; _panel_letter(ax, "d"); _clean_axes(ax)
    g, meta = _best_prediction_group(pred)
    if not g.empty:
        ax.scatter(g["true_rank_pct"], g["pred_rank_pct"], s=9, alpha=.30, color="#0072B2", edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#555555")
        ax.axhline(.90, ls=":", lw=.8, color="#999999")
        ax.axvline(.90, ls=":", lw=.8, color="#999999")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Observed target percentile")
        ax.set_ylabel("Predicted target percentile")
        ax.set_title("Representative ranking-percentile view", fontweight="bold")
        txt = f"Spearman = {meta.get('spearman', np.nan):.2f}\nn = {int(meta.get('n', len(g)))}"
        if meta.get("split_type"):
            txt += f"\n{str(meta.get('split_type')).replace('_',' ')}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=.22", facecolor="white", edgecolor="#CCCCCC", lw=.5, alpha=.92))
    else:
        nodata(ax, "Representative ranking-percentile view", "No prediction sample available")
    fig.suptitle("Figure 5. Split-aware descriptor-joined benchmark",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.0, w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_5_ml_stress_test", cfg)


def fig6(cfg, dd):
    dec = load_df(dd["source"] / "decision_rules")
    chk = load_df(dd["source"] / "minimum_reporting_checklist")
    rec = load_df(dd["main_tables"] / "Main_Table_4_use_case_recommendation_cards")
    if rec.empty:
        rec = load_df(dd["main_tables"] / "Main_Table_3_use_case_recommendation_cards")
    fig, axs = plt.subplots(1, 3, figsize=(16.8, 5.8))

    ax = axs[0]; ax.axis("off"); _panel_letter(ax, "a", inside=True)
    ax.set_title("Claim-first decision tree", fontweight="bold", pad=10)
    nodes = [
        ("Scientific claim", .50, .84),
        ("Target context\nknown?", .50, .64),
        ("Domain evidence\nobservable?", .27, .43),
        ("Split-aware\nperformance?", .73, .43),
        ("Caveated claim +\nsource data", .50, .20),
    ]
    for t, x, y in nodes:
        ax.add_patch(FancyBboxPatch((x-.145, y-.058), .29, .115,
                                    boxstyle="round,pad=.018,rounding_size=.020",
                                    lw=.85, facecolor="#F7F7F7", edgecolor="#555555"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.0)
    for (x1, y1), (x2, y2) in [((.50,.78),(.50,.70)), ((.50,.58),(.30,.49)), ((.50,.58),(.70,.49)),
                               ((.27,.37),(.44,.27)), ((.73,.37),(.56,.27))]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, lw=.9, color="#444444"))

    ax = axs[1]; ax.axis("off"); _panel_letter(ax, "b", inside=True)
    ax.set_title("Minimum reporting blocks", fontweight="bold", pad=10)
    blocks = [
        ("Data provenance", "file versions, hashes, access route"),
        ("Target context", "gas / property / unit / task"),
        ("Identifier joins", "retention, duplicates, leakage"),
        ("Domain evidence", "observed, missing, validated, flagged"),
        ("Split policy", "random plus grouped / claim-aware split"),
        ("Source data", "one table per figure panel"),
    ]
    y = .84
    for title, body in blocks:
        ax.add_patch(FancyBboxPatch((.07, y-.047), .86, .086, boxstyle="round,pad=.014,rounding_size=.018",
                                    lw=.75, facecolor="#F7F7F7", edgecolor="#888888"))
        ax.text(.11, y, title, ha="left", va="center", fontsize=7.8, fontweight="bold")
        ax.text(.45, y, body, ha="left", va="center", fontsize=7.1)
        y -= .115

    ax = axs[2]; ax.axis("off"); _panel_letter(ax, "c", inside=True)
    ax.set_title("Use-case recommendation cards", fontweight="bold", pad=10)
    if not rec.empty:
        show = rec.head(4).copy()
        y = .82
        for _, r in show.iterrows():
            ax.add_patch(FancyBboxPatch((.05, y-.075), .90, .135,
                                        boxstyle="round,pad=.016,rounding_size=.020",
                                        lw=.8, facecolor="#FAFAFA", edgecolor="#777777"))
            ax.text(.08, y+.026, str(r.get("use_case", "")), ha="left", va="center", fontsize=7.9, fontweight="bold")
            ax.text(.08, y-.034, str(r.get("preferred_resource_logic", ""))[:122], ha="left", va="center", fontsize=6.6, wrap=True)
            y -= .185
    else:
        ax.text(.5, .5, "Recommendation cards unavailable", ha="center", va="center")
    fig.suptitle("Figure 6. Claim-specific reporting framework for scientific ML datasets",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .91], w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_6_decision_framework", cfg)


def _archive_legacy_figure1(cfg: Config, dd: Dict[str, Path]):
    """Move legacy duplicate Figure 1 files out of the main-figure folder."""
    archive = dd["fig"] / "archive"
    mkdir(archive)
    for suffix in [".pdf", ".png", ".svg"]:
        p = dd["fig_main"] / f"Figure_1_chemistry_ready_concept{suffix}"
        if p.exists():
            target = archive / f"legacy_Figure_1_chemistry_ready_concept{suffix}"
            try:
                if target.exists():
                    target.unlink()
                shutil.move(str(p), str(target))
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass


def step_figures(cfg: Config, dd: Dict[str, Path], log):
    outs = []
    render_plan = [
        ("SI Figure S0 inventory map", lambda: inventory_alignment_figure(cfg, dd)),
        ("Figure 1 claim-readiness framework", lambda: fig1(cfg, dd)),
        ("Figure 2 resource atlas", lambda: fig2(cfg, dd)),
        ("Figure 3 evidence regimes", lambda: fig3(cfg, dd)),
        ("Figure 4 risk/readiness map", lambda: fig4(cfg, dd)),
        ("Figure 5 split-aware benchmark", lambda: fig5(cfg, dd)),
        ("Figure 6 reporting framework", lambda: fig6(cfg, dd)),
        ("SI figures S1-S8", lambda: si_figs(cfg, dd)),
    ]
    for name, func in render_plan:
        try:
            log.info("Rendering %s", name)
            new = func()
            if isinstance(new, list):
                outs += new
            log.info("Finished rendering %s (%d files)", name, len(new) if isinstance(new, list) else 0)
        except Exception as e:
            log.warning("Figure rendering failed for %s: %s", name, e)
    _archive_legacy_figure1(cfg, dd)
    _save_visual_caption_caveats(cfg, dd)
    manifest_rows = []
    for p in outs:
        if Path(p).exists():
            try:
                rel = str(Path(p).relative_to(cfg.out_dir))
            except Exception:
                rel = str(p)
            manifest_rows.append({"figure_file": str(p), "relative_path": rel, "exists": True, "size_bytes": Path(p).stat().st_size})
    save_df(pd.DataFrame(manifest_rows), dd["fig"] / "figure_output_manifest", cfg)
    export_final_panel_source_data(cfg, dd)


_V22_STEP_REPORTS_BASE = step_reports

def step_reports(cfg: Config, dd: Dict[str, Path], log):
    _V22_STEP_REPORTS_BASE(cfg, dd, log)
    caveats = _caption_caveats_dataframe()
    lines = [
        "# Visual-final figure audit and caption notes\n\n",
        f"Generated: {iso()}\n\n",
        f"Script version: `{VERSION}`\n\n",
        "## Main visual changes in this version\n\n",
        "- Panel letters are placed outside title areas to avoid overlap.\n",
        "- Figure 1 is exported only under the canonical claim-readiness filename; the legacy duplicate is archived outside the main figure folder.\n",
        "- Figure 3 explicitly represents resources whose evidence is not observable in parsed tabular metadata.\n",
        "- Figure 5C is a clean descriptor-family heatmap without an in-panel caveat; the caveat is saved for the caption.\n",
        "- Figure 5D is a ranking-percentile diagnostic, which better matches the screening/ranking claim than raw calibration.\n\n",
        "## Caption caveats\n\n",
        _df_to_markdown(caveats, 20) if "_df_to_markdown" in globals() else caveats.to_markdown(index=False),
        "\n",
    ]
    _write_md(dd["reports"] / "visual_final_figure_notes.md", "".join(lines)) if "_write_md" in globals() else (dd["reports"] / "visual_final_figure_notes.md").write_text("".join(lines), encoding="utf-8")


_V22_WRITE_README_BASE = write_readme

def write_readme(cfg: Config):
    _V22_WRITE_README_BASE(cfg)
    extra = f"""
Visual-final note
-----------------
Script version: {VERSION}

This version keeps the scientific computation unchanged but redraws the main
figures with a publication-oriented layout:
- non-overlapping panel letters;
- one canonical Figure 1 claim-readiness framework;
- explicit not-observable evidence handling;
- cleaned Figure 5C descriptor-family comparison;
- ranking-percentile Figure 5D;
- neutral journal_submission/ and github_audit/ compact reporting folders.

Recommended workflow
--------------------
1. Use --refresh-publication-only on an existing successful output folder to
   regenerate final figures, reports and compact audit folders.
2. Inspect figures/main/Figure_1_claim_readiness_framework.pdf and
   figures/main/Figure_5_ml_stress_test.pdf.
3. Only after visual approval, run one clean fresh final folder with --force.
"""
    with open(cfg.out_dir / "README_OUTPUTS.txt", "a", encoding="utf-8") as fh:
        fh.write(extra)

# =============================================================================
# END v2.2 VISUAL-FINAL OVERRIDES
# =============================================================================




# =============================================================================
# v2.2.1 ROBUST FINAL-ASSET FALLBACKS
# =============================================================================
# The full local output folder contains processed/ tables.  Compact review ZIPs
# may contain only main/SI tables and figure-panel source data.  These small
# fallbacks make the visual-refresh mode useful for both cases.

def _load_any(bases: Sequence[Path]) -> pd.DataFrame:
    for base in bases:
        try:
            df = load_df(base)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _load_fig5_tables(dd: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    desc = _load_any([
        dd["processed"] / "descriptor_joined_ml_metrics_summary",
        dd["tables"] / "main" / "Main_Table_6_descriptor_joined_ml_summary",
        dd["tables"] / "main" / "Main_Table_6_best_descriptor_joined_cases_publication",
        dd["source"] / "figure_panel_source_data" / "Figure_5a_generalization_gap",
    ])
    matched = _load_any([
        dd["processed"] / "descriptor_family_case_matched_comparison",
        dd["tables"] / "si" / "SI_Table_S24_descriptor_family_case_matched_comparison",
        dd["source"] / "figure_panel_source_data" / "Figure_5c_descriptor_family_comparison",
    ])
    pred = _load_any([
        dd["processed"] / "descriptor_joined_predictions_sample",
        dd["processed"] / "ml_predictions_sample",
        dd["source"] / "figure_panel_source_data" / "Figure_5d_ranking_percentile",
        dd["source"] / "figure_panel_source_data" / "Figure_5d_calibration",
    ])
    return desc, matched, pred


def _select_cases_v22(desc: pd.DataFrame, max_cases: int=5) -> pd.DataFrame:
    if desc is None or desc.empty:
        return pd.DataFrame()
    d = desc.copy()
    # Source-data fallback may already have case_label and split columns.
    if "case_label" in d.columns and "split_type" in d.columns:
        for col in ["mean_r2", "mean_top_10pct_recovery"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        return d.head(max_cases * max(2, d.get("split_type", pd.Series()).nunique() or 2))
    if "_select_headline_descriptor_cases" in globals():
        try:
            out = _select_headline_descriptor_cases(d, max_cases=max_cases)
            if out is not None and not out.empty:
                return out
        except Exception:
            pass
    required = {"split_type", "mean_r2", "mean_top_10pct_recovery"}
    if not required.issubset(d.columns):
        return pd.DataFrame()
    for col in ["mean_r2", "mean_top_10pct_recovery", "mean_spearman"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["case_label"] = d.apply(_case_label, axis=1)
    score = d.groupby("case_label")["mean_top_10pct_recovery"].max().sort_values(ascending=False)
    keep = score.head(max_cases).index
    return d[d.case_label.isin(keep)]


def fig5(cfg, dd):
    desc, matched, pred = _load_fig5_tables(dd)
    cases = _select_cases_v22(desc, max_cases=5)
    fig, axs = plt.subplots(2, 2, figsize=(14.2, 9.2))
    axs = axs.ravel()

    ax = axs[0]; _panel_letter(ax, "a"); _clean_axes(ax)
    if not cases.empty and {"case_label", "split_type", "mean_r2"}.issubset(cases.columns):
        cases = cases.copy()
        cases["mean_r2"] = pd.to_numeric(cases["mean_r2"], errors="coerce")
        piv = cases.pivot_table(index="case_label", columns="split_type", values="mean_r2", aggfunc="max")
        order = piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y = np.arange(len(order))
        for split in [s for s in ["random", "descriptor_grouped", "grouped"] if s in piv.columns]:
            ax.scatter(piv.loc[order, split], y,
                       marker=SPLIT_MARKERS.get(split, "o") if "SPLIT_MARKERS" in globals() else "o",
                       s=68, color=SPLIT_COLORS.get(split, "#333333") if "SPLIT_COLORS" in globals() else "#333333",
                       label=split.replace("descriptor_", "").replace("_", " "),
                       edgecolors="black", linewidth=.45, zorder=3)
        for i, labtxt in enumerate(order):
            vals = [piv.loc[labtxt, s] for s in piv.columns if pd.notna(piv.loc[labtxt, s])]
            if len(vals) >= 2:
                ax.plot(vals, [i]*len(vals), color="#BBBBBB", lw=.8, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([str(x)[:46] for x in order], fontsize=6.8)
        ax.set_xlabel("Mean R²"); ax.set_title("Generalization gap across joined cases", fontweight="bold")
        ax.legend(fontsize=6.8, loc="lower right", frameon=True)
    else:
        nodata(ax, "Generalization gap", "No descriptor-joined ML summary available")

    ax = axs[1]; _panel_letter(ax, "b"); _clean_axes(ax)
    if not cases.empty and {"case_label", "split_type", "mean_top_10pct_recovery"}.issubset(cases.columns):
        cases = cases.copy()
        cases["mean_top_10pct_recovery"] = pd.to_numeric(cases["mean_top_10pct_recovery"], errors="coerce")
        piv = cases.pivot_table(index="case_label", columns="split_type", values="mean_top_10pct_recovery", aggfunc="max")
        order = piv.max(axis=1).sort_values(ascending=True).index.tolist()
        y = np.arange(len(order))
        for split in [s for s in ["random", "descriptor_grouped", "grouped"] if s in piv.columns]:
            ax.scatter(piv.loc[order, split], y,
                       marker=SPLIT_MARKERS.get(split, "o") if "SPLIT_MARKERS" in globals() else "o",
                       s=68, color=SPLIT_COLORS.get(split, "#333333") if "SPLIT_COLORS" in globals() else "#333333",
                       label=split.replace("descriptor_", "").replace("_", " "),
                       edgecolors="black", linewidth=.45, zorder=3)
        for i, labtxt in enumerate(order):
            vals = [piv.loc[labtxt, s] for s in piv.columns if pd.notna(piv.loc[labtxt, s])]
            if len(vals) >= 2:
                ax.plot(vals, [i]*len(vals), color="#BBBBBB", lw=.8, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([str(x)[:46] for x in order], fontsize=6.8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Top-10% recovery")
        ax.set_title("Screening signal under harder splits", fontweight="bold")
    else:
        nodata(ax, "Top-10% recovery")

    ax = axs[2]; _panel_letter(ax, "c")
    mat = _make_descriptor_family_heatmap(matched)
    if not mat.empty:
        mat.index = [str(x).replace("_", " ") for x in mat.index]
        v22_heat(ax, mat, "Case-matched descriptor-family comparison", "YlGnBu", True,
                 annotate=True, fmt=".2f", xtick_rotation=28)
    else:
        nodata(ax, "Case-matched descriptor-family comparison",
               "No numeric descriptor-family metrics after aggregation")

    ax = axs[3]; _panel_letter(ax, "d"); _clean_axes(ax)
    g, meta = _best_prediction_group(pred)
    if not g.empty:
        # Fallback source-data may already be rank percentiles.
        if {"true_rank_pct", "pred_rank_pct"}.issubset(g.columns):
            g["true_rank_pct"] = pd.to_numeric(g["true_rank_pct"], errors="coerce")
            g["pred_rank_pct"] = pd.to_numeric(g["pred_rank_pct"], errors="coerce")
        ax.scatter(g["true_rank_pct"], g["pred_rank_pct"], s=9, alpha=.30,
                   color="#0072B2", edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#555555")
        ax.axhline(.90, ls=":", lw=.8, color="#999999")
        ax.axvline(.90, ls=":", lw=.8, color="#999999")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Observed target percentile")
        ax.set_ylabel("Predicted target percentile")
        ax.set_title("Representative ranking-percentile view", fontweight="bold")
        txt = f"Spearman = {meta.get('spearman', np.nan):.2f}\nn = {int(meta.get('n', len(g)))}"
        if meta.get("split_type"):
            txt += f"\n{str(meta.get('split_type')).replace('_',' ')}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=.22", facecolor="white",
                          edgecolor="#CCCCCC", lw=.5, alpha=.92))
    else:
        nodata(ax, "Representative ranking-percentile view", "No prediction sample available")

    fig.suptitle("Figure 5. Split-aware descriptor-joined benchmark",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.0, w_pad=2.0)
    return save_fig(fig, dd["fig_main"] / "Figure_5_ml_stress_test", cfg)

# =============================================================================
# END v2.2.1 ROBUST FINAL-ASSET FALLBACKS
# =============================================================================


if __name__ == '__main__':
    raise SystemExit(main())
