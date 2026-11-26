#!/usr/bin/env python3
"""Rebuild the historical ``data_raw`` input folder from public sources, verify it, and zip it.

The historical run read 509 table-like files from a local ``<DATA_ROOT>/data_raw``
folder that is not distributed with this repository. Their relative paths and
byte sizes are recorded in
``reproduce/figures/data/si/SI_Table_S1_complete_file_level_profile.csv``.
This script:

1. downloads the open source datasets from Zenodo and Figshare (resumable,
   checksum-verified), and picks up any files you downloaded by hand
   (CCDC-hosted data needs a free login);
2. extracts the archives into a staging folder;
3. places each expected file at its historical path, matching by file name
   (case, ``-``/``_`` and spaces normalised, so ``landfill-CO2.csv`` matches
   ``landfill_co2.csv``) and preferring the candidate whose folder path and
   byte size match best;
4. verifies every expected file by byte size and writes a report;
5. records SHA-256 checksums of the placed files (the historical run did not);
6. optionally zips the result.

Only the Python standard library is used.

Typical use (from the repository root)::

    python tools/fetch_historical_inputs.py --work ~/mof_inputs
    python tools/fetch_historical_inputs.py --work ~/mof_inputs --data-root ~/mof_data --zip

Run ``--help`` for all options. Re-running is safe: finished downloads are
skipped and interrupted ones resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproduce" / "figures" / "data" / "si" / "SI_Table_S1_complete_file_level_profile.csv"
ZENODO_API = "https://zenodo.org/api/records/"
FIGSHARE_API = "https://api.figshare.com/v2/articles/"
USER_AGENT = "claim-ready-mof-input-fetcher/1.0"
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")

# Versions identified by matching file names, byte sizes and timestamps in the
# historical file list. ``files``: exact names to download (None = all files).
SOURCES = [
    {
        "folder": "arc_mof",
        "label": "ARC-MOF v7",
        "kind": "zenodo",
        "id": "16802743",
        "files": [
            "all_topology_lists.csv", "ARC-MOF_Dim.csv", "flig-clusters.csv", "func-clusters.csv",
            "func-diverse-set.csv", "geo-clusters.csv", "geometric_properties.csv", "landfill-CH4.csv",
            "landfill-CO2.csv", "mc-clusters.csv", "mc-diverse-set.csv", "methane.csv",
            "methane_purification-CH4.csv", "methane_purification-CO2.csv", "overall_process.csv",
            "post_comb_vsa-CO2.csv", "post_comb_vsa-N2.csv", "pre_comb_4040-CO2.csv", "pre_comb_4040-H2.csv",
            "RACs.csv", "RDFs.csv",
        ],
        "optional_files": ["ARCMOF_20241004.tar.gz"],
    },
    {
        "folder": "core_mof_2024",
        "label": "CoRE MOF 2024 v1.1",
        "kind": "zenodo",
        "id": "15055758",
        "files": [
            "12089-recommended-screening-list.csv", "ASR_data_SI_20250204.csv", "ASR_FSR_check.csv",
            "FSR_data_SI_20250204.csv", "ION_data_SI_20250204.csv", "mofid-v2.zip", "NCR_ASR_SI.xlsx",
            "NCR_FSR_SI.xlsx", "NCR_ION_SI.xlsx", "TSA.zip", "unmodified_check_for_NCR_SI.xlsx", "water.zip",
        ],
    },
    {
        "folder": "core_mof_2025",
        "label": "CoRE MOF 2025 v1.0",
        "kind": "zenodo",
        "id": "15621349",
        "files": ["CR_meta_data_SI.json", "NCR_meta_data_SI.json"],
    },
    {
        "folder": "mosaec_db",
        "label": "MOSAEC-DB v1.0.0-release",
        "kind": "zenodo",
        "id": "15808197",
        "files": ["mosaec-db.csv", "mosaec-db.xlsx", "descriptors.tar.gz", "subsets.tar.gz", "misc_data.tar.gz"],
    },
    {
        "folder": "mofsimplify",
        "label": "MOFSimplify v1.2.0",
        "kind": "zenodo",
        "id": "5737968",
        "files": ["SciData.zip"],
    },
    {
        "folder": "arc_mof",
        "label": "clean_data.csv (ARC-MOF-derived benchmark table, Asgari-research/mof-chemical-holdouts)",
        "kind": "url",
        "url": "https://github.com/Asgari-research/mof-chemical-holdouts/raw/cf6ca6f2549932f1af7ee25db884ad2d7322c9a4/data/clean_data.zip",
        "name": "clean_data.zip",
        "size": 22997762,
        "md5": "6b5cd29977689b3526d05af9c9f36201",
    },
    {
        "folder": "qmof",
        "label": "QMOF Database (Figshare v18)",
        "kind": "figshare",
        "id": "13147324",
        "version": "18",
        "match": ("qmof_database", "qmof_thermo_database"),
    },
]

MANUAL = {
    "csd_mof_collection": (
        "CCDC downloads (free login): https://www.ccdc.cam.ac.uk/support-and-resources/downloads/\n"
        "      - the CSD MOF Collection archive (folder 'CSD_MOF_Collection', 4 CSV sheets)\n"
        "      - the CoRE MOF CSD-modified files: CR_data_CSD_modified_20250227.csv and\n"
        "        NCR_data_CSD_modified_20250227.xlsx (or the archive containing them)"
    ),
    "arc_mof": ("clean_data.csv comes from github.com/Asgari-research/mof-chemical-holdouts (data/clean_data.zip)\n"
                "      or Zenodo 22080792; put clean_data.zip or clean_data.csv here if the download failed."),
}

NOT_PUBLIC = {"PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt"}


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(msg, flush=True)


def norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def http_json(url: str, retries: int = 6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
                log(f"    server busy ({e.code}); retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None, md5: str | None, retries: int = 8) -> None:
    """Resumable download with size and md5 verification."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        if md5 is None or file_md5(dest) == md5:
            log(f"    ok (already downloaded) {dest.name}")
            return
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:
            if md5 is None or file_md5(part) == md5:
                part.replace(dest)
                log(f"    downloaded {dest.name} ({human(size)})")
                return
            part.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0  # server ignored Range; start again
                mode = "ab" if have else "wb"
                total = size or (have + int(r.headers.get("Content-Length") or 0))
                done, last = have, time.time()
                with part.open(mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 5:
                            pct = f" ({100 * done / total:.0f}%)" if total else ""
                            log(f"      {dest.name}: {human(done)}{pct}")
                            last = time.time()
            got = part.stat().st_size
            if size is not None and got != size:
                if got > size:
                    part.unlink()
                raise IOError(f"size {got} != expected {size}")
            if md5 is not None and file_md5(part) != md5:
                part.unlink()
                raise IOError("md5 checksum mismatch")
            part.replace(dest)
            log(f"    downloaded {dest.name} ({human(dest.stat().st_size)})")
            return
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            code = getattr(e, "code", None)
            if code == 416:  # range not satisfiable: partial file is complete or corrupt
                part.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}: {e}") from e
            wait = 10 * (attempt + 1)
            log(f"    retry {attempt + 1}/{retries - 1} for {dest.name} in {wait}s ({e})")
            time.sleep(wait)


# ------------------------------------------------------------------ source listing
def zenodo_files(src, args):
    meta = http_json(args.zenodo_api + src["id"])
    out = []
    for f in meta.get("files", []):
        name = f.get("key") or f.get("filename")
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("content")
        checksum = f.get("checksum") or ""
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        out.append({"name": name, "size": f.get("size"), "md5": md5, "url": link})
    return out


def figshare_files(src, args):
    url = f"{args.figshare_api}{src['id']}/versions/{args.qmof_version or src['version']}"
    meta = http_json(url)
    out = []
    for f in meta.get("files", []):
        out.append({"name": f["name"], "size": f.get("size"), "md5": f.get("computed_md5") or None, "url": f["download_url"]})
    return out


def select_files(src, listing, args):
    if src["kind"] == "figshare":
        keys = src["match"]
        chosen = [f for f in listing if f["name"].lower().endswith(".zip") and any(k in f["name"].lower() for k in keys)]
    else:
        wanted = list(src["files"]) + (src.get("optional_files", []) if args.arc_tarball else [])
        by_name = {f["name"]: f for f in listing}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            log(f"    WARNING: not found in record {src['id']}: {', '.join(missing)}")
        chosen = [by_name[w] for w in wanted if w in by_name]
    return chosen


# ------------------------------------------------------------------- extraction
def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                if not str(target).startswith(str(root) + os.sep) and target != root:
                    raise RuntimeError(f"unsafe path in {archive.name}: {info.filename}")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                for m in t.getmembers():
                    target = (dest / m.name).resolve()
                    if not str(target).startswith(str(root) + os.sep) or m.issym() or m.islnk():
                        raise RuntimeError(f"unsafe member in {archive.name}: {m.name}")
                t.extractall(dest)


def archive_stem(name: str) -> str:
    low = name.lower()
    for suf in ARCHIVE_SUFFIXES:
        if low.endswith(suf):
            return name[: -len(suf)]
    return name


def stage_source(folder: str, inputs: list[Path], staging: Path, nested: bool = False) -> None:
    """Extract archives and link plain files.

    Archives found inside extracted archives (e.g. QMOF's relaxed_structures.zip)
    hold structure files the pipeline does not read, so they are only extracted
    when ``nested`` is True.
    """
    base = staging / folder
    marker = base / ".staged"
    sig = "\n".join(sorted(f"{p.name}:{p.stat().st_size}" for p in inputs))
    if marker.exists() and marker.read_text() == sig:
        log(f"    ok (already extracted) {folder}")
        return
    if base.exists():
        shutil.rmtree(base)
    (base / "files").mkdir(parents=True)
    queue = list(inputs)
    while queue:
        p = queue.pop(0)
        if p.name.lower().endswith(ARCHIVE_SUFFIXES):
            dest = base / "extracted" / norm(archive_stem(p.name))
            log(f"    extracting {p.name}")
            safe_extract(p, dest)
            if nested:
                for inner in dest.rglob("*"):
                    if inner.is_file() and inner.name.lower().endswith(ARCHIVE_SUFFIXES):
                        queue.append(inner)
        else:
            link_or_copy(p, base / "files" / p.name)
    marker.write_text(sig)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# -------------------------------------------------------------------- placement
def load_manifest(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rel = r["relative_path"].replace("\\", "/")
            rows.append({"rel": rel, "size": int(float(r["size_bytes"]))})
    return rows


def best_candidate(rel: str, size: int, candidates: list[Path]):
    parts = [norm(p) for p in PurePosixPath(rel).parts]
    scored = []
    for c in candidates:
        cparts = [norm(p) for p in c.parts]
        k = 0
        while k < min(len(parts), len(cparts)) and parts[-1 - k] == cparts[-1 - k]:
            k += 1
        csize = c.stat().st_size
        scored.append((csize == size, k, -len(c.parts), str(c), c, csize))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    top = scored[0]
    ties = [s for s in scored if s[0] == top[0] and s[1] == top[1] and s[5] == top[5]]
    return top[4], top[5], len(ties) > 1


def place_all(manifest, staging: Path, out_root: Path):
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for folder_dir in staging.iterdir() if staging.exists() else []:
        if folder_dir.is_dir():
            for f in folder_dir.rglob("*"):
                if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".part"):
                    index[(folder_dir.name, norm(f.name))].append(f)
    report = []
    for row in manifest:
        rel, size = row["rel"], row["size"]
        parts = PurePosixPath(rel).parts
        entry = {"relative_path": rel, "expected_bytes": size, "found_bytes": "", "status": "", "source_file": ""}
        if PurePosixPath(rel).name in NOT_PUBLIC:
            entry["status"] = "NOT_PUBLIC"
            report.append(entry)
            continue
        folder = parts[1] if len(parts) > 2 and parts[0] == "data_raw" else parts[0]
        cands = index.get((folder, norm(parts[-1])), [])
        if not cands:
            entry["status"] = "MISSING"
            report.append(entry)
            continue
        src, found, ambiguous = best_candidate(rel, size, cands)
        dst = out_root / rel
        link_or_copy(src, dst)
        entry["found_bytes"] = found
        entry["source_file"] = str(src.relative_to(staging))
        entry["status"] = ("OK" if found == size else "SIZE_MISMATCH") + ("_AMBIGUOUS" if ambiguous else "")
        report.append(entry)
    return report


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="working folder for downloads and extraction (needs ~40 GB free)")
    ap.add_argument("--data-root", help="folder in which to build data_raw/ (default: WORK/build); pass this folder to the pipeline as --data-root")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="historical file list (SI Table S1)")
    ap.add_argument("--only", nargs="*", help="limit to these data_raw folders, e.g. arc_mof qmof")
    ap.add_argument("--skip-download", action="store_true", help="use files already in WORK/downloads and WORK/manual")
    ap.add_argument("--arc-tarball", action="store_true", help="also fetch ARCMOF_20241004.tar.gz (670 MB) and search it")
    ap.add_argument("--qmof-version", help="Figshare version of QMOF to use (default 18)")
    ap.add_argument("--nested", action="store_true", help="also extract archives found inside archives (not needed for the 509 files)")
    ap.add_argument("--no-checksums", action="store_true", help="skip SHA-256 of placed files")
    ap.add_argument("--zip", action="store_true", help="zip WORK/data_raw when every public file verifies")
    ap.add_argument("--zip-anyway", action="store_true", help="zip even if some files are missing or differ")
    ap.add_argument("--zenodo-api", default=ZENODO_API, help=argparse.SUPPRESS)
    ap.add_argument("--figshare-api", default=FIGSHARE_API, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    work = Path(args.work).expanduser().resolve()
    downloads, manual, staging = work / "downloads", work / "manual", work / "staging"
    out_root = Path(args.data_root).expanduser().resolve() if args.data_root else work / "build"
    if args.data_root and (work == out_root or out_root in work.parents):
        ap.error("--work must not be inside --data-root: the pipeline scans every table under its data root")
    for d in (downloads, manual, staging, out_root):
        d.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    sources = [s for s in SOURCES if not args.only or s["folder"] in args.only]
    folders = {s["folder"] for s in sources} | ({f for f in MANUAL if not args.only or f in args.only})
    if args.only:
        manifest = [m for m in manifest if len(PurePosixPath(m["rel"]).parts) > 2 and PurePosixPath(m["rel"]).parts[1] in args.only]

    free = shutil.disk_usage(work).free
    log(f"Working folder: {work}  (free space {human(free)})")
    if free < 40e9 and not args.only:
        log("WARNING: under 40 GB free; downloads + extraction + build need roughly 35-40 GB.")
    for folder in sorted(MANUAL):
        (manual / folder).mkdir(parents=True, exist_ok=True)

    # 1. download
    for src in sources:
        log(f"\n[{src['folder']}] {src['label']}")
        dest_dir = downloads / src["folder"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_download:
            continue
        if src["kind"] == "url":
            download(src["url"], dest_dir / src["name"], src.get("size"), src.get("md5"))
            continue
        listing = zenodo_files(src, args) if src["kind"] == "zenodo" else figshare_files(src, args)
        chosen = select_files(src, listing, args)
        if not chosen:
            log("    WARNING: nothing selected for download")
        for f in chosen:
            download(f["url"], dest_dir / f["name"], f["size"], f["md5"])
        if src["kind"] == "zenodo":
            time.sleep(2)  # be gentle with Zenodo's rate limit

    # 2. stage
    log("\nExtracting")
    for folder in sorted(folders):
        inputs = [p for p in (downloads / folder).glob("*") if p.is_file() and not p.name.endswith(".part")]
        inputs += [p for p in (manual / folder).rglob("*") if p.is_file() and not p.name.startswith(".")]
        if inputs:
            stage_source(folder, inputs, staging, nested=args.nested)
        else:
            log(f"    nothing to extract for {folder}")

    # 3. place + verify
    log("\nPlacing files at their historical paths")
    if (out_root / "data_raw").exists():
        shutil.rmtree(out_root / "data_raw")
    report = place_all(manifest, staging, out_root)

    # 4. checksums
    if not args.no_checksums:
        log("Computing SHA-256 of placed files (a few minutes for ~13 GB)")
        for e in report:
            p = out_root / e["relative_path"]
            e["sha256"] = file_sha256(p) if p.exists() else ""

    report_path = work / "input_verification_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["relative_path", "status", "expected_bytes", "found_bytes", "source_file"] + ([] if args.no_checksums else ["sha256"])
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(report)

    # 5. summary
    by = defaultdict(lambda: defaultdict(int))
    for e in report:
        parts = PurePosixPath(e["relative_path"]).parts
        folder = parts[1] if len(parts) > 2 else "(root)"
        by[folder][e["status"].replace("_AMBIGUOUS", "")] += 1
    log("\nSummary (files per historical folder)")
    log(f"  {'folder':<20}{'OK':>6}{'SIZE_DIFF':>11}{'MISSING':>9}{'NOT_PUBLIC':>12}")
    for folder in sorted(by):
        c = by[folder]
        log(f"  {folder:<20}{c['OK']:>6}{c['SIZE_MISMATCH']:>11}{c['MISSING']:>9}{c['NOT_PUBLIC']:>12}")
    n_ok = sum(c["OK"] for c in by.values())
    n_bad = sum(c["SIZE_MISMATCH"] + c["MISSING"] for c in by.values())
    log(f"\n  {n_ok} of {len(report)} files identical in size to the historical run; {n_bad} need attention.")
    log(f"  Full report: {report_path}")
    if any(e["status"].endswith("_AMBIGUOUS") for e in report):
        log("  Note: some files had several equally good candidates; see '_AMBIGUOUS' rows.")
    missing_folders = {PurePosixPath(e['relative_path']).parts[1] for e in report if e["status"] == "MISSING"}
    for folder in sorted(missing_folders & set(MANUAL)):
        log(f"\n  Manual download needed for {folder}: put files in {manual / folder}\n      {MANUAL[folder]}")
    if any(e["status"] == "NOT_PUBLIC" for e in report):
        log("\n  PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt was the original author's own inventory file; it is not public.")

    data_raw = out_root / "data_raw"
    log(f"\nRebuilt input folder: {data_raw}")
    log(f"Use it with:  --data-root {out_root}")

    # 6. zip
    if args.zip or args.zip_anyway:
        if n_bad and not args.zip_anyway:
            log("\nNot zipping: some files are missing or differ (use --zip-anyway to zip regardless).")
            return 1
        zpath = work / "data_raw.zip"
        log(f"\nZipping to {zpath} (this takes a while)")
        tmp = zpath.with_suffix(".zip.part")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as z:
            for p in sorted(data_raw.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out_root).as_posix())
            z.write(report_path, "input_verification_report.csv")
        tmp.replace(zpath)
        log(f"Wrote {zpath} ({human(zpath.stat().st_size)})")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
