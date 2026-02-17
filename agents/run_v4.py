from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests

from policy_search import search_policy_pages


# --------------------------
# Name normalization & matching
# --------------------------

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

def normalize_name(s: str) -> str:
    """Simple, robust-ish normalization for institution names."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # common noise
    s = s.replace("&", " and ")
    s = s.replace("univ ", "university ")
    s = s.replace("univ.", "university ")
    s = s.replace("*", " ")
    s = _NORMALIZE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_match_maps(ipeds_df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Map normalized institution name -> list of row indices in ipeds_df
    (Sometimes multiple campuses share similar names; keep all.)
    """
    m: dict[str, list[int]] = {}
    for idx, name in ipeds_df["INSTNM"].astype(str).items():
        key = normalize_name(name)
        if not key:
            continue
        m.setdefault(key, []).append(idx)
    return m

def match_scimago_to_ipeds(
    scimago_df: pd.DataFrame,
    ipeds_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      matched_scimago: SciMAGO rows with added UNITID/INSTNM/WEBADDR
      unmatched_scimago: SciMAGO rows that could not be matched
    """
    name_map = build_match_maps(ipeds_df)

    matched_rows = []
    unmatched_rows = []

    for _, r in scimago_df.iterrows():
        inst = str(r["Institution"])
        key = normalize_name(inst)
        idxs = name_map.get(key, [])
        if not idxs:
            unmatched_rows.append(r.to_dict())
            continue

        # If multiple, pick the one with a non-empty WEBADDR, else first
        best_idx = None
        for idx in idxs:
            web = ipeds_df.at[idx, "WEBADDR"]
            if pd.notna(web) and str(web).strip():
                best_idx = idx
                break
        if best_idx is None:
            best_idx = idxs[0]

        matched_rows.append({
            **r.to_dict(),
            "UNITID": int(ipeds_df.at[best_idx, "UNITID"]),
            "INSTNM_IPEDS": str(ipeds_df.at[best_idx, "INSTNM"]),
            "WEBADDR": str(ipeds_df.at[best_idx, "WEBADDR"]) if pd.notna(ipeds_df.at[best_idx, "WEBADDR"]) else "",
        })

    matched_scimago = pd.DataFrame(matched_rows)
    unmatched_scimago = pd.DataFrame(unmatched_rows)
    return matched_scimago, unmatched_scimago


# --------------------------
# HTML download
# --------------------------

@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int]
    content_type: str
    saved_path: str
    error: str

def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

def download_html(
    url: str,
    out_dir: Path,
    session: requests.Session,
    timeout: int = 25,
    max_retries: int = 2,
    sleep_s: float = 0.3,
) -> FetchResult:
    """
    Download a single URL and save if it's HTML.
    Returns FetchResult; if not HTML or failed, ok=False but still recorded.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; policy-research-bot/0.1; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            status = resp.status_code
            ctype = (resp.headers.get("content-type") or "").lower()

            # Only save HTML
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                return FetchResult(
                    url=url,
                    ok=False,
                    status_code=status,
                    content_type=ctype,
                    saved_path="",
                    error="skipped_non_html",
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_url_hash(url)}.html"
            path = out_dir / filename
            path.write_bytes(resp.content)

            return FetchResult(
                url=url,
                ok=True,
                status_code=status,
                content_type=ctype,
                saved_path=str(path),
                error="",
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            return FetchResult(
                url=url,
                ok=False,
                status_code=None,
                content_type="",
                saved_path="",
                error=last_err,
            )


# --------------------------
# Main pipeline
# --------------------------

def build_selected_500(
    ipeds_path: Path,
    scimago_path: Path,
    top_n: int,
    random_n: int,
    seed: int,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Build selected_500.csv with columns:
      UNITID, INSTNM, WEBADDR, group, scimago_rank(optional), scimago_institution(optional)
    """
    ipeds = pd.read_csv(ipeds_path)
    needed = {"UNITID", "INSTNM", "WEBADDR"}
    missing = needed - set(ipeds.columns)
    if missing:
        raise ValueError(f"IPEDS file missing required columns: {missing}")

    ipeds = ipeds[["UNITID", "INSTNM", "WEBADDR"]].copy()
    ipeds["UNITID"] = ipeds["UNITID"].astype(int)
    ipeds["INSTNM"] = ipeds["INSTNM"].astype(str)
    ipeds["WEBADDR"] = ipeds["WEBADDR"].fillna("").astype(str)

    sc = pd.read_csv(scimago_path, sep=";", encoding="latin1")
    if "Rank" not in sc.columns or "Institution" not in sc.columns:
        raise ValueError("SciMAGO file must contain 'Rank' and 'Institution' columns.")

    sc["Rank"] = pd.to_numeric(sc["Rank"], errors="coerce")
    sc = sc.dropna(subset=["Rank", "Institution"]).copy()
    sc["Rank"] = sc["Rank"].astype(int)
    sc["Institution"] = sc["Institution"].astype(str)

    # Match ALL SciMAGO (for exclusion pool)
    sc_matched_all, sc_unmatched_all = match_scimago_to_ipeds(sc, ipeds)

    # Top N by rank, then match
    sc_top = sc.sort_values("Rank").head(top_n).copy()
    sc_matched_top, sc_unmatched_top = match_scimago_to_ipeds(sc_top, ipeds)

    if len(sc_matched_top) < top_n:
        print(f"[WARN] Only matched {len(sc_matched_top)}/{top_n} SciMAGO top rows to IPEDS by simple name match.")
        if not sc_unmatched_top.empty:
            print("[WARN] Example unmatched (first 10):")
            print(sc_unmatched_top[["Rank", "Institution"]].head(10).to_string(index=False))

    top_df = pd.DataFrame({
        "UNITID": sc_matched_top["UNITID"].astype(int),
        "INSTNM": sc_matched_top["INSTNM_IPEDS"].astype(str),
        "WEBADDR": sc_matched_top["WEBADDR"].astype(str),
        "group": "scimago_top",
        "scimago_rank": sc_matched_top["Rank"].astype(int),
        "scimago_institution": sc_matched_top["Institution"].astype(str),
    })

    # Exclusion set = any IPEDS institutions that are in SciMAGO list (matched)
    scimago_unitids = set(sc_matched_all["UNITID"].astype(int).tolist())

    # Random sample from IPEDS excluding SciMAGO unitids
    pool = ipeds[~ipeds["UNITID"].isin(scimago_unitids)].copy()
    pool = pool[pool["WEBADDR"].astype(str).str.strip() != ""].copy()  # must have website for domain restriction to work well

    if len(pool) < random_n:
        raise ValueError(f"Random pool too small ({len(pool)}) to sample {random_n} universities.")

    random.seed(seed)
    sampled_unitids = random.sample(pool["UNITID"].tolist(), random_n)
    rand_df = pool[pool["UNITID"].isin(sampled_unitids)].copy()
    rand_df["group"] = "random_not_in_scimago"
    rand_df["scimago_rank"] = ""
    rand_df["scimago_institution"] = ""

    selected = pd.concat([top_df, rand_df[["UNITID", "INSTNM", "WEBADDR", "group", "scimago_rank", "scimago_institution"]]], ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipeds", type=str, default="hd2024_data_stata 2.csv", help="Path to IPEDS CSV (must include UNITID, INSTNM, WEBADDR).")
    ap.add_argument("--scimago", type=str, default="ScimagoIR 2025 - Overall Rank - Universities - USA.csv", help="Path to SciMAGO CSV.")
    ap.add_argument("--top-n", type=int, default=250)
    ap.add_argument("--random-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-results", type=int, default=100, help="num_results per Exa query template (None not supported via CLI; set a number).")
    ap.add_argument("--search-type", type=str, default="deep", choices=["neural", "keyword", "deep"])
    ap.add_argument("--restrict-domain", action="store_true", default=True, help="Restrict to official domain (default True). Use --no-restrict-domain to disable.")
    ap.add_argument("--no-restrict-domain", action="store_false", dest="restrict_domain")

    ap.add_argument("--outdir", type=str, default="data/v4", help="Base output directory.")
    ap.add_argument("--skip-html", action="store_true", help="Collect links but do not download HTML.")
    ap.add_argument("--max-urls-per-uni", type=int, default=60, help="Limit HTML downloads per university (after dedupe).")
    ap.add_argument("--sleep-between-unis", type=float, default=0.7, help="Seconds to sleep between universities (politeness).")

    args = ap.parse_args()

    ipeds_path = Path(args.ipeds)
    scimago_path = Path(args.scimago)
    outdir = Path(args.outdir)

    if not ipeds_path.exists():
        raise FileNotFoundError(f"IPEDS CSV not found: {ipeds_path.resolve()}")
    if not scimago_path.exists():
        raise FileNotFoundError(f"SciMAGO CSV not found: {scimago_path.resolve()}")

    sample_csv = outdir / "sample" / "selected_500.csv"
    selected = build_selected_500(
        ipeds_path=ipeds_path,
        scimago_path=scimago_path,
        top_n=args.top_n,
        random_n=args.random_n,
        seed=args.seed,
        out_csv=sample_csv,
    )
    print(f"[OK] Wrote sample: {sample_csv} (n={len(selected)})")

    raw_dir = outdir / "exa_raw"
    html_dir = outdir / "html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()

    for i, row in selected.iterrows():
        unitid = int(row["UNITID"])
        instnm = str(row["INSTNM"])
        web = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""
        group = str(row["group"])

        print(f"\n[{i+1}/{len(selected)}] {unitid} | {instnm} | {group}")

        # 1) Exa retrieval (links + text)
        try:
            hits = search_policy_pages(
                university=instnm,
                website=web,
                num_results=args.num_results,
                restrict_domain=args.restrict_domain,
                search_type=args.search_type,
            )
        except Exception as e:
            print(f"[ERROR] Exa search failed for {instnm}: {type(e).__name__}: {e}")
            hits = []

        # Save raw JSON
        raw_path = raw_dir / f"{unitid}.json"
        raw_path.write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] raw json: {raw_path} (hits={len(hits)})")

        # 2) Download HTMLs
        if args.skip_html:
            continue

        # dedupe + cap
        urls = []
        seen = set()
        for h in hits:
            u = h.get("url", "")
            if not u or u in seen:
                continue
            seen.add(u)
            urls.append(u)
        urls = urls[: max(0, int(args.max_urls_per_uni))]

        uni_html_dir = html_dir / str(unitid)
        manifest_path = uni_html_dir / "manifest.csv"
        uni_html_dir.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["unitid", "instnm", "url", "ok", "status_code", "content_type", "saved_path", "error"],
            )
            w.writeheader()

            ok_count = 0
            for u in urls:
                fr = download_html(u, uni_html_dir, sess)
                if fr.ok:
                    ok_count += 1
                w.writerow({
                    "unitid": unitid,
                    "instnm": instnm,
                    "url": fr.url,
                    "ok": fr.ok,
                    "status_code": fr.status_code if fr.status_code is not None else "",
                    "content_type": fr.content_type,
                    "saved_path": fr.saved_path,
                    "error": fr.error,
                })

        print(f"[OK] html manifest: {manifest_path} | saved_html={ok_count}/{len(urls)}")

        time.sleep(max(0.0, float(args.sleep_between_unis)))

    print("\nDone.")
    print(f"- Sample: {sample_csv}")
    print(f"- Raw JSON: {raw_dir}")
    print(f"- HTML: {html_dir}")


if __name__ == "__main__":
    main()
