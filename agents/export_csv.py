import json
import csv
from pathlib import Path

RAW_DIR = Path("data/v4/exa_raw")          # run_v4.py default outdir is data/v4
OUT_CSV = Path("data/v4/exa_links_all.csv")

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dir not found: {RAW_DIR.resolve()}")

    rows = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        unitid = fp.stem  # filename like 123456.json
        try:
            hits = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            hits = []

        for rank, h in enumerate(hits, start=1):
            rows.append({
                "unitid": unitid,
                #"rank_in_file": rank,
                "university": h.get("university", ""),
                "domain": h.get("domain", ""),
                #"domain_restricted": h.get("domain_restricted", ""),
                "query": h.get("query", ""),
                "title": h.get("title", ""),
                "url": h.get("url", ""),
                "length": h.get("length", ""),
                "raw_file": fp.name,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "unitid",
                #"rank_in_file",
                "university",
                "domain",
                #"domain_restricted",
                "query",
                "title",
                "url",
                "length",
                "raw_file",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {OUT_CSV} (rows={len(rows)})")

if __name__ == "__main__":
    main()
