import json
import csv
from pathlib import Path

RAW_DIR = Path("data/compilation/exa_raw")
OUT_CSV = Path("data/compilation/exa_links_all.csv")

# This already exists if you already "built the universities"
SAMPLE_CSV = Path("data/compilation/sample/selected_500.csv")


def load_scimago_rank_map():
    """
    Reads selected_500.csv and returns { unitid(str) -> scimago_rank(str) }.
    Random universities will typically have blank scimago_rank.
    """
    rank_map = {}
    if not SAMPLE_CSV.exists():
        print(f"[WARN] Sample CSV not found: {SAMPLE_CSV.resolve()}")
        return rank_map

    with SAMPLE_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # expected columns: UNITID, ..., scimago_rank, ...
        for row in r:
            unitid = str(row.get("UNITID", "")).strip()
            sc_rank = str(row.get("scimago_rank", "")).strip()
            if unitid:
                rank_map[unitid] = sc_rank

    return rank_map


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dir not found: {RAW_DIR.resolve()}")

    scimago_rank_map = load_scimago_rank_map()

    rows = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        unitid = fp.stem  # filename like 123456.json

        try:
            hits = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            hits = []

        scimago_rank = scimago_rank_map.get(unitid, "")

        for rank, h in enumerate(hits, start=1):
            rows.append({
                "unitid": unitid,
                "scimago_rank": scimago_rank,   # <-- merged in
                "university": h.get("university", ""),
                "domain": h.get("domain", ""),
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
                "scimago_rank",
                "university",
                "domain",
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