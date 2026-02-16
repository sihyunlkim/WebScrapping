import json
from pathlib import Path
import pandas as pd

from policy_search import search_policy_pages

DATA_PATH = "hd2024_data_stata 2.csv"

TARGETS = {
    "New York University": "NYU",
    "University of Washington-Seattle Campus": "UW",
}

SEARCH_TYPE = "deep"
RESTRICT_DOMAIN = True


def side_by_side(urls_a, urls_b, col_a, col_b):
    n = max(len(urls_a), len(urls_b))
    rows = []
    for i in range(n):
        rows.append({
            "rank": i + 1,
            col_a: urls_a[i] if i < len(urls_a) else "",
            col_b: urls_b[i] if i < len(urls_b) else "",
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[["UNITID", "INSTNM", "WEBADDR"]].copy()

    # ✅ NYU & UW만 (정확히 일치)
    df2 = df[df["INSTNM"].isin(TARGETS.keys())].copy()
    if df2.empty:
        raise ValueError("NYU/UW rows not found. Check INSTNM exact strings in the CSV.")

    Path("data/compare/raw").mkdir(parents=True, exist_ok=True)
    Path("data/compare").mkdir(parents=True, exist_ok=True)

    long_rows = []
    per_query_rows = []
    summary_rows = []

    for _, row in df2.iterrows():
        unitid = int(row["UNITID"])
        uni = str(row["INSTNM"])
        short = TARGETS[uni]
        web = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""

        # --- condition A: max=100
        cand_100 = search_policy_pages(
            university=uni,
            website=web,
            num_results=100,
            restrict_domain=RESTRICT_DOMAIN,
            search_type=SEARCH_TYPE,
        )

        # --- condition B: no max specified (numResults 미지정)
        cand_nomax = search_policy_pages(
            university=uni,
            website=web,
            num_results=None,
            restrict_domain=RESTRICT_DOMAIN,
            search_type=SEARCH_TYPE,
        )

        # raw 저장
        with open(f"data/compare/raw/{unitid}_{short}_max100.json", "w", encoding="utf-8") as f:
            json.dump(cand_100, f, ensure_ascii=False, indent=2)
        with open(f"data/compare/raw/{unitid}_{short}_nomax.json", "w", encoding="utf-8") as f:
            json.dump(cand_nomax, f, ensure_ascii=False, indent=2)

        # long format (doc/table 만들기 쉬움)
        for rank, c in enumerate(cand_100, start=1):
            long_rows.append({
                "UNITID": unitid,
                "university": short,
                "condition": "max_100",
                "rank": rank,
                "query": c.get("query", ""),
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "domain": c.get("domain", ""),
            })
        for rank, c in enumerate(cand_nomax, start=1):
            long_rows.append({
                "UNITID": unitid,
                "university": short,
                "condition": "no_max_specified",
                "rank": rank,
                "query": c.get("query", ""),
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "domain": c.get("domain", ""),
            })

        # side-by-side (대학별) CSV
        urls_100 = [c.get("url", "") for c in cand_100 if c.get("url")]
        urls_nm = [c.get("url", "") for c in cand_nomax if c.get("url")]
        sbs = side_by_side(urls_100, urls_nm, "max_100", "no_max_specified")
        sbs.to_csv(f"data/compare/{short}_side_by_side.csv", index=False)

        # overall overlap summary
        set_100, set_nm = set(urls_100), set(urls_nm)
        summary_rows.append({
            "university": short,
            "max_100_count": len(set_100),
            "no_max_count": len(set_nm),
            "overlap": len(set_100 & set_nm),
            "only_in_max_100": len(set_100 - set_nm),
            "only_in_no_max": len(set_nm - set_100),
        })

        # per-query overlap summary (query prompt별 비교)
        # long_rows에 query가 있으니, 여기서 직접 계산해도 되고 아래에서 전체 DF로 계산해도 됨.
        # 여기선 간단히 대학별로만 누적해두고, 아래에서 groupby로 계산.
        print(f"Done: {short} | max100={len(cand_100)} | nomax={len(cand_nomax)}")

    out_long = pd.DataFrame(long_rows)
    out_long.to_csv("data/compare/nyu_uw_links_long.csv", index=False)

   # ---- Non-overlap links (overall per university) ----
    diff_rows = []

    for uni in ["NYU", "UW"]:
        g = out_long[out_long["university"] == uni].copy()

        urls_100 = set(g[g["condition"] == "max_100"]["url"].dropna().tolist())
        urls_nm  = set(g[g["condition"] == "no_max_specified"]["url"].dropna().tolist())
        urls_100.discard("")
        urls_nm.discard("")

        only_max = sorted(urls_100 - urls_nm)
        only_nm  = sorted(urls_nm - urls_100)

        for u in only_max:
            diff_rows.append({
                "university": uni,
                "diff_side": "only_in_max_100",
                "url": u,
            })
        for u in only_nm:
            diff_rows.append({
                "university": uni,
                "diff_side": "only_in_no_max",
                "url": u,
            })

    pd.DataFrame(diff_rows).to_csv("data/compare/nyu_uw_nonoverlap_links.csv", index=False)
    print(" - data/compare/nyu_uw_nonoverlap_links.csv")

    # per-query summary (대학+query 단위)
    per_query = []
    if not out_long.empty:
        for (uni, q), g in out_long.groupby(["university", "query"]):
            urls_100 = set(g[g["condition"] == "max_100"]["url"].dropna().tolist())
            urls_nm = set(g[g["condition"] == "no_max_specified"]["url"].dropna().tolist())
            urls_100.discard("")
            urls_nm.discard("")
            per_query.append({
                "university": uni,
                "query": q,
                "max_100_count": len(urls_100),
                "no_max_count": len(urls_nm),
                "overlap": len(urls_100 & urls_nm),
                "only_in_max_100": len(urls_100 - urls_nm),
                "only_in_no_max": len(urls_nm - urls_100),
            })
    pd.DataFrame(per_query).to_csv("data/compare/nyu_uw_overlap_per_query.csv", index=False)

    pd.DataFrame(summary_rows).to_csv("data/compare/nyu_uw_overlap_summary.csv", index=False)

    print("\nSaved:")
    print(" - data/compare/NYU_side_by_side.csv")
    print(" - data/compare/UW_side_by_side.csv")
    print(" - data/compare/nyu_uw_links_long.csv")
    print(" - data/compare/nyu_uw_overlap_per_query.csv")
    print(" - data/compare/nyu_uw_overlap_summary.csv")
    print("Raw JSON under:")
    print(" - data/compare/raw/*")


if __name__ == "__main__":
    main()
