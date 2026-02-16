import json
from pathlib import Path
import pandas as pd

from policy_search import search_policy_pages
from policy_extract_gemini import extract_university_policy_with_gemini

DATA_PATH = "hd2024_data_stata 2.csv"  # 너 파일명 유지

def main():
    df = pd.read_csv(DATA_PATH)

    # pilot 20개만 (처음엔 무조건 작게)
    pilot = df.sample(20, random_state=42)[["UNITID", "INSTNM", "WEBADDR"]].copy()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    policy_rows = []
    candidate_rows = []

    for _, row in pilot.iterrows():
        unitid = int(row["UNITID"])
        uni = row["INSTNM"]
        web = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""

        # 1) domain-restricted search (landing page 기반)
        candidates = search_policy_pages(uni, web, num_results=100, restrict_domain=True)

        # raw candidates 저장
        raw_path = f"data/raw/{unitid}_policy_candidates.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

        # 후보 링크 CSV용 rows
        for i, c in enumerate(candidates, start=1):
            candidate_rows.append({
                "UNITID": unitid,
                "university_name": uni,
                "website": web,
                "domain": c.get("domain", ""),
                "rank": i,
                "query": c.get("query", ""),
                "title": c.get("title", ""),
                "url": c.get("url", ""),
            })

        # 2) (선택) Gemini로 university-wide policy 구조화
        extracted = extract_university_policy_with_gemini(uni, candidates)

        ext_path = f"data/extracted/{unitid}_university_policies.json"
        with open(ext_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, ensure_ascii=False, indent=2)

        for p in extracted.get("policies", []):
            policy_rows.append({
                "UNITID": unitid,
                "university_name": extracted.get("university_name", uni),
                "scope": p.get("scope", "university-wide"),
                "policy_title": p.get("policy_title", ""),
                "policy_url": p.get("policy_url", ""),
                "publisher": p.get("publisher", ""),
                "last_updated": p.get("last_updated", ""),
                "policy_types": "|".join(p.get("policy_types", [])) if isinstance(p.get("policy_types"), list) else "",
                "summary_bullets": " || ".join(p.get("summary_bullets", [])) if isinstance(p.get("summary_bullets"), list) else "",
            })

        print(f"Done: {unitid} - {uni} ({len(candidates)} candidates)")

    pd.DataFrame(candidate_rows).to_csv("data/policy_candidates_pilot.csv", index=False)
    pd.DataFrame(policy_rows).to_csv("data/university_policies_pilot.csv", index=False)

    print("\nSaved:")
    print(" - data/policy_candidates_pilot.csv")
    print(" - data/university_policies_pilot.csv")
    print(" - data/raw/*.json and data/extracted/*.json")

if __name__ == "__main__":
    main()
