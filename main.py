import json
from pathlib import Path
import pandas as pd

from agents.org_search import search_org_pages
from agents.org_extract_gemini import extract_units_with_gemini

def run_one(university: str):
    pages = search_org_pages(university, num_results=4)

    # raw evidence 저장 (디버깅용)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open(f"data/raw/org_pages_{university.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    org = extract_units_with_gemini(university, pages)

    # extracted JSON 저장
    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    with open(f"data/extracted/org_{university.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(org, f, ensure_ascii=False, indent=2)

    # CSV로 평탄화
    rows = []
    for u in org["units"]:
        rows.append({
            "university_name": org.get("university_name", university),
            "unit_name": u.get("unit_name", ""),
            "unit_type": u.get("unit_type", ""),
            "unit_url": u.get("unit_url", ""),
            "parent_unit_name": u.get("parent_unit_name", ""),
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    uni = "New York University"
    df = run_one(uni)
    print(df.head(30))
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/university_units.csv", index=False)
    print("\nSaved -> data/university_units.csv")
