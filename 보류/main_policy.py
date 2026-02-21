import json
from pathlib import Path
import pandas as pd

from policy_search import search_policy_pages
from agents.policy_extract_gemini import extract_university_policy_with_gemini

def run_one(university: str):
    pages = search_policy_pages(university, num_results=4)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    raw_path = f"data/raw/policy_pages_{university.replace(' ', '_')}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    extracted = extract_university_policy_with_gemini(university, pages)

    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    ext_path = f"data/extracted/policies_{university.replace(' ', '_')}.json"
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    rows = []
    for p in extracted.get("policies", []):
        rows.append({
            "university_name": extracted.get("university_name", university),
            "scope": p.get("scope", "university-wide"),
            "policy_title": p.get("policy_title", ""),
            "policy_url": p.get("policy_url", ""),
            "publisher": p.get("publisher", ""),
            "last_updated": p.get("last_updated", ""),
            "policy_types": "|".join(p.get("policy_types", [])) if isinstance(p.get("policy_types"), list) else "",
            "summary_bullets": " || ".join(p.get("summary_bullets", [])) if isinstance(p.get("summary_bullets"), list) else "",
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    universities = [
        "New York University",
        "University of Washington",
        "Carnegie Mellon University",
        "Stanford University",
        "Harvard University",
    ]

    all_df = []
    for uni in universities:
        df = run_one(uni)
        print(f"\n=== {uni} ===")
        print(df[["policy_title", "policy_url"]].head(5))
        all_df.append(df)

    out = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/university_policies.csv", index=False)
    print("\nSaved -> data/university_policies.csv")
