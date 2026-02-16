import json
from pathlib import Path
import pandas as pd

from policy_search import search_policy_pages 

DATA_PATH = "hd2024_data_stata 2.csv"  

def main():
    df = pd.read_csv(DATA_PATH)

    # 필수 컬럼만
    df = df[["UNITID", "INSTNM", "WEBADDR"]].copy()
    
    target_universities = [
        #"Harvard University",
        #"Carnegie Mellon University",
        "New York University",
        "University of Washington-Seattle Campus",
        #"Stanford University"
    ]
    #testing 5

    pilot = df[df["INSTNM"].str.contains("|".join(target_universities), case=False, na=False)]
    

    print(f"Found {len(pilot)} universities:")
    print(pilot[["INSTNM"]].to_string())
    print()



    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    candidate_rows = []

    for _, row in pilot.iterrows():
        unitid = int(row["UNITID"])
        uni = str(row["INSTNM"])
        web = str(row["WEBADDR"]) if pd.notna(row["WEBADDR"]) else ""

        # domain-restricted search만 수행
        candidates_domain = search_policy_pages(
            university=uni,
            website=web,
            #num_results=100
            search_type= "deep", 
            restrict_domain=True
        )

        # raw 저장 (content 포함)
        with open(f"data/raw/{unitid}_candidates_domain.json", "w", encoding="utf-8") as f:
            json.dump(candidates_domain, f, ensure_ascii=False, indent=2)

        # CSV row로 정리
        for rank, c in enumerate(candidates_domain, start=1):
            candidate_rows.append({
                "UNITID": unitid,
                "university_name": uni,
                "website": web,
                "rank": rank,
                "query": c.get("query", ""),
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "domain": c.get("domain", ""),
                "domain_restricted": c.get("domain_restricted", False),
                "text_len": len(c.get("text", "") or ""),
            })

        print(f"Done: {unitid} - {uni}")
        print(f"  domain candidates: {len(candidates_domain)}")

    out = pd.DataFrame(candidate_rows)
    out.to_csv("data/policy_candidates_pilot.csv", index=False)
    print("\nSaved -> data/policy_candidates_pilot.csv")
    print("Raw JSON saved under -> data/raw/")

if __name__ == "__main__":
    main()