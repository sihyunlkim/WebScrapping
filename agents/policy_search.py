import os
from pathlib import Path
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

EXA_KEY = os.getenv("EXA_API_KEY")
assert EXA_KEY, "EXA_API_KEY not found"

exa = Exa(EXA_KEY)

POLICY_QUERIES = [
    "{uni} generative AI policy",
    "{uni} AI policy academic integrity",
    "{uni} ChatGPT policy students",
    "{uni} provost generative AI guidance",
    "{uni} teaching and learning generative AI guidance",
]

def search_policy_pages(university: str, num_results: int = 5):
    hits = []
    for template in POLICY_QUERIES:
        q = template.format(uni=university)
        res = exa.search(
            query=q,
            num_results=num_results,
            contents={"text": True},  # 본문 텍스트 가져오기
        )
        for r in res.results:
            hits.append({
                "query": q,
                "title": r.title,
                "url": r.url,
                "text": (r.text or "")[:7000],
            })

    # URL dedupe
    seen = set()
    deduped = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            deduped.append(h)
    return deduped
