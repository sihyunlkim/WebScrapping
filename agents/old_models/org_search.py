import os
from pathlib import Path
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

EXA_KEY = os.getenv("EXA_API_KEY")
assert EXA_KEY, "EXA_API_KEY not found"

exa = Exa(EXA_KEY)

ORG_QUERIES = [
    "{uni} schools and colleges",
    "{uni} colleges and schools",
    "{uni} academic units",
    "{uni} departments directory",
    "{uni} schools colleges departments list",
]

def search_org_pages(university: str, num_results: int = 5):
    hits = []
    for template in ORG_QUERIES:
        q = template.format(uni=university)
        res = exa.search(
            query=q,
            num_results=num_results,
            contents={"text": True},  # 본문 text 가져오기
        )
        for r in res.results:
            hits.append({
                "query": q,
                "title": r.title,
                "url": r.url,
                "text": (r.text or "")[:6000],
            })

    # URL dedupe
    seen = set()
    deduped = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            deduped.append(h)
    return deduped
