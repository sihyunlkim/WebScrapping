import re
from urllib.parse import urlparse

# 너무 일반적인 단어/메뉴 항목 필터
BLACKLIST_EXACT = {
    "home", "about", "academics", "admissions", "research", "news", "events",
    "contact", "directory", "privacy", "accessibility", "careers", "apply",
    "student life", "campus life", "giving", "alumni", "library", "libraries",
}

# school/college 느낌 나는 키워드
UNIT_KEYWORDS = (
    "school", "college", "faculty", "division", "institute", "center", "centre",
    "academy", "graduate", "undergraduate"
)

def _clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("•*-–—|:;,.()[]{}")
    return s

def _looks_like_unit(name: str) -> bool:
    n = name.lower()
    if len(name) < 6 or len(name) > 120:
        return False
    if n in BLACKLIST_EXACT:
        return False
    if any(k in n for k in UNIT_KEYWORDS):
        return True
    return False

def _guess_unit_type(name: str) -> str:
    n = name.lower()
    if "college" in n:
        return "college"
    if "school" in n:
        return "school"
    if "faculty" in n:
        return "faculty"
    if "division" in n:
        return "division"
    if "institute" in n or "center" in n or "centre" in n:
        return "other"
    return "other"

def extract_units_fallback(university: str, pages: list[dict]) -> dict:
    """
    LLM 없이 규칙 기반으로 school/college/division 후보를 뽑는다.
    pages: [{"title","url","text"}...]
    """
    units = []
    seen = set()

    for p in pages:
        text = p.get("text") or ""
        base_url = p.get("url") or ""

        # 텍스트를 줄 단위로 보고, 학교/단과대 같은 라인만 후보로
        for raw in text.splitlines():
            line = _clean_line(raw)
            if not line:
                continue

            # 너무 긴 문장 제외
            if line.count(" ") > 16:
                continue

            if not _looks_like_unit(line):
                continue

            key = (line.lower(), base_url)
            if key in seen:
                continue
            seen.add(key)

            units.append({
                "unit_name": line,
                "unit_type": _guess_unit_type(line),
                "unit_url": "",              # fallback이라 URL 매칭은 비워둠
                "parent_unit_name": ""
            })

    # 너무 많은 잡음이면 상위 N개만 (나중에 튜닝)
    units = units[:80]

    return {
        "university_name": university,
        "units": units
    }
