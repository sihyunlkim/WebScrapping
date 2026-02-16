import os, json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_KEY, "GEMINI_API_KEY not found"

client = genai.Client(api_key=GEMINI_KEY)

def extract_university_policy_with_gemini(university: str, pages: list[dict]) -> dict:
    evidence = [
        {"title": p["title"], "url": p["url"], "text": p["text"]}
        for p in pages
        if p.get("text")
    ]

    schema_hint = {
        "university_name": "string",
        "policies": [
            {
                "policy_title": "string",
                "policy_url": "string",
                "publisher": "string (office/unit that issued it, if visible)",
                "last_updated": "string (date if visible, else empty)",
                "scope": "university-wide",
                "policy_types": ["integrity|teaching-guidance|student-guidance|data-privacy|research|other"],
                "summary_bullets": ["string", "string", "string"]
            }
        ]
    }

    prompt = f"""
You are extracting OFFICIAL university-wide AI / generative AI policies or guidance for {university}.

Rules:
- Select ONLY authoritative, official pages (e.g., provost, academic integrity office, teaching & learning center).
- Prefer the most current and most authoritative sources.
- Return up to 3 best policy/guidance pages total.
- Do NOT include course-level policies.
- Do NOT invent. Use evidence only.
- Return ONLY valid JSON (no markdown).
- scope must be exactly "university-wide".
- last_updated should be a date string if visible, else "".

JSON shape example:
{json.dumps(schema_hint, indent=2)}

Evidence pages:
{json.dumps(evidence)}
"""

    resp = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )

    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.replace("json", "", 1).strip()

    return json.loads(text)
