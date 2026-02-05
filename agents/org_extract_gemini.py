import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_KEY, "GEMINI_API_KEY not found"

client = genai.Client(api_key=GEMINI_KEY)

def extract_units_with_gemini(university: str, pages: list[dict]) -> dict:
    evidence = [
        {"title": p["title"], "url": p["url"], "text": p["text"]}
        for p in pages
        if p.get("text")
    ]

    schema_hint = {
        "university_name": "string",
        "units": [
            {
                "unit_name": "string",
                "unit_type": "school|college|division|faculty|department|other",
                "unit_url": "string (may be empty)",
                "parent_unit_name": "string (may be empty)"
            }
        ]
    }

    prompt = f"""
Extract the official organizational units (schools/colleges/divisions) for: {university}.

Rules:
- Prioritize top-level academic units (schools/colleges). Include departments ONLY if clearly listed under a specific school.
- Do NOT invent units. Use only evidence.
- Return ONLY valid JSON (no markdown, no commentary).
- Always include unit_url and parent_unit_name keys (use "" if unknown).

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

    # Sometimes it wraps in ```json ... ```
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.replace("json", "", 1).strip()

    return json.loads(text)
