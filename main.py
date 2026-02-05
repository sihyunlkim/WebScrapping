import os
from dotenv import load_dotenv
from exa_py import Exa
from openai import OpenAI

# load API keys
load_dotenv()

assert os.getenv("EXA_API_KEY"), "EXA_API_KEY not found"

exa = Exa(os.environ["EXA_API_KEY"])
#client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def search_ai_policy(university: str):
    query = f"{university} generative AI policy"
    results = exa.search(
        query=query,
        num_results=5,
        contents={"text": True}
    )

    print(f"\n=== {university} ===")
    for r in results.results:
        print("-", r.title)
        print(" ", r.url)

if __name__ == "__main__":
    universities = [
        "New York University",
        "University of Washington",
        "Carnegie Mellon University",
        "Stanford University",
        "Harvard University",
    ]
    for u in universities:
        search_ai_policy(u)
