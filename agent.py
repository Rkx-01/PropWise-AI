import os
import json
import re
import textwrap
import logging
from typing import Optional

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME   = "llama-3.3-70b-versatile"   # highly capable, fast, and supported for free on Groq
TEMPERATURE  = 0.3
MAX_TOKENS   = 1500

SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are a cautious, professional real estate advisor with expertise in
    property valuation and market analysis.

    STRICT RULES you must always follow:
    1. Never make specific investment guarantees or promises of returns.
    2. Never invent market statistics not supported by the data provided.
    3. Always remind the reader that valuations are model-based estimates.
    4. Use measured, evidence-based language (e.g. "may", "suggests", "typically").
    5. Recommend consulting a licensed real estate professional for final decisions.
    6. Base every insight solely on the property data and predicted price supplied.
""")

REPORT_SCHEMA = {
    "property_summary"       : "A concise 2-3 sentence factual overview of the property.",
    "price_interpretation"   : "What the predicted price implies about the property's market position.",
    "market_trend_insights"  : "General market factors that typically influence this type of property.",
    "recommended_actions"    : "Practical next steps for a buyer or seller, without investment guarantees.",
    "supporting_references"  : "A JSON array of 3-5 short generic real-estate guidance references (no URLs needed).",
    "legal_disclaimer"       : "A brief disclaimer that this is an AI-generated estimate, not financial advice.",
}

def _build_prompt(property_data: dict, predicted_price: float) -> str:
    prop_lines = "\n".join(f"  • {k.replace('_', ' ').title()}: {v}" for k, v in property_data.items())
    schema_lines = "\n".join(f'  "{key}": <{hint}>' for key, hint in REPORT_SCHEMA.items())
    return textwrap.dedent(f"""\
        ## Property Details
        {prop_lines}

        ## Model-Predicted Price
          ₹ {predicted_price:,.2f}   (machine-learning estimate, ±15% typical margin)

        ---

        Generate a structured real-estate advisory report for this property.
        Return ONLY a valid JSON object with exactly these keys
        (no markdown fences, no extra keys, no trailing commas):

        {{
        {schema_lines}
        }}

        For "supporting_references" return a JSON array of plain strings, e.g.:
        ["Reference 1", "Reference 2", "Reference 3"]
        
        Remember: Do not add any text before or after the JSON.
    """)

def _fallback_report(reason: str) -> dict:
    return {
        "property_summary": f"Advisory report unavailable. Reason: {reason}",
        "price_interpretation": "Unable to interpret price at this time.",
        "market_trend_insights": "Market insights could not be generated.",
        "recommended_actions": "Please retry later or consult a licensed real estate professional.",
        "supporting_references": ["General Buyer/Seller Guide", "Local guidelines"],
        "legal_disclaimer": "This is an AI-assisted tool. All outputs are indicative estimates only and do not constitute financial, legal, or investment advice.",
        "error": reason,
    }

def _parse_response(raw_text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match: return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    refs = data.get("supporting_references", [])
    if isinstance(refs, str):
        try: refs = json.loads(refs)
        except Exception: refs = [refs]
    data["supporting_references"] = refs if isinstance(refs, list) else [str(refs)]
    return data

def generate_advisory_report(property_data: dict, predicted_price: float) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return _fallback_report("GROQ_API_KEY not found. Set it in your environment or a .env file.")

    try:
        client = Groq(api_key=api_key)
    except Exception as exc:
        return _fallback_report(f"Groq client init failed: {exc}")

    prompt = _build_prompt(property_data, predicted_price)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        raw_text = response.choices[0].message.content
    except Exception as exc:
        return _fallback_report(f"Groq API error: {exc}")

    report = _parse_response(raw_text)
    if report is None:
        return _fallback_report("Could not parse structured response from LLM.")
    return report

if __name__ == "__main__":
    pass
