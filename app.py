import os
import base64
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Shared item structure ─────────────────────────────────────────────────────

ITEM_STRUCTURE = """
Each item in the items array must follow this EXACT structure:
{
  "name": "JUST the food name — NO numbers, NO quantities. e.g. 'Idli' not '2 Idlis'",
  "unit": "what ONE unit looks like — e.g. '1 piece (45g)' or '1 bowl (200ml)' or '1 roti'",
  "qty": <integer — how many of that unit the user ate. e.g. 2 for 2 idlis>,
  "calories": <kcal for ONE unit>,
  "protein": <grams for ONE unit>,
  "carbs":   <grams for ONE unit>,
  "fat":     <grams for ONE unit>,
  "fiber":   <grams for ONE unit>,
  "vitamins": {
    "vitamin_a": "e.g. 120ug" or null,
    "vitamin_c": "e.g. 15mg" or null,
    "vitamin_d": "e.g. 2ug" or null,
    "vitamin_b12": "e.g. 0.5ug" or null,
    "vitamin_b6": "e.g. 0.3mg" or null,
    "folate": "e.g. 40ug" or null
  },
  "minerals": {
    "iron": "e.g. 2mg" or null,
    "calcium": "e.g. 80mg" or null,
    "potassium": "e.g. 420mg" or null,
    "magnesium": "e.g. 30mg" or null,
    "zinc": "e.g. 1mg" or null,
    "sodium": "e.g. 200mg" or null
  }
}

CRITICAL QUANTITY RULES:
- 'name' must NEVER contain numbers or quantity words (no '2 Idlis', no 'two pieces')
- 'unit' describes ONE single piece/bowl/cup
- 'qty' is the count of units the user ate (integer, minimum 1)
- ALL macros (calories, protein, carbs, fat, fiber) are for ONE unit only
- The app will multiply by qty automatically — do NOT pre-multiply
- Example: user ate 2 idlis → name='Idli', unit='1 piece (45g)', qty=2, calories=77 (for 1 idli)
"""

MEAL_SCHEMA = f"""
Return ONLY valid JSON. No markdown. No explanation outside the JSON.

{{
  "meal_name": "Descriptive meal name",
  "confidence": "high" or "medium" or "low",
  "notes": "One sentence about estimation.",
  "items": [ <one entry per food item> ]
}}

{ITEM_STRUCTURE}

Rules:
- List EVERY individual food item as a SEPARATE entry in items[]
- 'name' = food name only, zero numbers or quantity words
- 'unit' = description of ONE piece/bowl/serving
- 'qty' = integer count of how many units visible/described
- All macros are PER ONE UNIT, plain numbers, no unit strings
- Use standard Indian home-cooked / restaurant portion sizes
- null for vitamins/minerals you cannot estimate confidently
"""

def get_meal_number(key, logs):
    """Return next meal number for the day."""
    existing = logs.get(key, [])
    return len(existing) + 1

def parse_response(raw):
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def normalise_items(data):
    """
    Ensure every item uses the name/unit/qty convention.
    Falls back gracefully if the AI used the old 'quantity' field.
    """
    import re
    for item in data.get("items", []):
        # --- qty ---
        if "qty" not in item or not isinstance(item.get("qty"), (int, float)):
            # Try to extract a number from the old 'quantity' string
            qty_str = str(item.get("quantity") or item.get("unit") or "1")
            nums = re.findall(r"\d+\.?\d*", qty_str)
            item["qty"] = int(float(nums[0])) if nums else 1
        item["qty"] = max(1, int(item["qty"]))

        # --- unit ---
        if "unit" not in item:
            item["unit"] = item.get("quantity", "1 serving")

        # --- name: strip leading numbers e.g. "2 Idlis" → "Idli" ---
        name = item.get("name", "Food")
        name = re.sub(r"^\d+\s+", "", name)          # "2 Idlis" → "Idlis"
        name = re.sub(r"s$", "", name) if len(name) > 3 else name  # "Idlis" → "Idli"
        item["name"] = name.strip()

        # --- remove old 'quantity' key to avoid confusion ---
        item.pop("quantity", None)

    return data

def add_meta(data, meal_num):
    data["meal_num"] = meal_num
    data["time"] = datetime.now().strftime("%I:%M %p")
    data["id"] = str(int(datetime.now().timestamp() * 1000))
    return data

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze a food photo — full per-item breakdown."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    meal_num = int(request.form.get("meal_num", 1))

    try:
        b64 = base64.standard_b64encode(file.read()).decode("utf-8")
        media_type = file.content_type or "image/jpeg"

        prompt = f"""You are a professional nutritionist analyzing a food photo.
Identify EVERY individual food item visible. Give each item its own full nutrition breakdown.
{MEAL_SCHEMA}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        data = normalise_items(parse_response(response.choices[0].message.content))
        return jsonify({"success": True, "data": add_meta(data, meal_num)})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    """Analyze a manually typed food description."""
    body = request.get_json() or {}
    food = body.get("food", "").strip()
    meal_num = int(body.get("meal_num", 1))
    if not food:
        return jsonify({"error": "No food description provided"}), 400

    try:
        prompt = f"""You are a professional nutritionist.
The user ate: "{food}"
List every component as a separate item with its own full nutrition breakdown.
{MEAL_SCHEMA}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        data = normalise_items(parse_response(response.choices[0].message.content))
        return jsonify({"success": True, "data": add_meta(data, meal_num)})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/correct-item", methods=["POST"])
def correct_item():
    """Re-analyze a single wrong item."""
    body = request.get_json() or {}
    correct_name = body.get("correct_name", "").strip()
    if not correct_name:
        return jsonify({"error": "No correction provided"}), 400

    try:
        prompt = f"""You are a professional nutritionist.
Full nutrition breakdown for a single serving of: "{correct_name}"

Return ONLY valid JSON, no markdown:
{{
  "name": "{correct_name}",
  "quantity": "standard single serving",
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "fiber": number,
  "vitamins": {{
    "vitamin_a": "value or null", "vitamin_c": "value or null",
    "vitamin_d": "value or null", "vitamin_b12": "value or null",
    "vitamin_b6": "value or null", "folate": "value or null"
  }},
  "minerals": {{
    "iron": "value or null", "calcium": "value or null",
    "potassium": "value or null", "magnesium": "value or null",
    "zinc": "value or null", "sodium": "value or null"
  }}
}}
All macros are numbers in grams. Calories in kcal. null for unknowns."""

        response = client.chat.completions.create(
            model="gpt-4o", max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        item = parse_response(response.choices[0].message.content)
        return jsonify({"success": True, "item": item})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/protein-powder", methods=["POST"])
def protein_powder():
    """Look up protein powder nutrition by brand/flavour/scoop size."""
    body = request.get_json() or {}
    brand   = str(body.get("brand",   "")).strip()
    flavour = str(body.get("flavour", "")).strip()
    scoop_g = str(body.get("scoop_g", "")).strip()   # safely cast — may arrive as number
    scoops  = int(body.get("scoops",  1))

    if not brand:
        return jsonify({"error": "Brand name required"}), 400

    # build a human-readable description for the prompt
    desc = brand
    if flavour: desc += f" {flavour} flavour"
    if scoop_g: desc += f", {scoop_g}g per scoop"

    # build quantity string safely
    try:
        total_g = int(scoop_g) * scoops
        qty_str = f"{scoops} scoop(s) ({total_g}g)"
    except (ValueError, TypeError):
        qty_str = f"{scoops} scoop(s)"

    try:
        prompt = f"""You are a sports nutrition expert with access to real product label data.
Look up the exact nutrition for: "{desc}"
The user consumed {scoops} scoop(s).

Return ONLY valid JSON, no markdown, no extra text:
{{
  "name": "{brand}{' - ' + flavour if flavour else ''} Protein Powder",
  "quantity": "{qty_str}",
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "fiber": number,
  "vitamins": {{
    "vitamin_b12": "value with unit or null",
    "vitamin_d":   "value with unit or null",
    "vitamin_b6":  "value with unit or null",
    "folate": null, "vitamin_a": null, "vitamin_c": null
  }},
  "minerals": {{
    "calcium":   "value with unit or null",
    "iron":      "value with unit or null",
    "potassium": "value with unit or null",
    "magnesium": "value with unit or null",
    "zinc":      "value with unit or null",
    "sodium":    "value with unit or null"
  }}
}}
Rules:
- All macros (protein, carbs, fat, fiber) are plain numbers in grams
- calories is a plain number in kcal
- Scale ALL values to {scoops} scoop(s)
- If you recognise this brand and flavour, use the actual label data
- If unknown, estimate from typical Indian whey protein (e.g. MuscleBlaze, Whole Truth)
- null for any vitamin/mineral you cannot estimate"""

        response = client.chat.completions.create(
            model="gpt-4o", max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        item = parse_response(response.choices[0].message.content)
        return jsonify({"success": True, "item": item})

    except json.JSONDecodeError:
        return jsonify({"error": "Could not look up this product. Try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)