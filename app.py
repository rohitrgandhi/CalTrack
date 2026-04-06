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

# ── Schemas ───────────────────────────────────────────────────────────────────

ITEM_SCHEMA = """
{
  "name": "Item name e.g. Matki Sabji",
  "quantity": "Visible portion e.g. 1 cup (150g) or 2 rotis or 1 small bowl",
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "fiber": number,
  "vitamins": {
    "vitamin_a": "e.g. 120μg" or null,
    "vitamin_c": "e.g. 15mg" or null,
    "vitamin_d": "e.g. 2μg" or null,
    "vitamin_b12": "e.g. 0.5μg" or null,
    "vitamin_b6": "e.g. 0.3mg" or null,
    "folate": "e.g. 40μg" or null
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
"""

MEAL_SCHEMA = """
Return ONLY a valid JSON object. No markdown. No explanation. Just JSON.

{
  "meal_name": "Full descriptive meal name e.g. Indian Lunch Thali",
  "confidence": "high" or "medium" or "low",
  "notes": "One sentence about how estimates were made.",
  "items": [ <one entry per food item, using the item structure below> ]
}

Each item in the items array must follow this structure:
{
  "name": "Item name e.g. Matki Sabji",
  "quantity": "Visible portion e.g. 1 cup (150g) or 2 rotis or 1 small bowl",
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "fiber": number,
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

Rules:
- List EVERY individual food item visible as a SEPARATE entry in items[]
- protein, carbs, fat, fiber are numbers in grams (no unit string)
- calories are numbers in kcal (no unit string)
- quantity must clearly describe the visible portion size
- Use standard Indian home-cooked or restaurant portion sizes where applicable
- Only include vitamins/minerals you can reasonably estimate; null for others
- confidence: high=clearly identifiable, medium=some uncertainty, low=hard to tell
"""

def get_meal_slot():
    h = datetime.now().hour
    if h < 6:   return "Early Morning"
    if h < 10:  return "Breakfast"
    if h < 12:  return "Mid Morning"
    if h < 15:  return "Lunch"
    if h < 17:  return "Afternoon Snack"
    if h < 20:  return "Dinner"
    return "Late Night"

def parse_response(raw_text):
    text = raw_text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def add_meta(data):
    data["meal_slot"] = get_meal_slot()
    data["time"] = datetime.now().strftime("%I:%M %p")
    data["id"] = str(int(datetime.now().timestamp() * 1000))
    return data

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze a food photo — returns per-item breakdown."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

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

        data = parse_response(response.choices[0].message.content)
        return jsonify({"success": True, "data": add_meta(data)})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    """Re-analyze full meal from a text description."""
    body = request.get_json() or {}
    food = body.get("food", "").strip()
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

        data = parse_response(response.choices[0].message.content)
        return jsonify({"success": True, "data": add_meta(data)})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/correct-item", methods=["POST"])
def correct_item():
    """Re-analyze a single wrong item — returns just that one item's data."""
    body = request.get_json() or {}
    correct_name = body.get("correct_name", "").strip()
    if not correct_name:
        return jsonify({"error": "No correction provided"}), 400

    try:
        prompt = f"""You are a professional nutritionist.
Give a full nutrition breakdown for a single food item: "{correct_name}"

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "name": "{correct_name}",
  "quantity": "standard single serving description",
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "fiber": number,
  "vitamins": {{
    "vitamin_a": "value with unit or null",
    "vitamin_c": "value with unit or null",
    "vitamin_d": "value with unit or null",
    "vitamin_b12": "value with unit or null",
    "vitamin_b6": "value with unit or null",
    "folate": "value with unit or null"
  }},
  "minerals": {{
    "iron": "value with unit or null",
    "calcium": "value with unit or null",
    "potassium": "value with unit or null",
    "magnesium": "value with unit or null",
    "zinc": "value with unit or null",
    "sodium": "value with unit or null"
  }}
}}

Rules:
- protein, carbs, fat, fiber are numbers in grams
- calories are numbers in kcal
- Use null for vitamins/minerals you cannot estimate confidently
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        item = parse_response(response.choices[0].message.content)
        return jsonify({"success": True, "item": item})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)