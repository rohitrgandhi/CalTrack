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

# ── Shared nutrition prompt structure ─────────────────────────────────────────

NUTRITION_SCHEMA = """
Return ONLY a valid JSON object. No markdown. No explanation. Just the JSON.

{
  "name": "Full descriptive meal name",
  "items": ["item 1", "item 2", "item 3"],
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
  },
  "confidence": "high" or "medium" or "low",
  "notes": "One sentence about how this estimate was made."
}

Rules:
- protein, carbs, fat, fiber are in grams (numbers only, no unit)
- calories are in kcal (number only)
- For Indian food, use standard restaurant or home-cooked portion sizes
- Only include vitamins/minerals you can reasonably estimate; use null for others
- confidence: high = clearly identifiable food, medium = some uncertainty, low = hard to tell
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

def parse_nutrition_response(raw_text):
    """Clean and parse JSON from Claude's response."""
    text = raw_text.strip()
    # Strip markdown fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Step 1: Analyze a food photo."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        image_bytes = file.read()
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        media_type = file.content_type or "image/jpeg"

        prompt = f"""You are a professional nutritionist analyzing a food photo.
Identify every food item visible, estimate portion sizes, and calculate nutrition.
{NUTRITION_SCHEMA}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        data = parse_nutrition_response(response.choices[0].message.content)
        data["meal_slot"] = get_meal_slot()
        data["time"] = datetime.now().strftime("%I:%M %p")
        data["id"] = str(int(datetime.now().timestamp() * 1000))
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned an unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    """Step 2: Re-analyze based on user-corrected food name."""
    body = request.get_json()
    food_description = (body or {}).get("food", "").strip()

    if not food_description:
        return jsonify({"error": "No food description provided"}), 400

    try:
        prompt = f"""You are a professional nutritionist.
The user is describing a meal they just ate: "{food_description}"

Identify all likely components, estimate a standard serving portion, and calculate full nutrition.
{NUTRITION_SCHEMA}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        data = parse_nutrition_response(response.choices[0].message.content)
        data["meal_slot"] = get_meal_slot()
        data["time"] = datetime.now().strftime("%I:%M %p")
        data["id"] = str(int(datetime.now().timestamp() * 1000))
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned an unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)