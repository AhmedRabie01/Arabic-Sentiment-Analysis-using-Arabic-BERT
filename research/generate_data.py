
# ================================
# 2) IMPORTS
# ================================
import google.generativeai as genai
import json, os, time, re
import pandas as pd
from tqdm import tqdm

# ================================
# 3) CONFIG
# ================================
API_KEY = "AIzaSyC8gVwbxyQDgnmNihAUF164q9MUZA80wO4"

MODEL_NAME = "models/gemini-2.5-flash"

TARGET_TOTAL = 1000          # غيرها لـ 2000 لو حابب
SLEEP_SECONDS = 1.2          # مهم جدًا لتجنب rate limit

CHECKPOINT = "pharmacy_checkpoint.jsonl"
OUT_CSV = "pharmacy_messages.csv"

INTENTS = ["Complaint", "Inquiry", "Praise"]

TOPICS = {
    "Complaint": ["delivery","staff_behavior","waiting_time","price","availability"],
    "Inquiry": ["availability","price","insurance","delivery"],
    "Praise": ["staff_behavior","delivery","availability"]
}

SENTIMENT = {
    "Complaint": "negative",
    "Inquiry": "neutral",
    "Praise": "positive"
}

# ================================
# 4) INIT GEMINI
# ================================
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ================================
# 5) PROMPT
# ================================
def build_prompt(intent):
    topics = ", ".join(TOPICS[intent])
    return f"""
أنت خبير في تحليل رسائل عملاء صيدليات مصرية.

المطلوب:
أنشئ رسالة واحدة فقط لعميل صيدلية.

القواعد:
- لهجة مصرية عامية
- جملة واحدة
- لا تشخيص طبي
- لا تذكر اسم التصنيف

التصنيف: {intent}
المواضيع المحتملة: {topics}

أخرج JSON فقط بهذا الشكل:
{{
  "text": "...",
  "intent": "{intent}",
  "topic": "...",
  "sentiment": "{SENTIMENT[intent]}"
}}
""".strip()

# ================================
# 6) HELPERS
# ================================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None

def valid_row(r, intent):
    return (
        isinstance(r, dict)
        and r.get("intent") == intent
        and isinstance(r.get("text"), str)
        and len(r["text"]) > 8
        and r.get("topic") in TOPICS[intent]
    )

# ================================
# 7) LOAD CHECKPOINT
# ================================
rows = []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print("✅ Loaded checkpoint:", len(rows))

# ================================
# 8) GENERATION LOOP
# ================================
pbar = tqdm(total=TARGET_TOTAL, initial=len(rows), desc="Generating")

while len(rows) < TARGET_TOTAL:

    counts = {i: sum(1 for r in rows if r["intent"] == i) for i in INTENTS}
    intent = min(counts, key=counts.get)

    prompt = build_prompt(intent)

    try:
        response = model.generate_content(prompt)
        text = response.text
    except Exception as e:
        print("⚠️ API Error:", e)
        time.sleep(5)
        continue

    row = extract_json(text)

    if row and valid_row(row, intent):
        rows.append(row)
        with open(CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pbar.update(1)

    time.sleep(SLEEP_SECONDS)

pbar.close()

# ================================
# 9) SAVE FINAL CSV
# ================================
df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("✅ DONE")
print(df["intent"].value_counts())
df.head()
