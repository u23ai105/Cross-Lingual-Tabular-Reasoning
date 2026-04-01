import os
import json
import base64
import time
import re
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import io

# ================= CONFIGURATION =================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

BASE_IMAGE_DIR = os.environ.get("BASE_IMAGE_DIR", "images")

# 🧪 OPEN WEIGHT MODELS (Must use Vision/VL versions)
MODELS_TO_TEST = [
    # "qwen/qwen2.5-vl-72b-instruct",  # 🏆 Best Open Model
    # "meta-llama/llama-3.2-90b-vision-instruct",  # 🥈 Best Reasoning
    # "mistralai/pixtral-12b",  # 🥉 Best Fast/Efficient
      "google/gemma-3-27b-it",
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ================= HELPER FUNCTIONS =================

def resize_and_encode_image(image_path, max_size=(1024, 1024)):
    """
    Resizes image to max 1024x1024 to reduce tokens/latency,
    then encodes to base64.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            img.thumbnail(max_size)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def get_image_path(folder_path, image_ref):
    if not image_ref: return None
    candidates = [image_ref, image_ref + ".png", image_ref + ".jpg", image_ref + ".jpeg"]
    for cand in candidates:
        full_path = os.path.join(folder_path, cand)
        if os.path.exists(full_path):
            return full_path
    return None


def extract_json_from_text(text):
    """
    Manually extracts JSON from text using regex.
    """
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def evaluate_with_retry(model_id, image_b64, question, options):
    """
    Sends request to API.
    Runs fast, but pauses/retries if the provider (DeepInfra) gets overloaded.
    """
    options_text = "\n".join([f"- {opt}" for opt in options])

    system_prompt = (
        "You are an expert data analyst. Analyze the chart/table. "
        "Answer with a JSON object containing 'reasoning' and 'final_answer'. "
        "The 'final_answer' must be one of the provided options."
    )
    user_prompt = f"Question: {question}\nOptions:\n{options_text}"

    max_retries = 5  # Increased to 5 to handle temporary provider outages

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content
            parsed = extract_json_from_text(content)

            if parsed:
                return parsed
            else:
                # If model returns garbage, treat as failure and maybe retry
                # For now, we return the raw content so you can see the error
                return {"reasoning": "Failed to parse JSON", "final_answer": content}

        except Exception as e:
            error_str = str(e).lower()

            # SPECIFIC HANDLING FOR RATE LIMITS (429) OR OVERLOADS
            if "429" in error_str or "rate limit" in error_str:
                wait_time = 10  # Wait 10 seconds to let the provider cool down
                print(f"\n🛑 Provider Busy (429). Pausing for {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️ API Error: {e}")
                # Don't retry on other fatal errors (like 400 Bad Request)
                if "400" in error_str:
                    return {"reasoning": f"Fatal API Error: {str(e)}", "final_answer": "Error"}
                time.sleep(1)  # Small safety wait for network blips

    return {"reasoning": "Max retries exceeded", "final_answer": "Error"}


# ================= MAIN EXECUTION =================

def main():
    results = []

    if not os.path.exists(BASE_IMAGE_DIR):
        print(f"❌ Error: Directory '{BASE_IMAGE_DIR}' not found.")
        return

    print(f"🚀 Starting HIGH SPEED Evaluation on {len(MODELS_TO_TEST)} models...")

    for root, _, files in os.walk(BASE_IMAGE_DIR):
        json_files = [f for f in files if f.endswith(".json") and ("gemini" in f or "gpt" in f)]

        for j_file in json_files:
            path = os.path.join(root, j_file)
            print(f"\n📂 Processing: {j_file}...")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content: continue
                    data = json.loads(content)
            except Exception as e:
                print(f"⚠️ Skipping {j_file}: {e}")
                continue

            # Process questions one by one
            for item in tqdm(data, desc="Evaluating"):
                question = item.get("question")
                options = item.get("options")
                image_ref = item.get("image_reference")
                correct_answer = item.get("correct_answer")

                if not question or not options or not image_ref: continue

                img_path = get_image_path(root, image_ref)
                if not img_path: continue

                img_b64 = resize_and_encode_image(img_path)
                if not img_b64: continue

                for model in MODELS_TO_TEST:
                    res = evaluate_with_retry(model, img_b64, question, options)

                    ans_clean = str(res.get("final_answer", "")).lower().strip()
                    corr_clean = str(correct_answer).lower().strip()
                    is_correct = corr_clean in ans_clean and len(ans_clean) > 0

                    results.append({
                        "model": model,
                        "file": j_file,
                        "question": question,
                        "correct_answer": correct_answer,
                        "model_answer": res.get("final_answer", ""),
                        "is_correct": is_correct,
                        "reasoning": res.get("reasoning", "")
                    })

                    # 🚀 NO SLEEP HERE - We rely on evaluate_with_retry to handle brakes

    # --- SUMMARY ---
    print("\n" + "=" * 40)
    print("📊 FINAL RESULTS")
    print("=" * 40)

    model_stats = {}
    for r in results:
        m = r["model"]
        if m not in model_stats: model_stats[m] = {"correct": 0, "total": 0, "errors": 0}

        if r["model_answer"] == "Error":
            model_stats[m]["errors"] += 1
        else:
            model_stats[m]["total"] += 1
            if r["is_correct"]: model_stats[m]["correct"] += 1

    for m, stats in model_stats.items():
        total = stats["total"]
        if total > 0:
            acc = (stats["correct"] / total) * 100
            print(f"Model: {m}")
            print(f"  Accuracy: {acc:.2f}% ({stats['correct']}/{total})")
            print(f"  API Errors: {stats['errors']}")
            print("-" * 20)

    with open("open_weights_results2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✅ Saved to 'open_weights_results2.json'")


if __name__ == "__main__":
    main()


# ========================================
# 📊 FINAL RESULTS
# ========================================
# Model: qwen/qwen2.5-vl-72b-instruct
#   Accuracy: 62.22% (84/135)
#   API Errors: 0
# --------------------
# Model: mistralai/pixtral-12b
#   Accuracy: 39.26% (53/135)
#   API Errors: 0
# --------------------
# Model: google/gemma-3-27b-it
#   Accuracy: 51.11% (69/135)
#   API Errors: 0
# --------------------