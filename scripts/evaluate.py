"""
QA Evaluator  —  Qwen3-VL-235B  (DeepInfra)
=============================================
Questions were generated from the ENGLISH image.
Evaluation tests the model using the HINDI image for the same chart.

Directory structure expected (same as generation script):
    base_dir/
    ├── echapter/          ← English images (used for generation)
    │   ├── page_101.jpg
    │   └── ...
    ├── hechapter/         ← Hindi images (used for evaluation)
    │   ├── page_101.jpg   ← same filename, Hindi text
    │   └── ...
    └── verified_bloom_QA.json

Usage:
    pip install openai pillow python-dotenv tqdm

    # .env file:
    DEEPINFRA_API_KEY=<your_key>

    python evaluate_hindi_qa.py \
        --qa_file   /path/to/verified_bloom_QA.json \
        --base_dir  /path/to/base_dir \
        --output_dir ./results
"""

import os
import sys
import json
import base64
import re
import time
import io
import argparse
from pathlib import Path
from typing import Optional

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

MODEL       = os.environ.get("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
MAX_RETRIES = 3

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=DEEPINFRA_API_KEY,
)

SYSTEM_PROMPT = (
    "You will be shown a chart or table image and a multiple-choice question with four options.\n\n"
    "Your task:\n"
    "1. Carefully read the image and the question.\n"
    "2. Select the single correct option.\n"
    "3. Provide a concise explanation for why that option is correct.\n\n"
    "Respond ONLY in this exact JSON format (no markdown, no extra text):\n"
    '{\n'
    '  "selected_option": "<exact text of the chosen option>",\n'
    '  "explanation": "<your reasoning>"\n'
    '}'
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. IDENTIFY ENGLISH / HINDI FOLDERS
#    (exact same logic as generation script)
# ─────────────────────────────────────────────────────────────────────────────
def identify_english_folder(folder1_name: str, folder2_name: str) -> tuple[str, str]:
    """
    Returns (english_folder_name, hindi_folder_name).

    Detection priority:
      1. Folder name contains 'hin' / 'hindi'  → that is Hindi
      2. Folder name contains 'eng' / 'english' → that is English
      3. Alphabetical fallback (echapter < hechapter → echapter = English)
    """
    hindi_keywords   = ['hin', 'hindi']
    english_keywords = ['eng', 'english']

    f1_lower = folder1_name.lower()
    f2_lower = folder2_name.lower()

    if any(kw in f1_lower for kw in hindi_keywords):
        return folder2_name, folder1_name

    if any(kw in f2_lower for kw in hindi_keywords):
        return folder1_name, folder2_name

    if any(kw in f1_lower for kw in english_keywords):
        return folder1_name, folder2_name

    if any(kw in f2_lower for kw in english_keywords):
        return folder2_name, folder1_name

    # Alphabetical fallback
    sorted_folders = sorted([folder1_name, folder2_name])
    print(f"  [WARN] Cannot detect language from folder names "
          f"'{folder1_name}' / '{folder2_name}'. "
          f"Defaulting: '{sorted_folders[0]}' = English.")
    return sorted_folders[0], sorted_folders[1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD ALL QUESTIONS
#    No subfolder filter — generation now produces one set per image (English only)
# ─────────────────────────────────────────────────────────────────────────────
def load_all_questions(qa_file: str) -> list[dict]:
    """
    Loads all questions from verified_bloom_QA.json.
    Since the generation script now uses only the English image,
    all questions have source_subfolder = English folder name.
    """
    with open(qa_file, "r", encoding="utf-8") as f:
        all_qs = json.load(f)

    subfolder_counts = {}
    for q in all_qs:
        sf = q.get("source_subfolder", "unknown")
        subfolder_counts[sf] = subfolder_counts.get(sf, 0) + 1

    print(f"[INFO] Loaded {len(all_qs)} questions:")
    for sf, count in sorted(subfolder_counts.items()):
        print(f"         {sf}: {count} questions")

    return all_qs


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE HELPERS  (identical to generation script)
# ─────────────────────────────────────────────────────────────────────────────
def resize_and_encode_image(image_path: str, max_size: tuple = (1024, 1024)) -> Optional[str]:
    """Resizes image and encodes to base64 — identical to generation code."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(max_size)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"  Error processing image {image_path}: {e}")
        return None


def find_image(folder_path: str, filename: str) -> Optional[str]:
    """Locate image in folder; case-insensitive fallback."""
    direct = Path(folder_path) / filename
    if direct.exists():
        return str(direct)
    for f in Path(folder_path).iterdir():
        if f.name.lower() == filename.lower():
            return str(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON EXTRACTOR  (identical to generation script)
# ─────────────────────────────────────────────────────────────────────────────
def extract_json_from_text(text: str):
    """Robust JSON extractor — mirrors generation code."""
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    match_dict = re.search(r'\{.*\}', text, re.DOTALL)
    if match_dict:
        try:
            return json.loads(match_dict.group())
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. CALL THE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def call_model(question: str, options: list[str], image_b64: str) -> dict:
    """
    Sends Hindi image + question + 4 options to Qwen3-VL on DeepInfra.
    Handles 429 rate limits the same way as generation code.
    """
    options_text = "\n".join(f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options))
    user_text = (
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Look at the image carefully and pick the correct option."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text",      "text": user_text},
        ]},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            raw    = response.choices[0].message.content
            parsed = extract_json_from_text(raw)

            if parsed and isinstance(parsed, dict):
                if "selected_option" in parsed and "explanation" in parsed:
                    return parsed

            print(f"  Invalid JSON (Attempt {attempt + 1}). Retrying...")
            time.sleep(1)

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"  Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  API Error: {e}")
                return {"selected_option": None, "explanation": f"API error: {e}"}

    return {"selected_option": None, "explanation": f"All {MAX_RETRIES} retries failed."}


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHECK CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────
def is_correct(selected: Optional[str], correct: str, options: list[str]) -> bool:
    if selected is None:
        return False

    sel = selected.strip().lower()
    cor = correct.strip().lower()

    if sel == cor:
        return True

    # If correct_answer is stored as a letter (A/B/C/D), map to option text
    if len(correct.strip()) == 1 and correct.strip().upper() in "ABCD":
        idx = ord(correct.strip().upper()) - ord("A")
        if 0 <= idx < len(options) and sel == options[idx].strip().lower():
            return True

    # Substring containment
    if cor in sel or sel in cor:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(questions: list[dict], base_dir: str, hindi_folder: str, output_dir: str):
    """
    For every question:
      1. Look up the image from the HINDI folder  (base_dir/hindi_folder/image_reference)
         Questions were generated from English images — we test using Hindi images.
      2. Encode the image
      3. Send to Qwen3-VL → selected_option + explanation
      4. Compare against correct_answer
      5. Save correctly_answered.json and wrongly_answered.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hindi_folder_path = os.path.join(base_dir, hindi_folder)

    correct_list = []
    wrong_list   = []
    skipped      = []

    print(f"\n{'='*60}")
    print(f"  Model        : {MODEL}")
    print(f"  Total Qs     : {len(questions)}")
    print(f"  Hindi folder : {hindi_folder_path}")
    print(f"{'='*60}\n")

    for q in tqdm(questions, desc="Evaluating"):
        q_id      = q.get("id", "unknown")
        img_ref   = q.get("image_reference", "")
        question  = q.get("question", "")
        options   = q.get("options", [])
        correct   = q.get("correct_answer", "")
        reasoning = q.get("reasoning", "")

        # ── Find image in the HINDI folder ────────────────────────────────
        img_path = find_image(hindi_folder_path, img_ref)
        if not img_path:
            print(f"  [SKIP] Image not found in Hindi folder: {img_ref}")
            skipped.append({
                "id"             : q_id,
                "image_reference": img_ref,
                "reason"         : f"not found in {hindi_folder}",
            })
            continue

        # ── Encode image ──────────────────────────────────────────────────
        img_b64 = resize_and_encode_image(img_path)
        if not img_b64:
            skipped.append({
                "id"             : q_id,
                "image_reference": img_ref,
                "reason"         : "encoding failed",
            })
            continue

        # ── Query model with Hindi image ──────────────────────────────────
        output      = call_model(question, options, img_b64)
        selected    = output.get("selected_option")
        explanation = output.get("explanation", "")

        # ── Build result record ───────────────────────────────────────────
        record = {
            "id"                    : q_id,
            "image_reference"       : img_ref,
            "image_used_for_eval"   : f"{hindi_folder}/{img_ref}",   # makes it explicit
            "source_subfolder"      : q.get("source_subfolder", ""), # English folder (generation)
            "taxonomy_level"        : q.get("taxonomy_level", ""),
            "question"              : question,
            "options"               : options,
            "correct_answer"        : correct,
            "model_selected"        : selected,
            "model_explanation"     : explanation,
            "verified_reasoning"    : reasoning,
        }

        if is_correct(selected, correct, options):
            correct_list.append(record)
        else:
            wrong_list.append({
                **record,
                "wrong_option_picked": selected,
                "correct_option"     : correct,
            })

        time.sleep(2)   # same courtesy delay as generation code

    # ── Accuracy ──────────────────────────────────────────────────────────
    attempted = len(correct_list) + len(wrong_list)
    accuracy  = (len(correct_list) / attempted * 100) if attempted > 0 else 0.0

    # ── Save correctly_answered.json ──────────────────────────────────────
    correct_path = Path(output_dir) / "correctly_answered.json"
    with open(correct_path, "w", encoding="utf-8") as f:
        json.dump({
            "model"                  : MODEL,
            "evaluated_using_folder" : hindi_folder,
            "total_questions"        : len(questions),
            "total_attempted"        : attempted,
            "total_correct"          : len(correct_list),
            "accuracy_percent"       : round(accuracy, 2),
            "skipped"                : len(skipped),
            "questions"              : correct_list,
        }, f, ensure_ascii=False, indent=2)

    # ── Save wrongly_answered.json ────────────────────────────────────────
    wrong_path = Path(output_dir) / "wrongly_answered.json"
    with open(wrong_path, "w", encoding="utf-8") as f:
        json.dump({
            "model"                  : MODEL,
            "evaluated_using_folder" : hindi_folder,
            "total_questions"        : len(questions),
            "total_attempted"        : attempted,
            "total_wrong"            : len(wrong_list),
            "accuracy_percent"       : round(accuracy, 2),
            "skipped"                : len(skipped),
            "questions"              : wrong_list,
        }, f, ensure_ascii=False, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {MODEL}")
    print(f"{'-'*60}")
    print(f"  Questions generated from : English folder (echapter)")
    print(f"  Questions evaluated with : Hindi folder  ({hindi_folder})")
    print(f"{'-'*60}")
    print(f"  Total questions          : {len(questions)}")
    print(f"  Attempted                : {attempted}")
    print(f"  Correct                  : {len(correct_list)}")
    print(f"  Wrong                    : {len(wrong_list)}")
    print(f"  Skipped                  : {len(skipped)}")
    print(f"  Accuracy                 : {accuracy:.2f}%")
    print(f"{'-'*60}")
    print(f"  Correct file  -> {correct_path}")
    print(f"  Wrong file    -> {wrong_path}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on QA using Hindi images (DeepInfra)."
    )
    p.add_argument(
        "--qa_file",
        required=True,
        help="Absolute path to verified_bloom_QA.json",
    )
    p.add_argument(
        "--base_dir",
        required=True,
        help="Parent directory that contains both echapter/ and hechapter/ subfolders",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Where to save correctly_answered.json and wrongly_answered.json",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.qa_file).exists():
        print(f"[ERROR] QA file not found: {args.qa_file}"); sys.exit(1)
    if not Path(args.base_dir).exists():
        print(f"[ERROR] Base directory not found: {args.base_dir}"); sys.exit(1)

    # ── Discover the two subfolders ───────────────────────────────────────
    inner_folders = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])

    if len(inner_folders) != 2:
        print(f"[ERROR] Expected exactly 2 subfolders in '{args.base_dir}', "
              f"found {len(inner_folders)}: {inner_folders}")
        sys.exit(1)

    # ── Identify which is English, which is Hindi ─────────────────────────
    english_folder, hindi_folder = identify_english_folder(
        inner_folders[0], inner_folders[1]
    )
    print(f"[INFO] English folder (generation) : {english_folder}")
    print(f"[INFO] Hindi folder   (evaluation) : {hindi_folder}")

    # ── Load all questions ────────────────────────────────────────────────
    questions = load_all_questions(args.qa_file)
    if not questions:
        print("[ERROR] No questions found in QA file."); sys.exit(1)

    evaluate(questions, args.base_dir, hindi_folder, args.output_dir)


if __name__ == "__main__":
    main()
