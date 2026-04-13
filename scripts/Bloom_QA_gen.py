import os
import json
import base64
import time
import re
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIGURATION =================
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

BASE_IMAGE_DIR        = os.environ.get("BASE_IMAGE_DIR", "images")
MODEL                 = os.environ.get("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
FOLDERS_TO_PROCESS    = int(os.environ.get("FOLDERS_TO_PROCESS", "25"))
MAX_IMAGES_PER_FOLDER = int(os.environ.get("MAX_IMAGES_PER_FOLDER", "10"))

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=DEEPINFRA_API_KEY,
)


# ================= HELPER FUNCTIONS =================

def resize_and_encode_image(image_path, max_size=(1024, 1024)):
    """Resizes image and encodes to base64."""
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


def extract_json_from_text(text):
    """Robust JSON extractor."""
    try:
        return json.loads(text)
    except Exception:
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


def identify_english_folder(folder1_name: str, folder2_name: str) -> tuple[str, str]:
    """
    Given two subfolder names, returns (english_folder_name, hindi_folder_name).

    Detection logic (in priority order):
      1. If a folder name contains 'hin' (case-insensitive) → it is Hindi
      2. If a folder name contains 'eng' or starts with 'e'  → it is English
      3. Alphabetical fallback: the first folder alphabetically is treated as English
         (echapter < hechapter, so echapter = English)

    The main folder is guaranteed to have exactly two subfolders.
    """
    hindi_keywords   = ['hin', 'hindi']
    english_keywords = ['eng', 'english']

    f1_lower = folder1_name.lower()
    f2_lower = folder2_name.lower()

    # Check folder 1 for Hindi markers
    if any(kw in f1_lower for kw in hindi_keywords):
        return folder2_name, folder1_name   # folder2 = English, folder1 = Hindi

    # Check folder 2 for Hindi markers
    if any(kw in f2_lower for kw in hindi_keywords):
        return folder1_name, folder2_name   # folder1 = English, folder2 = Hindi

    # Check folder 1 for explicit English markers
    if any(kw in f1_lower for kw in english_keywords):
        return folder1_name, folder2_name

    # Check folder 2 for explicit English markers
    if any(kw in f2_lower for kw in english_keywords):
        return folder2_name, folder1_name

    # Alphabetical fallback — echapter < hechapter so first = English
    sorted_folders = sorted([folder1_name, folder2_name])
    print(f"  [WARN] Could not detect language from folder names "
          f"'{folder1_name}' / '{folder2_name}'. "
          f"Defaulting to alphabetical order: '{sorted_folders[0]}' = English.")
    return sorted_folders[0], sorted_folders[1]


def check_if_chart_or_table_exists(image_b64):
    """Checks if the image contains a chart, graph, or table. Returns (bool, reason)."""
    prompt = (
        "Analyze this image carefully. Does it contain a data chart, graph, plot, or data table?\n"
        'Output ONLY a valid JSON object:\n'
        '{"is_valid": true/false, "reason": "brief explanation"}'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content
        parsed  = extract_json_from_text(content)

        if parsed and "is_valid" in parsed:
            return bool(parsed["is_valid"]), parsed.get("reason", "")
    except Exception as e:
        print(f"Error during pre-check: {e}")

    return True, "API check failed, defaulting to valid"


def generate_questions_for_image(image_b64, filename, language_context):
    """Generates 5 English questions based on Bloom's Taxonomy."""

    system_prompt = (
        "Your goal is to create high-quality "
        "multiple-choice assessment questions based ONLY on the data tables or charts in the images.\n\n"
        "CRITICAL RULE: The image may contain paragraphs of text around the table or chart. "
        "You MUST completely IGNORE all surrounding text. Base your questions, options, and reasoning "
        "STRICTLY on the numbers, labels, and data inside the table or chart itself.\n\n"
        "Generate exactly 5 questions based on Bloom's Taxonomy levels to test different cognitive skills:\n"
        "1. Remembering: Recall specific facts or data points directly from the image.\n"
        "2. Understanding: Explain or interpret what a specific trend or data subset means.\n"
        "3. Applying: Use the data provided to calculate a new metric or solve a practical problem.\n"
        "4. Analyzing: Compare, contrast, or break down the data to find relationships or anomalies.\n"
        "5. Evaluating: Make a judgment or assess a conclusion based on the overall data evidence.\n\n"
        "Ensure the questions and options are challenging and cannot be guessed blindly."
    )

    user_prompt = f"""
    The provided image contains a chart/table with context in {language_context}.
    Generate exactly 5 multiple-choice questions in ENGLISH representing the first 5 levels of Bloom's Taxonomy.
    Remember: ONLY use the data from the chart/table. Ignore all other text.

    Structure your response as a valid JSON Array:
    [
        {{
            "id": "unique_id",
            "image_reference": "{filename}",
            "taxonomy_level": "Remembering/Understanding/Applying/Analyzing/Evaluating",
            "question": "The question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option X",
            "reasoning": "Detailed step-by-step reasoning used to arrive at the answer."
        }}
    ]

    Output ONLY the JSON array.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                temperature=0.7,
            )

            content     = response.choices[0].message.content
            parsed_data = extract_json_from_text(content)

            if parsed_data and isinstance(parsed_data, list) and len(parsed_data) > 0:
                for idx, item in enumerate(parsed_data):
                    clean_name = re.sub(r'\W+', '', filename)
                    item['id']               = f"{clean_name}_q{idx + 1}"
                    item['image_reference']  = filename
                    item['original_language'] = language_context
                return parsed_data
            else:
                print(f"Invalid JSON (Attempt {attempt + 1}). Retrying...")
                time.sleep(1)

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"API Error: {e}")
                return []

    return []


def verify_questions_bundled(image_b64, questions):
    """
    Verifies all questions for an image in a single API call.
    Returns a list of verification results, one per question.
    """
    questions_block = ""
    for i, q in enumerate(questions):
        options_str = json.dumps(q["options"])
        questions_block += (
            f"\nQuestion {i+1} (id: {q['id']}):\n"
            f"  Question: {q['question']}\n"
            f"  Options: {options_str}\n"
            f"  Claimed correct answer: {q['correct_answer']}\n"
            f"  Claimed reasoning: {q['reasoning']}\n"
        )

    system_prompt = (
        "Your job is to independently verify "
        "multiple-choice questions about charts and tables.\n\n"
        "For each question:\n"
        "1. Look at the image data independently - ignore the claimed answer.\n"
        "2. Solve the question yourself step-by-step.\n"
        "3. Check if the claimed correct answer matches your answer.\n"
        "4. Check if any other option could also be valid (ambiguity).\n"
        "5. Check if the reasoning provided is sound.\n"
        "6. If the question does not use any information from the chart or table, flag it.\n"
        "7. If the question does not fall into the correct taxonomy level category, flag it.\n"
    )

    user_prompt = f"""
    Review the following {len(questions)} questions about the chart/table in the image.
    Ignore any surrounding text in the image - focus only on the data.

    {questions_block}

    For each question, output your verification. Return a JSON array:
    [
        {{
            "question_id": "the id from above",
            "your_answer": "The exact option string you believe is correct",
            "agrees_with_claimed": true/false,
            "is_ambiguous": true/false,
            "reasoning": "Your step-by-step reasoning",
            "using_chart_or_table": true/false,
            "does_taxonomy_level_match": true/false
        }}
    ]

    Output ONLY the JSON array.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
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
            parsed  = extract_json_from_text(content)

            if parsed and isinstance(parsed, list):
                return parsed
            else:
                print(f"Invalid verification JSON (Attempt {attempt + 1}). Retrying...")
                time.sleep(1)

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Verification API Error: {e}")
                return []

    return []


def attach_verification(questions, verification_results):
    """Attaches verification results to the corresponding questions."""
    verification_map = {v.get("question_id", ""): v for v in verification_results}

    for q in questions:
        v = verification_map.get(q["id"], {})
        q["verification"] = {
            "verifier_answer"          : v.get("your_answer", ""),
            "agrees_with_claimed"      : v.get("agrees_with_claimed", None),
            "is_ambiguous"             : v.get("is_ambiguous", None),
            "verifier_reasoning"       : v.get("reasoning", ""),
            "using_chart_or_table"     : v.get("using_chart_or_table", None),
            "does_taxonomy_level_match": v.get("does_taxonomy_level_match", None),
        }


# ================= MAIN EXECUTION =================

def main():
    if not os.path.exists(BASE_IMAGE_DIR):
        print(f"Error: Directory '{BASE_IMAGE_DIR}' not found.")
        return

    print(f"Starting Processing (model: {MODEL})...")

    all_subdirs = [d for d in os.listdir(BASE_IMAGE_DIR)
                   if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))]
    all_subdirs.sort()

    target_folders = all_subdirs[:FOLDERS_TO_PROCESS]

    for folder_name in target_folders:
        parent_path = os.path.join(BASE_IMAGE_DIR, folder_name)

        inner_folders = [d for d in os.listdir(parent_path)
                         if os.path.isdir(os.path.join(parent_path, d))]
        inner_folders.sort()

        # ── Must have exactly 2 subfolders ────────────────────────────────
        if len(inner_folders) != 2:
            print(f"Skipping '{folder_name}': Expected exactly 2 subfolders, "
                  f"found {len(inner_folders)}.")
            continue

        # ── Identify which subfolder is English, which is Hindi ───────────
        english_folder_name, hindi_folder_name = identify_english_folder(
            inner_folders[0], inner_folders[1]
        )

        english_folder_path = os.path.join(parent_path, english_folder_name)
        hindi_folder_path   = os.path.join(parent_path, hindi_folder_name)

        print(f"\nProcessing Parent Folder : {folder_name}")
        print(f"  English folder (use)   : {english_folder_name}")
        print(f"  Hindi folder  (skip)   : {hindi_folder_name}")

        # ── Collect image filenames from both folders ─────────────────────
        english_files = set(
            f for f in os.listdir(english_folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        hindi_files = set(
            f for f in os.listdir(hindi_folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

        # ── Only proceed with images present in BOTH folders ──────────────
        common_files = sorted(english_files.intersection(hindi_files))

        if not common_files:
            print(f"  No common images found between the two folders. Skipping.")
            continue

        print(f"  Common images found    : {len(common_files)}")

        parent_qa_data    = []
        skipped_images    = []
        valid_processed_count = 0

        for img_file in common_files:
            print(f"\n   Image pair found : {img_file}")
            print(f"   Using            : {english_folder_name}/{img_file}")
            print(f"   Ignoring         : {hindi_folder_name}/{img_file}")

            # ── Use only the English image path ───────────────────────────
            english_img_path = os.path.join(english_folder_path, img_file)

            img_b64 = resize_and_encode_image(english_img_path)
            if not img_b64:
                skipped_images.append({"image": img_file, "reason": "encoding failed"})
                continue

            # ── Check if image actually contains a chart or table ─────────
            print(f"      Checking if {img_file} contains a chart or table...")
            is_valid, skip_reason = check_if_chart_or_table_exists(img_b64)

            if not is_valid:
                print(f"      Skipping {img_file}: {skip_reason}")
                skipped_images.append({"image": img_file, "reason": skip_reason})
                continue

            # ── Generate questions using the English image only ───────────
            print(f"      Generating QA for {english_folder_name}/{img_file}...")
            generated_data = generate_questions_for_image(img_b64, img_file, "English")

            if generated_data:
                for q_item in generated_data:
                    q_item["source_subfolder"] = english_folder_name

                # ── Verify all questions in one API call ──────────────────
                print(f"      Verifying {len(generated_data)} questions...")
                verification_results = verify_questions_bundled(img_b64, generated_data)
                attach_verification(generated_data, verification_results)

                parent_qa_data.extend(generated_data)

            time.sleep(2)

            valid_processed_count += 1
            print(f"   Processed {valid_processed_count}/{MAX_IMAGES_PER_FOLDER} valid images.")

            if valid_processed_count >= MAX_IMAGES_PER_FOLDER:
                print(f"   Reached limit of {MAX_IMAGES_PER_FOLDER}. Moving to next folder.")
                break

        # ── Save results ──────────────────────────────────────────────────
        if parent_qa_data:
            output_path = os.path.join(parent_path, "verified_bloom_QA.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(parent_qa_data, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(parent_qa_data)} verified questions to: {output_path}")
            except Exception as e:
                print(f"Error saving JSON: {e}")

        if skipped_images:
            skip_path = os.path.join(parent_path, "skipped_images.json")
            try:
                with open(skip_path, "w", encoding="utf-8") as f:
                    json.dump(skipped_images, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(skipped_images)} skipped images to: {skip_path}")
            except Exception as e:
                print(f"Error saving skipped images log: {e}")

    print("\nRun completed!")


if __name__ == "__main__":
    main()
