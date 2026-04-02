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

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "google/gemini-3-pro-preview")

# High-tier vision models for verification
VERIFICATION_MODELS = [
    "openai/gpt-5.2",
    "x-ai/grok-4.1-fast",
    "anthropic/claude-sonnet-4.5"
]

FOLDERS_TO_PROCESS = 25

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
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


def check_if_chart_or_table_exists(image_b64):
    """Checks if the image contains a chart, graph, or table."""
    prompt = (
        "Analyze this image carefully. Does it contain a data chart, graph, plot, or data table?\n"
        'Output ONLY a valid JSON object with a single boolean key "is_valid".\n'
        'Example: {"is_valid": true} or {"is_valid": false}'
    )
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=2500
        )
        content = response.choices[0].message.content
        parsed = extract_json_from_text(content)

        if parsed and "is_valid" in parsed:
            return bool(parsed["is_valid"])
    except Exception as e:
        print(f"Error during pre-check: {e}")

    # Default to True if the API fails so we don't skip good images
    return True


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
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            parsed_data = extract_json_from_text(content)

            if parsed_data and isinstance(parsed_data, list) and len(parsed_data) > 0:
                for idx, item in enumerate(parsed_data):
                    clean_name = re.sub(r'\W+', '', filename)
                    item['id'] = f"{clean_name}_q{idx + 1}"
                    item['image_reference'] = filename
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


def verify_question_with_models(image_b64, question_data):
    """Sends the generated question to 3 different models to verify correctness."""
    question_text = question_data.get("question")
    options = question_data.get("options")
    expected_answer = question_data.get("correct_answer", "")

    prompt = f"""
    Look at the chart/table in the provided image and solve the following multiple-choice question.
    Ignore any surrounding text in the image.

    Question: {question_text}
    Options: {options}

    Analyze the image carefully, determine the correct option, and provide your reasoning.
    You must output a valid JSON object matching this schema exactly:
    {{
        "selected_option": "The exact string of the option you chose from the list",
        "reasoning": "Your step-by-step logic to arrive at this answer"
    }}
    Output ONLY the JSON object.
    """

    models_correct = []
    models_wrong = []
    model_reasonings = {}
    model_selected_options = {}

    for model_id in VERIFICATION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content
            result = extract_json_from_text(content)

            if result and "selected_option" in result:
                chosen = str(result["selected_option"]).strip()
                model_reasonings[model_id] = result.get("reasoning", "No reasoning provided.")

                full_option = next(
                    (opt for opt in options if chosen.lower() in opt.lower() or opt.lower() in chosen.lower()),
                    chosen
                )
                model_selected_options[model_id] = full_option

                if expected_answer.lower() in chosen.lower() or chosen.lower() in expected_answer.lower():
                    models_correct.append(model_id)
                else:
                    models_wrong.append(model_id)
            else:
                models_wrong.append(model_id)
                model_reasonings[model_id] = "Failed to return valid JSON."
                model_selected_options[model_id] = "No valid option returned."

        except Exception as e:
            print(f"Verification failed for {model_id}: {e}")
            models_wrong.append(model_id)
            model_reasonings[model_id] = f"API Error: {str(e)}"
            model_selected_options[model_id] = f"API Error: {str(e)}"

    score_str = f"{len(models_correct)}/{len(VERIFICATION_MODELS)}"

    return {
        "score": score_str,
        "models_correct": models_correct,
        "models_wrong": models_wrong,
        "model_reasonings": model_reasonings,
        "model_selected_options": model_selected_options
    }


# ================= MAIN EXECUTION =================

def main():
    if not os.path.exists(BASE_IMAGE_DIR):
        print(f"Error: Directory '{BASE_IMAGE_DIR}' not found.")
        return

    print("Starting Processing (Up to 10 valid images in the first folder)...")

    all_subdirs = [d for d in os.listdir(BASE_IMAGE_DIR) if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))]
    all_subdirs.sort()

    target_folders = all_subdirs[:FOLDERS_TO_PROCESS]

    for folder_name in target_folders:
        parent_path = os.path.join(BASE_IMAGE_DIR, folder_name)

        inner_folders = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
        inner_folders.sort()

        if len(inner_folders) < 2:
            print(f"Skipping '{folder_name}': Needs at least 2 subfolders.")
            continue

        folder1_path = os.path.join(parent_path, inner_folders[0])
        folder2_path = os.path.join(parent_path, inner_folders[1])

        files1 = set(f for f in os.listdir(folder1_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        files2 = set(f for f in os.listdir(folder2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

        common_files = list(files1.intersection(files2))
        common_files.sort()

        if not common_files:
            continue

        print(f"\nProcessing Parent Folder: {folder_name}")
        parent_qa_data = []
        valid_processed_count = 0

        for img_file in common_files:
            print(f"   Found matching image pair: {img_file}")

            img1_path = os.path.join(folder1_path, img_file)
            img2_path = os.path.join(folder2_path, img_file)

            img_b64_check = resize_and_encode_image(img1_path)
            if not img_b64_check:
                continue

            print(f"      Checking if {img_file} contains a chart or table...")
            is_valid_image = check_if_chart_or_table_exists(img_b64_check)

            if not is_valid_image:
                print(f"      No chart/table detected. Deleting {img_file} from both directories.")
                try:
                    if os.path.exists(img1_path): os.remove(img1_path)
                    if os.path.exists(img2_path): os.remove(img2_path)
                except Exception as e:
                    print(f"      Failed to delete files: {e}")
                continue

            for folder_path, sub_folder_name in [(folder1_path, inner_folders[0]), (folder2_path, inner_folders[1])]:
                img_path = os.path.join(folder_path, img_file)
                lang_ctx = "Hindi" if 'hin' in img_path.lower() else "English"

                img_b64 = resize_and_encode_image(img_path)
                if not img_b64:
                    continue

                print(f"      Generating QA for {sub_folder_name}/{img_file}...")
                generated_data = generate_questions_for_image(img_b64, img_file, lang_ctx)

                if generated_data:
                    print(f"      Verifying {len(generated_data)} questions with consensus models...")
                    for q_item in generated_data:
                        q_item["source_subfolder"] = sub_folder_name
                        verification_results = verify_question_with_models(img_b64, q_item)
                        q_item["verification"] = verification_results

                    parent_qa_data.extend(generated_data)

                time.sleep(2)

            valid_processed_count += 1
            print(f"   Successfully processed {valid_processed_count}/10 valid pairs.")

            if valid_processed_count >= 10:
                print("   Reached 10 valid image pairs. Stopping search in this folder.")
                break

        if parent_qa_data:
            output_path = os.path.join(parent_path, "verified_bloom_QA_10_images.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(parent_qa_data, f, indent=2, ensure_ascii=False)
                print(f"Saved verified questions to: {output_path}")
            except Exception as e:
                print(f"Error saving JSON: {e}")

        print("\nStopping process completely after the first folder, as requested.")
        break

    print("\nRun completed!")


if __name__ == "__main__":
    main()
