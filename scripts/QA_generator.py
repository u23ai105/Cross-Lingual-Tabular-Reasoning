# import os
# import json
# import base64
# import time
# import re
# from openai import OpenAI
# from tqdm import tqdm
# from PIL import Image
# import io
#
# # ================= CONFIGURATION =================
# OPENROUTER_API_KEY = "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"  # YOUR KEY
#
# BASE_IMAGE_DIR = "/Users/muzammilmohammad/Documents/CSAB/csab/Python/WikiMixQA/scripts/malayalam/final_results"
#
# # We use Gemini 2.0 Flash because it is the best free instruction-following VLM
# MODEL_ID = "openai/gpt-4o-2024-11-20"
#
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
# )
#
#
# # ================= HELPER FUNCTIONS =================
#
# def resize_and_encode_image(image_path, max_size=(1024, 1024)):
#     """Resizes image to max 1024x1024 and encodes to base64."""
#     try:
#         with Image.open(image_path) as img:
#             if img.mode in ('RGBA', 'P'):
#                 img = img.convert('RGB')
#             img.thumbnail(max_size)
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG", quality=85)
#             return base64.b64encode(buffered.getvalue()).decode('utf-8')
#     except Exception as e:
#         print(f"❌ Error processing image {image_path}: {e}")
#         return None
#
#
# def extract_json_from_text(text):
#     """Robust JSON extractor for chatty models."""
#     try:
#         # Fast path: try parsing directly
#         return json.loads(text)
#     except:
#         # Look for a list [ ... ]
#         # FIX: Removed the escape '\' before the closing ']'
#         match = re.search(r'\[.*]', text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except:
#                 pass
#     return None
#
#
# def generate_questions_for_image(image_b64, filename):
#     """
#     Asks the model to generate 5 questions for a specific image.
#     Returns a list of 5 question objects.
#     """
#
#     # Prompt designed to output the EXACT format your eval script needs
#     system_prompt = (
#         "You are an expert educational content creator. Analyze the provided image (chart, table, or diagram). "
#         "Generate exactly 5 multiple-choice questions based on the data in the image. "
#         "Return the output as a valid JSON Array."
#     )
#
#     user_prompt = f"""
#     Generate 5 multiple-choice questions for this image.
#
#     Each question must strictly follow this JSON structure:
#     {{
#         "id": "unique_id_here",
#         "image_reference": "{filename}",
#         "question": "The question text here?",
#         "options": ["Option A", "Option B", "Option C", "Option D"],
#         "correct_answer": "Option B",
#         "reasoning": "Brief explanation of why this is correct based on the image."
#     }}
#
#     Output ONLY the JSON array containing 5 such objects. Do not wrap in markdown code blocks.
#     """
#
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_ID,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": user_prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
#                     ]}
#                 ],
#                 temperature=0.7,  # Slightly creative to generate diverse questions
#             )
#
#             content = response.choices[0].message.content
#             parsed_data = extract_json_from_text(content)
#
#             if parsed_data and isinstance(parsed_data, list) and len(parsed_data) > 0:
#                 # Post-processing to ensure ID uniqueness
#                 for idx, item in enumerate(parsed_data):
#                     clean_name = re.sub(r'\W+', '', filename)  # Remove symbols for ID
#                     item['id'] = f"{clean_name}_q{idx + 1}"
#                     # Ensure image_reference matches exactly what get_image_path expects
#                     item['image_reference'] = filename
#                 return parsed_data
#             else:
#                 print(f"⚠️  Model output invalid JSON (Attempt {attempt + 1}). Retrying...")
#                 time.sleep(1)
#
#         except Exception as e:
#             if "429" in str(e):
#                 wait_time = (attempt + 1) * 20
#                 print(f"⏳ Rate limit. Waiting {wait_time}s...")
#                 time.sleep(wait_time)
#             else:
#                 print(f"❌ API Error: {e}")
#                 return []
#
#     return []
#
#
# # ================= MAIN EXECUTION =================
#
# def main():
#     if not os.path.exists(BASE_IMAGE_DIR):
#         print(f"❌ Error: Directory '{BASE_IMAGE_DIR}' not found.")
#         return
#
#     print(f"🚀 Starting Question Generation using {MODEL_ID}...")
#
#     # 1. Walk through all folders
#     for root, _, files in os.walk(BASE_IMAGE_DIR):
#         # Filter for images
#         image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#         if not image_files:
#             continue
#
#         print(f"\n📂 Processing folder: {root} ({len(image_files)} images found)")
#
#         folder_questions = []
#
#         # 2. Process each image
#         for img_file in tqdm(image_files, desc="Generating Q&A"):
#             img_path = os.path.join(root, img_file)
#
#             # Encode
#             img_b64 = resize_and_encode_image(img_path)
#             if not img_b64: continue
#
#             # Generate
#             generated_data = generate_questions_for_image(img_b64, img_file)
#
#             if generated_data:
#                 folder_questions.extend(generated_data)
#                 # print(f"   ✅ Generated {len(generated_data)} Qs for {img_file}")
#             else:
#                 print(f"   ❌ Failed to generate Qs for {img_file}")
#
#             # 🛑 RATE LIMIT PAUSE (Crucial for Free Tier)
#             # 10 seconds sleep between images to prevent 429 errors
#             time.sleep(10)
#
#         # 3. Save JSON in the specific folder
#         if folder_questions:
#             output_path = os.path.join(root, "gemini_gen_QA.json")
#             try:
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     json.dump(folder_questions, f, indent=2, ensure_ascii=False)
#                 print(f"💾 Saved {len(folder_questions)} questions to: {output_path}")
#             except Exception as e:
#                 print(f"❌ Error saving JSON: {e}")
#
#     print("\n✅ Generation Complete!")
#
#
# if __name__ == "__main__":
#     main()












# import os
# import json
# import base64
# import time
# import re
# from openai import OpenAI
# from tqdm import tqdm
# from PIL import Image
# import io
#
# # ================= CONFIGURATION =================
# OPENROUTER_API_KEY = "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"  # YOUR KEY
#
# BASE_IMAGE_DIR = "images"
#
# # Using GPT-4o (Paid Tier)
# MODEL_ID = "openai/gpt-4o-2024-11-20"
#
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
# )
#
#
# # ================= HELPER FUNCTIONS =================
#
# def resize_and_encode_image(image_path, max_size=(1024, 1024)):
#     """Resizes image to max 1024x1024 and encodes to base64."""
#     try:
#         with Image.open(image_path) as img:
#             if img.mode in ('RGBA', 'P'):
#                 img = img.convert('RGB')
#             img.thumbnail(max_size)
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG", quality=85)
#             return base64.b64encode(buffered.getvalue()).decode('utf-8')
#     except Exception as e:
#         print(f"❌ Error processing image {image_path}: {e}")
#         return None
#
#
# def extract_json_from_text(text):
#     """Robust JSON extractor for chatty models."""
#     try:
#         return json.loads(text)
#     except:
#         # Regex to find JSON array [ ... ]
#         match = re.search(r'\[.*]', text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except:
#                 pass
#     return None
#
#
# def generate_questions_for_image(image_b64, filename):
#     system_prompt = (
#         "You are an expert educational content creator. Analyze the provided image (chart, table, or diagram). "
#         "Generate exactly 5 multiple-choice questions based on the data in the image. "
#         "Return the output as a valid JSON Array."
#     )
#
#     user_prompt = f"""
#     Generate 5 multiple-choice questions for this image.
#
#     Each question must strictly follow this JSON structure:
#     {{
#         "id": "unique_id_here",
#         "image_reference": "{filename}",
#         "question": "The question text here?",
#         "options": ["Option A", "Option B", "Option C", "Option D"],
#         "correct_answer": "Option B",
#         "reasoning": "Brief explanation of why this is correct based on the image."
#     }}
#
#     Output ONLY the JSON array containing 5 such objects. Do not wrap in markdown code blocks.
#     """
#
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_ID,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": user_prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
#                     ]}
#                 ],
#                 temperature=0.7,
#             )
#
#             content = response.choices[0].message.content
#             parsed_data = extract_json_from_text(content)
#
#             if parsed_data and isinstance(parsed_data, list) and len(parsed_data) > 0:
#                 for idx, item in enumerate(parsed_data):
#                     clean_name = re.sub(r'\W+', '', filename)
#                     item['id'] = f"{clean_name}_q{idx + 1}"
#                     item['image_reference'] = filename
#                 return parsed_data
#             else:
#                 print(f"⚠️  Invalid JSON (Attempt {attempt + 1}). Retrying...")
#                 time.sleep(1)
#
#         except Exception as e:
#             if "429" in str(e):
#                 print(f"⏳ Rate limit hit. Waiting 5s...")
#                 time.sleep(5)
#             else:
#                 print(f"❌ API Error: {e}")
#                 return []
#     return []
#
#
# # ================= MAIN EXECUTION =================
#
# def main():
#     if not os.path.exists(BASE_IMAGE_DIR):
#         print(f"❌ Error: Directory '{BASE_IMAGE_DIR}' not found.")
#         return
#
#     print(f"🚀 Starting SINGLE IMAGE TEST using {MODEL_ID}...")
#
#     # 1. Walk through all folders
#     for root, _, files in os.walk(BASE_IMAGE_DIR):
#         image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#         if not image_files:
#             continue
#
#         print(f"\n📂 Scanning folder: {root}")
#
#         folder_questions = []
#
#         # 2. Process images until ONE succeeds
#         for img_file in tqdm(image_files, desc="Processing"):
#             img_path = os.path.join(root, img_file)
#
#             # Encode
#             img_b64 = resize_and_encode_image(img_path)
#             if not img_b64: continue
#
#             # Generate
#             generated_data = generate_questions_for_image(img_b64, img_file)
#
#             if generated_data:
#                 folder_questions.extend(generated_data)
#
#                 # --- SUCCESS: SAVE AND STOP ---
#                 print(f"\n✅ Success! Generated {len(generated_data)} questions for {img_file}")
#
#                 output_path = os.path.join(root, "gpt4o_gen_QA_TEST.json")
#                 try:
#                     with open(output_path, "w", encoding="utf-8") as f:
#                         json.dump(folder_questions, f, indent=2, ensure_ascii=False)
#                     print(f"💾 Saved TEST results to: {output_path}")
#                     print("🛑 Stopping execution immediately as requested.")
#                     return  # <--- EXITS THE ENTIRE SCRIPT
#                 except Exception as e:
#                     print(f"❌ Error saving JSON: {e}")
#                     return
#             else:
#                 print(f"   ❌ Failed to generate Qs for {img_file}")
#
#             # No sleep needed for paid tier single-file test
#
#     print("\n⚠️ No images were successfully processed.")
#
#
# if __name__ == "__main__":
#     main()


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
OPENROUTER_API_KEY = "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"
BASE_IMAGE_DIR = "/Users/muzammilmohammad/Documents/CSAB/csab/Python/WikiMixQA/scripts/hindi/final_results_hindi"
MODEL_ID = "google/gemini-3-pro-preview"
FOLDERS_TO_PROCESS = 10  # Limit to 10 folders as requested

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
        print(f"❌ Error processing image {image_path}: {e}")
        return None


def extract_json_from_text(text):
    """Robust JSON extractor."""
    try:
        return json.loads(text)
    except:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def generate_questions_for_image(image_b64, filename, language_context):
    """
    Generates 5 English questions based on a specific balanced difficulty taxonomy.
    """

    # We incorporate your 25% Direct Lookup requirement while keeping the rest challenging.
    system_prompt = (
        "You are an advanced Data Scientist. Your goal is to create high-quality "
        "assessment questions based on images of tables or charts. \n\n"
        "Follow this specific Taxonomy and Distribution for the 5 questions:\n"
        "1. DIRECT LOOKUP (25%): 'What is the value of [X] in [Y]?' Focus on the most detailed/small data.\n"
        "2. COMPARISON (25%): 'Which has higher/more [metric]?' Requires comparing 3+ variables.\n"
        "3. AGGREGATION (25%): 'What is the total/sum/average/difference?' Requires multi-step math.\n"
        "4. TREND/PATTERN (25%): 'Is [X] increasing or decreasing?' Identify the steepest change.\n\n"
        "CRITICAL: For non-lookup questions, it must be impossible to answer by looking at just one cell."
    )

    user_prompt = f"""
    The provided image is a chart/table in {language_context}. 
    Generate exactly 5 multiple-choice questions in ENGLISH. 

    Difficulty Guidelines:
    - Ensure a mix of questions based on the 25/25/25/25 distribution provided in the system prompt.
    - Even if the image text is in {language_context}, the output must be in ENGLISH.
    - For the Aggregation and Trend questions, ensure they are hard enough to challenge an AI model.

    Structure your response as a valid JSON Array:
    {{
        "id": "unique_id",
        "image_reference": "{filename}",
        "taxonomy_type": "Direct Lookup/Comparison/Aggregation/Trend",
        "question": "The question text?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option X",
        "reasoning": "Detailed logic or math used to find the answer from the image."
    }}

    Output ONLY the JSON array.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
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
                # Add unique IDs
                for idx, item in enumerate(parsed_data):
                    clean_name = re.sub(r'\W+', '', filename)
                    item['id'] = f"{clean_name}_q{idx + 1}"
                    item['image_reference'] = filename
                    item['original_language'] = language_context  # Metadata tag
                return parsed_data
            else:
                print(f"⚠️ Invalid JSON (Attempt {attempt + 1}). Retrying...")
                time.sleep(1)

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"⏳ Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ API Error: {e}")
                return []

    return []


# ================= MAIN EXECUTION =================

def main():
    if not os.path.exists(BASE_IMAGE_DIR):
        print(f"❌ Error: Directory '{BASE_IMAGE_DIR}' not found.")
        return

    print(f"🚀 Starting Processing on first {FOLDERS_TO_PROCESS} folders...")

    # Get list of subdirectories (the pairs)
    all_subdirs = [d for d in os.listdir(BASE_IMAGE_DIR) if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))]
    all_subdirs.sort()  # Sort to ensure consistent processing order

    # Limit to the first 10
    target_folders = all_subdirs[:FOLDERS_TO_PROCESS]

    processed_count = 0

    for folder_name in target_folders:
        folder_path = os.path.join(BASE_IMAGE_DIR, folder_name)

        # Find images in this folder
        files = os.listdir(folder_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            continue

        print(f"\n📂 Processing Pair Folder ({processed_count + 1}/{FOLDERS_TO_PROCESS}): {folder_name}")

        folder_qa_data = []

        for img_file in tqdm(image_files, desc="Generating Q&A"):
            img_path = os.path.join(folder_path, img_file)

            # Simple heuristic to determine language context for the prompt
            # Adjust keywords if your filenames are different (e.g., 'mal' vs 'eng')
            if 'hindi' in img_file.lower() or 'hin' in img_file.lower():
                lang_ctx = "Hindi"
            else:
                lang_ctx = "English"

            # Encode
            img_b64 = resize_and_encode_image(img_path)
            if not img_b64: continue

            # Generate Questions
            generated_data = generate_questions_for_image(img_b64, img_file, lang_ctx)

            if generated_data:
                folder_qa_data.extend(generated_data)

            # Short sleep to be safe with rate limits
            time.sleep(2)

        # Save JSON inside the specific folder
        if folder_qa_data:
            output_path = os.path.join(folder_path, "new_gen_QA.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(folder_qa_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved {len(folder_qa_data)} questions to: {output_path}")
            except Exception as e:
                print(f"❌ Error saving JSON: {e}")

        processed_count += 1

    print("\n✅ Completed processing 10 folders!")


if __name__ == "__main__":
    main()