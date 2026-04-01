import os
import requests
import difflib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import urllib3

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
START_URL = "https://wb.gov.in/site-map.aspx"
DOWNLOAD_FOLDER = "Bengali_docs"
ALLOWED_DOMAINS = ["www.wb.gov.in", "wb.gov.in"]
MAX_DEPTH = 4


def get_similarity(a, b):
    """Calculates how similar two strings are (0.0 to 1.0)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_pairs_smartly(soup, page_url):
    """
    Finds pairs using HTML structure first, then falls back to fuzzy filename matching.
    """
    pairs = []
    processed_urls = set()

    # --- STRATEGY 1: HTML Structure (High Confidence) ---
    # Look for table rows or list items that contain exactly 2 links
    containers = soup.find_all(['tr', 'li', 'div'])

    for container in containers:
        # Find all PDF links strictly inside this container (direct children or shallow depth)
        links = container.find_all('a', href=True)
        pdf_links = []

        for link in links:
            full_url = urljoin(page_url, link['href'])
            if full_url.lower().endswith(".pdf") and full_url not in processed_urls:
                pdf_links.append(full_url)

        # If we found exactly 2 PDFs in one row/item, they are 99% likely a pair (Eng + Hin)
        if len(pdf_links) == 2:
            url1, url2 = pdf_links[0], pdf_links[1]

            # formatting for visual check
            name1 = url1.split('/')[-1]
            name2 = url2.split('/')[-1]

            # double check they are somewhat similar (avoid pairing completely different files)
            if get_similarity(name1, name2) > 0.4:
                pairs.append((url1, url2, "Structural-Match"))
                processed_urls.add(url1)
                processed_urls.add(url2)

    # --- STRATEGY 2: Fuzzy Filename Matching (Fallback) ---
    # Collect all remaining unpaired PDF links
    all_links = soup.find_all('a', href=True)
    remaining_pdfs = []
    for link in all_links:
        full_url = urljoin(page_url, link['href'])
        if full_url.lower().endswith(".pdf") and full_url not in processed_urls:
            remaining_pdfs.append(full_url)

    # Compare every file against every other file
    remaining_pdfs.sort()  # Sorting helps close matches stay near each other

    while remaining_pdfs:
        current = remaining_pdfs.pop(0)
        curr_name = current.split("/")[-1]

        best_match = None
        best_score = 0.0

        for candidate in remaining_pdfs:
            cand_name = candidate.split("/")[-1]
            score = get_similarity(curr_name, cand_name)

            # Threshold: Files must be >80% similar to be considered a pair
            # This catches "chapter1.pdf" vs "hchapter1.pdf" (score usually > 0.9)
            if score > 0.8 and score > best_score:
                best_score = score
                best_match = candidate

        if best_match:
            pairs.append((current, best_match, f"Fuzzy-Match ({int(best_score * 100)}%)"))
            processed_urls.add(current)
            processed_urls.add(best_match)
            remaining_pdfs.remove(best_match)
        else:
            # No pair found, treat as single
            # pairs.append((current, None, "Single")) # Uncomment to keep singles
            pass

    return pairs


def download_file(url, folder_path):
    if not url: return
    filename = url.split("/")[-1]
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath): return

    try:
        # print(f"      Downloading: {filename}...")
        r = requests.get(url, verify=False, stream=True, timeout=10)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    except:
        pass


def recursive_crawl():
    if not os.path.exists(DOWNLOAD_FOLDER): os.makedirs(DOWNLOAD_FOLDER)
    queue = [(START_URL, 0)]
    visited = set()
    headers = {'User-Agent': 'Mozilla/5.0'}

    while queue:
        current_url, depth = queue.pop(0)
        if current_url in visited: continue
        visited.add(current_url)

        # Domain Check
        if not any(d in urlparse(current_url).netloc for d in ALLOWED_DOMAINS): continue

        print(f"\n[{depth}] Scanning: {current_url}")

        try:
            response = requests.get(current_url, headers=headers, verify=False, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # FIND AND DOWNLOAD PAIRS
            file_pairs = find_pairs_smartly(soup, current_url)

            if file_pairs:
                page_folder = current_url.split("/")[-2] if len(current_url.split("/")) > 1 else "misc"
                save_dir = os.path.join(DOWNLOAD_FOLDER, page_folder)
                os.makedirs(save_dir, exist_ok=True)

                for p1, p2, method in file_pairs:
                    name1 = p1.split('/')[-1]
                    name2 = p2.split('/')[-1]
                    print(f"   >>> [{method}] {name1}  <-->  {name2}")
                    download_file(p1, save_dir)
                    download_file(p2, save_dir)

            # CRAWL DEEPER
            if depth < MAX_DEPTH:
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link['href']).split("#")[0]
                    if not full_url.lower().endswith((".pdf", ".jpg", ".zip")) and full_url not in visited:
                        if any(d in urlparse(full_url).netloc for d in ALLOWED_DOMAINS):
                            queue.append((full_url, depth + 1))

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    recursive_crawl()