
import json
import getpass
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://kimhannom.fit.hcmus.edu.vn/api/web/clc-sinonom"

INPUT_JSON = Path("pilot/api_output/hn_001_response.json")
OUTPUT_DIR = Path("pilot/api_output")

NORMALIZE_URL = f"{BASE_URL}/unicode-normalize"
TRANSLITERATION_URL = f"{BASE_URL}/sinonom-transliteration"

# Custom User-Agent: do not use the default python-requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/158.0.0.0 Safari/537.36"
)


def check_response(response, step_name):
    """Check HTTP status and API-level success."""
    print(f"\n[{step_name}] HTTP status: {response.status_code}")
    response.raise_for_status()

    try:
        result = response.json()
    except ValueError:
        raise RuntimeError(
            f"{step_name}: API không trả về JSON.\n"
            f"Response: {response.text[:1000]}"
        )

    if not result.get("is_success"):
        raise RuntimeError(
            f"{step_name} thất bại: "
            f"code={result.get('code')}, "
            f"message={result.get('message')}"
        )

    return result


def extract_ocr_lines(data):
    """Extract OCR text lines from the saved image-ocr response."""
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        payload = data["data"]
    elif isinstance(data, dict) and "result_ocr_text" in data:
        payload = data
    else:
        raise KeyError("JSON OCR không có trường 'data' hoặc 'result_ocr_text'.")

    ocr_text = payload.get("result_ocr_text")

    if ocr_text is None:
        raise KeyError("Không tìm thấy result_ocr_text.")

    if isinstance(ocr_text, list):
        lines = [str(item) for item in ocr_text]
    elif isinstance(ocr_text, str):
        lines = ocr_text.splitlines()
    else:
        raise TypeError(
            "result_ocr_text không phải list hoặc string: "
            f"{type(ocr_text).__name__}"
        )

    lines = [line for line in lines if line.strip()]

    if not lines:
        raise ValueError("Không có dòng OCR nào để xử lý.")

    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load saved OCR result
    # --------------------------------------------------------

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file OCR: {INPUT_JSON.resolve()}"
        )

    with INPUT_JSON.open("r", encoding="utf-8") as file:
        ocr_response = json.load(file)

    
    has_ocr_data = bool(ocr_response.get("result_ocr_text")) or bool(
    (ocr_response.get("details") or {}).get("details"))
    
    if not has_ocr_data:
        raise RuntimeError("File OCR không có dữ liệu nhận diện hợp lệ.")

    original_lines = extract_ocr_lines(ocr_response)

    print(f"[OCR] Đã đọc {len(original_lines)} dòng từ file JSON.")

    # --------------------------------------------------------
    # 2. Authentication
    # --------------------------------------------------------

    token = getpass.getpass("Nhập Kim API Bearer token: ").strip()

    if not token:
        raise ValueError("Token không được để trống.")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    session = requests.Session()
    session.headers.update(headers)

    # --------------------------------------------------------
    # 3. Unicode normalization
    # --------------------------------------------------------

    print("\n[1/2] Đang gọi unicode-normalize...")

    normalize_payload = {
        "texts": original_lines
    }

    normalize_http = session.post(
        NORMALIZE_URL,
        json=normalize_payload,
        timeout=60,
    )

    normalize_response = check_response(
        normalize_http,
        "unicode-normalize",
    )

    normalize_items = normalize_response["data"]["texts"]

    if len(normalize_items) != len(original_lines):
        raise RuntimeError(
            "Số dòng trả về từ unicode-normalize không khớp "
            "với số dòng OCR."
        )

    normalized_lines = [
        item["normalized"] for item in normalize_items
    ]

    print(
        f"[unicode-normalize] Thành công: "
        f"{len(normalized_lines)} dòng."
    )

    # Save normalization response
    normalize_output = OUTPUT_DIR / "hn_001_normalize_response.json"

    with normalize_output.open("w", encoding="utf-8") as file:
        json.dump(
            normalize_response,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 4. Sino-Nom transliteration
    # --------------------------------------------------------

    print("\n[2/2] Đang gọi sinonom-transliteration...")

    transliteration_payload = {
        "font_type": 1,
        "lang_type": 2,
        "text": "\n".join(normalized_lines),
    }

    transliteration_http = session.post(
        TRANSLITERATION_URL,
        json=transliteration_payload,
        timeout=60,
    )

    transliteration_response = check_response(
        transliteration_http,
        "sinonom-transliteration",
    )

    result_data = transliteration_response["data"]

    hannom_result = result_data["result_hannom_text"]
    transcription_result = result_data["result_text_transcription"]

    if len(hannom_result) != len(transcription_result):
        raise RuntimeError(
            "Số dòng Hán Nôm và Quốc ngữ trả về không khớp."
        )

    print(
        f"[transliteration] Thành công: "
        f"{len(transcription_result)} dòng."
    )

    # --------------------------------------------------------
    # 5. Save outputs
    # --------------------------------------------------------

    transliteration_output = (
        OUTPUT_DIR / "hn_001_transliteration_response.json"
    )

    with transliteration_output.open("w", encoding="utf-8") as file:
        json.dump(
            transliteration_response,
            file,
            ensure_ascii=False,
            indent=2,
        )

    normalized_text_output = OUTPUT_DIR / "hn_001_normalized.txt"

    with normalized_text_output.open("w", encoding="utf-8") as file:
        file.write("\n".join(normalized_lines))

    hannom_output = OUTPUT_DIR / "hn_001_transliterated_hannom.txt"

    with hannom_output.open("w", encoding="utf-8") as file:
        file.write("\n".join(hannom_result))

    transcription_output = (
        OUTPUT_DIR / "hn_001_quoc_ngu.txt"
    )

    with transcription_output.open("w", encoding="utf-8") as file:
        file.write("\n".join(transcription_result))

    # Save paired line alignment for inspection
    aligned_output = OUTPUT_DIR / "hn_001_aligned.txt"

    with aligned_output.open("w", encoding="utf-8") as file:
        for index, (han, quoc_ngu) in enumerate(
            zip(hannom_result, transcription_result),
            start=1,
        ):
            file.write(f"[Dòng {index}]\n")
            file.write(f"Hán Nôm: {han}\n")
            file.write(f"Quốc ngữ: {quoc_ngu}\n\n")

    print("\n========== PIPELINE COMPLETED ==========")
    print(f"Số dòng OCR: {len(original_lines)}")
    print(f"Số dòng normalized: {len(normalized_lines)}")
    print(f"Số dòng phiên âm: {len(transcription_result)}")
    print("\nCác file đã lưu:")

    for path in [
        normalize_output,
        transliteration_output,
        normalized_text_output,
        hannom_output,
        transcription_output,
        aligned_output,
    ]:
        print(f" - {path}")

    print("========================================")


if __name__ == "__main__":
    main()
