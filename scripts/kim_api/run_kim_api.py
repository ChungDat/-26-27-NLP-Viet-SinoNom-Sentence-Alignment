import json
import getpass
import time
from pathlib import Path
import requests

BASE = "https://kimhannom.fit.hcmus.edu.vn"


ROOT = Path(__file__).resolve().parents[2]

IMAGE = ROOT / "pilot" / "images" / "han_nom" / "hn_001.png"

OUT = ROOT / "pilot" / "api_output"
OUT.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
    ),
    "Authorization": "Bearer " + getpass.getpass("Nhap API token: ")
}

s = requests.Session()
s.headers.update(headers)

def check(r):
    r.raise_for_status()
    data = r.json()
    if data.get("is_success") is not True or data.get("code") != "000000":
        raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2))
    return data["data"]

# 1. Upload
print("\n[1/3] UPLOAD IMAGE")

with IMAGE.open("rb") as f:
    uploaded = check(s.post(
        BASE + "/api/web/clc-sinonom/image-upload",
        files={"image_file": (IMAGE.name, f, "image/png")},
        timeout=120
    ))

file_name = uploaded["file_name"]
print("file_name:", file_name)

time.sleep(2)

# 2. Image OCR - dung truc tiep tham so da xac dinh
print("\n[2/2] IMAGE OCR")

time.sleep(2)

payload = {
    "ocr_id": 1,
    "lang_type": 2,
    "reading_direction": 1,
    "file_name": file_name
}

print("OCR payload:", json.dumps(payload, ensure_ascii=False))

ocr = check(s.post(
    BASE + "/api/web/clc-sinonom/image-ocr",
    json=payload,
    timeout=180
))

(OUT / "hn_001_response.json").write_text(
    json.dumps(ocr, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

texts = ocr.get("result_ocr_text", [])

(OUT / "hn_001_ocr.txt").write_text(
    "\n".join(texts),
    encoding="utf-8"
)

print("\nOCR SUCCESS")
print("So dong:", len(texts))
print("\n".join(texts))
print("\nDa luu ket qua tai:", OUT.resolve())
